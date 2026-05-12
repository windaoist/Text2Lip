import os
import sys
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
from facenet_pytorch import MTCNN
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lip_sync.models.text_to_viseme import TextToVisemeProcessor
from lip_sync.inference.echomimic_backend import EchoMimicBackend

# Global instances for workers
_worker_face_mesh = None
_worker_mtcnn = None
_worker_audio_processor = None

def init_worker():
    """Initialize heavy models once per process."""
    global _worker_face_mesh, _worker_mtcnn, _worker_audio_processor
    
    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    _worker_face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # MTCNN - Use CPU for workers if multiple processes are running to avoid GPU contention
    # or limit GPU memory if using CUDA. Here we default to CPU for robustness in parallel.
    device = torch.device('cpu') 
    _worker_mtcnn = MTCNN(image_size=64, margin=20, post_process=False, device=device)
    
    # Audio Processor (Whisper)
    backend = EchoMimicBackend()
    _worker_audio_processor = backend.audio_processor

def extract_faus_from_landmarks(landmarks):
    faus = np.zeros(16, dtype=np.float32)
    faus[0] = landmarks[14].y - landmarks[13].y
    faus[1] = landmarks[291].x - landmarks[61].x
    faus[2] = landmarks[145].y - landmarks[159].y
    faus[3] = landmarks[374].y - landmarks[386].y
    faus[4] = landmarks[1].y - ((landmarks[52].y + landmarks[65].y) / 2)
    faus[5] = landmarks[1].y - ((landmarks[282].y + landmarks[295].y) / 2)
    faus = faus * 10.0 
    return faus

def parse_align(align_path):
    words = []
    if not os.path.exists(align_path): return ""
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            word = parts[2]
            if word != 'sil': words.append(word)
    return " ".join(words)

def process_single_file(file_info):
    """Worker function to process one video/audio pair."""
    audio_path, align_path, video_path, output_path, fps, skip_factor = file_info
    
    try:
        # 1. Text
        text = parse_align(align_path)
        
        # 2. Audio Features
        raw_features = _worker_audio_processor.audio2feat(audio_path)
        whisper_chunks = _worker_audio_processor.feature2chunks(raw_features, fps=fps)
        whisper_features = whisper_chunks.reshape(whisper_chunks.shape[0], -1)
        
        # 3. Video & FAUs
        gt_frames = None
        fau_signals_list = []
        if os.path.exists(video_path):
            frames = []
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            
            last_faus = np.zeros(16, dtype=np.float32)
            last_face = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Frame Skipping Logic
                # Only run heavy detection every `skip_factor` frames
                should_compute = (frame_idx % skip_factor == 0)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe FAUs
                if should_compute:
                    results = _worker_face_mesh.process(frame_rgb)
                    if results.multi_face_landmarks:
                        last_faus = extract_faus_from_landmarks(results.multi_face_landmarks[0].landmark)
                    else:
                        last_faus = np.zeros(16, dtype=np.float32)
                fau_signals_list.append(last_faus)
                
                # MTCNN Face
                if should_compute:
                    pil_img = Image.fromarray(frame_rgb)
                    face = _worker_mtcnn(pil_img)
                    if face is not None:
                        last_face = (face / 127.5) - 1.0
                    else:
                        last_face = frames[-1] if len(frames) > 0 else torch.zeros(3, 64, 64) - 1.0
                
                frames.append(last_face if last_face is not None else torch.zeros(3, 64, 64) - 1.0)
                frame_idx += 1
                
            cap.release()
            
            if len(frames) > 0:
                gt_frames = torch.stack(frames)
                fau_signals = torch.tensor(np.array(fau_signals_list))
                
                # Align T_v with audio T_a
                T_a = whisper_features.shape[0]
                T_v = gt_frames.shape[0]
                if T_v > T_a:
                    gt_frames, fau_signals = gt_frames[:T_a], fau_signals[:T_a]
                elif T_v < T_a:
                    padding_f = gt_frames[-1].unsqueeze(0).repeat(T_a - T_v, 1, 1, 1)
                    gt_frames = torch.cat([gt_frames, padding_f], dim=0)
                    padding_s = fau_signals[-1].unsqueeze(0).repeat(T_a - T_v, 1)
                    fau_signals = torch.cat([fau_signals, padding_s], dim=0)
        
        # 4. Save
        save_data = {
            "text": text,
            "whisper_features": torch.from_numpy(whisper_features).float()
        }
        if gt_frames is not None:
            save_data["gt_frames"] = gt_frames
            save_data["fau_signals"] = fau_signals
            
        torch.save(save_data, output_path)
        return True
    except Exception as e:
        print(f"[-] Error processing {audio_path}: {e}")
        return False

def preprocess_speaker(speaker_id="s1", config_path="configs/text_driven_config.yaml", 
                       max_samples=None, num_workers=4, skip_factor=2):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    data_root = config['dataset']['grid_path']
    audio_dir = os.path.join(data_root, speaker_id)
    align_dir = os.path.join(data_root, "align")
    video_dir = os.path.join(data_root, speaker_id + "_video")
    output_dir = os.path.join(data_root, "preprocessed", speaker_id)
    os.makedirs(output_dir, exist_ok=True)
    
    fps = config['inference']['fps']
    
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    if max_samples:
        audio_files = audio_files[:max_samples]
        
    tasks = []
    for f in audio_files:
        base_name = os.path.splitext(f)[0]
        tasks.append((
            os.path.join(audio_dir, f),
            os.path.join(align_dir, base_name + ".align"),
            os.path.join(video_dir, base_name + ".mpg"),
            os.path.join(output_dir, base_name + ".pt"),
            fps,
            skip_factor
        ))
        
    print(f"[*] Starting multiprocess preprocessing with {num_workers} workers...")
    print(f"[*] Skip factor: {skip_factor} (compute every {skip_factor} frames)")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks)):
            if future.result():
                success_count += 1
                
    print(f"[*] Done. Successfully processed {success_count}/{len(tasks)} samples.")

if __name__ == "__main__":
    import argparse
    # multiprocessing support for Windows
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default="s1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip", type=int, default=2, help="Compute FAUs/Face every N frames")
    args = parser.parse_args()
    
    preprocess_speaker(speaker_id=args.speaker, max_samples=args.limit, 
                       num_workers=args.workers, skip_factor=args.skip)
