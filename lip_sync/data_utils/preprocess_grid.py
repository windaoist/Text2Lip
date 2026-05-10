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

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lip_sync.models.text_to_viseme import TextToVisemeProcessor
from lip_sync.inference.echomimic_backend import EchoMimicBackend


def load_config(config_path="configs/text_driven_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_align(align_path):
    """Parse .align file to get the full sentence text."""
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word = parts[2]
            if word != 'sil':
                words.append(word)
    return " ".join(words)


def preprocess_speaker(speaker_id="s1", config=None):
    if config is None:
        config = load_config()

    backend = EchoMimicBackend()
    if backend.audio_processor is None:
        print("[-] Error: Audio model not found, cannot extract features.")
        return

    # Initialize face detector for GT video extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=64, margin=20, post_process=False, device=device)

    data_root = config['dataset']['grid_path']
    audio_dir = os.path.join(data_root, speaker_id)
    align_dir = os.path.join(data_root, "align")
    # Typical pattern if videos are downloaded separately
    video_dir = os.path.join(data_root, speaker_id + "_video")

    output_dir = os.path.join(data_root, "preprocessed", speaker_id)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(audio_dir) or not os.path.exists(align_dir):
        print(
            f"[-] Error: Could not find audio dir {audio_dir} or align dir {align_dir}")
        return

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    print(f"[*] Preprocessing {len(audio_files)} files for {speaker_id}...")
    print(f"[*] Output directory: {output_dir}")

    count = 0
    for f in tqdm(audio_files):
        base_name = os.path.splitext(f)[0]
        audio_path = os.path.join(audio_dir, f)
        align_path = os.path.join(align_dir, base_name + ".align")

        # Guess potential video paths
        video_path = audio_path.replace(".wav", ".mpg")
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, base_name + ".mpg")

        if not os.path.exists(align_path):
            continue

        try:
            # 1. Extract text
            text = parse_align(align_path)

            # 2. Extract features
            raw_features = backend.audio_processor.audio2feat(audio_path)
            fps = config['inference']['fps']
            whisper_chunks = backend.audio_processor.feature2chunks(
                raw_features, fps=fps)
            # Flatten to (T, 19200) for training
            whisper_features = whisper_chunks.reshape(
                whisper_chunks.shape[0], -1)

            # 3. Extract Ground Truth Video Frames
            gt_frames = None
            if os.path.exists(video_path):
                frames = []
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    # Detect and crop face (returns tensor in [0, 255])
                    face = mtcnn(pil_img)
                    if face is not None:
                        # Normalize to [-1, 1]
                        face = (face / 127.5) - 1.0
                        frames.append(face)
                    else:
                        # Fallback blank frame if detection fails
                        blank = torch.zeros(3, 64, 64) - 1.0
                        frames.append(frames[-1] if len(frames) > 0 else blank)
                cap.release()

                if len(frames) > 0:
                    gt_frames = torch.stack(frames)  # (T_v, 3, 64, 64)

                    # Align T_v with audio T_a
                    T_a = whisper_features.shape[0]
                    T_v = gt_frames.shape[0]
                    if T_v > T_a:
                        gt_frames = gt_frames[:T_a]
                    elif T_v < T_a:
                        padding = gt_frames[-1].unsqueeze(
                            0).repeat(T_a - T_v, 1, 1, 1)
                        gt_frames = torch.cat([gt_frames, padding], dim=0)

            # 4. Save
            save_data = {
                "text": text,
                "whisper_features": torch.from_numpy(whisper_features).float()
            }
            if gt_frames is not None:
                save_data["gt_frames"] = gt_frames

            torch.save(save_data, os.path.join(output_dir, base_name + ".pt"))
            count += 1
        except Exception as e:
            print(f"[-] Error processing {f}: {e}")

    print(
        f"[*] Preprocessing complete. Successfully processed {count} samples.")


if __name__ == "__main__":
    preprocess_speaker("s1")