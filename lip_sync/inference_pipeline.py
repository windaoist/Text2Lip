import os
import sys
import torch
import cv2
import numpy as np
import yaml
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core components from the new modular structure
try:
    from lip_sync.models.text_to_viseme import TextToVisemeProcessor
    from lip_sync.models.motion_gen import VisemeEncoder
    from lip_sync.inference.echomimic_backend import EchoMimicBackend
    ECHOMIMIC_AVAILABLE = True
except ImportError as e:
    print(f"[-] Warning: Failed to load backend or models: {e}")
    ECHOMIMIC_AVAILABLE = False


def load_config(config_path="configs/text_driven_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_video_from_text(text, reference_image_path, output_path, config=None, progress_callback=None):
    """
    End-to-end text-driven video generation (Direct text-to-viseme-to-video).
    
    Args:
        progress_callback: Optional callable(percent, stage, message) for real-time progress
    """
    if config is None:
        config = load_config()

    print(f"\n[*] Starting text-driven pipeline...")
    print(f"[*] Input text: '{text}'")

    if not (ECHOMIMIC_AVAILABLE):
        print(f"[-] Error: EchoMimic backend is not available.")
        return None

    # ==========================================
    # Step 1: Text -> Visemes with Phoneme Duration Prediction
    # ==========================================
    if progress_callback:
        progress_callback(0, 'text_to_viseme', '正在将文本转换为口型序列...')
    
    print(f"[*] [Step 1] Processing text to visemes with duration prediction...")
    processor = TextToVisemeProcessor()
    
    # Strategy: Use process_with_duration() to compute the linguistically-
    # accurate target frame count, but keep the original (unexpanded) viseme
    # sequence as input. The VisemeEncoder's cross-attention decoder will
    # align N visemes to T target frames automatically.
    _, total_frames, frame_counts, phonemes = processor.process_with_duration(text)
    
    # Get base viseme IDs (unexpanded) for the model input
    # This is what the model was trained on — short viseme sequences
    base_viseme_ids = processor.process(text)
    
    # target_frames_len is the sum of per-phoneme durations
    target_frames_len = total_frames

    viseme_ids = base_viseme_ids.unsqueeze(0)  # (1, N) — short sequence
    estimated_seconds = target_frames_len / config['inference']['fps']
    old_frames = len(phonemes) * config['inference']['frames_per_viseme']
    print(
        f"[*] Phonemes: {phonemes}")
    print(
        f"[*] Frame counts per phoneme: {frame_counts}")
    print(
        f"[*] Viseme sequence length (input): {viseme_ids.shape[1]}")
    print(
        f"[*] Target frames (output): {target_frames_len} (estimated {estimated_seconds:.1f}s at {config['inference']['fps']}fps)")
    print(
        f"[*] (OLD method: {old_frames} frames = {old_frames / config['inference']['fps']:.1f}s)")

    # ==========================================
    # Step 2 & 3: Visemes -> Features -> Video
    # ==========================================
    if progress_callback:
        progress_callback(5, 'initializing_backend', '正在初始化视频生成引擎...')
    
    print(f"[*] [Step 2&3] Mapping features + Generating video...")
    backend = EchoMimicBackend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result_path = backend.generate_from_text(
        reference_image_path,
        viseme_ids,
        output_path,
        target_frames_len=target_frames_len,
        width=config['inference']['width'],
        height=config['inference']['height'],
        steps=config['inference']['steps'],
        cfg=config['inference']['cfg'],
        fps=config['inference']['fps'],
        seed=config['inference']['seed'],
        progress_callback=progress_callback
    )

    print(f"[*] [Success] Video generated: {result_path}")
    return result_path


if __name__ == "__main__":
    # Test with the user's example sentence
    input_text = "I am a text-driven talking head. No audio needed."
    ref_image = "data/reference/test_face.png"
    output_file = "output/text_driven_result/sample_video.mp4"

    generate_video_from_text(input_text, ref_image, output_file)
