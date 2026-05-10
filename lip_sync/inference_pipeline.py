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

def generate_video_from_text(text, reference_image_path, output_path, config=None):
    """
    End-to-end text-driven video generation (Direct text-to-viseme-to-video).
    """
    if config is None:
        config = load_config()

    print(f"\n[*] Starting text-driven pipeline...")
    print(f"[*] Input text: '{text}'")

    if not (ECHOMIMIC_AVAILABLE):
        print(f"[-] Error: EchoMimic backend is not available.")
        return None

    # ==========================================
    # Step 1: Text -> Visemes
    # ==========================================
    print(f"[*] [Step 1] Processing text to visemes...")
    processor = TextToVisemeProcessor()
    base_viseme_ids = processor.process(text)
    
    # Timing prediction (Current simple implementation: constant multiplier)
    frames_per_viseme = config['inference']['frames_per_viseme']
    target_frames_len = len(base_viseme_ids) * frames_per_viseme
        
    viseme_ids = base_viseme_ids.unsqueeze(0) # (1, N)
    print(f"[*] Viseme sequence length: {viseme_ids.shape[1]}, Target frames: {target_frames_len}")
    
    # ==========================================
    # Step 2 & 3: Visemes -> Features -> Video
    # ==========================================
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
        seed=config['inference']['seed']
    )
        
    print(f"[*] [Success] Video generated: {result_path}")
    return result_path

if __name__ == "__main__":
    input_text = "I am a text-driven talking head. No audio needed."
    ref_image = "data/reference/test_face.png"
    output_file = "output/text_driven_result/sample_video.mp4"
    
    generate_video_from_text(input_text, ref_image, output_file)
