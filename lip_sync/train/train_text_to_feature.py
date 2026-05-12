import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lip_sync.models.text_to_viseme import TextToVisemeProcessor
from lip_sync.models.motion_gen import VisemeEncoder
from lip_sync.models.aux_renderer import AuxiliaryRenderer
from lip_sync.models.syncnet import PretrainedSyncNet
try:
    import lpips
except ImportError:
    pass

# Load config
def load_config(config_path="configs/text_driven_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class VisemeToFeatureDataset(Dataset):
    """
    Dataset class: Loads preprocessed .pt files from GRID dataset.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.processor = TextToVisemeProcessor()
        self.samples = []
        
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith(".pt"):
                    self.samples.append(os.path.join(data_dir, f))
        
        print(f"[*] Loaded {len(self.samples)} samples from {data_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        data = torch.load(sample_path)
        
        text = data["text"]
        target_feature = data["whisper_features"] # (T, 19200)
        
        viseme_ids = self.processor.process(text) # (N,)
        
        T = target_feature.shape[0]
        
        # Try loading real GT frames and FAUs; fallback to dummy if not preprocessed
        if "gt_frames" in data:
            gt_frames = data["gt_frames"] # Shape: (T, 3, 64, 64)
            if "fau_signals" in data:
                 fau_signals = data["fau_signals"] # Shape: (T, 16)
            else:
                 fau_signals = torch.zeros((T, 16), dtype=torch.float)
        else:
            gt_frames = torch.zeros((T, 3, 64, 64), dtype=torch.float)
            fau_signals = torch.zeros((T, 16), dtype=torch.float)
            
        return viseme_ids, target_feature, gt_frames, fau_signals

def collate_fn(batch):
    viseme_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]
    gt_frames_list = [item[2] for item in batch]
    fau_signals_list = [item[3] for item in batch]

    max_viseme_len = max(v.size(0) for v in viseme_list)

    padded_visemes = torch.zeros(len(batch), max_viseme_len, dtype=torch.long)
    viseme_padding_mask = torch.ones(len(batch), max_viseme_len, dtype=torch.bool)
    
    for i, v in enumerate(viseme_list):
        padded_visemes[i, :v.size(0)] = v
        viseme_padding_mask[i, :v.size(0)] = False # False means valid (not padded)

    max_target_len = max(t.size(0) for t in target_list)
    feature_dim = target_list[0].size(1)
    
    padded_targets = torch.zeros(len(batch), max_target_len, feature_dim, dtype=torch.float)
    padded_gt_frames = torch.zeros(len(batch), max_target_len, 3, 64, 64, dtype=torch.float)
    padded_faus = torch.zeros(len(batch), max_target_len, 16, dtype=torch.float)
    
    for i, t in enumerate(target_list):
        padded_targets[i, :t.size(0), :] = t
        padded_gt_frames[i, :gt_frames_list[i].size(0), :, :, :] = gt_frames_list[i]
        padded_faus[i, :fau_signals_list[i].size(0), :] = fau_signals_list[i]

    return padded_visemes, viseme_padding_mask, padded_targets, padded_gt_frames, padded_faus

def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting training on {device}")

    processor = TextToVisemeProcessor()
    model = VisemeEncoder(
        num_visemes=processor.vocab_size, 
        d_model=config['model']['d_model'], 
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        out_dim=config['model']['out_dim']
    ) 
    model.to(device)

    # Initialize Auxiliary modules and losses
    aux_renderer = AuxiliaryRenderer().to(device)
    sync_net = PretrainedSyncNet().to(device)
    try:
        import lpips
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    except Exception as e:
        print(f"[-] Warning: lpips failed to load ({e}). Will skip LPIPS loss.")
        lpips_loss_fn = None

    # Include the audio_adapter of PretrainedSyncNet in the optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + 
        list(aux_renderer.parameters()) + 
        list(sync_net.audio_adapter.parameters()), 
        lr=1e-4
    )
    criterion = nn.MSELoss()

    data_dir = config['dataset']['preprocessed_path']
    if not os.path.exists(data_dir):
        print(f"[-] Error: Preprocessed directory {data_dir} not found.")
        return

    dataset = VisemeToFeatureDataset(data_dir=data_dir)
    if len(dataset) == 0:
        print("[-] Error: Dataset is empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        aux_renderer.train()
        sync_net.audio_adapter.train() # train adapter, keep wav2lip frozen
        total_loss = 0
        total_loss_mse = 0
        total_loss_lpips = 0
        total_loss_sync = 0
        
        for visemes, viseme_padding_mask, targets, gt_frames, fau_signals in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            visemes = visemes.to(device)
            viseme_padding_mask = viseme_padding_mask.to(device)
            targets = targets.to(device)
            gt_frames = gt_frames.to(device)
            fau_signals = fau_signals.to(device)
            
            optimizer.zero_grad()
            B, T, _ = targets.shape # Actual target frame length
            
            # We now pass the real FAUs extracted during preprocessing
            outputs = model(visemes, target_frames_len=T, viseme_padding_mask=viseme_padding_mask, fau_signals=fau_signals)
            
            # Base MSE Loss
            loss_mse = criterion(outputs, targets)
            
            # Auxiliary Losses
            pred_frames = aux_renderer(outputs) # (B, T, 3, 64, 64)
            loss_sync = sync_net(pred_frames, targets)
            
            loss_lpips = 0.0
            if lpips_loss_fn is not None:
                B, T, C, H, W = pred_frames.shape
                pred_frames_flat = pred_frames.view(B*T, C, H, W)
                gt_frames_flat = gt_frames.view(B*T, C, H, W)
                loss_lpips = lpips_loss_fn(pred_frames_flat, gt_frames_flat).mean()
            
            # Combine losses
            alpha_lpips = 0.1
            beta_sync = 0.1
            loss = loss_mse + alpha_lpips * loss_lpips + beta_sync * loss_sync
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_mse += loss_mse.item()
            total_loss_sync += loss_sync.item()
            if isinstance(loss_lpips, torch.Tensor):
                total_loss_lpips += loss_lpips.item()
            else:
                total_loss_lpips += loss_lpips
        
        num_batches = len(dataloader)
        print(f"[*] Epoch {epoch+1} "
              f"Total Loss: {total_loss/num_batches:.4f} | "
              f"MSE: {total_loss_mse/num_batches:.4f} | "
              f"LPIPS: {total_loss_lpips/num_batches:.4f} | "
              f"SYNC: {total_loss_sync/num_batches:.4f}")

    save_path = config['paths']['text_model_weights']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[*] Model saved to: {save_path}")

if __name__ == "__main__":
    train()
