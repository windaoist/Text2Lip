import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
from pathlib import Path
import argparse
from functools import lru_cache
import copy

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
    Optimized with caching for repeated access.
    """
    def __init__(self, data_dir, cache_size=256):
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

    @lru_cache(maxsize=256)
    def _load_sample(self, sample_path):
        """Cached sample loading to reduce I/O on repeated accesses."""
        return torch.load(sample_path)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        data = self._load_sample(sample_path)
        
        text = data["text"]
        target_feature = data["whisper_features"]  # (T, 19200)
        
        viseme_ids = self.processor.process(text)  # (N,)
        
        T = target_feature.shape[0]
        
        # Try loading real GT frames and FAUs; fallback to dummy if not preprocessed
        if "gt_frames" in data:
            gt_frames = data["gt_frames"]  # Shape: (T, 3, 64, 64)
            if "fau_signals" in data:
                fau_signals = data["fau_signals"]  # Shape: (T, 16)
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
        viseme_padding_mask[i, :v.size(0)] = False  # False means valid (not padded)

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


def get_args():
    parser = argparse.ArgumentParser(description="Train text-to-feature model with optimizations")
    parser.add_argument("--config", type=str, default="configs/text_driven_config.yaml",
                        help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32, adjust based on GPU VRAM)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers (default: 4)")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-lr", type=float, default=5e-4,
                        help="Max learning rate for OneCycle scheduler (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile for model acceleration (default: True)")
    parser.add_argument("--no-compile", action="store_false", dest="compile",
                        help="Disable torch.compile")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor (default: 4)")
    parser.add_argument("--cache-size", type=int, default=256,
                        help="Dataset sample cache size (default: 256)")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Early stopping patience (epochs, default: 10)")
    parser.add_argument("--matmul-precision", type=str, default="high",
                        choices=["highest", "high", "medium"],
                        help="torch matmul precision (default: high, enables Tensor Cores)")
    return parser.parse_args()


def build_models(config, processor, device):
    """Factory function to create fresh model instances."""
    model = VisemeEncoder(
        num_visemes=processor.vocab_size,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        out_dim=config['model']['out_dim']
    )
    model.to(device)

    aux_renderer = AuxiliaryRenderer().to(device)
    sync_net = PretrainedSyncNet().to(device)

    return model, aux_renderer, sync_net


def train():
    args = get_args()
    config = load_config(args.config)

    # ─── Precision optimization ───
    torch.set_float32_matmul_precision(args.matmul_precision)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting training on {device}")
    print(f"[*] Using float32 matmul precision: {args.matmul_precision}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ─── Model setup ───
    processor = TextToVisemeProcessor()
    model, aux_renderer, sync_net = build_models(config, processor, device)

    # ─── Try torch.compile, with full fallback on any failure ───
    compile_enabled = args.compile and hasattr(torch, 'compile')
    if compile_enabled:
        print("[*] Attempting to compile model with torch.compile (mode=reduce-overhead)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            aux_renderer = torch.compile(aux_renderer, mode="reduce-overhead")
            sync_net.audio_adapter = torch.compile(sync_net.audio_adapter, mode="reduce-overhead")
            print("[+] Model compilation registered. Will test on first forward pass.")
        except Exception as e:
            print(f"[-] torch.compile setup failed ({e}). Falling back to eager mode.")
            compile_enabled = False
            # Rebuild uncompiled models
            model, aux_renderer, sync_net = build_models(config, processor, device)

    try:
        import lpips
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    except Exception as e:
        print(f"[-] Warning: lpips failed to load ({e}). Will skip LPIPS loss.")
        lpips_loss_fn = None

    # ─── Optimizer ───
    optimizer = optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": aux_renderer.parameters(), "lr": args.lr},
            {"params": sync_net.audio_adapter.parameters(), "lr": args.lr * 0.5},
        ],
        lr=args.lr,
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()

    # ─── Dataset & DataLoader ───
    data_dir = config['dataset']['preprocessed_path']
    if not os.path.exists(data_dir):
        print(f"[-] Error: Preprocessed directory {data_dir} not found.")
        return

    dataset = VisemeToFeatureDataset(data_dir=data_dir, cache_size=args.cache_size)
    if len(dataset) == 0:
        print("[-] Error: Dataset is empty.")
        return

    temp_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    steps_per_epoch = len(temp_loader)
    del temp_loader

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"[*] Dataset size: {len(dataset)} samples")
    print(f"[*] Batch size: {args.batch_size}")
    print(f"[*] Gradient accumulation steps: {args.accum_steps}")
    print(f"[*] Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"[*] Steps per epoch: {steps_per_epoch}")
    effective_steps = math.ceil(steps_per_epoch / args.accum_steps) if args.accum_steps > 1 else steps_per_epoch
    print(f"[*] Effective steps per epoch: {effective_steps}")

    # ─── Learning Rate Scheduler ───
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.max_lr, args.max_lr, args.max_lr * 0.5],
        steps_per_epoch=effective_steps,
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # ─── Mixed Precision (AMP) ───
    use_amp = device.type == "cuda"
    try:
        if use_amp:
            scaler = torch.amp.GradScaler('cuda')
        else:
            # Plain GradScaler with amp disabled for CPU or non-CUDA devices
            scaler = torch.cuda.amp.GradScaler(enabled=False)
    except (AttributeError, RuntimeError):
        # Fallback for environments where torch.cuda.amp is unavailable
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        use_amp = False

    # ─── Training Loop ───
    epochs = args.epochs
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    compiled_failed = False

    for epoch in range(epochs):
        model.train()
        aux_renderer.train()
        sync_net.audio_adapter.train()

        total_loss = 0.0
        total_loss_mse = 0.0
        total_loss_lpips = 0.0
        total_loss_sync = 0.0
        num_batches = 0

        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for step_idx, (visemes, viseme_padding_mask, targets, gt_frames, fau_signals) in enumerate(progress_bar):
            # Move data to GPU
            visemes = visemes.to(device, non_blocking=True)
            viseme_padding_mask = viseme_padding_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            gt_frames = gt_frames.to(device, non_blocking=True)
            fau_signals = fau_signals.to(device, non_blocking=True)

            # ─── Forward pass ───
            try:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    B, T, _ = targets.shape

                    outputs = model(
                        visemes,
                        target_frames_len=T,
                        viseme_padding_mask=viseme_padding_mask,
                        fau_signals=fau_signals
                    )

                    # Base MSE Loss
                    loss_mse = criterion(outputs, targets)

                    # Auxiliary Losses
                    pred_frames = aux_renderer(outputs)  # (B, T, 3, 64, 64)
                    loss_sync = sync_net(pred_frames, targets)

                    loss_lpips = 0.0
                    if lpips_loss_fn is not None:
                        B_actual, T_actual, C, H, W = pred_frames.shape
                        pred_frames_flat = pred_frames.reshape(B_actual * T_actual, C, H, W)
                        gt_frames_flat = gt_frames.reshape(B_actual * T_actual, C, H, W)
                        loss_lpips = lpips_loss_fn(pred_frames_flat, gt_frames_flat).mean()

                    # Combine losses
                    alpha_lpips = 0.1
                    beta_sync = 0.1
                    loss = loss_mse + alpha_lpips * loss_lpips + beta_sync * loss_sync

                    # Scale loss for gradient accumulation
                    loss = loss / args.accum_steps

            except Exception as e:
                # ─── If torch.compile fails at runtime, fall back to eager mode ───
                if compile_enabled and not compiled_failed:
                    print(f"\n[-] torch.compile forward pass failed ({type(e).__name__}: {e}).")
                    print("[*] Disabling compile and falling back to eager mode...")
                    compile_enabled = False
                    compiled_failed = True
                    
                    # Rebuild uncompiled models and re-create optimizer
                    model, aux_renderer, sync_net = build_models(config, processor, device)
                    
                    # Re-create optimizer
                    optimizer = optim.AdamW(
                        [
                            {"params": model.parameters(), "lr": args.lr},
                            {"params": aux_renderer.parameters(), "lr": args.lr},
                            {"params": sync_net.audio_adapter.parameters(), "lr": args.lr * 0.5},
                        ],
                        lr=args.lr,
                        weight_decay=1e-5,
                    )
                    
                    # Re-create scheduler
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=[args.max_lr, args.max_lr, args.max_lr * 0.5],
                        steps_per_epoch=effective_steps,
                        epochs=args.epochs,
                        pct_start=0.1,
                        div_factor=25.0,
                        final_div_factor=1e4,
                    )
                    
                    # Update progress bar message
                    progress_bar.set_postfix_str("Falling back to eager mode...")
                    
                    # Re-run this step with eager mode
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(
                            visemes,
                            target_frames_len=T,
                            viseme_padding_mask=viseme_padding_mask,
                            fau_signals=fau_signals
                        )
                        loss_mse = criterion(outputs, targets)
                        pred_frames = aux_renderer(outputs)
                        loss_sync = sync_net(pred_frames, targets)
                        loss_lpips = 0.0
                        if lpips_loss_fn is not None:
                            B_actual, T_actual, C, H, W = pred_frames.shape
                            pred_frames_flat = pred_frames.reshape(B_actual * T_actual, C, H, W)
                            gt_frames_flat = gt_frames.reshape(B_actual * T_actual, C, H, W)
                            loss_lpips = lpips_loss_fn(pred_frames_flat, gt_frames_flat).mean()
                        loss = loss_mse + alpha_lpips * loss_lpips + beta_sync * loss_sync
                        loss = loss / args.accum_steps
                else:
                    raise e

            # ─── Backward pass ───
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(aux_renderer.parameters()) +
                    list(sync_net.audio_adapter.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                num_batches += 1

            # Track losses
            total_loss += loss.item() * args.accum_steps
            total_loss_mse += loss_mse.item()
            total_loss_sync += loss_sync.item()
            if isinstance(loss_lpips, torch.Tensor):
                total_loss_lpips += loss_lpips.item()
            else:
                total_loss_lpips += loss_lpips

            # Update progress bar
            if compile_enabled:
                status = "compiled"
            elif compiled_failed:
                status = "eager"
            else:
                status = "eager"
            progress_bar.set_postfix({
                'loss': f"{loss.item() * args.accum_steps:.4f}",
                'mse': f"{loss_mse.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'mode': status
            })

        # Handle remaining gradients at end of epoch
        if (step_idx + 1) % args.accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            num_batches += 1

        # ─── Epoch summary ───
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_loss_mse / max(num_batches, 1)
        avg_lpips = total_loss_lpips / max(num_batches, 1)
        avg_sync = total_loss_sync / max(num_batches, 1)

        print(f"[*] Epoch {epoch+1}/{epochs} | "
              f"Total: {avg_loss:.4f} | "
              f"MSE: {avg_mse:.4f} | "
              f"LPIPS: {avg_lpips:.4f} | "
              f"SYNC: {avg_sync:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # ─── Early stopping ───
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_path = config['paths']['text_model_weights']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # If model was compiled, extract the underlying model
            if compile_enabled and hasattr(model, '_orig_mod'):
                torch.save(model._orig_mod.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"[+] Best model saved to: {save_path} (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[*] Early stopping triggered after {epoch+1} epochs (no improvement for {patience_counter} epochs)")
                break

    print(f"\n[*] Training complete! Best loss: {best_loss:.4f}")
    print(f"[*] Final model weights: {config['paths']['text_model_weights']}")


if __name__ == "__main__":
    train()
