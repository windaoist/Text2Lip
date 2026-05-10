import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryRenderer(nn.Module):
    """
    A lightweight auxiliary CNN to map 1D Whisper features (19200 dims)
    into low-resolution (64x64) RGB frames.
    This enables the computation of perceptual losses (LPIPS) and video-based Sync Loss.
    """
    def __init__(self, in_features=19200, out_size=64):
        super().__init__()
        self.out_size = out_size
        
        # We first project the massive 19200 vector down to a manageable size, say 1024
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Then we decode it spatially: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.Tanh() # Output in range [-1, 1] which is typical for LPIPS/images
        )

    def forward(self, features):
        """
        features: (B, T, 19200)
        Returns: (B, T, 3, 64, 64) predicted frames
        """
        B, T, D = features.shape
        
        # Process frame by frame
        x = features.view(B * T, D)
        x = self.fc(x)
        x = x.view(B * T, 256, 4, 4)
        
        img = self.decoder(x)
        return img.view(B, T, 3, self.out_size, self.out_size)


class DummySyncNet(nn.Module):
    """
    A placeholder SyncNet to compute an audio-visual contrastive loss.
    In a real scenario, this would be a pre-trained SyncNet.
    Here we build a basic feature extractor for the predicted frames and audio,
    and compute cosine similarity.
    """
    def __init__(self):
        super().__init__()
        # Audio feature encoder (whisper features are 19200)
        self.audio_encoder = nn.Sequential(
            nn.Linear(19200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Video frame encoder (64x64 RGB -> 256)
        self.video_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )

    def forward(self, video_frames, audio_features):
        """
        video_frames: (B, T, 3, H, W)
        audio_features: (B, T, 19200)
        Returns: Contrastive sync loss
        """
        B, T, C, H, W = video_frames.shape
        
        # Flatten B and T
        v = video_frames.view(B * T, C, H, W)
        a = audio_features.view(B * T, -1)
        
        # Extract embeddings
        v_emb = self.video_encoder(v)
        a_emb = self.audio_encoder(a)
        
        # Normalize
        v_emb = F.normalize(v_emb, p=2, dim=1)
        a_emb = F.normalize(a_emb, p=2, dim=1)
        
        # Maximize cosine similarity between matching pairs (diagonal)
        # Minimize for non-matching pairs (off-diagonal)
        # For simplicity, we just do a direct cosine embedding loss with target 1
        similarity = F.cosine_similarity(v_emb, a_emb)
        loss = 1.0 - similarity.mean()
        
        return loss
