import torch
from torch import nn
from torch.nn import functional as F
import math


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences):
        """
        audio_sequences := (B, 1, 80, 16) # For Wav2Lip it's mel-spectrogram
        face_sequences := (B, 15, 96, 96) # For Wav2Lip it's 5 frames * 3 channels
        """
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.reshape(audio_embedding.size(0), -1)
        face_embedding = face_embedding.reshape(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding


class PretrainedSyncNet(nn.Module):
    """
    Adapter to use Wav2Lip's pre-trained SyncNet in our pipeline.
    It handles input reshaping and normalization to match Wav2Lip's expectations.
    """

    def __init__(self, weights_path="pretrained_weights/lipsync_expert.pth"):
        super().__init__()
        self.syncnet = SyncNet_color()
        if torch.cuda.is_available():
            # Ensure safe loading or explicit map location
            checkpoint = torch.load(weights_path, map_location='cuda')
        else:
            checkpoint = torch.load(weights_path, map_location='cpu')

        # Wav2Lip checkpoints often save state_dict inside "state_dict" key
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.syncnet.load_state_dict(state_dict)
        # Freeze SyncNet parameters as we only use it for loss
        for param in self.syncnet.parameters():
            param.requires_grad = False
        self.syncnet.eval()

        # We need a small linear adapter to map our 19200-dim Whisper features
        # into the (1, 80, 16) shape that Wav2Lip's audio encoder expects.
        # Alternatively, if we just want a "Sync Loss", we can train this adapter
        # alongside our main network.
        self.audio_adapter = nn.Sequential(
            nn.Linear(19200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 80 * 16)
        )
        # Wav2Lip face encoder outputs different dimensions depending on input resolution.
        # With 96x96 lower-half input, it outputs 2048 (512 * 4 * 1).
        # With classic 96x96 full face (48x96 lower-half resized), it outputs 512.
        # This adapter normalizes the face embedding to 512 for cosine similarity.
        self.face_adapter = nn.Sequential(
            nn.Linear(2048, 512),
        )

    def forward(self, video_frames, audio_features):
        """
        video_frames: (B, T, 3, H, W)  [Our AuxiliaryRenderer outputs 64x64]
        audio_features: (B, T, 19200)
        """
        B, T, C, H, W = video_frames.shape

        loss = 0.0
        valid_windows = 0

        # SyncNet expects a window of 5 frames (T_s = 5)
        window_size = 5
        if T < window_size:
            return torch.tensor(0.0, device=video_frames.device, requires_grad=True)

        # Iterate over sliding windows
        for i in range(T - window_size + 1):
            # 1. Prepare video window: (B, 5, 3, H, W)
            v_window = video_frames[:, i:i+window_size, :, :, :]
            # Resize to 96x96 (Wav2Lip expectation) using interpolate

            v_window_flat = v_window.reshape(B * window_size, C, H, W)

            # Wav2Lip style: first crop lower half (H//2:), then resize to 96x96
            # Input: 64x64 -> crop to 32:64 (lower half: 32x64) -> resize to 96x96
            half_h = H // 2
            v_window_cropped = v_window_flat[:, :, half_h:, :]  # (B*5, 3, 32, 64)
            v_window_resized = F.interpolate(
                v_window_cropped, size=(96, 96),
                mode='bilinear', align_corners=False
            )  # (B*5, 3, 96, 96)

            v_window_resized = v_window_resized.reshape(
                B, window_size, C, 96, 96)

            # Prepare audio window
            # SyncNet expects the audio corresponding to the center frame of the video window
            center_idx = i + window_size // 2
            a_center = audio_features[:, center_idx, :]  # (B, 19200)

            # Project to expected shape
            a_projected = self.audio_adapter(a_center)  # (B, 80 * 16)
            a_input = a_projected.view(B, 1, 80, 16)

            # Format for SyncNet: (B, C*5, 96, 96)
            v_input = v_window_resized.permute(
                0, 2, 1, 3, 4).contiguous().view(B, C * window_size, 96, 96)
            # 3. Get embeddings
            a_emb, v_emb = self.syncnet(a_input, v_input)

            # Project face embedding from 2048 → 512 to match audio embedding dim
            v_emb = self.face_adapter(v_emb)

            # 4. Compute contrastive loss (maximize cosine similarity for matching pair)
            similarity = F.cosine_similarity(v_emb, a_emb)
            loss += (1.0 - similarity).mean()
            valid_windows += 1

        if valid_windows > 0:
            loss = loss / valid_windows
        else:
            loss = torch.tensor(
                0.0, device=video_frames.device, requires_grad=True)

        return loss
