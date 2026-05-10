import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class VisemeEncoder(nn.Module):
    """
    Enhanced VisemeEncoder using Cross-Attention to dynamically align 
    input viseme sequences of length N with target video frame length T.
    """
    def __init__(self, num_visemes, d_model=256, nhead=8, num_layers=4, out_dim=None, fau_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_visemes, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Self-attention for visemes to capture context
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention decoder to align N visemes to T frames
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # Using fewer layers for decoder is often sufficient
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Learnable positional embeddings for target frames
        self.frame_query_embed = nn.Embedding(5000, d_model) 
        
        # FAU conditioning: project action units to d_model space
        self.fau_projection = nn.Linear(fau_dim, d_model)
        
        self.d_model = d_model
        self.output_projection = nn.Linear(d_model, out_dim) if out_dim is not None else nn.Identity()

    def forward(self, viseme_ids, target_frames_len=None, viseme_padding_mask=None, fau_signals=None):
        """
        viseme_ids: (B, N)
        target_frames_len: scalar integer T, the desired number of output frames
        viseme_padding_mask: (B, N) boolean mask where True means padding
        fau_signals: (B, T, fau_dim) micro-expression control signals
        """
        B, N = viseme_ids.shape
        T = target_frames_len if target_frames_len is not None else N

        # 1. Encode viseme sequences (B, N, d_model)
        x = self.embedding(viseme_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=viseme_padding_mask)

        # 2. Generate frame queries (B, T, d_model)
        frame_positions = torch.arange(T, device=viseme_ids.device).unsqueeze(0).expand(B, T)
        frame_queries = self.frame_query_embed(frame_positions) * math.sqrt(self.d_model)
        
        # 3. Add FAU conditioning if provided
        if fau_signals is not None:
            fau_feat = self.fau_projection(fau_signals)
            frame_queries = frame_queries + fau_feat
            
        # We also add standard sinusoidal positional encoding to provide robust relative positioning
        frame_queries = self.pos_encoder(frame_queries)

        # 4. Cross-attention alignment: Queries are T frames, Keys/Values are N visemes
        out = self.transformer_decoder(
            tgt=frame_queries, 
            memory=memory, 
            memory_key_padding_mask=viseme_padding_mask
        )

        return self.output_projection(out)

if __name__ == "__main__":
    # 测试
    B, N = 1, 15
    T = 30 # We want 30 output frames from 15 visemes
    viseme_ids = torch.randint(0, 15, (B, N))
    
    encoder = VisemeEncoder(num_visemes=16, out_dim=19200)
    features = encoder(viseme_ids, target_frames_len=T)
    
    print(f"Step 2 - Viseme IDs Shape: {viseme_ids.shape}")
    print(f"Step 2 - Generated Features Shape: {features.shape}")
