# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# Utility: TransformerEncoder block
# -------------------------
def build_transformer_encoder(dim, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.1):
    """
    Simple Transformer encoder (motion temporal modeling)
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True  # input shape: [B, S, D]
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# -------------------------
# Video Encoder for LLM-assisted VideoQA
# -------------------------
class VideoEncoder(nn.Module):
    """
    Encode motion + appearance into a single vector per video.
    This embedding can be used for retrieval or fed to LLM as summary.
    """

    def __init__(
        self,
        motion_dim: int = 2304,
        appearance_dim: int = 768,
        hidden_dim: int = 1024,
        motion_proj_dim: int = 1024,
        num_motion_layers: int = 2,
        motion_heads: int = 8,
        dropout: float = 0.1,
        use_mean_pool_motion: bool = True,
    ):
        super().__init__()
        self.use_mean_pool_motion = use_mean_pool_motion

        # Motion projection + transformer
        self.motion_proj = nn.Linear(motion_dim, motion_proj_dim)
        self.motion_ln = nn.LayerNorm(motion_proj_dim)
        self.motion_encoder = build_transformer_encoder(
            dim=motion_proj_dim,
            num_layers=num_motion_layers,
            num_heads=motion_heads,
            dropout=dropout,
        )

        # Appearance projection
        self.appearance_proj = nn.Linear(appearance_dim, motion_proj_dim)
        self.appearance_ln = nn.LayerNorm(motion_proj_dim)

        # Final fusion -> video embedding
        self.video_proj = nn.Linear(motion_proj_dim * 2, hidden_dim)
        self.video_ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        motion_feats: torch.Tensor,          # [B, T, D_m]
        motion_mask: Optional[torch.Tensor] = None, # [B, T] (1 for valid)
        appearance_feats: Optional[torch.Tensor] = None  # [B, D_a]
    ):
        B = motion_feats.shape[0]

        # --- Motion ---
        mot = self.motion_proj(motion_feats)  # [B, T, M]
        mot = self.motion_ln(mot)

        if self.use_mean_pool_motion:
            # simple mean pooling
            if motion_mask is not None:
                denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1).to(mot.dtype)
                mot_pooled = (mot * motion_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, M]
            else:
                mot_pooled = mot.mean(dim=1)
            mot_tokens = mot_pooled  # [B, M]
        else:
            # transformer encoder
            key_padding_mask = (motion_mask == 0) if motion_mask is not None else None
            mot_encoded = self.motion_encoder(mot, src_key_padding_mask=key_padding_mask)
            # mean pool temporal dimension
            if motion_mask is not None:
                denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1)
                mot_tokens = (mot_encoded * motion_mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                mot_tokens = mot_encoded.mean(dim=1)  # [B, M]

        # --- Appearance ---
        if appearance_feats is not None:
            app = self.appearance_proj(appearance_feats)
            app = self.appearance_ln(app)
        else:
            # fallback zero vector
            app = torch.zeros_like(mot_tokens)
        
        # --- Combine ---
        combined = torch.cat([mot_tokens, app], dim=-1)  # [B, 2*M]
        video_emb = self.video_ln(self.video_proj(combined))  # [B, hidden_dim]

        return video_emb

# -------------------------
# Optional: Retrieval + Contrastive Loss
# -------------------------
def contrastive_loss(video_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float = 0.07):
    """
    Simple contrastive loss (InfoNCE)
    video_emb, text_emb: [B, D]
    """
    video_emb = F.normalize(video_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = video_emb @ text_emb.T / temperature  # [B, B]
    labels = torch.arange(video_emb.size(0), device=video_emb.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2.0


class VideoTextContrastive(nn.Module):
    def __init__(self, video_dim=1024, text_dim=768, temperature=0.07):
        super().__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.temperature = temperature
        # Linear projection text -> video_dim
        self.text_proj = nn.Linear(text_dim, video_dim)

    def forward(self, video_emb: torch.Tensor, text_emb: torch.Tensor):
        # normalize embeddings
        video_emb = F.normalize(video_emb, dim=-1)           # [B, video_dim]
        text_emb = F.normalize(self.text_proj(text_emb), dim=-1)  # [B, video_dim]

        logits = video_emb @ text_emb.T / self.temperature   # [B, B]
        labels = torch.arange(video_emb.size(0), device=video_emb.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2.0