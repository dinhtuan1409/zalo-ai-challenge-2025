import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# Transformer Encoder Utility
# -------------------------
def build_transformer_encoder(dim, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


# -------------------------
# HME Multi-choice VideoQA
# -------------------------
class HME_MC(nn.Module):
    def __init__(
        self,
        motion_dim: int = 2304,
        appearance_dim: int = 768,
        text_dim: int = 1536,       # MPNet + Video-LLaVA caption
        hidden_dim: int = 1024,
        motion_proj_dim: int = 1024,
        num_motion_layers: int = 2,
        motion_heads: int = 8,
        fusion_heads: int = 8,
        dropout: float = 0.1,
        use_mean_pool_motion: bool = False,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.appearance_dim = appearance_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_mean_pool_motion = use_mean_pool_motion

        # --- Project features into common hidden space ---
        self.motion_proj = nn.Linear(motion_dim, motion_proj_dim)
        self.motion_ln = nn.LayerNorm(motion_proj_dim)
        self.motion_encoder = build_transformer_encoder(
            dim=motion_proj_dim,
            num_layers=num_motion_layers,
            num_heads=motion_heads,
            dropout=dropout,
        )

        self.appearance_proj = nn.Linear(appearance_dim, motion_proj_dim)
        self.appearance_ln = nn.LayerNorm(motion_proj_dim)

        self.text_proj = nn.Linear(text_dim, motion_proj_dim)
        self.text_ln = nn.LayerNorm(motion_proj_dim)

        # --- Cross-attention: text queries video tokens ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=motion_proj_dim,
            num_heads=fusion_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_ln = nn.LayerNorm(motion_proj_dim)

        self.fusion_ff = nn.Sequential(
            nn.Linear(motion_proj_dim, motion_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motion_proj_dim, motion_proj_dim),
        )
        self.fusion_ln = nn.LayerNorm(motion_proj_dim)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(motion_proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        motion_feats: torch.Tensor,      # [B, T, 2304]
        motion_mask: Optional[torch.Tensor],  # [B, T]
        appearance_feats: torch.Tensor,  # [B, 768]
        text_feats: torch.Tensor,        # [B, C, 1536] (MPNet + Video-LLaVA)
        labels: Optional[torch.Tensor] = None,
    ):
        B = motion_feats.shape[0]
        device = motion_feats.device

        # --- Project motion ---
        mot = self.motion_proj(motion_feats)
        mot = self.motion_ln(mot)

        # --- Motion encoder / mean pooling ---
        if self.use_mean_pool_motion:
            if motion_mask is not None:
                denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1).to(mot.dtype)
                mot_pooled = (mot * motion_mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                mot_pooled = mot.mean(dim=1)
            mot_tokens = mot_pooled.unsqueeze(1)  # [B, 1, M]
        else:
            key_padding_mask = None
            if video_mask is not None:
                key_padding_mask = (video_mask == 0)        # bool mask
                key_padding_mask = key_padding_mask.repeat_interleave(C, dim=0)

            mot_tokens = self.motion_encoder(mot, src_key_padding_mask=key_padding_mask)  # [B, T, M]

        # --- Appearance ---
        app = self.appearance_proj(appearance_feats)
        app = self.appearance_ln(app)
        app_tok = app.unsqueeze(1)  # [B, 1, M]

        # --- Video tokens + mask ---
        video_tokens = torch.cat([app_tok, mot_tokens], dim=1)  # [B, T+1, M]
        if motion_mask is not None:
            app_mask = torch.ones((B,1), dtype=torch.bool, device=device)
            video_mask = torch.cat([app_mask, motion_mask], dim=1)
        else:
            video_mask = None

        # --- Text projection ---
        B, C, _ = text_feats.shape
        txt = self.text_proj(text_feats)       # [B, C, M]
        txt = self.text_ln(txt)

        # --- Cross-attention: each text choice queries video tokens ---
        queries = txt.reshape(B * C, 1, -1)                       # [B*C, 1, M]
        keys = video_tokens.repeat_interleave(C, dim=0)           # [B*C, T+1, M]
        key_padding_mask = video_mask.repeat_interleave(C, dim=0) if video_mask is not None else None

        attn_out, _ = self.cross_attn(queries, keys, keys, key_padding_mask=key_padding_mask)
        attn_out = self.cross_attn_ln(attn_out.squeeze(1))       # [B*C, M]
        fused = self.fusion_ln(self.fusion_ff(attn_out))         # [B*C, M]

        # --- Classifier ---
        txt_flat = txt.reshape(B * C, -1)
        combined = torch.cat([txt_flat, fused], dim=-1)          # [B*C, 2M]
        logits = self.classifier(combined).view(B, C)            # [B, C]

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)

        return output
