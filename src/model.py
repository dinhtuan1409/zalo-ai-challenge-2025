# model_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# Utility: TransformerEncoder block (simple)
# -------------------------
def build_transformer_encoder(dim, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,  # (B, S, D)
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# -------------------------
# HME-style Multi-choice VideoQA with Memory + Iterative Reasoning
# -------------------------
class HME_MC_v2(nn.Module):
    def __init__(
        self,
        motion_dim: int = 2304,
        appearance_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 1024,
        motion_proj_dim: int = 1024,
        num_motion_layers: int = 2,
        motion_heads: int = 8,
        fusion_heads: int = 8,
        reasoning_steps: int = 2,
        memory_size: int = 5,
        dropout: float = 0.1,
        use_mean_pool_motion: bool = False
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.appearance_dim = appearance_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.motion_proj_dim = motion_proj_dim
        self.reasoning_steps = reasoning_steps
        self.use_mean_pool_motion = use_mean_pool_motion
        self.memory_size = memory_size

        # --- project raw features ---
        self.motion_proj = nn.Linear(motion_dim, motion_proj_dim)
        self.motion_ln = nn.LayerNorm(motion_proj_dim)
        self.motion_encoder = build_transformer_encoder(motion_proj_dim, num_motion_layers, motion_heads, dropout=dropout)

        self.appearance_proj = nn.Linear(appearance_dim, motion_proj_dim)
        self.appearance_ln = nn.LayerNorm(motion_proj_dim)

        self.text_proj = nn.Linear(text_dim, motion_proj_dim)
        self.text_ln = nn.LayerNorm(motion_proj_dim)

        # --- memory modules ---
        self.appearance_memory = nn.Parameter(torch.zeros(memory_size, motion_proj_dim))
        self.motion_memory = nn.Parameter(torch.zeros(memory_size, motion_proj_dim))

        # --- cross attention: text queries attend video+memory ---
        self.cross_attn = nn.MultiheadAttention(embed_dim=motion_proj_dim, num_heads=fusion_heads, dropout=dropout, batch_first=True)
        self.cross_attn_ln = nn.LayerNorm(motion_proj_dim)

        # --- fusion feed-forward ---
        self.fusion_ff = nn.Sequential(
            nn.Linear(motion_proj_dim, motion_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motion_proj_dim, motion_proj_dim)
        )
        self.fusion_ln = nn.LayerNorm(motion_proj_dim)

        # --- final classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(motion_proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        motion_feats: torch.Tensor,         # [B, T, D_m]
        motion_mask: Optional[torch.Tensor],# [B, T]
        appearance_feats: torch.Tensor,     # [B, D_a]
        text_feats: torch.Tensor,           # [B, C, D_t]
        labels: Optional[torch.Tensor] = None
    ):
        B, device = motion_feats.size(0), motion_feats.device

        # --- project features ---
        mot = self.motion_ln(self.motion_proj(motion_feats))
        app = self.appearance_ln(self.appearance_proj(appearance_feats))
        txt = self.text_ln(self.text_proj(text_feats))

        # --- motion temporal encoder ---
        if self.use_mean_pool_motion:
            if motion_mask is not None:
                denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1).to(mot.dtype)
                mot_tokens = ((mot * motion_mask.unsqueeze(-1)).sum(dim=1) / denom).unsqueeze(1)
            else:
                mot_tokens = mot.mean(dim=1, keepdim=True)
        else:
            key_padding_mask = (motion_mask == 0) if motion_mask is not None else None
            mot_tokens = self.motion_encoder(mot, src_key_padding_mask=key_padding_mask)

        # --- add appearance token ---
        app_tok = app.unsqueeze(1)
        video_tokens = torch.cat([app_tok, mot_tokens], dim=1)
        video_mask = torch.ones((B, video_tokens.size(1)), dtype=torch.long, device=device)

        # --- iterative reasoning over memory ---
        mem_app = self.appearance_memory.unsqueeze(0).expand(B, -1, -1)
        mem_mot = self.motion_memory.unsqueeze(0).expand(B, -1, -1)
        video_tokens = torch.cat([video_tokens, mem_app, mem_mot], dim=1)
        video_mask = torch.cat([video_mask, torch.ones((B, mem_app.size(1)+mem_mot.size(1)), device=device)], dim=1)

        B, C, M = txt.shape
        queries = txt.view(B*C, 1, M)
        keys = video_tokens.repeat_interleave(C, dim=0)
        key_padding_mask = (video_mask == 0).repeat_interleave(C, dim=0)

        # iterative cross-attention
        x = queries
        for _ in range(self.reasoning_steps):
            attn_out, _ = self.cross_attn(x, keys, keys, key_padding_mask=key_padding_mask)
            x = self.cross_attn_ln(x + attn_out)
            x = self.fusion_ln(self.fusion_ff(x))

        # --- combine text and fused video-attended ---
        combined = torch.cat([queries, x], dim=-1).squeeze(1)
        logits = self.classifier(combined).view(B, C)

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss
        return output
