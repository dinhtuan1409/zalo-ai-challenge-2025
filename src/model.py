# model.py
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
        batch_first=True,  # use (B, S, D)
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


# -------------------------
# HME-style Multi-choice VideoQA Model
# -------------------------
class HME_MC(nn.Module):
    """
    HME-style multi-choice VideoQA model (practical / efficient).
    Inputs:
      - motion_feats: [B, T, D_m] (clip-level motion embeddings)
      - motion_mask:  [B, T] (1 for valid, 0 for pad)
      - appearance_feats: [B, D_a] (single vector per video)
      - text_feats: [B, C, D_t] (MPNet embeddings per choice)
    Output:
      - logits: [B, C] (scores for each choice)
    """

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
        dropout: float = 0.1,
        use_mean_pool_motion: bool = False,
        treat_text_as_query: bool = True,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.appearance_dim = appearance_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_mean_pool_motion = use_mean_pool_motion
        self.treat_text_as_query = treat_text_as_query

        # --- project raw features into common hidden space ---
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

        # Project text into same hidden dim
        self.text_proj = nn.Linear(text_dim, motion_proj_dim)
        self.text_ln = nn.LayerNorm(motion_proj_dim)

        # Cross-modal attention: text queries attend to video tokens (motion+appearance)
        # We'll implement as multihead attention modules.
        self.cross_attn = nn.MultiheadAttention(embed_dim=motion_proj_dim, num_heads=fusion_heads, dropout=dropout, batch_first=True)
        self.cross_attn_ln = nn.LayerNorm(motion_proj_dim)

        # Small feed-forward after fusion
        self.fusion_ff = nn.Sequential(
            nn.Linear(motion_proj_dim, motion_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motion_proj_dim, motion_proj_dim),
        )
        self.fusion_ln = nn.LayerNorm(motion_proj_dim)

        # Final scoring head: take per-choice representation -> score
        # We'll produce per-choice vector by attending text (choice) over video, then combine
        self.classifier = nn.Sequential(
            nn.Linear(motion_proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # one logit per choice
        )

    def forward(
        self,
        motion_feats: torch.Tensor,         # [B, T, D_m] or [B, 1, D_m] when single
        motion_mask: Optional[torch.Tensor],# [B, T]  (1 for valid)
        appearance_feats: torch.Tensor,     # [B, D_a] or [B, 1, D_a]
        text_feats: torch.Tensor,           # [B, C, D_t]
        labels: Optional[torch.Tensor] = None,
    ):
        B = motion_feats.shape[0]
        device = motion_feats.device

        # --- Project motion ---
        mot = self.motion_proj(motion_feats)           # [B, T, M]
        mot = self.motion_ln(mot)

        # Apply motion encoder (with mask)
        if self.use_mean_pool_motion:
            # skip temporal encoder if we just mean pool
            if motion_mask is not None:
                denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1).to(mot.dtype)
                mot_pooled = (mot * motion_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, M]
            else:
                mot_pooled = mot.mean(dim=1)
            # use mot_pooled as the single token
            mot_tokens = mot_pooled.unsqueeze(1)  # [B,1,M]
        else:
            # Build key_padding_mask for nn.TransformerEncoder: True for positions to mask
            if motion_mask is not None:
                # TransformerEncoder expects key_padding_mask shape [B, T] with True at PAD positions
                key_padding_mask = (motion_mask == 0)  # pad positions True
            else:
                key_padding_mask = None
            mot_tokens = self.motion_encoder(mot, src_key_padding_mask=key_padding_mask)  # [B, T, M]

        # --- Appearance projection: add as an extra token to video tokens ---
        app = self.appearance_proj(appearance_feats)  # [B, M] or [B,1,M]
        app = self.appearance_ln(app)
        # ensure app shape [B, 1, M]
        if app.dim() == 2:
            app_tok = app.unsqueeze(1)
        else:
            app_tok = app

        if self.use_mean_pool_motion:
            # mot_tokens: [B, 1, M]
            video_tokens = torch.cat([app_tok, mot_tokens], dim=1)  # [B, 2, M]
            video_mask = torch.ones((B, video_tokens.shape[1]), dtype=torch.long, device=device)
        else:
            # mot_tokens: [B, T, M]; cat app at front
            video_tokens = torch.cat([app_tok, mot_tokens], dim=1)  # [B, T+1, M]
            if motion_mask is not None:
                # combine app valid + motion_mask
                app_mask = torch.ones((B,1), dtype=motion_mask.dtype, device=device)
                video_mask = torch.cat([app_mask, motion_mask], dim=1)  # [B, T+1]
            else:
                video_mask = None

        # --- Project text ---
        # text_feats: [B, C, D_t] -> project to [B, C, M]
        txt = self.text_proj(text_feats)
        txt = self.text_ln(txt)

        # For each choice, let text vector (single token per choice) query video tokens.
        # We will treat text choice as query; video tokens as key/value.
        # Prepare cross-attn inputs for MultiheadAttention (batch_first=True)
        # Flatten batch+choices to compute in one call.
        B, C, M = txt.shape
        # txt reshape -> [B*C, 1, M] (we use mean of choice tokens as single query)
        # Here txt is already one vector per choice; if it had sequence, you'd pool first.
        queries = txt.view(B*C, 1, M)  # [B*C, 1, M]
        keys = video_tokens.repeat_interleave(C, dim=0)  # [B*C, S, M]

        if video_mask is not None:
            # key_padding_mask for multihead attn expects shape [B*C, S] with True for PAD
            key_padding_mask = (video_mask == 0).repeat_interleave(C, dim=0)  # [B*C, S]
        else:
            key_padding_mask = None

        # MultiheadAttention (query, key, value). returns (attended, weights)
        attn_out, _ = self.cross_attn(queries, keys, keys, key_padding_mask=key_padding_mask)
        attn_out = self.cross_attn_ln(attn_out.squeeze(1))  # [B*C, M]

        # fusion feed-forward
        fused = self.fusion_ln(self.fusion_ff(attn_out))  # [B*C, M]

        # Combine text vector (txt) and fused video-attended vector
        txt_flat = txt.view(B*C, M)
        combined = torch.cat([txt_flat, fused], dim=-1)  # [B*C, 2M]

        logits_per_choice = self.classifier(combined).view(B, C)  # [B, C]

        output = {"logits": logits_per_choice}

        if labels is not None:
            loss = F.cross_entropy(logits_per_choice, labels)
            output["loss"] = loss

        return output
