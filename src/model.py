import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Simple Transformer Block
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


# -----------------------
# Cross Attention
# -----------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # q: [B,1,dim]
        # kv: [B,T,dim]
        out, _ = self.attn(q, kv, kv)
        return self.ln(q + out)


# -----------------------
# Improved Model
# -----------------------
class SimpleVideoQAMC(nn.Module):
    def __init__(
        self,
        motion_dim=2304,
        appearance_dim=768,
        text_dim=768,
        hidden_dim=1024,
        motion_proj_dim=1024,
        num_motion_layers=2,
        num_heads=8
    ):
        super().__init__()

        # ----- Projections -----
        self.motion_proj = nn.Linear(motion_dim, motion_proj_dim)
        self.appearance_proj = nn.Linear(appearance_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # ----- Motion encoder -----
        self.motion_transformer = nn.ModuleList([
            TransformerBlock(motion_proj_dim, num_heads=num_heads)
            for _ in range(num_motion_layers)
        ])

        # ----- Fuse motion + appearance -----
        self.motion_to_app = CrossAttention(hidden_dim, num_heads)

        # ----- Text cross-attention -----
        self.text_attend_video = CrossAttention(hidden_dim, num_heads)

        # ----- Classifier -----
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, motion_feats, motion_mask, appearance_feats, text_feats, labels=None):
        """
        motion_feats: [B,T,2304]
        appearance_feats: [B,768]
        text_feats:  [B,C,768]
        """

        # Project features
        motion = self.motion_proj(motion_feats)       # [B,T,1024]
        app = self.appearance_proj(appearance_feats)  # [B,1024]
        text = self.text_proj(text_feats)             # [B,C,1024]

        # ----- Encode motion -----
        for blk in self.motion_transformer:
            motion = blk(motion, mask=motion_mask)

        # Summary token for motion
        motion_token = motion.mean(dim=1, keepdim=True)   # [B,1,1024]

        # ----- Fuse appearance -----
        app = app.unsqueeze(1)                            # [B,1,1024]
        fused_video = self.motion_to_app(app, motion)     # app attends to motion

        # ----- Text attends to video -----
        fused_text = self.text_attend_video(text, fused_video)

        # Mean pooling
        video_repr = fused_text.mean(dim=1)

        logits = self.cls_head(video_repr)

        return {"logits": logits}
