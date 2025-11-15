import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVideoQAMC(nn.Module):
    """
    Giải pháp 2 - VideoQA Multi-Choice đơn giản
    Dùng mean pool motion + appearance + text → MLP
    """

    def __init__(self,
                 motion_dim=2304,
                 appearance_dim=768,
                 text_dim=768,
                 hidden1=512,
                 hidden2=128,
                 dropout=0.2):
        super().__init__()

        # Tổng kích thước vector video
        self.video_dim = motion_dim + appearance_dim

        # MLP phân loại cho mỗi lựa chọn
        self.classifier = nn.Sequential(
            nn.Linear(self.video_dim + text_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)  # logit 1 chiều
        )

    def forward(self,
                motion_feats,      # [B, T, Dm]
                motion_mask,       # [B, T]
                appearance_feats,  # [B, Da]
                text_feats,        # [B, C, Dt]
                labels=None):

        B, C = text_feats.shape[:2]

        # -------- 1. Mean pool motion (mask-aware) ----------
        if motion_mask is not None:
            denom = motion_mask.sum(dim=1, keepdim=True).clamp(min=1).to(motion_feats.dtype)
            motion_mean = (motion_feats * motion_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            motion_mean = motion_feats.mean(dim=1)

        # -------- 2. Video vector = concat(motion_mean, appearance) ----------
        video_vec = torch.cat([motion_mean, appearance_feats], dim=-1)   # [B, Dv]

        # Lặp B lần → [B*C, Dv]
        video_rep = video_vec.unsqueeze(1).repeat(1, C, 1).view(B*C, -1)

        # -------- 3. Flatten text choices ----------
        text_rep = text_feats.view(B*C, -1)   # [B*C, Dt]

        # -------- 4. Combine video & text ----------
        combine = torch.cat([video_rep, text_rep], dim=-1)   # [B*C, Dv+Dt]

        # -------- 5. MLP → logits ----------
        logits = self.classifier(combine)       # [B*C, 1]
        logits = logits.view(B, C)              # [B, C]

        out = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss

        return out