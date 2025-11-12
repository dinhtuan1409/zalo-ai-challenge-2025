# hme_mc_clip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from typing import Optional, List

def build_transformer_encoder(dim, num_layers=2, num_heads=8, dropout=0.1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
        dropout=dropout, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# --- Dynamic Memory Module ---
class DynamicMemory(nn.Module):
    def __init__(self, memory_size, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, feats):  # [B, T, D]
        B, T, D = feats.shape
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # Read
        keys = self.proj(feats)
        attn = F.softmax(torch.bmm(keys, mem.transpose(1,2)) / (D**0.5), dim=-1)
        read = torch.bmm(attn, mem)  # [B, T, D]

        # Write
        write = attn.mean(1)  # [B, M]
        gate = self.gate(feats.mean(1))  # [B, 1]
        new_mem = mem + gate.unsqueeze(1) * write.unsqueeze(2)
        return read, new_mem

# --- HME-MC v3 (CLIP-Aligned) ---
class HME_MC(nn.Module):
    def __init__(
        self,
        motion_dim=2304,
        clip_dim=768,
        hidden_dim=1024,
        proj_dim=768,
        memory_size=8,
        reasoning_steps=3,
        clip_model_name="openai/clip-vit-base-patch32",
        freeze_clip=True
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.reasoning_steps = reasoning_steps
        self.memory_size = memory_size

        # --- CLIP ---
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # --- Motion Proj ---
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # --- Motion Encoder ---
        self.motion_encoder = build_transformer_encoder(proj_dim, num_layers=2)

        # --- Dynamic Memories ---
        self.app_memory = DynamicMemory(memory_size, proj_dim)
        self.mot_memory = DynamicMemory(memory_size, proj_dim)

        # --- Cross Attention ---
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.norm_attn = nn.LayerNorm(proj_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim, proj_dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim*4, proj_dim)
        )
        self.norm_ffn = nn.LayerNorm(proj_dim)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def encode_text(self, questions: List[str], answers: List[List[str]]) -> torch.Tensor:
        """
        Input:
            questions: [B] strings
            answers:   [B, C_i] strings (C_i có thể khác nhau)
        Output:
            text_feats: [total_choices, 768]
        """
        texts = []
        self.choice_counts = []  # Lưu số lựa chọn mỗi sample
        for q, ans_list in zip(questions, answers):
            for a in ans_list:
                texts.append(f"{q} {a}")
            self.choice_counts.append(len(ans_list))

        if not texts:
            return torch.zeros(0, self.proj_dim, device=next(self.parameters()).device)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.clip_model.parameters()).device)

        with torch.no_grad():
            text_feats = self.clip_model.get_text_features(**inputs)  # [N, 768]
        return text_feats

    def forward(
        self,
        motion_feats: torch.Tensor,          # [B, T, 2304]
        motion_mask: torch.Tensor,           # [B, T] - bool
        clip_img_feats: torch.Tensor,        # [B, 768]
        questions: List[str],
        answers: List[List[str]],
        labels: Optional[torch.Tensor] = None
    ):
        B, device = motion_feats.size(0), motion_feats.device

        # --- 1. Project motion ---
        mot = self.motion_proj(motion_feats)  # [B, T, 768]

        # --- 2. Motion Encoder ---
        mot_tokens = self.motion_encoder(
            mot,
            src_key_padding_mask=~motion_mask  # ~bool → True = pad
        )  # [B, T, 768]

        # --- 3. Dynamic Memory ---
        mot_read, _ = self.mot_memory(mot_tokens)
        app_read, _ = self.app_memory(
            clip_img_feats.unsqueeze(1).expand(-1, mot_tokens.size(1), -1)
        )

        # --- 4. Video Tokens ---
        video_tokens = torch.cat([mot_tokens, mot_read, app_read], dim=1)  # [B, 3T, 768]
        video_mask = torch.ones(B, video_tokens.size(1), device=device, dtype=torch.bool)
        video_mask[:, :mot_tokens.size(1)] = motion_mask  # Chỉ motion có mask thật

        # --- 5. Encode Text ---
        text_feats = self.encode_text(questions, answers)  # [N, 768]
        N = text_feats.size(0)
        queries = text_feats.unsqueeze(1)  # [N, 1, 768]

        # --- 6. Repeat video ---
        video_tokens_rep = video_tokens.repeat_interleave(
            torch.tensor(self.choice_counts, device=device), dim=0
        )  # [N, 3T, 768]
        video_mask_rep = video_mask.repeat_interleave(
            torch.tensor(self.choice_counts, device=device), dim=0
        )  # [N, 3T]

        # --- 7. Iterative Reasoning ---
        x = queries
        for _ in range(self.reasoning_steps):
            attn_out, _ = self.cross_attn(
                x, video_tokens_rep, video_tokens_rep,
                key_padding_mask=~video_mask_rep  # bool mask
            )
            x = self.norm_attn(x + attn_out)
            x = self.norm_ffn(x + self.ffn(x))

        # --- 8. Classification ---
        combined = torch.cat([queries, x], dim=-1).squeeze(1)  # [N, 1536]
        logits = self.classifier(combined).squeeze(-1)  # [N]

        # --- 9. Reshape to [B, max_C] ---
        max_C = max(self.choice_counts)
        logits_padded = torch.full((B, max_C), float('-inf'), device=device)
        idx = 0
        for i, c in enumerate(self.choice_counts):
            logits_padded[i, :c] = logits[idx:idx+c]
            idx += c

        output = {"logits": logits_padded}
        if labels is not None:
            # Chỉ tính loss trên các lựa chọn hợp lệ
            valid_mask = labels != -1
            if valid_mask.any():
                loss = F.cross_entropy(logits_padded[valid_mask], labels[valid_mask], label_smoothing=0.1)
                output["loss"] = loss
        return output