# hme_mc_clip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
from typing import Optional

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
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)

        # Read
        keys = self.proj(feats)
        attn = F.softmax(torch.bmm(keys, mem.transpose(1,2)) / (D**0.5), dim=-1)
        read = torch.bmm(attn, mem)

        # Write (EMA)
        write = attn.mean(1)
        gate = self.gate(feats.mean(1))
        new_mem = mem + gate.unsqueeze(1) * write.unsqueeze(2)
        return read, new_mem

# --- HME-MC v3 (CLIP-Aligned) ---
class HME_MC(nn.Module):
    def __init__(
        self,
        motion_dim=2304,
        clip_dim=768,
        hidden_dim=1024,
        proj_dim=768,           # CLIP space
        memory_size=8,
        reasoning_steps=3,
        clip_model_name="openai/clip-vit-base-patch32",
        freeze_clip=True
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.reasoning_steps = reasoning_steps

        # --- CLIP Model (frozen) ---
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # --- Project Motion to CLIP space ---
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # --- Motion Temporal Encoder ---
        self.motion_encoder = build_transformer_encoder(proj_dim, num_layers=2)

        # --- Dynamic Memories (in CLIP space) ---
        self.app_memory = DynamicMemory(memory_size, proj_dim)
        self.mot_memory = DynamicMemory(memory_size, proj_dim)

        # --- Cross Attention (text query video) ---
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads=8, batch_first=True, dropout=0.1)
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

    def encode_text(self, questions, answers):
        # questions: [B], answers: [B, C]
        B, C = answers.shape[0], answers.shape[1]
        texts = [f"{q} {a}" for q in questions for a in answers[q]]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=77).to(next(self.clip_model.parameters()).device)
        with torch.no_grad():
            text_feats = self.clip_model.get_text_features(**inputs)  # [B*C, 768]
        return text_feats

    def forward(
        self,
        motion_feats: torch.Tensor,      # [B, T, 2304] from SlowFast
        motion_mask: Optional[torch.Tensor],
        clip_img_feats: torch.Tensor,    # [B, 768] from CLIP (global pool)
        questions: list,                 # [B] strings
        answers: list,                   # [B, C] strings
        labels: Optional[torch.Tensor] = None
    ):
        B, device = motion_feats.size(0), motion_feats.device

        # --- 1. Project motion to CLIP space ---
        mot = self.motion_proj(motion_feats)  # [B, T, 768]

        # --- 2. Motion Encoder ---
        if motion_mask is not None:
            key_padding_mask = (motion_mask == 0)
        else:
            key_padding_mask = None
        mot_tokens = self.motion_encoder(mot, src_key_padding_mask=key_padding_mask)  # [B, T, 768]

        # --- 3. Dynamic Memory Read ---
        mot_read, _ = self.mot_memory(mot_tokens)
        app_read, _ = self.app_memory(clip_img_feats.unsqueeze(1).expand(-1, mot_tokens.size(1), -1))

        # --- 4. Video Tokens: motion + memory ---
        video_tokens = torch.cat([mot_tokens, mot_read, app_read], dim=1)  # [B, 3T, 768]
        video_mask = torch.ones(B, video_tokens.size(1), device=device, dtype=torch.long)

        # --- 5. Encode Text (CLIP) ---
        text_feats = self.encode_text(questions, answers)  # [B*C, 768]
        queries = text_feats.unsqueeze(1)  # [B*C, 1, 768]

        # --- 6. Repeat video for each choice ---
        keys = video_tokens.repeat_interleave(len(answers[0]), dim=0)
        key_padding_mask = video_mask.repeat_interleave(len(answers[0]), dim=0)

        # --- 7. Iterative Reasoning ---
        x = queries
        for _ in range(self.reasoning_steps):
            attn_out, _ = self.cross_attn(x, keys, keys, key_padding_mask=key_padding_mask)
            x = self.norm_attn(x + attn_out)
            x = self.norm_ffn(x + self.ffn(x))

        # --- 8. Classification ---
        combined = torch.cat([queries, x], dim=-1).squeeze(1)  # [B*C, 1536]
        logits = self.classifier(combined).view(B, -1)  # [B, C]

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            output["loss"] = loss
        return output