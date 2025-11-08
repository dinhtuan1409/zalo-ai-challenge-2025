import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class VideoTextLLMQA(nn.Module):
    def __init__(self,
                 video_dim=2304,
                 text_dim=768,
                 hidden_dim=512,
                 nhead=8,
                 num_layers=2,
                 num_video_tokens=4,
                 dropout=0.2,
                 llm_model_name="meta-llama/Llama-2-7b-hf",
                 device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_video_tokens = num_video_tokens

        # ---- Video Projection ----
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Text Projection ----
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- FiLM modulation ----
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        # ---- Cross-Attention for video->text ----
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=nhead,
                                                batch_first=True,
                                                dropout=dropout)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- LLM ----
        self.llm = AutoModel.from_pretrained(llm_model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # ---- Final Classifier (on LLM embeddings) ----
        self.classifier = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, video_feats, text_feats, questions=None):
        """
        video_feats: (B, D_v)
        text_feats: (B, C, D_t)
        questions: list of strings, length B (optional, for LLM prompting)
        """
        B, C, _ = text_feats.shape

        # 1. Multi-frame video tokens
        v = self.video_proj(video_feats).unsqueeze(1).repeat(1, self.num_video_tokens, 1)  # (B, N, H)

        # 2. Text projection
        t = self.text_proj(text_feats)  # (B, C, H)

        # 3. FiLM modulation
        scale = self.film_scale(v.mean(1, keepdim=True))
        shift = self.film_shift(v.mean(1, keepdim=True))
        t_mod = t * scale + shift

        # 4. Cross-Attention: video queries text
        attn_out, _ = self.cross_attn(query=v, key=t_mod, value=t_mod)  # (B, N, H)

        # 5. Concat video + text -> Transformer
        x = torch.cat([attn_out, t_mod], dim=1)  # (B, N+C, H)
        x = self.transformer(x)

        # 6. Lấy text embeddings (bỏ video tokens)
        choice_embs = x[:, self.num_video_tokens:, :]  # (B, C, H)

        # 7. LLM reasoning
        # Tạo prompt: question + choices
        if questions is None:
            questions = ["Answer the question based on the video."] * B
        prompts = []
        for i in range(B):
            choices_text = " | ".join([f"{j+1}: choice" for j in range(C)])
            prompts.append(f"Question: {questions[i]}\nChoices: {choices_text}")

        # Tokenize
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Pass through LLM
        with torch.no_grad():
            llm_output = self.llm(**tokenized).last_hidden_state  # (B, seq_len, hidden_size)
        
        # Pooling: mean over sequence
        llm_pooled = llm_output.mean(1)  # (B, hidden_size)

        # 8. Final classifier: combine LLM pooled + choice embeddings
        # Simple way: apply classifier on choice_embs individually
        logits = self.classifier(choice_embs).squeeze(-1)  # (B, C)
        return logits
