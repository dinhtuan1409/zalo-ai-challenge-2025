import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List

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
        # Chúng ta vẫn tải LLM, nhưng không nhất thiết phải ở device_map="auto"
        # nếu chúng ta chỉ dùng nó để trích xuất đặc trưng (với no_grad)
        # device_map="auto" vẫn ổn nếu bạn có đủ VRAM
        self.llm = AutoModel.from_pretrained(llm_model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- MỚI: Lớp chiếu cho LLM output ----
        # Chiếu output của LLM (llm.config.hidden_size) về hidden_dim
        self.llm_proj = nn.Linear(self.llm.config.hidden_size, hidden_dim)

        # ---- THAY ĐỔI: Final Classifier (để kết hợp cả hai luồng) ----
        # Input sẽ là:
        # 1. choice_embs (H)
        # 2. llm_pooled (đã chiếu) (H)
        # 3. Tương tác (H)
        # Tổng cộng: hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # THAY ĐỔI: Input là hidden_dim * 3
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    # THAY ĐỔI: Signature của hàm forward
    def forward(self, video_feats, text_feats, questions: List[str], choice_texts: List[List[str]]):
        """
        video_feats: (B, D_v)
        text_feats: (B, C, D_t) - Đây là feature embeddings (ví dụ: CLIP)
        questions: list[str], length B - Văn bản thô của câu hỏi
        choice_texts: list[list[str]], shape (B, C) - Văn bản thô của các lựa chọn
        """
        B, C, _ = text_feats.shape

        # ==================================================
        # LUỒNG 1: FUSION ĐA PHƯƠNG TIỆN (Video + Text Embeddings)
        # ==================================================
        
        # 1. Multi-frame video tokens
        v = self.video_proj(video_feats).unsqueeze(1).repeat(1, self.num_video_tokens, 1) # (B, N, H)

        # 2. Text projection (Sử dụng text_feats)
        t = self.text_proj(text_feats) # (B, C, H)

        # 3. FiLM modulation
        scale = self.film_scale(v.mean(1, keepdim=True))
        shift = self.film_shift(v.mean(1, keepdim=True))
        t_mod = t * scale + shift

        # 4. Cross-Attention: video queries text
        attn_out, _ = self.cross_attn(query=v, key=t_mod, value=t_mod) # (B, N, H)

        # 5. Concat video + text -> Transformer
        x = torch.cat([attn_out, t_mod], dim=1) # (B, N+C, H)
        x = self.transformer(x)

        # 6. Lấy text embeddings (bỏ video tokens)
        choice_embs = x[:, self.num_video_tokens:, :] # (B, C, H)

        # ==================================================
        # LUỒNG 2: SUY LUẬN NGÔN NGỮ (Question + Choice Texts)
        # ==================================================

        # 7. LLM reasoning (Sử dụng questions + choice_texts)
        if not choice_texts or len(choice_texts) != B:
            raise ValueError("choice_texts phải được cung cấp và có batch size B")
        
        prompts = []
        for i in range(B):
            # Tạo prompt từ văn bản thô
            choices_str = " | ".join([f"{j+1}: {choice_texts[i][j]}" for j in range(C)])
            prompts.append(f"Question: {questions[i]}\nChoices: {choices_str}")

        # Tokenize
        tokenized = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=256 # Giới hạn max_length để tiết kiệm bộ nhớ
        ).to(self.device)

        # Pass through LLM (không cần tính gradient)
        with torch.no_grad():
            llm_output = self.llm(**tokenized).last_hidden_state # (B, seq_len, llm_hidden_size)
        
        # Pooling: Lấy embedding của token cuối cùng (thường là EOS hoặc token quan trọng)
        # Hoặc dùng mean pooling
        llm_pooled = llm_output.mean(1) # (B, llm_hidden_size)

        # ==================================================
        # KẾT HỢP HAI LUỒNG
        # ==================================================
        
        # 8. MỚI: Chiếu LLM output về hidden_dim
        # llm_pooled đại diện cho "suy luận" về câu hỏi VÀ tất cả các lựa chọn
        reasoning_vec = self.llm_proj(llm_pooled) # (B, H)

        # 9. MỚI: Mở rộng reasoning_vec để khớp với số choices (C)
        # (B, H) -> (B, 1, H) -> (B, C, H)
        reasoning_vec_expanded = reasoning_vec.unsqueeze(1).repeat(1, C, 1)

        # 10. MỚI: Kết hợp đặc trưng
        # Kết hợp đặc trưng fusion (choice_embs) và đặc trưng suy luận (reasoning_vec_expanded)
        combined_features = torch.cat([
            choice_embs,                  # "Lựa chọn này khớp video không?"
            reasoning_vec_expanded,       # "Bối cảnh chung của câu hỏi/lựa chọn là gì?"
            choice_embs * reasoning_vec_expanded # Tương tác
        ], dim=2) # (B, C, H*3)

        # 11. MỚI: Đưa qua classifier đã được cập nhật
        logits = self.classifier(combined_features).squeeze(-1) # (B, C)
        
        return logits