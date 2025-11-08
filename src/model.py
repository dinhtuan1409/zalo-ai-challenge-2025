import torch
import torch.nn as nn
# Import các thư viện cần thiết, đặc biệt là BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import List 

class VideoTextLLMQA_V2(nn.Module):
    def __init__(self,
                 video_dim=2304,
                 text_dim=768,
                 hidden_dim=512,
                 nhead=8,
                 num_layers=2,
                 num_video_tokens=4,
                 dropout=0.2,
                 llm_model_name="mistralai/Mistral-7B-v0.1",
                 device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_video_tokens = num_video_tokens

        # ======================================================
        # LUỒNG 1: FUSION ĐA PHƯƠNG TIỆN (Video + Text Embeddings)
        # ======================================================
        
        # Video Projection
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text Projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # FiLM modulation (cho video mod text)
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=nhead,
                                                batch_first=True,
                                                dropout=dropout)

        # Transformer Encoder (Fusion)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ======================================================
        # LUỒNG 2: LLM (PEFT Injection)
        # ======================================================

        # ❗ Cấu hình 4-bit Quantization để tiết kiệm VRAM ❗
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # Normal Float 4-bit
            # Sử dụng bfloat16 nếu GPU hỗ trợ, nếu không thì dùng float16
            bnb_4bit_compute_dtype=torch.bfloat16, 
        )

        # Tải mô hình CausalLM và áp dụng LoRA
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            quantization_config=bnb_config, # Áp dụng quantization
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cấu hình và áp dụng LoRA
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM 
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # Lớp chiếu đặc trưng đa phương tiện (từ H -> LLM_H)
        llm_hidden_size = self.llm.config.hidden_size
        self.mm_proj = nn.Sequential(
            nn.Linear(hidden_dim, llm_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(llm_hidden_size)
        )

        # Classifier cuối cùng trên output của LLM
        self.final_answer_proj = nn.Linear(llm_hidden_size, 1)

    def forward(self, video_feats, text_feats, questions: List[str], choice_texts: List[List[str]]):
        B, C, _ = text_feats.shape

        # ==================================================
        # 1. TÍNH TOÁN `choice_embs` (Fusion)
        # ==================================================
        
        # 1. Video Tokens
        v = self.video_proj(video_feats).unsqueeze(1).repeat(1, self.num_video_tokens, 1) # (B, N, H)

        # 2. Text Projection + FiLM modulation
        t = self.text_proj(text_feats) # (B, C, H)
        scale = self.film_scale(v.mean(1, keepdim=True))
        shift = self.film_shift(v.mean(1, keepdim=True))
        t_mod = t * scale + shift

        # 3. Cross-Attention
        attn_out, _ = self.cross_attn(query=v, key=t_mod, value=t_mod) # (B, N, H)

        # 4. Transformer (Fusion)
        x = torch.cat([attn_out, t_mod], dim=1) # (B, N+C, H)
        x = self.transformer(x)
        choice_embs = x[:, self.num_video_tokens:, :] # (B, C, H)

        # ==================================================
        # 2. "TIÊM" VÀO LLM
        # ==================================================

        # 5. Chiếu `choice_embs` lên không gian của LLM
        projected_choice_embs = self.mm_proj(choice_embs) # (B, C, LLM_H)

        # 6. Chuẩn bị input text (Câu hỏi)
        prompts = [
            f"Question: {q}\nBased on the video, which of the following is correct?" 
            for q in questions
        ]
        
        tokenized_text = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Lấy text embeddings
        text_embeddings = self.llm.get_input_embeddings()(tokenized_text.input_ids) # (B, T, LLM_H)
        text_attention_mask = tokenized_text.attention_mask # (B, T)
        
        # 7. Kết hợp (Inject) Text và Choice Embeddings
        inputs_embeds = torch.cat([
            text_embeddings, 
            projected_choice_embs
        ], dim=1) # (B, T+C, LLM_H)
        
        # Tạo attention mask cho các choice token
        choice_attention_mask = torch.ones(B, C, device=self.device).long()
        attention_mask = torch.cat([
            text_attention_mask, 
            choice_attention_mask
        ], dim=1) # (B, T+C)

        # 8. Forward qua LLM
        llm_output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_states = llm_output.last_hidden_state # (B, T+C, LLM_H)

        # 9. Lấy output tại vị trí các "choice token"
        choice_outputs = last_hidden_states[:, -C:, :] # (B, C, LLM_H)

        # 10. Áp dụng classifier
        logits = self.final_answer_proj(choice_outputs).squeeze(-1) # (B, C)

        return logits
