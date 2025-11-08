import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType # MỚI: Cần thư viện PEFT

class VideoTextLLMQA_V2(nn.Module):
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

        # ======================================================
        # LUỒNG 1: FUSION ĐA PHƯƠNG TIỆN (Giữ nguyên)
        # ======================================================
        
        self.video_proj = nn.Sequential(...) # Giống như cũ
        self.text_proj = nn.Sequential(...)  # Giống như cũ
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(...) # Giống như cũ
        encoder_layer = nn.TransformerEncoderLayer(...) # Giống như cũ
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ======================================================
        # LUỒNG 2: LLM (THAY ĐỔI LỚN)
        # ======================================================

        # THAY ĐỔI: Tải mô hình CausalLM để có thể huấn luyện
        # device_map="auto" rất quan trọng khi huấn luyện mô hình lớn
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            device_map="auto"
            # torch_dtype=torch.bfloat16 # Thêm dòng này để tiết kiệm VRAM
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # MỚI: Thiết lập LoRA để fine-tune LLM hiệu quả
        # Bạn có thể cần điều chỉnh 'target_modules' cho mô hình của mình
        lora_config = LoraConfig(
            r=16, # Rank của ma trận adapter (phổ biến là 8, 16, 32)
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # Các lớp attention để áp dụng LoRA
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM # Hoặc FEATURE_EXTRACTION nếu bạn chỉ muốn lấy features
        )
        # Bọc LLM bằng PEFT
        self.llm = get_peft_model(self.llm, lora_config)
        
        # MỚI: Lớp chiếu đặc trưng đa phương tiện
        # Chiếu từ `hidden_dim` (512) lên `llm.config.hidden_size` (ví dụ: 4096)
        llm_hidden_size = self.llm.config.hidden_size
        self.mm_proj = nn.Sequential(
            nn.Linear(hidden_dim, llm_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(llm_hidden_size)
        )

        # THAY ĐỔI: Classifier
        # Xóa classifier cũ. Classifier mới sẽ hoạt động trên output của LLM
        # Chúng ta sẽ áp dụng một lớp linear lên output của LLM tại vị trí các "choice token"
        self.final_answer_proj = nn.Linear(llm_hidden_size, 1)

    def forward(self, video_feats, text_feats, questions: List[str], choice_texts: List[List[str]]):
        """
        Input vẫn như cũ (hoặc bạn có thể bỏ 'choice_texts' nếu không dùng)
        """
        B, C, _ = text_feats.shape

        # ==================================================
        # LUỒNG 1: TÍNH TOÁN `choice_embs` (Giữ nguyên)
        # ==================================================
        
        v = self.video_proj(video_feats).unsqueeze(1).repeat(1, self.num_video_tokens, 1) # (B, N, H)
        t = self.text_proj(text_feats) # (B, C, H)
        scale = self.film_scale(v.mean(1, keepdim=True))
        shift = self.film_shift(v.mean(1, keepdim=True))
        t_mod = t * scale + shift
        attn_out, _ = self.cross_attn(query=v, key=t_mod, value=t_mod) # (B, N, H)
        x = torch.cat([attn_out, t_mod], dim=1) # (B, N+C, H)
        x = self.transformer(x)
        choice_embs = x[:, self.num_video_tokens:, :] # (B, C, H)

        # ==================================================
        # LUỒNG 2: "TIÊM" VÀO LLM
        # ==================================================

        # 1. MỚI: Chiếu `choice_embs` lên không gian của LLM
        # Đây là các "soft token" đại diện cho lựa chọn đã fusion với video
        projected_choice_embs = self.mm_proj(choice_embs) # (B, C, LLM_H)

        # 2. MỚI: Chuẩn bị input text (chỉ câu hỏi)
        # Tạo một prompt template
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
        
        # Lấy text embeddings từ LLM
        text_embeddings = self.llm.get_input_embeddings()(tokenized_text.input_ids) # (B, T, LLM_H)
        text_attention_mask = tokenized_text.attention_mask # (B, T)
        
        T = text_embeddings.shape[1] # Độ dài chuỗi text

        # 3. MỚI: Kết hợp (Inject) Text Embeddings và Choice Embeddings
        # [Text_Token_1, ..., Text_Token_T, Choice_Token_1, ..., Choice_Token_C]
        inputs_embeds = torch.cat([
            text_embeddings, 
            projected_choice_embs
        ], dim=1) # (B, T+C, LLM_H)
        
        # Tạo attention mask tương ứng
        choice_attention_mask = torch.ones(B, C, device=self.device).long()
        attention_mask = torch.cat([
            text_attention_mask, 
            choice_attention_mask
        ], dim=1) # (B, T+C)

        # 4. MỚI: Forward qua LLM (BỎ `torch.no_grad()`)
        # Cần bật training (ví dụ: model.train())
        llm_output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_states = llm_output.last_hidden_state # (B, T+C, LLM_H)

        # 5. MỚI: Lấy output tại vị trí các "choice token"
        # Chúng ta chỉ quan tâm đến C token cuối cùng
        choice_outputs = last_hidden_states[:, -C:, :] # (B, C, LLM_H)

        # 6. MỚI: Áp dụng classifier cuối cùng
        # Dự đoán điểm số cho từng choice token
        logits = self.final_answer_proj(choice_outputs).squeeze(-1) # (B, C)

        return logits