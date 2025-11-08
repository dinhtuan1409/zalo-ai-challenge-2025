import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from peft import PeftModel # ❗ CẦN THIẾT cho việc tải LoRA adapter

# --- Import các file code của bạn ---
from dataset import FeatureVideoQADatasetMPNET, collate_fn_mpnet, load_text_encoder
# Sử dụng tên class mới đã được PEFT hóa
from model import VideoTextLLMQA_V2 

# =============================
# CONFIG
# =============================
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
VIDEO_FEAT_DIR = "Feature/public_test" 

VIDEO_FEAT_DIM = 2304 

# ❗ SỬA LỖI: Điểm checkpoint là thư mục adapter, không phải file .pt
CHECKPOINT = "/kaggle/working/best_adapter" 

OUTPUT_FILE = "/kaggle/working/submission.json"
BATCH_SIZE = 32 
NUM_WORKERS = os.cpu_count()
USE_FP16 = True
# =============================


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# Load text encoder
# -----------------------------------
print("🔄 Loading text encoder...")
text_encoder = load_text_encoder(device)

# -----------------------------------
# Load model & Adapter (QUAN TRỌNG)
# -----------------------------------
print("🔄 Building model (VideoTextLLMQA_V2)...")
model = VideoTextLLMQA_V2(
    video_dim=VIDEO_FEAT_DIM, 
    text_dim=768,
    hidden_dim=512,
    device=device
).to(device) # Chuyển các lớp fusion sang GPU

# ❗ BƯỚC SỬA LỖI CHÍNH: Tải adapter LoRA vào submodule LLM ❗
if os.path.exists(CHECKPOINT):
    print(f"Loading LoRA adapter from {CHECKPOINT}...")
    # Tải adapter LoRA vào self.llm (là mô hình Mistral 7B đã được lượng tử hóa)
    model.llm = PeftModel.from_pretrained(model.llm, CHECKPOINT)
    print("✅ Adapter loaded successfully.")
else:
    raise FileNotFoundError(f"Adapter checkpoint not found at {CHECKPOINT}. Did training complete?")


model.eval()

# -----------------------------------
# Load test dataset
# -----------------------------------
print("🔄 Loading test dataset...")
test_ds = FeatureVideoQADatasetMPNET(
    json_path=TEST_JSON,
    video_feat_dir=VIDEO_FEAT_DIR,
    video_feat_dim=VIDEO_FEAT_DIM, 
    text_encoder=text_encoder, 
    preload_text=True,
    is_test=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_mpnet,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

results = []

# -----------------------------------
# Inference loop
# -----------------------------------
with torch.no_grad():
    pbar = tqdm(test_loader, desc="🚀 Inference")
    for batch in pbar:

        video_feats = batch["video_feats"].to(device)
        text_feats = batch["text_feats"].to(device)
        questions = batch["questions"]       
        choice_texts = batch["choice_texts"] 
        ids  = batch["ids"]

        # Dùng autocast (FP16)
        with autocast(enabled=USE_FP16):
            # Cần truyền đủ 4 đối số cho model V2
            logits = model(video_feats, text_feats, questions, choice_texts) 
        
        preds = logits.argmax(dim=1).cpu().tolist()

        for qid, p in zip(ids, preds):
            results.append({
                "id": qid,
                "answer": int(p) 
            })

# -----------------------------------
# Save submission
# -----------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    output_data = {"data": results}
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ DONE! Saved submission to {OUTPUT_FILE}")