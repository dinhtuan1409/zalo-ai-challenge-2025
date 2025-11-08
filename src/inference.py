import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

# --- Import các file code của bạn ---
# (Giả sử chúng nằm trong các file .py tương ứng)
from dataset import FeatureVideoQADatasetMPNET, collate_fn_mpnet, load_text_encoder
from model import ContextTransformerQAModel 

# =============================
#  CONFIG
# =============================
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
VIDEO_FEAT_DIR = "Feature/public_test" # folder chứa .pt video test

# ❗ ĐỒNG BỘ: Phải khớp với script train
VIDEO_FEAT_DIM = 2304 

# ❗ ĐỒNG BỘ: Phải khớp với nơi script train lưu model
CHECKPOINT = "/kaggle/working/best_model.pt" 

OUTPUT_FILE = "/kaggle/working/submission.json"
BATCH_SIZE = 32 # Có thể tăng BATCH_SIZE khi inference
NUM_WORKERS = os.cpu_count()
USE_FP16 = True
# =============================


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# Load text encoder (giống train)
# -----------------------------------
print("🔄 Loading text encoder...")
text_encoder = load_text_encoder(device)

# -----------------------------------
# Load model
# -----------------------------------
print("🔄 Loading model...")
model = ContextTransformerQAModel(
    video_dim=VIDEO_FEAT_DIM, # ❗ ĐỒNG BỘ: Dùng 2304
    text_dim=768
).to(device)

# 💡 TỐI ƯU: (PyTorch 2.0+) Tăng tốc model
if hasattr(torch, 'compile'):
    print("Compiling model (PyTorch 2.0+)...")
    model = torch.compile(model)

print(f"Loading checkpoint from {CHECKPOINT}...")
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model"], strict=True) # Dùng strict=True để đảm bảo
model.eval()

# -----------------------------------
# Load test dataset
# -----------------------------------
print("🔄 Loading test dataset...")
test_ds = FeatureVideoQADatasetMPNET(
    json_path=TEST_JSON,
    video_feat_dir=VIDEO_FEAT_DIR,
    video_feat_dim=VIDEO_FEAT_DIM, # ❗ ĐỒNG BỘ
    text_encoder=text_encoder,     # ❗ ĐỒNG BỘ: Dùng text encoder đã load
    preload_text=True,
    is_test=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_mpnet,
    num_workers=NUM_WORKERS, # 💡 TỐI ƯU
    pin_memory=True
)

results = []

# -----------------------------------
# Inference loop
# -----------------------------------
with torch.no_grad():
    pbar = tqdm(test_loader, desc="🚀 Inference")
    for batch in pbar:

        video = batch["video_feats"].to(device)
        text  = batch["text_feats"].to(device)
        ids   = batch["ids"]

        # 💡 TỐI ƯU: Dùng autocast
        with autocast(enabled=USE_FP16):
            logits = model(video, text) # (B, C)
        
        preds = logits.argmax(dim=1).cpu().tolist()

        for qid, p in zip(ids, preds):
            results.append({
                "id": qid,
                "answer": int(p) # Submit index (0, 1, 2, 3...)
            })

# -----------------------------------
# Save submission
# -----------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # Định dạng output chuẩn cho Zalo: {"data": [...]}
    output_data = {"data": results}
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ DONE! Saved submission to {OUTPUT_FILE}")