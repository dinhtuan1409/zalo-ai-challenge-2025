# inference_hme.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from peft import PeftModel  # Cho LoRA adapter

# --- Import các file code của bạn ---
from dataset_test import FeatureVideoQAHME_MC, collate_fn_hme, load_text_encoder
from model import HME_MC  # HME multi-choice model

# =============================
# CONFIG
# =============================
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
FEATURE_DIR_APP = "/kaggle/working/feature_motion_appeare/test_appear"
FEATURE_DIR_MOT = "/kaggle/working/feature_motion_appeare/test_motion"

CHECKPOINT = "/kaggle/working/hme_ckpt"
OUTPUT_FILE = "/kaggle/working/submission.json"
BATCH_SIZE = 16
USE_FP16 = True
NUM_WORKERS = os.cpu_count()
MOTION_DIM = 2304
APPEARANCE_DIM = 768
TEXT_DIM = 768
MOTION_PROJ_DIM = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Load text encoder
# =============================
print("🔄 Loading text encoder...")
text_encoder = load_text_encoder(device=device)

# =============================
# Build HME_MC model
# =============================
print("🔄 Building HME_MC model...")
model = HME_MC(
    motion_dim=MOTION_DIM,
    appearance_dim=APPEARANCE_DIM,
    text_dim=TEXT_DIM,
    motion_proj_dim=MOTION_PROJ_DIM
).to(device)

# Load LoRA adapter vào submodule LLM nếu có
if os.path.exists(CHECKPOINT):
    print(f"Loading LoRA adapter from {CHECKPOINT}...")
    model.llm = PeftModel.from_pretrained(model.llm, CHECKPOINT)
    print("✅ Adapter loaded successfully.")
else:
    print("⚠️ No adapter found, using base model.")

model.eval()

# =============================
# Load test dataset
# =============================
print("🔄 Loading test dataset...")
test_ds = FeatureVideoQAHME_MC(
    json_path=TEST_JSON,
    feature_dir_appearance=FEATURE_DIR_APP,
    feature_dir_motion=FEATURE_DIR_MOT,
    text_encoder=text_encoder,
    preload_text=True,
    is_test=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_hme,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =============================
# Inference loop
# =============================
results = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc="🚀 Inference")
    for batch in pbar:
        # Unpack
        motion_feats = batch["motion_feats"].to(device)           # [B, T, 2304]
        motion_mask = batch["motion_mask"].to(device)             # [B, T]
        appearance_feats = batch["appearance_feats"].to(device)   # [B, 768]
        text_feats = batch["text_feats"].to(device)               # [B, C, 768]
        ids = batch["ids"]

        # FP16 autocast
        with autocast(enabled=USE_FP16):
            out = model(
                motion_feats=motion_feats,
                motion_mask=motion_mask,
                appearance_feats=appearance_feats,
                text_feats=text_feats
            )

        logits = out["logits"]          # [B, C]
        preds = logits.argmax(dim=1).cpu().tolist()

        for qid, p in zip(ids, preds):
            results.append({"id": qid, "answer": int(p)})

# =============================
# Save submission
# =============================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"data": results}, f, ensure_ascii=False, indent=2)

print(f"✅ DONE! Saved submission to {OUTPUT_FILE}")
