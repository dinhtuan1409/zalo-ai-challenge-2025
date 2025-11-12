# ============================================================
# ✅ EXTRACT APPEARANCE FEATURES (CLIP ViT-L/14@336px)
# For public_test (no support_frames)
# Compatible with HME-VideoQA setup
# ============================================================

import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
JSON_PATH = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
VIDEO_ROOT = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/videos"
OUTPUT_DIR = "/kaggle/working/features_clip_test/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-L/14@336px"

NUM_SAMPLE_FRAMES = 16  # số khung hình sẽ sample đều toàn video
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD CLIP MODEL
# ============================================================
print(f"🔄 Loading CLIP model: {CLIP_MODEL}")
model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
model.eval()

# ============================================================
# LOAD JSON DATA
# ============================================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

print(f"📄 Loaded {len(data)} test samples from {JSON_PATH}")

# ============================================================
# MAIN LOOP
# ============================================================
for item in tqdm(data, desc="Extracting CLIP features"):
    vid_path = os.path.join("/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test", item["video_path"])
    vid_id = item["id"]

    if not os.path.exists(vid_path):
        print(f"⚠️ Missing video: {vid_path}")
        continue

    try:
        # Load video bằng decord
        vr = VideoReader(vid_path, ctx=cpu(0))
        num_frames = len(vr)

        if num_frames == 0:
            print(f"⚠️ Empty video: {vid_id}")
            continue

        # Uniform sample N frames toàn video
        if num_frames <= NUM_SAMPLE_FRAMES:
            frame_indices = list(range(num_frames))
        else:
            frame_indices = np.linspace(0, num_frames - 1, NUM_SAMPLE_FRAMES, dtype=int)

        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in frame_indices]

        # Tiền xử lý cho CLIP
        inputs = torch.stack([preprocess(im) for im in frames]).to(DEVICE)

        # Encode bằng CLIP
        with torch.no_grad():
            feats = model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

        # Mean pooling
        feat_mean = feats.mean(dim=0, keepdim=True).cpu()

        # Lưu tensor [1, 768]
        torch.save(feat_mean, os.path.join(OUTPUT_DIR, f"{vid_id}_appearance.pt"))

    except Exception as e:
        print(f"❌ Error processing {vid_id}: {e}")
        continue

print(f"\n✅ Done! CLIP appearance features saved to: {OUTPUT_DIR}")
