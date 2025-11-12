import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
JSON_PATH = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
VIDEO_ROOT = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test"
OUTPUT_DIR = "/kaggle/working/features/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-L/14@336px"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD CLIP MODEL
# ==========================
print(f"🔄 Loading CLIP model: {CLIP_MODEL}")
model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
model.eval()

# ==========================
# LOAD JSON DATA
# ==========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

print(f"📄 Loaded {len(data)} samples from {JSON_PATH}")

# ==========================
# MAIN LOOP
# ==========================
for item in tqdm(data, desc="Extracting features"):
    vid_path = os.path.join(VIDEO_ROOT, item["video_path"])
    support_times = item["support_frames"]
    vid_id = item["id"]

    # Kiểm tra video tồn tại
    if not os.path.exists(vid_path):
        print(f"⚠️ Video not found: {vid_path}")
        continue

    try:
        # Mở video
        vr = VideoReader(vid_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        
        if fps is None or fps <= 0:
            fps = 30.0  # fallback an toàn
        
        # Quy đổi giây → frame index, có clamp về trong phạm vi hợp lệ
        raw_indices = [int(t * fps) for t in support_times]
        frame_indices = []
        for i in raw_indices:
            if i < 0:
                frame_indices.append(0)
            elif i >= num_frames:
                frame_indices.append(num_frames - 1)
            else:
                frame_indices.append(i)
        frame_indices = sorted(list(set(frame_indices)))

        if len(frame_indices) == 0:
            print(f"⚠️ No valid frames for {vid_id} (max frame {num_frames})")
            continue

        # Đọc frame tương ứng
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in frame_indices]
        inputs = torch.stack([preprocess(im) for im in frames]).to(DEVICE)

        # Trích xuất đặc trưng CLIP
        with torch.no_grad():
            feats = model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 norm

        # Mean pooling (gộp frame)
        feat_mean = feats.mean(dim=0, keepdim=True).cpu()

        # Lưu
        save_path = os.path.join(OUTPUT_DIR, f"{vid_id}.pt")
        torch.save(feat_mean, save_path)

    except Exception as e:
        print(f"❌ Error processing {vid_id}: {e}")
        continue

print(f"\n✅ Done! All features saved to: {OUTPUT_DIR}")
