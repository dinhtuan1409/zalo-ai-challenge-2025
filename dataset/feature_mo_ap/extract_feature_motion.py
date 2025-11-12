# ============================================================
# ✅ EXTRACT MOTION FEATURES (2304D) FROM SlowFast-R50
# Compatible with HME-VideoQA (Fan et al.)
# ============================================================

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from pytorchvideo.models.hub import slowfast_r50

# ============================================================
# CONFIG
# ============================================================
VIDEO_ROOT = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/videos"
OUTPUT_DIR = "/kaggle/working/features_motion_slowfast_test/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 32
ALPHA = 4  # Slow path temporal ratio
BATCH_SAVE = True  # save per video as .pt
torch.set_grad_enabled(False)

# ============================================================
# LOAD MODEL
# ============================================================
print("🔄 Loading SlowFast-R50 pretrained on Kinetics-400...")
model = slowfast_r50(pretrained=True).to(DEVICE)
model.eval()

# Hook feature before classification head (2304-D embedding)
embedding = None
def hook_fn(module, inp, out):
    global embedding
    embedding = inp[0]

head_layer = model.blocks[6]  # last block (head)
head_layer.register_forward_hook(hook_fn)

# ============================================================
# PREPROCESS FUNCTION
# ============================================================
preprocess = T.Compose([
    T.ToTensor(),  # (H,W,C) -> (C,H,W), [0,1]
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def load_all_frames(video_path):
    """Read all RGB frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(f))
    cap.release()
    return frames  # list of tensors (C,H,W)

def sample_uniform(frames, num_samples=32):
    """Uniformly sample N frames."""
    if len(frames) == 0:
        raise RuntimeError("Video has no frames!")
    idx = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    return [frames[i] for i in idx]

def pack_pathways(frames_tensor):
    """Pack SlowFast dual pathway."""
    fast = frames_tensor
    slow = frames_tensor[:, :, ::ALPHA]
    return [slow, fast]

# ============================================================
# MAIN EXTRACTION LOOP
# ============================================================
video_files = sorted([f for f in os.listdir(VIDEO_ROOT) if f.endswith(".mp4")])
print(f"📁 Found {len(video_files)} videos")

for vid_file in tqdm(video_files, desc="Extracting motion features"):
    vid_id = os.path.splitext(vid_file)[0]
    vid_path = os.path.join(VIDEO_ROOT, vid_file)

    try:
        frames = load_all_frames(vid_path)
        if len(frames) == 0:
            print(f"⚠️ Skipping empty video: {vid_file}")
            continue

        embeddings = []
        # Split into clips of 32 frames
        for i in range(0, len(frames), CLIP_LEN):
            clip_frames = frames[i:i+CLIP_LEN]
            # Duplicate last frame if short
            if len(clip_frames) < CLIP_LEN:
                clip_frames += [clip_frames[-1]] * (CLIP_LEN - len(clip_frames))

            # Uniform sample (ensures temporal consistency)
            clip_frames = sample_uniform(clip_frames, CLIP_LEN)

            # Stack to tensor [1, C, T, H, W]
            clip = torch.stack(clip_frames, dim=1).unsqueeze(0).to(DEVICE)
            inputs = pack_pathways(clip)

            # Forward pass
            embedding = None
            _ = model(inputs)
            if embedding is None:
                raise RuntimeError("⚠️ Hook failed to capture embedding")

            embeddings.append(embedding.squeeze().cpu())

        # (num_clips, 2304)
        feat = torch.stack(embeddings)
        # Normalize L2
        feat = feat / feat.norm(p=2, dim=1, keepdim=True)

        if BATCH_SAVE:
            torch.save(feat, os.path.join(OUTPUT_DIR, f"{vid_id}_motion.pt"))

    except Exception as e:
        print(f"❌ Error processing {vid_file}: {e}")
        continue

print(f"\n✅ Done! Motion features (2304D) saved in {OUTPUT_DIR}")
