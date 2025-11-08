import torch
from pytorchvideo.models.hub import slowfast_r50
import cv2
import numpy as np
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = slowfast_r50(pretrained=True).to(DEVICE)
model.eval()

# ---- HOOK LAYER 2304-D ----
embedding = None
def hook_fn(module, inp, out):
    global embedding
    embedding = inp[0]   # input of classifier head

head_layer = model.blocks[6]      # SlowFast head
head_layer.register_forward_hook(hook_fn)


# ---- PREPROCESS ----
preprocess = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225,0.225,0.225])
])


# ✅ ĐỌC TOÀN BỘ VIDEO -> MẢNG FRAME (RGB)
def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(f))
    cap.release()
    return frames   # list of Tensors (C,H,W)


# ✅ PACK PATHWAYS SLOWFAST
def pack_pathways(frames):
    fast = frames             # (B, C, 32, H, W)
    slow = frames[:, :, ::4]  # (B, C, 8, H, W)
    return [slow, fast]


# ✅ TRÍCH XUẤT TOÀN VIDEO → DANH SÁCH EMBEDDING 2304D
def extract_full_video(video_path, clip_len=32):
    global embedding
    all_frames = load_all_frames(video_path)

    if len(all_frames) == 0:
        raise RuntimeError("Không đọc được frame video")

    embeddings = []

    # Chia video thành các clip 32 frames
    for i in range(0, len(all_frames), clip_len):
        clip_frames = all_frames[i:i+clip_len]

        # duplicate cho đủ 32 frames
        if len(clip_frames) < clip_len:
            clip_frames += [clip_frames[-1]] * (clip_len - len(clip_frames))

        # Tensor shape (C, T, H, W)
        clip = torch.stack(clip_frames, dim=1).unsqueeze(0).to(DEVICE)

        inputs = pack_pathways(clip)

        embedding = None
        with torch.no_grad():
            _ = model(inputs)

        if embedding is None:
            raise RuntimeError("Hook không chạy – kiểm tra pathways")

        embeddings.append(embedding.squeeze().cpu())

    # output: (num_clips, 2304)
    return torch.stack(embeddings)
