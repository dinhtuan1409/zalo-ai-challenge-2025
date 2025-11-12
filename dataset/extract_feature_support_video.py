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
    embedding = inp[0]   # input of the head = (B, 2304)

head_layer = model.blocks[6]      # classifier head
head_layer.register_forward_hook(hook_fn)


# ---- PREPROCESS ----
preprocess = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225,0.225,0.225])
])


# ✅ LOAD CLIP QUANH SUPPORT FRAME
def load_clip_around_support(video_path, support_time, clip_len=32):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame trung tâm
    center = int(support_time * fps)

    start = max(0, center - clip_len // 2)
    end = min(total, start + clip_len)

    # điều chỉnh nếu end-start < clip_len
    if end - start < clip_len:
        start = max(0, end - clip_len)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    while len(frames) < clip_len:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(f))

    cap.release()

    # duplicate frame cuối nếu thiếu
    while len(frames) < clip_len:
        frames.append(frames[-1])

    x = torch.stack(frames, dim=1)        # (C, T, H, W)
    return x.unsqueeze(0)                 # (1, C, T, H, W)


# ✅ PACK PATHWAYS CHUẨN SLOWFAST
def pack_pathways(frames):
    fast = frames             # (B, C, 32, H, W)
    slow = frames[:, :, ::4]  # (B, C, 8, H, W)
    return [slow, fast]


# ✅ EXTRACT FEATURE TỪ SUPPORT FRAME
def extract_with_support(video_path, support_time):
    global embedding
    embedding = None

    clip = load_clip_around_support(video_path, support_time, clip_len=32)
    clip = clip.to(DEVICE)

    inputs = pack_pathways(clip)

    with torch.no_grad():
        _ = model(inputs)

    if embedding is None:
        raise RuntimeError("Hook không chạy – kiểm tra pathways")

    return embedding.squeeze().cpu()     # vector (2304,)
