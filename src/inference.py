# inference_hme_clip.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

from dataset import FeatureVideoQAHME_MC, collate_fn_hme, load_text_encoder, load_clip_encoder
from model import HME_MC

# --- Self-attention pooling helper (optional) ---
class AttentionPooling(torch.nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        pooled = attn_out.mean(dim=1)
        return pooled

def inference_hme_clip(
    json_path: str,
    feature_dir_app: str,
    feature_dir_mot: str,
    checkpoint_path: str,
    output_file: str,
    batch_size: int = 16,
    use_fp16: bool = True,
    motion_dim: int = 2304,
    appearance_dim: int = 768,
    text_dim: int = 1536,  # MPNet + CLIP
    motion_proj_dim: int = 1024,
    hidden_dim: int = 1024,
    num_motion_layers: int = 2,
    num_workers: int = None,
    preload_text: bool = True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = num_workers or os.cpu_count()

    # --- Load encoders ---
    print("🔄 Loading MPNet + CLIP encoders...")
    mpnet = load_text_encoder(device="cpu")
    clip_model, clip_processor = load_clip_encoder(device="cpu")

    # --- Build dataset ---
    print("🔄 Loading dataset...")
    dataset = FeatureVideoQAHME_MC(
        json_path=json_path,
        feature_dir_appearance=feature_dir_app,
        feature_dir_motion=feature_dir_mot,
        text_encoder=mpnet,
        clip_model=clip_model,
        clip_processor=clip_processor,
        preload_text=preload_text,
        is_test=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hme,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Build model ---
    print("🔄 Building HME_MC model...")
    model = HME_MC(
        motion_dim=motion_dim,
        appearance_dim=appearance_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        motion_proj_dim=motion_proj_dim,
        num_motion_layers=num_motion_layers
    ).to(device)

    # --- Load checkpoint ---
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    attn_pool = AttentionPooling(motion_proj_dim).to(device)
    attn_pool.eval()

    # --- Inference ---
    results = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="🚀 Inference")
        for batch in pbar:
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            text = batch["text_feats"].to(device)  # [B, C, 1536]
            motion_mask = batch["motion_mask"].to(device)
            ids = batch["ids"]

            with autocast(enabled=use_fp16):
                out = model(
                    motion_feats=motion,
                    motion_mask=motion_mask,
                    appearance_feats=appearance,
                    text_feats=text
                )
                logits = out["logits"]  # [B, C]

            preds = logits.argmax(dim=1).cpu().tolist()
            for qid, p in zip(ids, preds):
                results.append({"id": qid, "answer": int(p)})

    # --- Save results ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"data": results}, f, ensure_ascii=False, indent=2)

    print(f"✅ DONE! Saved submission to {output_file}")
    return results

# --- CLI ---
if __name__ == "__main__":
    inference_hme_clip(
        json_path="/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json",
        feature_dir_app="/kaggle/working/feature_motion_appeare/test_appear",
        feature_dir_mot="/kaggle/working/feature_motion_appeare/test_motion",
        checkpoint_path="/kaggle/working/hme_ckpt/best_model.pt",
        output_file="/kaggle/working/submission_clip.json",
        batch_size=16
    )
