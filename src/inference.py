# inference_hme_trainstyle.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

from dataset_test import FeatureVideoQAHME_MC, collate_fn_hme, load_text_encoder
from model import HME_MC


def inference_hme_trainstyle(
    json_path: str,
    feature_dir_app: str,
    feature_dir_mot: str,
    checkpoint_path: str,
    output_file: str,
    batch_size: int = 16,
    use_fp16: bool = True,
    motion_dim: int = 2304,
    appearance_dim: int = 768,
    text_dim: int = 768,
    motion_proj_dim: int = 1024,
    hidden_dim: int = 1024,
    num_motion_layers: int = 2,
    num_workers: int = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = num_workers or os.cpu_count()

    # --- Load text encoder ---
    print("🔄 Loading text encoder...")
    text_encoder = load_text_encoder(device=device)

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

    # --- Load dataset ---
    print("🔄 Loading dataset...")
    dataset = FeatureVideoQAHME_MC(
        json_path=json_path,
        feature_dir_appearance=feature_dir_app,
        feature_dir_motion=feature_dir_mot,
        text_encoder=text_encoder,
        preload_text=True,
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

    # --- Inference ---
    results = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="🚀 Inference")
        for batch in pbar:
            motion_feats = batch["motion_feats"].to(device)       # [B, T, 2304]
            motion_mask = batch["motion_mask"].to(device)         # [B, T]
            appearance_feats = batch["appearance_feats"].to(device)  # [B, 768]
            text_feats = batch["text_feats"].to(device)           # [B, C, 768]
            ids = batch["ids"]

            # FP16 autocast
            with autocast(enabled=use_fp16):
                out = model(
                    motion_feats=motion_feats,
                    motion_mask=motion_mask,
                    appearance_feats=appearance_feats,
                    text_feats=text_feats
                )

            logits = out["logits"]    # [B, C]
            preds = logits.argmax(dim=1).cpu().tolist()

            for qid, p in zip(ids, preds):
                results.append({"id": qid, "answer": int(p)})

    # --- Save submission ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"data": results}, f, ensure_ascii=False, indent=2)

    print(f"✅ DONE! Saved submission to {output_file}")
    return results


if __name__ == "__main__":
    inference_hme_trainstyle(
        json_path="/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json",
        feature_dir_app="/kaggle/working/feature_motion_appeare/test_appear",
        feature_dir_mot="/kaggle/working/feature_motion_appeare/test_motion",
        checkpoint_path="/kaggle/working/hme_ckpt/best_model.pt",
        output_file="/kaggle/working/submission.json",
        batch_size=16
    )
