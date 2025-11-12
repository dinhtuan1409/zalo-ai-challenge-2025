# inference_hme_simple.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

from dataset_test import FeatureVideoQAHME_MC, collate_fn_hme, load_text_encoder
from model import HME_MC


def run_inference_hme_simple(
    json_path: str,
    feature_dir_app: str,
    feature_dir_mot: str,
    output_file: str,
    batch_size: int = 16,
    use_fp16: bool = True,
    motion_dim: int = 2304,
    appearance_dim: int = 768,
    text_dim: int = 768,
    motion_proj_dim: int = 1024,
    num_workers: int = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = num_workers or os.cpu_count()

    # --- Load text encoder ---
    print("🔄 Loading text encoder...")
    text_encoder = load_text_encoder(device=device)

    # --- Build HME_MC model ---
    print("🔄 Building HME_MC model...")
    model = HME_MC(
        motion_dim=motion_dim,
        appearance_dim=appearance_dim,
        text_dim=text_dim,
        motion_proj_dim=motion_proj_dim
    ).to(device)
    model.eval()

    # --- Load test dataset ---
    print("🔄 Loading test dataset...")
    test_ds = FeatureVideoQAHME_MC(
        json_path=json_path,
        feature_dir_appearance=feature_dir_app,
        feature_dir_motion=feature_dir_mot,
        text_encoder=text_encoder,
        preload_text=True,
        is_test=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hme,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Inference loop ---
    results = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="🚀 Inference")
        for batch in pbar:
            motion_feats = batch["motion_feats"].to(device)
            motion_mask = batch["motion_mask"].to(device)
            appearance_feats = batch["appearance_feats"].to(device)
            text_feats = batch["text_feats"].to(device)
            ids = batch["ids"]

            with autocast(enabled=use_fp16):
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

    # --- Save submission ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"data": results}, f, ensure_ascii=False, indent=2)

    print(f"✅ DONE! Saved submission to {output_file}")
    return results


if __name__ == "__main__":
    # Ví dụ chạy
    run_inference_hme_simple(
        json_path="/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json",
        feature_dir_app="/kaggle/working/feature_motion_appeare/test_appear",
        feature_dir_mot="/kaggle/working/feature_motion_appeare/test_motion",
        output_file="/kaggle/working/submission.json",
        batch_size=16
    )
