# train_llm_assisted.py
import os
import random
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# --- import modules ---
from model import VideoEncoder, VideoTextContrastive
from dataset import FeatureVideoQAHME_MC_CLIP, CLIPTextEncoder, collate_fn_hme_clip

# -------------------------
# Default args / hyperparams
# -------------------------
def get_default_args():
    return {
        "data_json": "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json",
        "feat_app_dir": "/kaggle/working/feature_motion_appeare/train_appear",
        "feat_mot_dir": "/kaggle/working/feature_motion_appeare/train_motion",
        "output_dir": "/kaggle/working/hme_ckpt_llm",
        "batch_size": 4,
        "accum_steps": 4,
        "epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "valid_split": 0.1,
        "seed": 42,
        "use_fp16": True,
        "earlystop_patience": 6,
        "num_workers": 4,
        "motion_dim": 2304,
        "appearance_dim": 768,
        "hidden_dim": 1024,
        "motion_proj_dim": 1024,
        "num_motion_layers": 2,
        "video_dim": 1024,   # embedding dim for contrastive
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# -------------------------
# Utility: seed
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Evaluate retrieval
# -------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion: VideoTextContrastive, device: str, use_fp16: bool) -> float:
    model.eval()
    total_acc = 0.0
    total_count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for batch in pbar:
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            text_emb = batch["text_feats"].to(device).mean(dim=1)  # [B, D], pool choices

            with autocast(enabled=use_fp16):
                video_emb = model(motion_feats=motion,
                                  motion_mask=batch.get("motion_mask", None),
                                  appearance_feats=appearance)
                video_emb = F.normalize(video_emb, dim=-1)
                text_emb = F.normalize(criterion.text_proj(text_emb), dim=-1)

                sims = video_emb @ text_emb.T  # [B, B]
                labels = torch.arange(video_emb.size(0), device=device)
                preds = sims.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                total_acc += acc
                total_count += 1

    return total_acc / total_count if total_count > 0 else 0.0

# -------------------------
# Training loop
# -------------------------
def train_loop(args):
    seed_everything(args["seed"])
    device = args["device"]

    # Text encoder
    print("Loading CLIP text encoder...")
    text_encoder = CLIPTextEncoder().to(device)

    # Dataset
    print("Preparing dataset...")
    full_ds = FeatureVideoQAHME_MC_CLIP(
        json_path=args["data_json"],
        feature_dir_appearance=args["feat_app_dir"],
        feature_dir_motion=args["feat_mot_dir"],
        text_encoder=text_encoder,
        preload_text=True,
        is_test=False
    )

    n = len(full_ds)
    n_val = max(1, int(n * args["valid_split"]))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True,
                              num_workers=args["num_workers"], pin_memory=True,
                              collate_fn=collate_fn_hme_clip)
    val_loader = DataLoader(val_ds, batch_size=max(1, args["batch_size"]//1), shuffle=False,
                            num_workers=args["num_workers"], pin_memory=True,
                            collate_fn=collate_fn_hme_clip)

    # Model + criterion
    print("Building VideoEncoder and contrastive criterion...")
    model = VideoEncoder(
        motion_dim=args["motion_dim"],
        appearance_dim=args["appearance_dim"],
        hidden_dim=args["hidden_dim"],
        motion_proj_dim=args["motion_proj_dim"],
        num_motion_layers=args["num_motion_layers"],
        use_mean_pool_motion=True
    ).to(device)

    criterion = VideoTextContrastive(video_dim=args["video_dim"], text_dim=text_encoder.hidden_size).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()),
                                  lr=args["lr"], weight_decay=args["weight_decay"])
    scaler = GradScaler(enabled=args["use_fp16"])

    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs(args["output_dir"], exist_ok=True)

    print("Start training...")
    for epoch in range(args["epochs"]):
        model.train()
        criterion.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']} train")
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            text_emb = batch["text_feats"].to(device).mean(dim=1)  # [B, D], pool choices

            with autocast(enabled=args["use_fp16"]):
                video_emb = model(motion_feats=motion,
                                  motion_mask=batch.get("motion_mask", None),
                                  appearance_feats=appearance)
                loss = criterion(video_emb, text_emb)
                loss_to_back = loss / args["accum_steps"]

            scaler.scale(loss_to_back).backward()

            if (step + 1) % args["accum_steps"] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}"})

        val_acc = evaluate(model, val_loader, criterion, device, args["use_fp16"])
        print(f"Epoch {epoch+1} - val_acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": args
            }
            torch.save(ckpt, os.path.join(args["output_dir"], "best_model.pt"))
            print(f"Saved best_model.pt (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{args['earlystop_patience']})")
            if epochs_no_improve >= args["earlystop_patience"]:
                print("Early stopping triggered.")
                break

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": best_val_acc,
        "args": args
    }, os.path.join(args["output_dir"], "last_model.pt"))
    print("Saved last_model.pt")
    print("Training finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    default_args = get_default_args()
    parser = argparse.ArgumentParser()
    for k, v in default_args.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    parsed = vars(parser.parse_args())
    parsed["use_fp16"] = bool(parsed["use_fp16"])
    parsed["seed"] = int(parsed["seed"])
    parsed["device"] = parsed["device"]
    parsed["num_workers"] = int(parsed["num_workers"])
    parsed["batch_size"] = int(parsed["batch_size"])
    parsed["accum_steps"] = int(parsed["accum_steps"])
    parsed["epochs"] = int(parsed["epochs"])
    parsed["earlystop_patience"] = int(parsed["earlystop_patience"])

    train_loop(parsed)
