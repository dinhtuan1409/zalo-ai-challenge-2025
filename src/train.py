import os
import math
import random
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# --- import your modules ---
from model import HME_MC
from dataset import FeatureVideoQAHME_MC, load_text_encoder, collate_fn_hme
from dataset import load_video_llava

# -------------------------
# Config / Hyperparameters
# -------------------------
def get_default_args():
    return {
        "data_json": "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json",
        "feat_app_dir": "/kaggle/working/feature_motion_appeare/train_appear",
        "feat_mot_dir": "/kaggle/working/feature_motion_appeare/train_motion",
        "output_dir": "/kaggle/working/hme_ckpt",
        "batch_size": 4,
        "accum_steps": 4,
        "epochs": 30,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "valid_split": 0.1,
        "seed": 42,
        "use_fp16": True,
        "earlystop_patience": 6,
        "num_workers": 4,
        "motion_dim": 2304,
        "appearance_dim": 768,
        "text_dim": 1536,           # MPNet+Video-LLaVA
        "motion_proj_dim": 1024,
        "hidden_dim": 1024,
        "num_motion_layers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: str, use_fp16: bool) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for batch in pbar:
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            text = batch["text_feats"].to(device)   # [B,C,1536]
            mask_motion = batch.get("motion_mask", None)
            if mask_motion is not None:
                mask_motion = mask_motion.to(device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(device)

            with autocast(enabled=use_fp16):
                out = model(
                    motion_feats=motion,
                    motion_mask=mask_motion,
                    appearance_feats=appearance,
                    text_feats=text,
                    labels=labels if (labels is not None and (labels!=-1).any()) else None
                )
                logits = out["logits"]
                loss = loss_fn(logits, labels) if labels is not None else torch.tensor(0.0, device=device)

            batch_size = appearance.size(0)
            total_loss += loss.item() * batch_size

            if labels is not None:
                valid_mask = labels != -1
                if valid_mask.sum() > 0:
                    preds = logits.argmax(dim=1)
                    total_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                    total_count += int(valid_mask.sum().item())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = (total_correct / total_count) if total_count > 0 else 0.0
    return avg_loss, avg_acc

# -------------------------
def train_loop(args):
    seed_everything(args["seed"])
    device = args["device"]

    # --- Load encoders ---
    print("Loading MPNet + Video-LLaVA encoders...")
    mpnet = load_text_encoder(device=device)
    vllava_model, vllava_tokenizer, vllava_processor = load_video_llava(device=device)

    # --- Load frames dict (video_name -> list of PIL.Image frames) ---
    # Bạn cần chuẩn bị dict frames_dict trước
    frames_dict = {}  # Ví dụ: {"video1": [PIL.Image,...], "video2": [...], ...}

    # --- Dataset ---
    print("Preparing dataset...")
    full_ds = FeatureVideoQAHME_MC(
        json_path=args["data_json"],
        feature_dir_appearance=args["feat_app_dir"],
        feature_dir_motion=args["feat_mot_dir"],
        text_encoder=mpnet,
        vllava_model=vllava_model,
        vllava_tokenizer=vllava_tokenizer,
        vllava_processor=vllava_processor,
        frames_dict=frames_dict,
        preload_text=True,
        is_test=False
    )

    # --- Train/Val split ---
    n = len(full_ds)
    n_val = max(1, int(n * args["valid_split"]))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=args["batch_size"],
                              shuffle=True,
                              num_workers=args["num_workers"],
                              pin_memory=True,
                              collate_fn=collate_fn_hme)
    val_loader = DataLoader(Subset(full_ds, val_idx),
                            batch_size=args["batch_size"],
                            shuffle=False,
                            num_workers=args["num_workers"],
                            pin_memory=True,
                            collate_fn=collate_fn_hme)

    # --- Model ---
    print("Building HME_MC model...")
    model = HME_MC(
        motion_dim=args["motion_dim"],
        appearance_dim=args["appearance_dim"],
        text_dim=args["text_dim"],      # 1536 dim
        hidden_dim=args["hidden_dim"],
        motion_proj_dim=args["motion_proj_dim"],
        num_motion_layers=args["num_motion_layers"]
    ).to(device)

    # --- Optimizer, Scheduler, Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)
    scaler = GradScaler(enabled=args["use_fp16"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(args["output_dir"], exist_ok=True)

    # --- Training ---
    for epoch in range(args["epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']} train")

        for step, batch in enumerate(pbar):
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            text = batch["text_feats"].to(device)
            mask_motion = batch.get("motion_mask", None)
            if mask_motion is not None:
                mask_motion = mask_motion.to(device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(device)

            with autocast(enabled=args["use_fp16"]):
                out = model(
                    motion_feats=motion,
                    motion_mask=mask_motion,
                    appearance_feats=appearance,
                    text_feats=text,
                    labels=labels if (labels is not None and (labels!=-1).any()) else None
                )
                logits = out["logits"]
                loss = loss_fn(logits, labels)
                loss_to_back = loss / args["accum_steps"]

            scaler.scale(loss_to_back).backward()

            if (step + 1) % args["accum_steps"] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}"})

        # --- Validation ---
        val_loss, val_acc = evaluate(model, val_loader, device, args["use_fp16"])
        print(f"Epoch {epoch+1} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": args
            }
            torch.save(ckpt, os.path.join(args["output_dir"], "best_model.pt"))
            print(f"Saved best_model.pt (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args["earlystop_patience"]:
                print("Early stopping triggered.")
                break

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
        "args": args
    }, os.path.join(args["output_dir"], "last_model.pt"))
    print("Saved last_model.pt")


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
