# train_optimized.py
import os
import random
import json
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from model import HME_MC
from dataset import FeatureVideoQAHME_CLIP, collate_fn_hme_clip
# -------------------------
# Default hyperparameters
# -------------------------
def get_default_args():
    return {
        "data_json": "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json",
        "feat_app_dir": "/kaggle/working/feature_motion_appeare/train_appear",
        "feat_mot_dir": "/kaggle/working/feature_motion_appeare/train_motion",
        "output_dir": "/kaggle/working/hme_clip_ckpt",
        "batch_size": 16,
        "accum_steps": 2,
        "epochs": 30,
        "lr": 3e-5,
        "weight_decay": 1e-2,
        "valid_split": 0.1,
        "seed": 42,
        "use_fp16": True,
        "earlystop_patience": 5,
        "num_workers": 4,
        "clip_model_name": "openai/clip-vit-base-patch32",
        "freeze_clip": True,
        "proj_dim": 768,
        "memory_size": 8,
        "reasoning_steps": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# -------------------------
# Seed & Utils
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, use_fp16: bool) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in tqdm(loader, desc="Eval", leave=False):
        appearance = batch["appearance_feats"].to(device)
        motion = batch["motion_feats"].to(device)
        motion_mask = batch["motion_mask"].to(device)
        questions = batch["questions"]
        choices = batch["choices"]
        labels = batch["labels"].to(device)

        with autocast(enabled=use_fp16):
            out = model(
                motion_feats=motion,
                motion_mask=motion_mask,
                clip_img_feats=appearance,
                questions=questions,
                answers=choices,
                labels=labels if (labels != -1).any() else None
            )
            loss = out.get("loss", torch.tensor(0.0, device=device))

        total_loss += loss.item() * appearance.size(0)
        preds = out["logits"].argmax(dim=1)
        valid = labels != -1
        total_correct += (preds[valid] == labels[valid]).sum().item()
        total_count += valid.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / total_count if total_count > 0 else 0.0
    return avg_loss, acc

# -------------------------
# Training Loop
# -------------------------
def train_loop(args):
    seed_everything(args["seed"])
    device = args["device"]
    os.makedirs(args["output_dir"], exist_ok=True)

    print("Loading dataset...")
    full_ds = FeatureVideoQAHME_CLIP(
        json_path=args["data_json"],
        feature_dir_appearance=args["feat_app_dir"],
        feature_dir_motion=args["feat_mot_dir"],
        preload_text=True
    )

    # Split
    n = len(full_ds)
    n_val = max(1, int(n * args["valid_split"]))
    indices = np.random.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True,
                              num_workers=args["num_workers"], pin_memory=True, collate_fn=collate_fn_hme_clip)
    val_loader = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False,
                            num_workers=args["num_workers"], pin_memory=True, collate_fn=collate_fn_hme_clip)

    print("Building HME_MC_CLIP model...")
    model = HME_MC(
        motion_dim=2304,
        clip_dim=768,
        proj_dim=args["proj_dim"],
        memory_size=args["memory_size"],
        reasoning_steps=args["reasoning_steps"],
        clip_model_name=args["clip_model_name"],
        freeze_clip=args["freeze_clip"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"])
    scaler = GradScaler(enabled=args["use_fp16"])

    best_val_acc = 0.0
    epochs_no_improve = 0

    print("Start training...")
    for epoch in range(args["epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")):
            appearance = batch["appearance_feats"].to(device)
            motion = batch["motion_feats"].to(device)
            motion_mask = batch["motion_mask"].to(device)
            questions = batch["questions"]
            choices = batch["choices"]
            labels = batch["labels"].to(device)

            with autocast(enabled=args["use_fp16"]):
                out = model(
                    motion_feats=motion,
                    motion_mask=motion_mask,
                    clip_img_feats=appearance,
                    questions=questions,
                    answers=choices,
                    labels=labels
                )
                loss = out["loss"] / args["accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % args["accum_steps"] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += out["loss"].item()

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device, args["use_fp16"])
        print(f"\nEpoch {epoch+1} - train_loss: {running_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        scheduler.step()

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": args
            }, os.path.join(args["output_dir"], "best_model.pt"))
            print(f"New best! Saved (acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args["earlystop_patience"]:
                print("Early stopping.")
                break

    # Save last
    torch.save(model.state_dict(), os.path.join(args["output_dir"], "last_model.pt"))
    print("Training finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    defaults = get_default_args()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()
    args = vars(args)
    args["use_fp16"] = bool(args["use_fp16"])
    args["freeze_clip"] = bool(args["freeze_clip"])
    args["batch_size"] = int(args["batch_size"])
    args["num_workers"] = int(args["num_workers"])
    args["epochs"] = int(args["epochs"])

    train_loop(args)