import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# --- Import các file code của bạn ---
# (Giả sử chúng nằm trong các file .py tương ứng)
from dataset import FeatureVideoQADatasetMPNET, collate_fn_mpnet, load_text_encoder
from model import EarlyFusionMPNetQA 

# ===========================
# CONFIG
# ===========================
DATA_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
VIDEO_FEAT_DIR = "Feature/train"

# ❗ SỬA LỖI: Đồng bộ video_dim ở đây
VIDEO_FEAT_DIM = 2304 # Cần khớp với feature của bạn (dataset đang là 2304)

BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 15
WEIGHT_DECAY = 0.01
VALID_SPLIT = 0.1
OUTPUT_DIR = "/kaggle/working/"

SEED = 42
USE_FP16 = True
ACCUM_STEPS = 1
EARLYSTOP_PATIENCE = 3
CLIP_NORM = 1.0

# 💡 TỐI ƯU: Tăng tốc độ load data
NUM_WORKERS = os.cpu_count() # Sử dụng tất cả các CPU core

# ===========================
# SEED
# ===========================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===========================
# EVALUATE
# ===========================
def evaluate(model, loader, loss_fn, device):
    """
    Tính toán loss và accuracy trên tập validation.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # 💡 TỐI ƯU: Thêm leave=False để thanh tqdm eval tự xóa
        pbar = tqdm(loader, desc="Eval", leave=False)
        for batch in pbar:
            video = batch["video_feats"].to(device)
            text = batch["text_feats"].to(device)
            labels = batch["labels"].to(device)

            # 💡 TỐI ƯU: Chạy eval với autocast
            with autocast(enabled=USE_FP16):
                logits = model(video, text)
                
                # 💡 TỐI ƯU: Dùng loss_fn đã khởi tạo (với ignore_index)
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * video.size(0)
            
            # --- Tính accuracy (giữ nguyên logic mask của bạn) ---
            preds = logits.argmax(dim=1)
            
            # 💡 TỐI ƯU: ignore_index=-1 cho cả accuracy
            mask = labels != -1 
            if mask.sum() > 0:
                total_correct += (preds[mask] == labels[mask]).sum().item()
                total_samples += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc

# ===========================
# MAIN TRAIN LOOP
# ===========================
def train_loop():
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------
    # Load dataset
    # 💡 TỐI ƯU: Load text_encoder 1 LẦN duy nhất ở main
    # --------------------------
    print("Loading text encoder...")
    text_encoder = load_text_encoder(device)
    
    print("Loading dataset...")
    full_ds = FeatureVideoQADatasetMPNET(
        json_path=DATA_JSON,
        video_feat_dir=VIDEO_FEAT_DIR,
        video_feat_dim=VIDEO_FEAT_DIM, # ❗ SỬA LỖI: Truyền video_dim vào dataset
        text_encoder=text_encoder,     # Truyền encoder đã load
        preload_text=True,
        is_test=False
    )

    n = len(full_ds)
    n_val = max(1, int(n * VALID_SPLIT))

    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_mpnet,
        num_workers=NUM_WORKERS, # 💡 TỐI ƯU
        pin_memory=True          # 💡 TỐI ƯU: Tăng tốc chuyển data sang GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2, # Thường có thể tăng batch_size khi eval
        shuffle=False,
        collate_fn=collate_fn_mpnet,
        num_workers=NUM_WORKERS, # 💡 TỐI ƯU
        pin_memory=True
    )

    # --------------------------
    # Build model
    # --------------------------
    print("Building model...")
    model = EarlyFusionMPNetQA(
        video_dim=VIDEO_FEAT_DIM, # ❗ SỬA LỖI: Dùng đúng video_dim
        text_dim=768,
        hidden_dim=512
    ).to(device)

    # 💡 TỐI ƯU: (PyTorch 2.0+) Tăng tốc model
    if hasattr(torch, 'compile'):
        print("Compiling model (PyTorch 2.0+)...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 💡 TỐI ƯU: Khởi tạo loss function 1 lần
    # Dùng ignore_index=-1 để tự động bỏ qua các sample không có label
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # 💡 TỐI ƯU: Theo dõi val_loss (ổn định hơn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',     # Theo dõi val_loss (thay vì max cho acc)
        factor=0.5,
        patience=1,
        verbose=True
    )

    scaler = GradScaler(enabled=USE_FP16)

    # --------------------------
    # Training loop
    # --------------------------
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting training ---")
    print(f"Device: {device}, FP16: {USE_FP16}, Accum Steps: {ACCUM_STEPS}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train")
        total_loss = 0.0
        
        # Đặt zero_grad ở đầu vòng lặp step
        optimizer.zero_grad() 

        for step, batch in enumerate(pbar):
            video = batch["video_feats"].to(device)
            text = batch["text_feats"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=USE_FP16):
                logits = model(video, text)
                loss = loss_fn(logits, labels)
                
                # ❗ SỬA LỖI: Phải scale loss trước khi backward()
                # khi dùng gradient accumulation
                loss_to_backward = loss / ACCUM_STEPS 

            scaler.scale(loss_to_backward).backward()

            # --- Optimizer step ---
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Tích lũy loss (chưa scale) để log
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # --------------------------
        # Evaluate
        # --------------------------
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"✅ Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 💡 TỐI ƯU: Step scheduler dựa trên val_loss
        scheduler.step(val_loss)

        # --------------------------
        # Save best
        # 💡 TỐI ƯU: Lưu model dựa trên val_loss
        # --------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                "model": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch
            }, save_path)
            print(f"💾 Saved best model! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= EARLYSTOP_PATIENCE:
                print("⛔ Early stopping triggered!")
                break

    print(f"✅ Training done. Best val_loss: {best_val_loss:.4f}")

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    train_loop()