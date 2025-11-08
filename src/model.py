import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import json # MỚI: Để lưu metadata
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer # MỚI: Cần cho LLM

# --- MỚI: Giả sử bạn đã cập nhật các file này ---
# (Bạn cần tự tạo các file này dựa trên hướng dẫn trước)
from model import VideoTextLLMQA_V2 # THAY ĐỔI: Import mô hình V2
from dataset import (
    FeatureVideoQADatasetMPNET, # Giả sử dataset này đã được sửa
    collate_fn_mpnet            # Giả sử collate_fn này đã được sửa
)

# ===========================
# CONFIG
# ===========================
DATA_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
VIDEO_FEAT_DIR = "Feature/train"

# --- Config cho mô hình V2 ---
LLM_MODEL_NAME = "mistralai/Mistral-7B-v0.1" # ĐÃ SỬA: Dùng Mistral 7B (Open Access)
VIDEO_FEAT_DIM = 2304 
TEXT_FEAT_DIM = 768 # Giữ nguyên dim của text_feats (MPNet/CLIP)

# --- Config huấn luyện (Điều chỉnh cho PEFT) ---
BATCH_SIZE = 4        # THAY ĐỔI: Giảm BS vì LLM tốn VRAM
ACCUM_STEPS = 4       # THAY ĐỔI: Tăng ACCUM (Effective BS = 4*4 = 16)
LR = 1e-4             # THAY ĐỔI: Learning rate phổ biến cho LoRA
EPOCHS = 10           # Giảm epochs, vì LLM hội tụ nhanh hơn
WEIGHT_DECAY = 0.01
VALID_SPLIT = 0.1
OUTPUT_DIR = "/kaggle/working/"

SEED = 42
USE_FP16 = True
EARLYSTOP_PATIENCE = 2 # Giảm patience
CLIP_NORM = 1.0
NUM_WORKERS = os.cpu_count()

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
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for batch in pbar:
            # THAY ĐỔI: Unpack batch cho mô hình V2
            video_feats = batch["video_feats"].to(device)
            text_feats = batch["text_feats"].to(device) # Vẫn cần text_feats
            labels = batch["labels"].to(device)
            questions = batch["questions"]       # MỚI: list[str]
            choice_texts = batch["choice_texts"] # MỚI: list[list[str]]

            with autocast(enabled=USE_FP16):
                # THAY ĐỔI: Truyền input mới cho model
                logits = model(video_feats, text_feats, questions, choice_texts)
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * video_feats.size(0)
            
            mask = labels != -1 
            if mask.sum() > 0:
                preds = logits.argmax(dim=1)
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
    # MỚI: Load Tokenizer (thay vì text_encoder)
    # --------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    full_ds = FeatureVideoQADatasetMPNET( # GIẢ SỬ file này đã được sửa
        json_path=DATA_JSON,
        video_feat_dir=VIDEO_FEAT_DIR,
        video_feat_dim=VIDEO_FEAT_DIM,
        # text_encoder bị xóa
        tokenizer=tokenizer, # MỚI: Truyền tokenizer vào dataset
        preload_text=True, # Giả sử bạn vẫn preload text_feats
        is_test=False
    )

    n = len(full_ds)
    n_val = max(1, int(n * VALID_SPLIT))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE, # Đã cập nhật
        shuffle=True,
        collate_fn=collate_fn_mpnet, # GIẢ SỬ file này đã được sửa
        num_workers=NUM_WORKERS,
        pin_memory=True 
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn_mpnet, # GIẢ SỬ file này đã được sửa
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------
    # Build model (THAY ĐỔI)
    # --------------------------
    print("Building model (V2)...")
    model = VideoTextLLMQA_V2(
        video_dim=VIDEO_FEAT_DIM,
        text_dim=TEXT_FEAT_DIM, # Dim của text_feats (MPNet/CLIP)
        hidden_dim=512,
        llm_model_name=LLM_MODEL_NAME, # Đã là Mistral
        device=device
        # Model V2 tự xử lý device_map và PEFT bên trong
    )

    # Tạm thời tắt torch.compile, nó có thể không tương thích tốt với PEFT/HF
    # if hasattr(torch, 'compile'):
    #     print("Compiling model (PyTorch 2.0+)...")
    #     model = torch.compile(model)

    # --------------------------
    # Optimizer (THAY ĐỔI LỚN)
    # --------------------------
    print("Setting up PEFT optimizer...")
    # MỚI: Chỉ lấy các tham số có thể huấn luyện (LoRA, projections, v.v.)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"Adding trainable param: {name}")

    optimizer = torch.optim.AdamW(
        trainable_params, # THAY ĐỔI: Chỉ truyền các params này
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    scaler = GradScaler(enabled=USE_FP16)

    # --------------------------
    # Training loop
    # --------------------------
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting PEFT training ---")
    print(f"Device: {device}, FP16: {USE_FP16}, Effective BS: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train")
        total_loss = 0.0
        
        optimizer.zero_grad() 

        for step, batch in enumerate(pbar):
            # THAY ĐỔI: Unpack batch cho mô hình V2
            video_feats = batch["video_feats"].to(device)
            text_feats = batch["text_feats"].to(device)
            labels = batch["labels"].to(device)
            questions = batch["questions"]       # MỚI
            choice_texts = batch["choice_texts"] # MỚI

            with autocast(enabled=USE_FP16):
                # THAY ĐỔI: Truyền input mới cho model
                logits = model(video_feats, text_feats, questions, choice_texts)
                loss = loss_fn(logits, labels)
                loss_to_backward = loss / ACCUM_STEPS 

            scaler.scale(loss_to_backward).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                # THAY ĐỔI: Clip grad norm chỉ cho các tham số huấn luyện
                torch.nn.utils.clip_grad_norm_(trainable_params, CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # --------------------------
        # Evaluate
        # --------------------------
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"✅ Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss)

        # --------------------------
        # Save best (THAY ĐỔI LỚN)
        # --------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # THAY ĐỔI: Lưu adapter (PEFT)
            adapter_path = os.path.join(OUTPUT_DIR, "best_adapter")
            model.save_pretrained(adapter_path) # Đây là cách lưu của PEFT
            
            # Lưu các thông tin khác
            meta_path = os.path.join(OUTPUT_DIR, "best_model_meta.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch
                }, f)
            
            print(f"💾 Saved best adapter to {adapter_path}!")
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