# dataset_hme_v2.py
import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from transformers import CLIPTokenizer, CLIPModel

# -------------------------
# Dataset: Trả về raw text + features
# -------------------------
class FeatureVideoQAHME_CLIP(Dataset):
    def __init__(
        self,
        json_path: str,
        feature_dir_appearance: str,
        feature_dir_motion: str,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        is_test: bool = False,
    ):
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "data" in data:
                self.items = data["data"]
            else:
                self.items = data  # nếu không có key "data"

        self.feat_app_dir = feature_dir_appearance
        self.feat_mot_dir = feature_dir_motion
        self.is_test = is_test

        # CLIP tokenizer (chỉ cần tokenizer, model sẽ ở trong HME_MC_CLIP)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        print(f"CLIP tokenizer loaded: {clip_model_name}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]  # List[str]

        # --- Load Appearance (CLIP image) ---
        path_app = os.path.join(self.feat_app_dir, f"{qid}.pt")
        if os.path.exists(path_app):
            feat_app = torch.load(path_app).squeeze(0)  # [768]
        else:
            print(f"Warning: Missing appearance {path_app}, using zero")
            feat_app = torch.zeros(768)

        # --- Load Motion (SlowFast) ---
        video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path_mot = os.path.join(self.feat_mot_dir, f"{video_name}.pt")
        if os.path.exists(path_mot):
            feat_mot = torch.load(path_mot).float()  # [T, 2304]
        else:
            print(f"Warning: Missing motion {path_mot}, using zero")
            feat_mot = torch.zeros(1, 2304)

        motion_mask = torch.ones(feat_mot.shape[0], dtype=torch.bool)

        # --- Label ---
        label = -1
        if not self.is_test and "answer" in item:
            ans = item["answer"]
            for i, c in enumerate(choices):
                if ans.strip().lower() == c.strip().lower():
                    label = i
                    break

        return {
            "id": qid,
            "appearance_feats": feat_app,        # [768]
            "motion_feats": feat_mot,            # [T, 2304]
            "motion_mask": motion_mask,          # [T]
            "question": question,
            "choices": choices,                  # List[str]
            "label": label
        }


# -------------------------
# Collate Function: Pad motion, không pad text
# -------------------------
def collate_fn_hme_clip(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)

    # --- Motion: pad to max_T ---
    motion_list = [b["motion_feats"] for b in batch]
    max_T = max(m.shape[0] for m in motion_list)
    motion_padded = torch.zeros(batch_size, max_T, 2304, dtype=torch.float32)
    motion_mask = torch.zeros(batch_size, max_T, dtype=torch.bool)

    for i, m in enumerate(motion_list):
        T = m.shape[0]
        motion_padded[i, :T] = m
        motion_mask[i, :T] = 1

    # --- Appearance: stack ---
    appearance_feats = torch.stack([b["appearance_feats"] for b in batch])  # [B, 768]

    # --- Text: list of strings ---
    questions = [b["question"] for b in batch]
    choices = [b["choices"] for b in batch]

    # --- Labels ---
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # --- IDs ---
    ids = [b["id"] for b in batch]

    return {
        "ids": ids,
        "appearance_feats": appearance_feats,      # [B, 768]
        "motion_feats": motion_padded,             # [B, T_max, 2304]
        "motion_mask": motion_mask,                # [B, T_max]
        "questions": questions,                    # List[str]
        "choices": choices,                        # List[List[str]]
        "labels": labels                           # [B]
    }