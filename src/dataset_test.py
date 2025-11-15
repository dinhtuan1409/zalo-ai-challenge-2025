# dataset_hme_v2.py
import os
import json
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# ================================
# Load text encoder
# ================================
def load_text_encoder(device: Optional[str] = None) -> SentenceTransformer:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", trust_remote_code=True)
    model.eval()
    model = model.to(device)
    print(f"Text encoder loaded on {device}")
    return model


# ================================
# Dataset for HME Multi-choice VideoQA
# ================================
class FeatureVideoQAHME_MC(Dataset):
    """
    Dataset HME-VideoQA Multi-choice:
      - Appearance feature: [768] per video ID
      - Motion feature: [T, 2304] per video_path
      - Text feature: [C, 768] per choice
    """

    def __init__(
        self,
        json_path: str,
        feature_dir_appearance: str,
        feature_dir_motion: str,
        text_encoder: Optional[SentenceTransformer] = None,
        preload_text: bool = True,
        is_test: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["data"]

        self.feature_dir_appearance = feature_dir_appearance
        self.feature_dir_motion = feature_dir_motion
        self.is_test = is_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = text_encoder or load_text_encoder(self.device)
        self.preload_text = preload_text

        if preload_text:
            print("⚡ Pre-encoding all text (MPNet)...")
            self.text_cache = self._pre_encode_all_text()
        else:
            self.text_cache = None
            print("⚠️ preload_text=False → encode text on-the-fly (slower).")

    def __len__(self):
        return len(self.items)

    @torch.no_grad()
    def _pre_encode_all_text(self) -> Dict[str, torch.Tensor]:
        cache = {}
        for item in self.items:
            qid = item["id"]
            question = item["question"]
            choices = item["choices"]
            texts = [question + " [SEP] " + c for c in choices]
            emb = self.text_encoder.encode(
                texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
            )
            cache[qid] = emb.float().cpu()  # [C, 768]
        return cache

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        emb = self.text_encoder.encode(
            texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
        )
        return emb.float().cpu()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]

        # --- Appearance feature (per video ID) ---
        path_app = os.path.join(self.feature_dir_appearance, f"{qid}_appearance.pt")
        if os.path.exists(path_app):
            feat_app = torch.load(path_app).squeeze(0)  # [768]
        else:
            feat_app = torch.zeros(768)

        # --- Motion feature (per video_path) ---
        video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path_mot = os.path.join(self.feature_dir_motion, f"{video_name}.pt")
        if os.path.exists(path_mot):
            feat_mot = torch.load(path_mot)           # [T, 2304]
        else:
            feat_mot = torch.zeros((1, 2304))
        motion_mask = torch.ones(feat_mot.shape[0], dtype=torch.long)

        # --- Text feature ---
        if self.preload_text:
            text_feats = self.text_cache[qid]          # [C, 768]
        else:
            texts = [question + " [SEP] " + c for c in choices]
            text_feats = self.encode_text(texts)       # [C, 768]

        # --- Label ---
        label = -1
        if not self.is_test:
            ans = item.get("answer")
            if ans:
                for i, c in enumerate(choices):
                    if ans.strip() == c.strip():
                        label = i
                        break

        return {
            "id": qid,
            "appearance_feat": feat_app,       # [768]
            "motion_feat": feat_mot,           # [T, 2304]
            "motion_mask": motion_mask,        # [T]
            "text_feats": text_feats,          # [C, 768]
            "question": question,
            "choice_texts": choices,
            "label": label
        }


# ================================
# Collate function
# ================================
def collate_fn_hme(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    # --- Motion ---
    max_T = max(b["motion_feat"].shape[0] for b in batch)
    motion_feats = torch.zeros((B, max_T, 2304))
    motion_mask = torch.zeros((B, max_T), dtype=torch.long)
    for i, b in enumerate(batch):
        T_b = b["motion_feat"].shape[0]
        motion_feats[i, :T_b] = b["motion_feat"]
        motion_mask[i, :T_b] = b["motion_mask"]

    # --- Appearance ---
    appearance_feats = torch.stack([b["appearance_feat"] for b in batch], dim=0)  # [B, 768]

    # --- Text ---
    max_C = max(b["text_feats"].shape[0] for b in batch)
    text_feats = torch.zeros((B, max_C, 768))
    for i, b in enumerate(batch):
        C_b = b["text_feats"].shape[0]
        text_feats[i, :C_b] = b["text_feats"]

    # --- Labels ---
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # --- Raw text ---
    questions = [b["question"] for b in batch]
    choice_texts = [b["choice_texts"] for b in batch]
    ids = [b["id"] for b in batch]

    return {
        "ids": ids,
        "appearance_feats": appearance_feats,
        "motion_feats": motion_feats,
        "motion_mask": motion_mask,
        "text_feats": text_feats,
        "labels": labels,
        "questions": questions,
        "choice_texts": choice_texts
    }



