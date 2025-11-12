import os
import json
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPModel
from typing import List, Dict, Any, Optional


# ================================
# Load CLIP Text Encoder (đồng bộ với ViT-L/14@336px)
# ================================
class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", trainable_layers=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.text_encoder = self.clip.text_model
        self.hidden_size = self.text_encoder.config.hidden_size

        # freeze toàn bộ
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # fine-tune n layer cuối
        if trainable_layers > 0:
            for layer in self.text_encoder.encoder.layers[-trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def encode_text(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        """Trích xuất text embedding [N, D]"""
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.text_encoder(**enc)
        text_embeds = outputs.last_hidden_state[:, 0, :]  # CLS token
        return text_embeds.float()


# ================================
# Dataset for HME Multi-choice VideoQA (CLIP text)
# ================================
class FeatureVideoQAHME_MC_CLIP(Dataset):
    """
    Dataset HME-VideoQA Multi-choice:
      - Appearance feature: [768] per video ID
      - Motion feature: [T, 2304] per video_path
      - Text feature: [C, hidden_size] (CLIP text encoder)
    """

    def __init__(
        self,
        json_path: str,
        feature_dir_appearance: str,
        feature_dir_motion: str,
        text_encoder: Optional[CLIPTextEncoder] = None,
        preload_text: bool = True,
        is_test: bool = False,
        device: Optional[str] = None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["data"]

        self.feature_dir_appearance = feature_dir_appearance
        self.feature_dir_motion = feature_dir_motion
        self.is_test = is_test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # text encoder
        self.text_encoder = text_encoder or CLIPTextEncoder().to(self.device)
        self.preload_text = preload_text

        if preload_text:
            print("⚡ Pre-encoding all text with CLIP text encoder...")
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
            emb = self.text_encoder.encode_text(texts, device=self.device)
            cache[qid] = emb.cpu()  # [C, D]
        return cache

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        emb = self.text_encoder.encode_text(texts, device=self.device)
        return emb.cpu()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]

        # --- Appearance feature ---
        path_app = os.path.join(self.feature_dir_appearance, f"{qid}_appearance.pt")
        if os.path.exists(path_app):
            feat_app = torch.load(path_app).squeeze(0)  # [768]
        else:
            feat_app = torch.zeros(768)

        # --- Motion feature ---
        video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path_mot = os.path.join(self.feature_dir_motion, f"{video_name}.pt")
        if os.path.exists(path_mot):
            feat_mot = torch.load(path_mot)  # [T, 2304]
        else:
            feat_mot = torch.zeros((1, 2304))
        motion_mask = torch.ones(feat_mot.shape[0], dtype=torch.long)

        # --- Text feature ---
        if self.preload_text:
            text_feats = self.text_cache[qid]  # [C, D]
        else:
            texts = [question + " [SEP] " + c for c in choices]
            text_feats = self.encode_text(texts)  # [C, D]

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
            "appearance_feat": feat_app,
            "motion_feat": feat_mot,
            "motion_mask": motion_mask,
            "text_feats": text_feats,
            "question": question,
            "choice_texts": choices,
            "label": label,
        }


# ================================
# Collate function
# ================================
def collate_fn_hme_clip(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
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
    feat_dim = batch[0]["text_feats"].shape[1]
    max_C = max(b["text_feats"].shape[0] for b in batch)
    text_feats = torch.zeros((B, max_C, feat_dim))
    for i, b in enumerate(batch):
        C_b = b["text_feats"].shape[0]
        text_feats[i, :C_b] = b["text_feats"]

    # --- Labels ---
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

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
        "choice_texts": choice_texts,
    }
