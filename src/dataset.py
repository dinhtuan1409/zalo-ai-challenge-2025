import os
import json
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"


# ===========================================================
# ✅ TEXT ENCODER
# ===========================================================
def load_text_encoder(device: Optional[str] = None) -> SentenceTransformer:
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Text encoder (MPNet) loaded on {device}")
    return model


# ===========================================================
# ✅ DATASET
# ===========================================================
class FeatureVideoQAHME_MC(Dataset):
    """
    Dataset tối ưu cho HME multiple-choice:
      - appearance_feat: [1, 768]
      - motion_feat: [T, 2304]
      - text_feats: [num_choices, 768]
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
            print("⚡ Pre-encoding text (MPNet)...")
            self.text_cache = self._pre_encode_all_text()
        else:
            self.text_cache = None

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
                texts, convert_to_tensor=True, device=self.device
            )
            cache[qid] = emb.float().cpu()
        return cache

    def _safe_load(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            return torch.zeros(1, 768)
        feat = torch.load(path, map_location="cpu")
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)
        return feat.float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]

        # Appearance theo id
        path_app = os.path.join(self.feature_dir_appearance, f"{qid}.pt")
        feat_app = self._safe_load(path_app)
        if feat_app.dim() == 2 and feat_app.shape[0] == 1:
            feat_app = feat_app.squeeze(0)  # [768]

        # Motion theo tên video_path
        video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path_motion = os.path.join(self.feature_dir_motion, f"{video_name}.pt")
        feat_mot = self._safe_load(path_motion)  # [T, 2304]

        # Text
        if self.preload_text:
            text_feats = self.text_cache[qid]
        else:
            texts = [question + " [SEP] " + c for c in choices]
            text_feats = self.text_encoder.encode(texts, convert_to_tensor=True)

        # Label
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
            "appearance_feat": feat_app,  # [768]
            "motion_feat": feat_mot,      # [T, 2304]
            "text_feats": text_feats,     # [num_choices, 768]
            "question": question,
            "choice_texts": choices,
            "label": label if not self.is_test else None,
        }


def collate_fn_hme_mc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_T = max(b["motion_feat"].shape[0] for b in batch)
    max_choice = max(b["text_feats"].shape[0] for b in batch)

    ids, app_feats, mot_feats, text_feats, labels = [], [], [], [], []
    questions, raw_choices = [], []

    has_label = batch[0]["label"] is not None

    for b in batch:
        ids.append(b["id"])
        questions.append(b["question"])
        raw_choices.append(b["choice_texts"])

        # appearance [768]
        app_feats.append(b["appearance_feat"])

        # motion [T,2304] → pad
        mot = b["motion_feat"]
        pad_len = max_T - mot.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, mot.shape[1])
            mot = torch.cat([mot, pad], dim=0)
        mot_feats.append(mot)

        # text [num_choices,768] → pad
        tf = b["text_feats"]
        if tf.shape[0] < max_choice:
            pad = torch.zeros(max_choice - tf.shape[0], tf.shape[1])
            tf = torch.cat([tf, pad], dim=0)
        text_feats.append(tf)

        if has_label:
            labels.append(b["label"])

    return {
        "ids": ids,
        "appearance_feats": torch.stack(app_feats).float(),  # [B,768]
        "motion_feats": torch.stack(mot_feats).float(),      # [B,T,2304]
        "text_feats": torch.stack(text_feats).float(),       # [B,num_choices,768]
        "labels": torch.tensor(labels, dtype=torch.long) if has_label else None,
        "questions": questions,
        "choice_texts": raw_choices,
        "mask_motion": torch.tensor(  # [B,T], 1 = real, 0 = pad
            [[1]*b["motion_feat"].shape[0] + [0]*(max_T - b["motion_feat"].shape[0]) for b in batch]
        ),
    }