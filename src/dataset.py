# ===========================================================
# dataset_hme_vllava.py
# ===========================================================

import os
import json
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from PIL import Image

# -------------------------
# Video-LLaVA
# -------------------------
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    AutoModelForCausalLM
)

# ===========================================================
# Load MPNet Text Encoder
# ===========================================================
def load_text_encoder(device: Optional[str] = None) -> SentenceTransformer:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.eval()
    model = model.to(device)
    print(f"MPNet text encoder loaded on {device}")
    return model


# ===========================================================
# Load Video-LLaVA
# ===========================================================
def load_video_llava(device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "LanguageBind/Video‑LLaVA‑7B-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    ).to(device)

    model.eval()

    print(f"Video-LLaVA loaded on {device}")
    return model, tokenizer, processor


# ===========================================================
# Video-LLaVA Caption Extraction
# ===========================================================
@torch.no_grad()
def videollava_caption(frames: List[Image.Image],
                       vllava_model,
                       vllava_tokenizer,
                       vllava_processor,
                       device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if len(frames) == 0:
        return "an empty video"

    # limit to 32 frames
    frames = frames[:32]

    inputs = vllava_processor(
        images=frames,
        text="Describe the video in detail.",
        return_tensors="pt"
    ).to(device)

    out = vllava_model.generate(**inputs, max_new_tokens=80)
    caption = vllava_tokenizer.decode(out[0], skip_special_tokens=True)
    return caption.strip()


# ===========================================================
# Encode caption → MPNet vector
# ===========================================================
@torch.no_grad()
def encode_caption_embedding(caption: str, text_encoder, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = text_encoder.encode(
        [caption],
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False
    ).float()
    return emb.squeeze(0).cpu()  # [768]


# ===========================================================
# Dataset with Video-LLaVA caption
# ===========================================================
class FeatureVideoQAHME_MC(Dataset):
    """
    Dataset cho HME Multi-choice VideoQA
      - Appearance feature: [768]
      - Motion feature: [T, 2304]
      - Text feature: [C, 1536] = MPNet(768) + Caption(768)
    """

    def __init__(
        self,
        json_path: str,
        feature_dir_appearance: str,
        feature_dir_motion: str,
        text_encoder: Optional[SentenceTransformer] = None,
        preload_text: bool = True,
        is_test: bool = False,
        # Video-LLaVA
        vllava_model=None,
        vllava_tokenizer=None,
        vllava_processor=None,
        frames_dict: Optional[Dict[str, List[Image.Image]]] = None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["data"]

        self.feature_dir_appearance = feature_dir_appearance
        self.feature_dir_motion = feature_dir_motion
        self.is_test = is_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_encoder = text_encoder or load_text_encoder(self.device)

        # Video-LLaVA
        self.vllava_model = vllava_model
        self.vllava_tokenizer = vllava_tokenizer
        self.vllava_processor = vllava_processor

        self.frames_dict = frames_dict or {}

        self.preload_text = preload_text
        if preload_text:
            print("⚡ Pre-encoding text + Video-LLaVA caption...")
            self.text_cache = self._pre_encode_all_text()
        else:
            print("⚠️ preload_text=False → encode text on the fly (slower).")
            self.text_cache = None

    def __len__(self):
        return len(self.items)

    # -------------------------------------------------------
    # Preload all text + caption
    # -------------------------------------------------------
    @torch.no_grad()
    def _pre_encode_all_text(self) -> Dict[str, torch.Tensor]:
        cache = {}

        for item in self.items:
            qid = item["id"]
            question = item["question"]
            choices = item["choices"]

            video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]

            # -----------------------------
            # MPNet: encode (question + choice)
            # -----------------------------
            texts = [question + " [SEP] " + c for c in choices]
            mpnet_emb = self.text_encoder.encode(
                texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
            ).float().cpu()  # [C, 768]

            # -----------------------------
            # Get Video caption
            # -----------------------------
            cap_emb = torch.zeros(768)

            if (
                self.vllava_model is not None
                and video_name in self.frames_dict
            ):
                caption = videollava_caption(
                    self.frames_dict[video_name],
                    self.vllava_model,
                    self.vllava_tokenizer,
                    self.vllava_processor,
                    device=self.device
                )

                cap_emb = encode_caption_embedding(
                    caption,
                    self.text_encoder,
                    device=self.device
                )  # [768]

            # expand for every choice
            cap_emb = cap_emb.unsqueeze(0).expand(mpnet_emb.shape[0], -1)

            # -----------------------------
            # Combine feature: MPNet + Caption
            # -----------------------------
            combined_emb = torch.cat([mpnet_emb, cap_emb], dim=1)  # [C, 1536]
            cache[qid] = combined_emb

        return cache

    # -------------------------------------------------------
    # On-the-fly encoding
    # -------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str], video_name: str) -> torch.Tensor:
        mpnet_emb = self.text_encoder.encode(
            texts, convert_to_tensor=True, device=self.device, show_progress_bar=False
        ).float().cpu()  # [C, 768]

        cap_emb = torch.zeros(768)
        if (
            self.vllava_model is not None
            and video_name in self.frames_dict
        ):
            caption = videollava_caption(
                self.frames_dict[video_name],
                self.vllava_model,
                self.vllava_tokenizer,
                self.vllava_processor,
                device=self.device
            )

            cap_emb = encode_caption_embedding(
                caption, self.text_encoder, device=self.device
            )

        cap_emb = cap_emb.unsqueeze(0).expand(mpnet_emb.shape[0], -1)
        return torch.cat([mpnet_emb, cap_emb], dim=1)

    # -------------------------------------------------------
    # Get item
    # -------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]

        # Appearance feature
        path_app = os.path.join(self.feature_dir_appearance, f"{qid}.pt")
        feat_app = torch.load(path_app).squeeze(0) if os.path.exists(path_app) else torch.zeros(768)

        # Motion feature
        video_name = os.path.splitext(os.path.basename(item["video_path"]))[0]
        path_mot = os.path.join(self.feature_dir_motion, f"{video_name}.pt")

        if os.path.exists(path_mot):
            feat_mot = torch.load(path_mot)  # [T, 2304]
        else:
            feat_mot = torch.zeros((1, 2304))

        motion_mask = torch.ones(feat_mot.shape[0], dtype=torch.long)

        # Text feature
        if self.preload_text:
            text_feats = self.text_cache[qid]  # [C, 1536]
        else:
            texts = [question + " [SEP] " + c for c in choices]
            text_feats = self.encode_text(texts, video_name)

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
            "appearance_feat": feat_app,
            "motion_feat": feat_mot,
            "motion_mask": motion_mask,
            "text_feats": text_feats,
            "question": question,
            "choice_texts": choices,
            "label": label,
        }


# ===========================================================
# Collate function (unchanged)
# ===========================================================
def collate_fn_hme(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)

    # Motion sequence padding
    max_T = max(b["motion_feat"].shape[0] for b in batch)
    motion_feats = torch.zeros((batch_size, max_T, 2304))
    motion_mask = torch.zeros((batch_size, max_T), dtype=torch.long)

    for i, b in enumerate(batch):
        T_b = b["motion_feat"].shape[0]
        motion_feats[i, :T_b] = b["motion_feat"]
        motion_mask[i, :T_b] = b["motion_mask"]

    appearance_feats = torch.stack([b["appearance_feat"] for b in batch], dim=0)

    max_C = max(b["text_feats"].shape[0] for b in batch)
    text_feats = torch.zeros((batch_size, max_C, batch[0]["text_feats"].shape[1]))

    for i, b in enumerate(batch):
        C_b = b["text_feats"].shape[0]
        text_feats[i, :C_b] = b["text_feats"]

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
