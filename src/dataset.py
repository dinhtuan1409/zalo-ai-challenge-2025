import os
import json
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Ngăn transformers tự ý thêm template, giữ cho [SEP] hoạt động
os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"

# ===========================================================
# ✅ LOAD TEXT ENCODER (MPNet)
# ===========================================================

def load_text_encoder(device: Optional[str] = None) -> SentenceTransformer:
    """
    Load SentenceTransformer MPNet.
    Chỉ load 1 lần duy nhất trong main process.
    """
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        trust_remote_code=True
    )
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    print(f"Text encoder (MPNet) loaded on device: {device}")
    return model


# ===========================================================
# ✅ DATASET – Video feature (.pt) + Text MPNet encode
# *BỔ SUNG RAW TEXT*
# ===========================================================

class FeatureVideoQADatasetMPNET(Dataset):
    """
    Video: load từ .pt (shape: D_video)
    Text: encode trực tiếp bằng MPNet (shape: num_choices x 768)
    Bổ sung: Raw question và choices (dùng cho LLM)
    """

    def __init__(self, 
                 json_path: str, 
                 video_feat_dir: str,
                 video_feat_dim: int = 2304,
                 text_encoder: Optional[SentenceTransformer] = None,
                 preload_text: bool = True,
                 is_test: bool = False):
        """
        preload_text=True: encode toàn bộ text trước → nhanh nhất, ổn định nhất
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)["data"]

        self.video_feat_dir = video_feat_dir
        self.video_feat_dim = video_feat_dim
        self.is_test = is_test

        # text encoder (MPNet)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ❗ LƯU Ý: Vẫn cần MPNet encoder để tạo ra text_feats cho Fusion Module
        self.text_encoder = text_encoder if text_encoder else load_text_encoder(self.device)

        # ✅ Preload text → encode 1 lần (khuyên dùng)
        self.preload_text = preload_text
        if preload_text:
            print("⚡ Encoding toàn bộ text bằng MPNet (1 lần duy nhất)...")
            self.text_cache = self._pre_encode_all_text()
        else:
            print("⚠️ Cảnh báo: `preload_text=False`. Text sẽ được encode on-the-fly, rất chậm.")
            self.text_cache = None

    def __len__(self) -> int:
        return len(self.items)

    # -------------------------------------------------------
    # ✅ Encode toàn bộ text trước
    # -------------------------------------------------------
    @torch.no_grad()
    def _pre_encode_all_text(self) -> Dict[str, torch.Tensor]:
        cache = {}

        for item in self.items:
            qid = item["id"]
            question = item["question"]
            choices = item["choices"]

            # Format: [ "Question [SEP] Choice 1", "Question [SEP] Choice 2", ... ]
            texts = [question + " [SEP] " + c for c in choices]

            emb = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
            
            # Quan trọng: Chuyển về CPU trước khi lưu vào cache
            cache[qid] = emb.float().cpu()

        print(f"✅ Pre-encoded và cached {len(cache)} text items.")
        return cache

    # -------------------------------------------------------
    # ✅ Encode text khi không preload (chậm, không khuyên dùng)
    # -------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        emb = self.text_encoder.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        return emb.float().cpu()

    # -------------------------------------------------------
    # ✅ GET ITEM
    # *BỔ SUNG RAW TEXT*
    # -------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        qid = item["id"]
        
        # Lấy text thô (MỚI)
        question = item["question"]
        choice_texts = item["choices"]

        # 1️⃣ Video feature
        vid = os.path.splitext(os.path.basename(item["video_path"]))[0]
        feat_path = os.path.join(self.video_feat_dir, f"{vid}.pt")

        if os.path.exists(feat_path):
            video_feat = torch.load(feat_path).float()
        else:
            video_feat = torch.zeros(self.video_feat_dim)

        # 2️⃣ Text feature (MPNet embeddings)
        if self.preload_text:
            text_feats = self.text_cache[qid]
        else:
            # Chế độ on-the-fly (rất chậm)
            texts = [question + " [SEP] " + c for c in choice_texts]
            text_feats = self.encode_text(texts)

        # 3️⃣ Label
        label = -1
        if not self.is_test:
            ans = item.get("answer")
            if ans:
                for i, c in enumerate(choice_texts): # Dùng choice_texts
                    if ans.strip() == c.strip():
                        label = i
                        break

        # 4️⃣ Return (BỔ SUNG RAW TEXT)
        base_return = {
            "id": qid,
            "video_feat": video_feat,
            "text_feats": text_feats,
            "question": question,         # MỚI: Text thô
            "choice_texts": choice_texts, # MỚI: List of strings thô
        }

        if not self.is_test:
            base_return["label"] = label
        
        return base_return


# ===========================================================
# ✅ COLLATE FN (PAD CHOICE)
# *XỬ LÝ RAW TEXT*
# ===========================================================

def collate_fn_mpnet(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hàm Collate tùy chỉnh:
    1. Pad text_feats (tensor MPNet)
    2. Gộp raw text (strings) thành list
    """
    
    # Tìm số lượng choices tối đa trong batch này
    max_choice = max(b["text_feats"].shape[0] for b in batch)

    ids, video_feats, text_feats, labels = [], [], [], []
    # MỚI: Lưu trữ raw text
    questions: List[str] = []
    raw_choice_texts: List[List[str]] = []

    has_labels = "label" in batch[0]

    for b in batch:
        ids.append(b["id"])
        video_feats.append(b["video_feat"])

        # MỚI: Lấy raw text (chỉ append list/string, KHÔNG STACK)
        questions.append(b["question"])
        raw_choice_texts.append(b["choice_texts"])

        # Xử lý padding cho text_feats (tensor MPNet embeddings)
        tf = b["text_feats"]
        num_c, dim = tf.shape

        if num_c < max_choice:
            # Tạo tensor padding (số_lượng_pad, dim)
            pad = torch.zeros((max_choice - num_c, dim), dtype=tf.dtype)
            tf = torch.cat([tf, pad], dim=0)

        text_feats.append(tf)

        if has_labels:
            labels.append(b["label"])

    # Stack tensors và trả về raw text
    return {
        "ids": ids,
        "video_feats": torch.stack(video_feats).float(),
        "text_feats": torch.stack(text_feats).float(),
        "labels": torch.tensor(labels, dtype=torch.long) if has_labels else None,
        # MỚI: Trả về list text thô
        "questions": questions,
        "choice_texts": raw_choice_texts,
    }