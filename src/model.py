import torch
import torch.nn as nn

class ContextTransformerQAModel(nn.Module):
    """
    Kết hợp video và text bằng cách xem video như một [CONTEXT] token
    và cho nó tương tác với các token của C choices.

    Nhận đầu vào:
        video_feats: (B, D_v)
        text_feats:  (B, C, 768)

    Trả ra:
        logits: (B, C)
    """
    def __init__(
        self,
        video_dim=2304,    # Feature video của bạn
        text_dim=768,      # MPNet fixed
        hidden_dim=512,    # Chiều không gian chung
        nhead=8,
        num_layers=2,      # Có thể thử 1, 2 hoặc 3
        dropout=0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim

        # 1. Chiếu video về hidden_dim
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Chiếu text về hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Transformer Encoder để kết hợp và so sánh
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, video_feats, text_feats):
        """
        video_feats: (B, D_v)
        text_feats:  (B, C, 768)
        """
        B, C, _ = text_feats.shape

        # 1. Chiếu video
        # (B, D_v) -> (B, H)
        v_proj = self.video_proj(video_feats)
        
        # Thêm 1 chiều sequence -> (B, 1, H)
        # Đây sẽ là [CONTEXT_TOKEN] của chúng ta
        v_token = v_proj.unsqueeze(1)

        # 2. Chiếu text
        # (B, C, 768) -> (B, C, H)
        t_proj = self.text_proj(text_feats)

        # 3. Nối video và text lại thành 1 sequence
        # (B, 1, H) + (B, C, H) -> (B, 1+C, H)
        x = torch.cat([v_token, t_proj], dim=1)

        # 4. Đưa qua Transformer
        # (B, 1+C, H) -> (B, 1+C, H)
        x_transformed = self.transformer_encoder(x)

        # 5. Lấy ra các embedding của C lựa chọn (bỏ qua video token)
        # (B, 1+C, H) -> (B, C, H)
        choice_embs = x_transformed[:, 1:, :] 
        
        # 6. Phân loại trên từng lựa chọn
        # (B, C, H) -> (B, C, 1) -> (B, C)
        logits = self.classifier(choice_embs).squeeze(-1)

        return logits