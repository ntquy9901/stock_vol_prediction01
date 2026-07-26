"""Dual-group news embedding baseline model.

HAR branch (reuses ParallelLSTMGNN.get_embeddings, read-only import) + news branch (1-layer LSTM
directly over the pre-aggregated per-day dual-group feature vector) + late concat fusion.

Isolated: no modification to src/lstm_gat_hybrid.

Simpler than `2026-07-07_embedding_baseline`'s news branch: that one pooled a variable-length SET
of raw article embeddings per day (ArticleSetAttentionPooling) because its input was one row per
article. Here each (ticker, date) is already ONE fixed-width vector (built upstream by
build_dual_group_panel.py: mean-pool + EWMA), so the news branch only needs a temporal encoder
over the 22-day window — no pooling/masking layer is needed.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN


class NewsFeatureLSTM(nn.Module):
    """1-layer LSTM over the seq (22-day) window, per stock, on the pre-aggregated per-day
    dual-group news feature vector.

    Input x_news: [B, T=seq, S=stocks, n_feat]
    Output      : [B, S=stocks, d_news]
    """

    def __init__(self, n_feat: int, d_news: int = 64, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_news)
        self.lstm = nn.LSTM(d_news, d_news, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_news: torch.Tensor) -> torch.Tensor:
        B, T, S, F = x_news.shape
        h = self.dropout(torch.relu(self.proj(x_news)))          # [B,T,S,d_news]
        h = h.permute(0, 2, 1, 3).reshape(B * S, T, -1)           # [B*S, T, d_news]
        _, (h_n, _) = self.lstm(h)
        return h_n[-1].reshape(B, S, -1)                          # [B, S, d_news]


class DualGroupNewsBaseline(nn.Module):
    """Parallel LSTM-GNN (HAR) + dual-group news-feature branch -> concat fusion -> [B, num_stocks].

    Note: `ParallelLSTMGNN` is instantiated for its LSTM+GAT feature extractors (used via
    `get_embeddings`). Its own internal fusion MLP is unused, matching `2026-07-07_embedding_baseline`.
    """

    def __init__(self, config: LSTMGATConfig, n_feat: int, d_news: int = 64, dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        self.news_branch = NewsFeatureLSTM(n_feat, d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har: torch.Tensor, adj: torch.Tensor, x_news: torch.Tensor) -> torch.Tensor:
        """
        x_har : [B, seq, num_stocks, 3]
        adj   : [B, num_stocks, num_stocks]
        x_news: [B, seq, num_stocks, n_feat]
        returns: [B, num_stocks]
        """
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,d_lstm], [B,S,d_gat]
        news_rep = self.news_branch(x_news)                   # [B,S,d_news]
        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
        return self.fusion(h).squeeze(-1)


def build_default_model(n_feat: int, d_news: int = 64, dropout: float = 0.5) -> DualGroupNewsBaseline:
    """Build DualGroupNewsBaseline with the project's default LSTMGATConfig (HAR 3 features)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only (NOT 5)
    return DualGroupNewsBaseline(config, n_feat=n_feat, d_news=d_news, dropout=dropout)
