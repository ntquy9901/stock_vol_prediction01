"""Embedding baseline model.

HAR branch (reuses ParallelLSTMGNN.get_embeddings, read-only import) + news branch
(Article-Set Attention Pooling + 1-L temporal LSTM) + late concat fusion.

Isolated: no modification to src/lstm_gat_hybrid.
"""
import sys
from pathlib import Path

# bootstrap paths (rule 3.F.4)
_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN


class ArticleSetAttentionPooling(nn.Module):
    """Aggregate a variable-length set of news embeddings per (stock, day).

    Permutation-invariant via a shared learnable query. Days with 0 real articles
    emit a learned `no_news_token` so the model can distinguish "no news" from "neutral".

    Note [MEDIUM-7]: for days with exactly 1 real article, softmax over a single entry = 1.0,
    so the query receives no gradient from that day and output = proj(that article). Accepted
    trade-off for sparse-data regimes (most VN stock-days have 0-1 articles); the query still
    trains on multi-article days. Switch to mean-pooling if this becomes limiting.

    Input:
        article_embs: [B, T, S, A, D]  (T=seq, S=stocks; matches x_har layout)
        mask        : [B, T, S, A]     (1 = real article, 0 = pad)
    Output:
        daily       : [B, T, S, d_news]
    """

    def __init__(self, emb_dim: int, d_news: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d_news)
        self.query = nn.Parameter(torch.randn(d_news) * 0.02)
        self.no_news_token = nn.Parameter(torch.randn(d_news) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, article_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(article_embs)                        # [B,S,T,A,d_news]
        scores = (h * self.query).sum(-1)                  # [B,S,T,A]
        # Finite large-negative (not -inf): softmax over all-masked day -> uniform (finite),
        # then has_news=0 swaps in no_news_token below. -inf would yield NaN.
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)               # [B,S,T,A]
        daily = (attn.unsqueeze(-1) * h).sum(-2)           # [B,S,T,d_news]
        has_news = (mask.sum(-1, keepdim=True) > 0).to(daily.dtype)  # [B,S,T,1]
        daily = has_news * daily + (1 - has_news) * self.no_news_token
        return self.dropout(daily)


class NewsTemporalEncoder(nn.Module):
    """1-layer LSTM over the seq (22-day) window, per stock. Vectorized across batch.

    Input daily: [B, T=seq, S=stocks, d_news]
    Output     : [B, S=stocks, d_news]
    """

    def __init__(self, d_news: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_news, d_news, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, daily: torch.Tensor) -> torch.Tensor:
        B, T, S, D = daily.shape
        # [B,T,S,D] -> [B,S,T,D] -> [B*S, T, D] so LSTM runs over T per stock
        x = self.dropout(daily.permute(0, 2, 1, 3).reshape(B * S, T, D))
        _, (h, _) = self.lstm(x)
        return h[-1].reshape(B, S, D)                      # [B, S, d_news]


class EmbeddingBaseline(nn.Module):
    """Parallel LSTM-GNN (HAR) + news embedding branch -> concat fusion -> [B, num_stocks].

    Note: `ParallelLSTMGNN` is instantiated for its LSTM+GAT feature extractors
    (used via `get_embeddings`). Its own internal fusion MLP is unused (~23K idle params,
    acceptable trade-off for clean read-only reuse vs duplicating ~60 lines).
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.config = config
        # HAR branch (reused). config.num_features_per_stock MUST be 3 (HAR only).
        self.har = ParallelLSTMGNN(config)
        # [MEDIUM-8 fix] ParallelLSTMGNN's internal fusion MLP is unused (we call get_embeddings).
        # Freeze it: no grad, no Adam state, no checkpoint bloat (~23K idle params excluded).
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        # News branch
        self.news_pool = ArticleSetAttentionPooling(emb_dim, d_news)
        self.news_temporal = NewsTemporalEncoder(d_news)

        # Fusion: HAR-LSTM (64) + HAR-GAT (256) + news (d_news) -> MLP
        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har: torch.Tensor, adj: torch.Tensor,
                x_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_har: [B, seq, num_stocks, 3]
        adj  : [B, num_stocks, num_stocks]
        x_emb: [B, seq, num_stocks, MAX_ARTICLES, emb_dim]
        mask : [B, seq, num_stocks, MAX_ARTICLES]
        returns: [B, num_stocks]
        """
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B, S_stocks, d_lstm], [B, S_stocks, d_gat]
        daily = self.news_pool(x_emb, mask)                   # [B, S_stocks, seq, d_news]
        news_rep = self.news_temporal(daily)                  # [B, S_stocks, d_news]
        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
        return self.fusion(h).squeeze(-1)


def build_default_model(emb_dim: int = 64, d_news: int = 64, dropout: float = 0.5) -> EmbeddingBaseline:
    """Build EmbeddingBaseline with the project's default LSTMGATConfig (HAR 3 features)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only (NOT 5)
    return EmbeddingBaseline(config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)
