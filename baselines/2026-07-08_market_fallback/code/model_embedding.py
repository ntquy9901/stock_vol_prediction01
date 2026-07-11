"""Market-fallback model.

HAR branch (reuse ParallelLSTMGNN.get_embeddings) + per-stock news (sparse) + market news
(dense) fused via a deterministic availability gate + temporal LSTM + concat.

Isolated: no modification to src/lstm_gat_hybrid or the embedding baseline.
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


class ArticleSetAttentionPooling(nn.Module):
    """Aggregate a variable-length set of news embeddings per (stock, day) — permutation-invariant.

    Note [MEDIUM-7]: 1-article days give query no gradient; accepted trade-off for sparse data.
    """

    def __init__(self, emb_dim: int, d_news: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d_news)
        self.query = nn.Parameter(torch.randn(d_news) * 0.02)
        self.no_news_token = nn.Parameter(torch.randn(d_news) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, article_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(article_embs)
        scores = (h * self.query).sum(-1)
        scores = scores.masked_fill(mask == 0, -1e9)   # finite (not -inf) -> no NaN on all-masked
        attn = torch.softmax(scores, dim=-1)
        daily = (attn.unsqueeze(-1) * h).sum(-2)
        has_news = (mask.sum(-1, keepdim=True) > 0).to(daily.dtype)
        daily = has_news * daily + (1 - has_news) * self.no_news_token
        return self.dropout(daily)


class NewsTemporalEncoder(nn.Module):
    """1-layer LSTM over the seq window, per stock. Input [B,T,S,D] -> output [B,S,D]."""

    def __init__(self, d_news: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_news, d_news, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, daily: torch.Tensor) -> torch.Tensor:
        B, T, S, D = daily.shape
        x = self.dropout(daily.permute(0, 2, 1, 3).reshape(B * S, T, D))
        _, (h, _) = self.lstm(x)
        return h[-1].reshape(B, S, D)


class MarketBranch(nn.Module):
    """Pool ALL articles of a day -> 1 market vector/day. Reuses ArticleSetAttentionPooling.

    Input : x_market [B, T, MAX_M, emb_dim], market_mask [B, T, MAX_M]
    Output: market_daily [B, T, d_news]  (one per day, shared across stocks)
    """

    def __init__(self, emb_dim: int, d_news: int):
        super().__init__()
        self.pool = ArticleSetAttentionPooling(emb_dim, d_news)   # reuse

    def forward(self, x_market: torch.Tensor, market_mask: torch.Tensor) -> torch.Tensor:
        # ArticleSetAttentionPooling operates on [..., A, D] with mask [..., A].
        # Here leading dims are [B, T] -> output [B, T, d_news].
        return self.pool(x_market, market_mask)


class GatedNewsFusion(nn.Module):
    """Fuse per-stock news (sparse) with market news (dense) via an availability gate.

    Deterministic MVP: g = has_news (1 if stock has news that day, else 0).
    -> uses stock-specific when available, market fallback when stock is news-blind.
    """

    def __init__(self):
        super().__init__()

    def forward(self, stock_daily: torch.Tensor, market_daily: torch.Tensor,
                has_news: torch.Tensor) -> torch.Tensor:
        # stock_daily: [B, T, S, d], market_daily: [B, T, d], has_news: [B, T, S, 1]
        market = market_daily.unsqueeze(2).expand_as(stock_daily)   # [B, T, S, d]
        g = has_news.to(stock_daily.dtype)
        return g * stock_daily + (1 - g) * market


class MarketFallbackBaseline(nn.Module):
    """HAR (LSTM+GAT) + gated [per-stock news | market news] -> concat fusion -> [B, num_stocks]."""

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():   # [MEDIUM-8] freeze unused ParallelLSTMGNN fusion
            p.requires_grad_(False)

        self.news_pool = ArticleSetAttentionPooling(emb_dim, d_news)   # per-stock
        self.market_branch = MarketBranch(emb_dim, d_news)             # market (reuses pooling)
        self.gated_fusion = GatedNewsFusion()
        self.news_temporal = NewsTemporalEncoder(d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har, adj, x_emb, mask, x_market, market_mask):
        """
        x_har      : [B, seq, num_stocks, 3]
        adj        : [B, num_stocks, num_stocks]
        x_emb      : [B, seq, num_stocks, MAX_ARTICLES, emb_dim]
        mask       : [B, seq, num_stocks, MAX_ARTICLES]
        x_market   : [B, seq, MAX_MARKET, emb_dim]
        market_mask: [B, seq, MAX_MARKET]
        returns    : [B, num_stocks]
        """
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)              # [B, S, d_lstm/gat]
        stock_daily = self.news_pool(x_emb, mask)                        # [B, T, S, d]
        market_daily = self.market_branch(x_market, market_mask)         # [B, T, d]
        has_news = (mask.sum(-1, keepdim=True) > 0).float()              # [B, T, S, 1]
        daily = self.gated_fusion(stock_daily, market_daily, has_news)   # [B, T, S, d]
        news_rep = self.news_temporal(daily)                             # [B, S, d]
        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
        return self.fusion(h).squeeze(-1)


def build_default_model(emb_dim: int = 64, d_news: int = 64, dropout: float = 0.5) -> MarketFallbackBaseline:
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    return MarketFallbackBaseline(config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)
