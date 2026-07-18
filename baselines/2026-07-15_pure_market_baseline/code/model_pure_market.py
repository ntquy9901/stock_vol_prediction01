"""Pure-market model: HAR branch + ONE market-news vector per day, BROADCAST to every stock.

No per-stock news branch, no ticker matching, no gate. Reuses ArticleSetAttentionPooling +
NewsTemporalEncoder from the 2026-07-07 embedding baseline (read-only import) via a
stocks-dim=1 adapter (those modules are dimension-agnostic on the "stocks" axis — they only
do Linear + softmax + weighted-sum on the last axes).

Isolated: no modification to src/lstm_gat_hybrid or any sibling baseline.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_EMB_CODE = _ROOT / "baselines" / "2026-07-07_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_EMB_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN
from model_embedding import ArticleSetAttentionPooling, NewsTemporalEncoder  # noqa: E402


class PureMarketBaseline(nn.Module):
    """Parallel LSTM-GNN (HAR) + market-news branch (broadcast, no ticker match) -> fusion.

    forward(x_har, adj, x_market, market_mask) -> [B, num_stocks]
      x_har      : [B, seq, num_stocks, 3]
      adj        : [B, num_stocks, num_stocks]
      x_market   : [B, seq, MAX_MARKET, emb_dim]   (SAME for every stock in the batch item)
      market_mask: [B, seq, MAX_MARKET]
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():  # unused internal fusion MLP — freeze (see sibling)
            p.requires_grad_(False)

        # Market branch: reused pooling modules, stocks-dim=1 (see forward())
        self.market_pool = ArticleSetAttentionPooling(emb_dim, d_news)
        self.market_temporal = NewsTemporalEncoder(d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har, adj, x_market, market_mask):
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,64], [B,S,256]
        num_stocks = h_lstm.shape[1]

        # stocks-dim=1 adapter: reuse per-stock pooling modules for the single market vector
        x_market_5d = x_market.unsqueeze(2)        # [B, seq, 1, MAX_MARKET, emb_dim]
        market_mask_4d = market_mask.unsqueeze(2)   # [B, seq, 1, MAX_MARKET]
        daily = self.market_pool(x_market_5d, market_mask_4d)   # [B, seq, 1, d_news]
        market_rep = self.market_temporal(daily).squeeze(1)     # [B, d_news] (1 vector/sample)

        # BROADCAST: identical market vector concatenated onto every stock (no gate, no per-stock)
        market_bc = market_rep.unsqueeze(1).expand(-1, num_stocks, -1)  # [B, S, d_news]

        h = torch.cat([h_lstm, h_gnn, market_bc], dim=-1)   # [B, S, 384]
        return self.fusion(h).squeeze(-1)                    # [B, S]


def build_default_model(emb_dim: int = 64, d_news: int = 64, dropout: float = 0.5) -> PureMarketBaseline:
    """Build PureMarketBaseline with project default LSTMGATConfig (HAR 3 features)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    return PureMarketBaseline(config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)
