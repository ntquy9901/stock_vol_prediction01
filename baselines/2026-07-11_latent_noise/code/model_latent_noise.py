"""Latent Noise Injection baseline (Tier A).

Subclass of `EmbeddingBaseline` (read-only import from sibling baseline
2026-07-07_embedding_baseline). Adds Gaussian noise to the news representation
during training only:  z' = z + noise_std * eps,  eps ~ N(0,1).

Eval mode -> no noise -> deterministic. No loss change (that's Tier B / VIB, deferred).

Isolated: no edit to src/ or to the embedding baseline folder.
"""
import sys
from pathlib import Path

# bootstrap paths (rule §3.F.4): project root + own code + sibling embedding-baseline code
_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_EMB_CODE = _ROOT / "baselines" / "2026-07-07_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_EMB_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# read-only reuse: EmbeddingBaseline from the sibling embedding baseline
from model_embedding import EmbeddingBaseline  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402


class LatentNoiseBaseline(EmbeddingBaseline):
    """EmbeddingBaseline + Gaussian noise on the news representation (train only).

    Rationale (teacher's hint): news is sparse (94.5% of stock-days are blind);
    injecting noise on the news latent discourages the model from over-relying on
    the few days that have ticker-specific news.

    forward(x_har, adj, x_emb, mask) -> [B, num_stocks]  (same I/O as EmbeddingBaseline)
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 dropout: float = 0.5, noise_std: float = 0.1):
        super().__init__(config=config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)
        self.noise_std = float(noise_std)

    def forward(self, x_har, adj, x_emb, mask):
        # HAR branch (unchanged) + news branch, then noise on news_rep (train only)
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B, S, 64], [B, S, 256]
        daily = self.news_pool(x_emb, mask)                   # [B, S, seq, d_news]
        news_rep = self.news_temporal(daily)                  # [B, S, d_news]
        if self.training and self.noise_std > 0:
            news_rep = news_rep + self.noise_std * torch.randn_like(news_rep)
        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)      # [B, S, 384]
        return self.fusion(h).squeeze(-1)                     # [B, S]


def build_default_model(emb_dim: int = 64, d_news: int = 64, dropout: float = 0.5,
                        noise_std: float = 0.1) -> LatentNoiseBaseline:
    """Build LatentNoiseBaseline with project default LSTMGATConfig (HAR 3 features)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    return LatentNoiseBaseline(config, emb_dim=emb_dim, d_news=d_news,
                               dropout=dropout, noise_std=noise_std)
