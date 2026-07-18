"""REST-TS baseline: Residual-Exclusive Supervision for Text in Time Series.

Based on "Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time
Series Forecasting" (Nguyen et al., Deakin, arXiv:2606.19413, 2026) — see
../requirements/requirements.md and ../design/design.md.

Two INDEPENDENT prediction heads instead of one shared concat+MLP fusion (what all 5 prior
news baselines in this project did):
  - har_head:  predicts y directly from HAR embeddings alone (never sees news).
  - news_head: predicts the RESIDUAL (y - har_pred, gradient-detached) — because no numerical
    pathway can reduce this residual loss, the news branch is structurally forced to extract
    genuine signal instead of collapsing to a content-independent constant.

Reuses ArticleSetAttentionPooling + NewsTemporalEncoder (read-only import) — same proven
pooling as the sibling baseline, only the fusion/supervision mechanism changes.

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


class RestTsBaseline(nn.Module):
    """forward(x_har, adj, x_emb, mask) -> (har_pred [B,S], news_pred [B,S]).

    Caller is responsible for combining (har_pred + news_pred) for the final prediction and
    for computing the REST-TS 2-part loss (see train_resttext.py) — kept out of forward() so
    the residual-detach step (which must happen in the TRAINING loop, not the model) is
    explicit and not hidden inside the module.
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        self.news_pool = ArticleSetAttentionPooling(emb_dim, d_news)
        self.news_temporal = NewsTemporalEncoder(d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        # har_head: independent, never sees news_rep
        self.har_head = nn.Sequential(
            nn.Linear(d_lstm + d_gat, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        # news_head: predicts the RESIDUAL only, from news_rep alone
        self.news_head = nn.Sequential(
            nn.Linear(d_news, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har, adj, x_emb, mask):
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,64], [B,S,256]
        har_pred = self.har_head(torch.cat([h_lstm, h_gnn], dim=-1)).squeeze(-1)  # [B,S]

        daily = self.news_pool(x_emb, mask)                   # [B,S,seq,d_news]... see note below
        news_rep = self.news_temporal(daily)                  # [B,S,d_news]
        news_pred = self.news_head(news_rep).squeeze(-1)       # [B,S]

        return har_pred, news_pred


def build_default_model(emb_dim: int = 64, d_news: int = 64, dropout: float = 0.5) -> RestTsBaseline:
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return RestTsBaseline(config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)
