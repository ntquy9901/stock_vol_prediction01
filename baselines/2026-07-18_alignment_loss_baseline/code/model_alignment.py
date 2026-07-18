"""M2VN-style alignment-loss baseline.

Based on M2VN — "Fusing Narrative Semantics for Financial Volatility Forecasting"
(Kong et al., Oxford/ICAIF'25, arXiv:2510.20699) — see ../requirements/requirements.md and
../design/design.md. Only the auxiliary alignment-loss idea is reproduced (M2VN's point-in-time
LLM, Time Machine GPT, has no available replacement here).

Prediction path is IDENTICAL to the sibling EmbeddingBaseline (concat + MLP fusion) — this
baseline isolates whether an auxiliary loss pulling the HAR and news latent spaces together
helps the SAME fusion mechanism learn a better representation, without changing the fusion
itself. 2 extra projection heads are used ONLY for the alignment loss at train time.

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
import torch.nn.functional as F

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN
from model_embedding import ArticleSetAttentionPooling, NewsTemporalEncoder  # noqa: E402


class AlignmentLossBaseline(nn.Module):
    """forward(x_har, adj, x_emb, mask) -> (pred [B,S], proj_har [B,S,d_align], proj_news [B,S,d_align]).

    `pred` is the final volatility forecast (same fusion as EmbeddingBaseline). `proj_har` /
    `proj_news` are only used by the training loop to compute the M2VN-style alignment loss —
    they do NOT feed back into `pred` (no architecture change to the prediction path itself).
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 d_align: int = 32, dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        self.news_pool = ArticleSetAttentionPooling(emb_dim, d_news)
        self.news_temporal = NewsTemporalEncoder(d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # alignment projection heads (train-time only auxiliary signal)
        self.align_har = nn.Linear(d_lstm + d_gat, d_align)
        self.align_news = nn.Linear(d_news, d_align)

    def forward(self, x_har, adj, x_emb, mask):
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,64], [B,S,256]
        har_embed = torch.cat([h_lstm, h_gnn], dim=-1)         # [B,S,320]

        daily = self.news_pool(x_emb, mask)
        news_rep = self.news_temporal(daily)                   # [B,S,64]

        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
        pred = self.fusion(h).squeeze(-1)                       # [B,S]

        proj_har = F.normalize(self.align_har(har_embed), dim=-1)     # [B,S,d_align]
        proj_news = F.normalize(self.align_news(news_rep), dim=-1)    # [B,S,d_align]

        return pred, proj_har, proj_news


def alignment_loss(proj_har: torch.Tensor, proj_news: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity (both inputs already L2-normalized -> cosine = dot product)."""
    return 1.0 - (proj_har * proj_news).sum(dim=-1).mean()


def build_default_model(emb_dim: int = 64, d_news: int = 64, d_align: int = 32,
                        dropout: float = 0.5) -> AlignmentLossBaseline:
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return AlignmentLossBaseline(config, emb_dim=emb_dim, d_news=d_news, d_align=d_align,
                                 dropout=dropout)
