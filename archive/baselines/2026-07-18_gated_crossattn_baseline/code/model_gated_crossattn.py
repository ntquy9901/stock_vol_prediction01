"""MSGCA-style gated cross-attention baseline.

Based on MSGCA — "Stock movement prediction with multimodal stable fusion via gated
cross-attention mechanism" (Complex & Intelligent Systems, 2025) — see
../requirements/requirements.md and ../design/design.md.

Replaces the naive concat+MLP fusion (used by EmbeddingBaseline and all prior news baselines)
with: HAR embedding (query) cross-attends into the news representation (key/value), then a
LEARNED sigmoid gate (conditioned on BOTH modalities, unlike the deterministic has_news binary
gate used in 2026-07-08_market_fallback) controls how much attended-news signal is added.

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
from model_embedding import ArticleSetAttentionPooling  # noqa: E402


class GatedCrossAttnBaseline(nn.Module):
    """forward(x_har, adj, x_emb, mask) -> pred [B, num_stocks].

    har_embed [B,S,320] attends (as a 1-token query) into the SEQUENCE of 22 daily-pooled news
    vectors per stock (as K/V, seq_len=22) via nn.MultiheadAttention.

    [FIX, code review 2026-07-18] An earlier version collapsed the news sequence to a single
    vector (via NewsTemporalEncoder's LSTM) BEFORE cross-attention, giving K/V a seq_len of 1.
    Softmax over a single key is always exactly 1.0 regardless of the query, so `attended` was
    mathematically independent of `har_embed`/`q_proj` — the cross-attention did not actually
    implement query-conditioned selection (q_proj received no useful gradient). Fixed by
    attending over the un-collapsed 22-day sequence (real multi-token K/V) instead — this also
    means NewsTemporalEncoder's LSTM is no longer used here; attention now does both the
    temporal aggregation AND the relevance selection MSGCA describes.
    """

    def __init__(self, config: LSTMGATConfig, emb_dim: int = 64, d_news: int = 64,
                 num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        self.news_pool = ArticleSetAttentionPooling(emb_dim, d_news)

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        d_har = d_lstm + d_gat   # 320

        self.q_proj = nn.Linear(d_har, d_news)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_news, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_har + d_news, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_har + d_news, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_har, adj, x_emb, mask):
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,64], [B,S,256]
        har_embed = torch.cat([h_lstm, h_gnn], dim=-1)         # [B,S,320]
        B, S, _ = har_embed.shape

        daily = self.news_pool(x_emb, mask)                    # [B,seq,S,d_news] (NOT collapsed)
        seq_len = daily.shape[1]
        # -> [B,S,seq,d_news] -> [B*S,seq,d_news]: real multi-token K/V per stock (fixes the
        # degenerate seq_len=1 softmax bug — see class docstring)
        kv = daily.permute(0, 2, 1, 3).reshape(B * S, seq_len, -1)

        q = self.q_proj(har_embed).reshape(B * S, 1, -1)        # [B*S,1,d_news] (1-token query)
        attended, _attn_w = self.cross_attn(q, kv, kv)           # [B*S,1,d_news]
        attended = attended.reshape(B, S, -1)                     # [B,S,d_news]

        gate = self.gate_mlp(torch.cat([har_embed, attended], dim=-1))   # [B,S,1] in (0,1)
        fused = torch.cat([har_embed, gate * attended], dim=-1)           # [B,S,320+d_news]

        return self.fusion(fused).squeeze(-1)                    # [B,S]


def build_default_model(emb_dim: int = 64, d_news: int = 64, num_heads: int = 4,
                        dropout: float = 0.5) -> GatedCrossAttnBaseline:
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return GatedCrossAttnBaseline(config, emb_dim=emb_dim, d_news=d_news, num_heads=num_heads,
                                  dropout=dropout)
