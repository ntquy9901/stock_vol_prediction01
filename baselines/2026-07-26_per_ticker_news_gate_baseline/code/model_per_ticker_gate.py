"""Per-ticker isolated-gradient news gate (see design.md §1).

Identical to the sibling `2026-07-25_dual_group_news_embedding_baseline`'s
`DualGroupNewsBaseline` (HAR branch via `ParallelLSTMGNN.get_embeddings` + `NewsFeatureLSTM` +
concat fusion), with exactly ONE addition: a per-ticker learnable scalar gate
(`gate_logits`, shape [num_stocks], NOT shared across tickers) that scales each ticker's own
news representation before fusion.

Unlike `2026-07-18_gated_crossattn_baseline`'s `gate_mlp` (a single MLP with weights SHARED
across all 30 tickers — its gradient is a sum over all tickers' loss terms each step, which lets
it learn cross-ticker/sector shortcuts instead of per-ticker news usefulness; see that baseline's
`analyze_gate_per_ticker.py` finding of near-zero, wrong-direction correlation with real
usefulness), `gate_logits[i]` here is used ONLY in computing `pred[:, i]` — no other ticker's
prediction or gate parameter touches it. `∂loss/∂gate_logits[i]` is therefore a function of
ticker i's own prediction error ONLY (proven in design.md §1, verified by
`test_gate_gradient_isolated_per_ticker`).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN
from model_dual_news import NewsFeatureLSTM  # noqa: E402 (sibling, read-only)


class PerTickerGatedNewsBaseline(nn.Module):
    """forward(x_har, adj, x_news) -> pred [B, num_stocks].

    `num_stocks` must be fixed and known at construction time (the gate is one scalar per
    stock-SLOT, not per stock-identity string — the caller is responsible for keeping the same
    stock ordering across train/val/test, which `create_dual_news_dataloaders` already guarantees
    via its shared `common_stocks` list).
    """

    def __init__(self, config: LSTMGATConfig, n_feat: int, num_stocks: int,
                 d_news: int = 64, dropout: float = 0.5):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks
        self.har = ParallelLSTMGNN(config)
        for p in self.har.fusion.parameters():
            p.requires_grad_(False)

        self.news_branch = NewsFeatureLSTM(n_feat, d_news)

        # One free scalar per ticker, init at 0 -> sigmoid(0) = 0.5 (neutral: neither forced
        # open nor closed at the start of training). NOT shared across tickers (design.md §1) —
        # this is the entire point of this baseline vs. gated_crossattn's shared gate_mlp.
        self.gate_logits = nn.Parameter(torch.zeros(num_stocks))

        d_lstm = config.lstm_hidden_dim
        d_gat = config.gat_num_heads * config.gat_hidden_dim
        fusion_in = d_lstm + d_gat + d_news
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def gate_values(self) -> torch.Tensor:
        """Current per-ticker gate values in (0,1), shape [num_stocks]. Detached, for logging."""
        return torch.sigmoid(self.gate_logits).detach()

    def forward(self, x_har: torch.Tensor, adj: torch.Tensor, x_news: torch.Tensor) -> torch.Tensor:
        """
        x_har : [B, seq, num_stocks, 3]
        adj   : [B, num_stocks, num_stocks]
        x_news: [B, seq, num_stocks, n_feat]
        returns: [B, num_stocks]
        """
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)   # [B,S,d_lstm], [B,S,d_gat]
        news_rep = self.news_branch(x_news)                   # [B,S,d_news]

        S = news_rep.shape[1]
        gate = torch.sigmoid(self.gate_logits).view(1, S, 1)   # [1,S,1], broadcasts over B and d_news
        gated_news = gate * news_rep                            # [B,S,d_news] — per-ticker scaling only

        h = torch.cat([h_lstm, h_gnn, gated_news], dim=-1)
        return self.fusion(h).squeeze(-1)


def build_default_model(n_feat: int, num_stocks: int, d_news: int = 64,
                        dropout: float = 0.5) -> PerTickerGatedNewsBaseline:
    """Build PerTickerGatedNewsBaseline with the project's default LSTMGATConfig (HAR 3 features)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return PerTickerGatedNewsBaseline(config, n_feat=n_feat, num_stocks=num_stocks,
                                      d_news=d_news, dropout=dropout)
