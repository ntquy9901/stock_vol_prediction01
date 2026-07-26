"""Selective News Gate baseline model.

Subclasses `DualGroupNewsBaseline` (read-only import from the sibling baseline
`2026-07-25_dual_group_news_embedding_baseline`) and adds a FIXED (non-learnable) per-stock mask
on `news_rep`, applied AFTER `NewsFeatureLSTM` and BEFORE the fusion concat. Masking after the
LSTM (not on the raw `x_news` input) guarantees an exactly-zero news contribution for masked
stocks regardless of the LSTM's bias terms (an all-zero input to an LSTM does not necessarily
produce an all-zero output).

Ticker list source: `docs/suggestion/2026-07-25_professor_report.md` (per-ticker HGB/XGBoost
delta-R^2 at the t+5 horizon specifically — NOT the 4-horizon average, which would mix in
signal from t+1/t+10/t+22). SHB excluded per explicit user decision (2026-07-25) despite a
positive delta-R^2, due to a suspected time-proxy artifact (the EDA report flags the same
mechanism inflating SHB's t+22 delta-R^2).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from model_dual_news import DualGroupNewsBaseline  # noqa: E402 (sibling baseline, read-only)

# 22 tickers with delta-R^2 >= 0.01 at t+5 (docs/suggestion/2026-07-25_professor_report.md,
# Phu luc table, t+5 column) -> news branch active.
NEWS_ON_TICKERS = {
    "ACB", "MWG", "VIB", "TPB", "SAB", "VJC", "MBB", "POW", "TCB", "MSN",
    "HPG", "VIC", "SSB", "BID", "SSI", "STB", "FPT", "VCB", "CTG", "GVR", "VNM", "HDB",
}
# 8 tickers with bias=0 (news branch forced off): SHB (user-excluded, time-proxy suspicion),
# GAS/PLX/NVL/BVH (negative delta-R^2 at t+5), VHM/BCM/PDR (positive but negligible, <0.01).
NEWS_OFF_TICKERS = {"SHB", "GAS", "PLX", "NVL", "BVH", "VHM", "BCM", "PDR"}
# [2026-07-25, discovered at first real run] the actual training pipeline's stock universe
# (`_load_raw_stock_data` / `_split_raw_data_by_date`) has 32 common stocks, 2 more than the
# EDA report's 30-ticker VN30 analysis: VPB, VRE. No delta-R^2 evidence exists for them either
# way -> default to the conservative direction (OFF), consistent with this baseline's whole
# premise of only turning news ON where there's positive evidence.
NEWS_OFF_TICKERS |= {"VPB", "VRE"}


def build_stock_mask(stock_names: list[str]) -> torch.Tensor:
    """[S] float tensor, 1.0 for NEWS_ON tickers, 0.0 for NEWS_OFF — in the EXACT order of
    `stock_names` (the dataset's actual per-batch stock ordering, not assumed fixed)."""
    unknown = [s for s in stock_names if s not in NEWS_ON_TICKERS and s not in NEWS_OFF_TICKERS]
    if unknown:
        raise ValueError(
            f"stock(s) not classified as NEWS_ON or NEWS_OFF: {unknown} — the ticker "
            "classification (from the EDA report) must cover every ticker the dataset trains on."
        )
    return torch.tensor([1.0 if s in NEWS_ON_TICKERS else 0.0 for s in stock_names], dtype=torch.float32)


class SelectiveGateNewsBaseline(DualGroupNewsBaseline):
    """DualGroupNewsBaseline + a fixed per-stock mask zeroing `news_rep` for NEWS_OFF tickers."""

    def __init__(self, config, n_feat: int, stock_names: list[str], d_news: int = 64, dropout: float = 0.5):
        super().__init__(config, n_feat=n_feat, d_news=d_news, dropout=dropout)
        self.stock_names = list(stock_names)
        # buffer (not nn.Parameter): fixed by domain knowledge, never updated by the optimizer.
        self.register_buffer("stock_mask", build_stock_mask(self.stock_names))

    def forward(self, x_har: torch.Tensor, adj: torch.Tensor, x_news: torch.Tensor) -> torch.Tensor:
        h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)
        news_rep = self.news_branch(x_news)
        news_rep = news_rep * self.stock_mask.view(1, -1, 1)   # [B,S,d_news] * [1,S,1]
        h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
        return self.fusion(h).squeeze(-1)


def build_default_model(n_feat: int, stock_names: list[str], d_news: int = 64,
                        dropout: float = 0.5) -> SelectiveGateNewsBaseline:
    from src.lstm_gat_hybrid.config import LSTMGATConfig

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return SelectiveGateNewsBaseline(config, n_feat=n_feat, stock_names=stock_names,
                                     d_news=d_news, dropout=dropout)
