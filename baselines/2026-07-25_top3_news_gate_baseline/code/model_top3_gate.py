"""Top-3 News Gate baseline model.

Subclasses `SelectiveGateNewsBaseline` (read-only import from the sibling baseline
`2026-07-25_selective_news_gate_baseline`), overriding ONLY the ticker classification: news is
active for just 3 tickers with the strongest, most consistent 4-horizon average delta-R^2 in
the EDA report (`docs/suggestion/2026-07-25_professor_report.md` SS4 "Nhom 1: Huong loi nhieu
nhat"), excluding SHB (highest avg delta-R^2 but suspected time-proxy artifact, per user decision
2026-07-25). Everything else (all 29 other tickers in the actual 32-ticker training universe,
including SHB/VPB/VRE) is bias=0.

Rationale for narrowing further: the prior baseline (22 ON tickers, delta-R^2>=0.01 at t+5)
contradicted its own hypothesis (NEWS_OFF group scored HIGHER DirAcc than NEWS_ON). Narrowing to
only the 3 strongest, most consistent signals tests whether a much higher evidence bar changes
the conclusion.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_GATE_CODE = _ROOT / "baselines" / "2026-07-25_selective_news_gate_baseline" / "code"
_SIBLING_DUAL_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_GATE_CODE), str(_SIBLING_DUAL_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from model_selective_gate import SelectiveGateNewsBaseline  # noqa: E402 (sibling, read-only)

# Avg delta-R^2 across t+1/t+5/t+10/t+22 (docs/suggestion/2026-07-25_professor_report.md SS4
# "Nhom 1"): VIB +0.914, ACB +0.707, MWG +0.560. SHB (+3.124, highest) excluded per user decision
# (suspected time-proxy artifact, same concern flagged for its t+22 delta-R^2 in the EDA report).
NEWS_ON_TICKERS = {"VIB", "ACB", "MWG"}


def build_stock_mask(stock_names: list[str]) -> torch.Tensor:
    """[S] float tensor, 1.0 only for {VIB, ACB, MWG}, 0.0 for every other ticker (including SHB,
    VPB, VRE — no classification lookup needed since this is a strict allowlist of 3)."""
    return torch.tensor([1.0 if s in NEWS_ON_TICKERS else 0.0 for s in stock_names], dtype=torch.float32)


class Top3NewsGateBaseline(SelectiveGateNewsBaseline):
    """SelectiveGateNewsBaseline with the ticker mask narrowed to just {VIB, ACB, MWG}."""

    def __init__(self, config, n_feat: int, stock_names: list[str], d_news: int = 64, dropout: float = 0.5):
        super().__init__(config, n_feat=n_feat, stock_names=stock_names, d_news=d_news, dropout=dropout)
        # Overwrite the buffer registered by the parent (which used the 22/10 classification)
        # with the narrower 3-ticker allowlist.
        self.stock_mask = build_stock_mask(self.stock_names).to(self.stock_mask.device)


def build_default_model(n_feat: int, stock_names: list[str], d_news: int = 64,
                        dropout: float = 0.5) -> Top3NewsGateBaseline:
    from src.lstm_gat_hybrid.config import LSTMGATConfig

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return Top3NewsGateBaseline(config, n_feat=n_feat, stock_names=stock_names,
                                d_news=d_news, dropout=dropout)
