"""Ablation-Derived News Gate baseline model.

Subclasses `SelectiveGateNewsBaseline` (read-only import from the sibling baseline
`2026-07-25_selective_news_gate_baseline`), overriding the ticker classification with the list
derived from `2026-07-25_news_usefulness_ablation` — a per-ticker delta_QLIKE comparison between
a fresh HAR-only reference and the all-32-stocks-ON dual-group model, BOTH trained for the same
10 epochs on the identical data pipeline (not borrowed from a different model family, unlike the
two prior gate baselines today).
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

# Derived from baselines/2026-07-25_news_usefulness_ablation/results/
# ablation_derived_ticker_classification.json (delta_qlike < 0, epoch-matched 10-vs-10).
NEWS_ON_TICKERS = {"HDB", "HPG", "MWG", "NVL", "PDR", "PLX", "SSI", "VHM", "VJC", "VPB", "VRE"}


def build_stock_mask(stock_names: list[str]) -> torch.Tensor:
    """[S] float tensor, 1.0 for the 11 ablation-derived ON tickers, 0.0 for everyone else."""
    return torch.tensor([1.0 if s in NEWS_ON_TICKERS else 0.0 for s in stock_names], dtype=torch.float32)


class AblationDerivedGateBaseline(SelectiveGateNewsBaseline):
    """SelectiveGateNewsBaseline with the mask narrowed to the 11 ablation-derived ON tickers."""

    def __init__(self, config, n_feat: int, stock_names: list[str], d_news: int = 64, dropout: float = 0.5):
        super().__init__(config, n_feat=n_feat, stock_names=stock_names, d_news=d_news, dropout=dropout)
        self.stock_mask = build_stock_mask(self.stock_names).to(self.stock_mask.device)


def build_default_model(n_feat: int, stock_names: list[str], d_news: int = 64,
                        dropout: float = 0.5) -> AblationDerivedGateBaseline:
    from src.lstm_gat_hybrid.config import LSTMGATConfig

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return AblationDerivedGateBaseline(config, n_feat=n_feat, stock_names=stock_names,
                                       d_news=d_news, dropout=dropout)
