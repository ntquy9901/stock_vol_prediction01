"""Test the core claim of this baseline: NEWS_OFF stocks get an EXACTLY zero news contribution,
NEWS_ON stocks don't.

Run: pytest baselines/2026-07-25_selective_news_gate_baseline/test/test_mask_correctness.py -v
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_selective_gate import (
    NEWS_OFF_TICKERS, NEWS_ON_TICKERS, SelectiveGateNewsBaseline, build_stock_mask,
)

pytestmark = pytest.mark.smoke


def _make_model(stock_names, n_feat=8, d_news=16):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = SelectiveGateNewsBaseline(config, n_feat=n_feat, stock_names=stock_names,
                                      d_news=d_news, dropout=0.0).eval()
    return model


def test_build_stock_mask_matches_classification():
    stock_names = ["ACB", "SHB", "GAS", "VIB"]   # ACB/VIB=ON, SHB/GAS=OFF
    mask = build_stock_mask(stock_names)
    assert mask.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_build_stock_mask_rejects_unclassified_ticker():
    with pytest.raises(ValueError, match="not classified"):
        build_stock_mask(["ACB", "NOT_A_REAL_TICKER"])


def test_news_off_stock_has_zero_news_contribution():
    """Changing x_news for a NEWS_OFF stock must NOT change that stock's prediction at all
    (exact equality, not just 'small difference') — the whole point of masking after the LSTM."""
    stock_names = ["ACB", "SHB"]   # ACB=ON, SHB=OFF
    assert stock_names[1] in NEWS_OFF_TICKERS
    model = _make_model(stock_names)

    B, T, S, F = 2, 5, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 1, :] = torch.randn(B, T, F)   # perturb ONLY the SHB (index 1, NEWS_OFF) column

    with torch.no_grad():
        pred_a = model(x_har, adj, x_news_a)
        pred_b = model(x_har, adj, x_news_b)

    # SHB (index 1) prediction must be numerically IDENTICAL despite different news input.
    assert torch.equal(pred_a[:, 1], pred_b[:, 1]), "NEWS_OFF stock's prediction changed with different news input"
    # ACB (index 0, NEWS_ON) prediction is untouched by SHB's own column changing (batched per-stock
    # LSTM has no cross-stock leakage) -- also must be identical here since only SHB's column changed.
    assert torch.equal(pred_a[:, 0], pred_b[:, 0])


def test_news_on_stock_is_affected_by_its_own_news():
    """Changing x_news for a NEWS_ON stock SHOULD change that stock's prediction (sanity check
    that the mask isn't accidentally zeroing everything)."""
    stock_names = ["ACB", "SHB"]
    assert stock_names[0] in NEWS_ON_TICKERS
    model = _make_model(stock_names)

    B, T, S, F = 2, 5, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 0, :] = torch.randn(B, T, F)   # perturb ONLY the ACB (index 0, NEWS_ON) column

    with torch.no_grad():
        pred_a = model(x_har, adj, x_news_a)
        pred_b = model(x_har, adj, x_news_b)

    assert not torch.equal(pred_a[:, 0], pred_b[:, 0]), "NEWS_ON stock's prediction should change with different news"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
