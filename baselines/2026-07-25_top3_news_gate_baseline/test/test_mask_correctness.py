"""Test the core claim of this baseline: only VIB/ACB/MWG get news; every other ticker gets an
EXACTLY zero news contribution.

Run: pytest baselines/2026-07-25_top3_news_gate_baseline/test/test_mask_correctness.py -v
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
from model_top3_gate import NEWS_ON_TICKERS, Top3NewsGateBaseline, build_stock_mask

pytestmark = pytest.mark.smoke


def test_only_three_tickers_are_on():
    assert NEWS_ON_TICKERS == {"VIB", "ACB", "MWG"}


def test_build_stock_mask_is_strict_allowlist():
    stock_names = ["ACB", "SHB", "GAS", "VIB", "MWG", "VPB"]
    mask = build_stock_mask(stock_names)
    assert mask.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_non_top3_stock_has_zero_news_contribution():
    stock_names = ["ACB", "SHB"]   # ACB=ON, SHB=OFF (not in top-3)
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = Top3NewsGateBaseline(config, n_feat=8, stock_names=stock_names, d_news=8, dropout=0.0).eval()
    assert model.stock_mask.tolist() == [1.0, 0.0]

    B, T, S, F = 2, 4, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 1, :] = torch.randn(B, T, F)   # perturb SHB's (index 1) news only

    with torch.no_grad():
        pred_a = model(x_har, adj, x_news_a)
        pred_b = model(x_har, adj, x_news_b)

    assert torch.equal(pred_a[:, 1], pred_b[:, 1]), "non-top-3 stock's prediction changed with different news"


def test_top3_stock_is_affected_by_its_own_news():
    stock_names = ["ACB", "SHB"]
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = Top3NewsGateBaseline(config, n_feat=8, stock_names=stock_names, d_news=8, dropout=0.0).eval()

    B, T, S, F = 2, 4, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 0, :] = torch.randn(B, T, F)   # perturb ACB's (index 0, top-3) news only

    with torch.no_grad():
        pred_a = model(x_har, adj, x_news_a)
        pred_b = model(x_har, adj, x_news_b)

    assert not torch.equal(pred_a[:, 0], pred_b[:, 0])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
