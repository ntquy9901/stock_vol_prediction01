"""Test the core claim of this baseline: only the 11 ablation-derived tickers get news; every
other ticker gets an EXACTLY zero news contribution.

Run: pytest baselines/2026-07-25_ablation_derived_gate_baseline/test/test_mask_correctness.py -v
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
from model_ablation_gate import NEWS_ON_TICKERS, AblationDerivedGateBaseline, build_stock_mask

pytestmark = pytest.mark.smoke


def test_eleven_tickers_are_on():
    assert NEWS_ON_TICKERS == {
        "HDB", "HPG", "MWG", "NVL", "PDR", "PLX", "SSI", "VHM", "VJC", "VPB", "VRE",
    }


def test_build_stock_mask_matches_ablation_list():
    stock_names = ["HDB", "ACB", "MWG", "SHB"]   # HDB/MWG=ON, ACB/SHB=OFF
    mask = build_stock_mask(stock_names)
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_off_stock_has_zero_news_contribution():
    stock_names = ["HDB", "ACB"]   # HDB=ON, ACB=OFF (per ablation, unlike the EDA-based baselines)
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = AblationDerivedGateBaseline(config, n_feat=8, stock_names=stock_names,
                                        d_news=8, dropout=0.0).eval()
    assert model.stock_mask.tolist() == [1.0, 0.0]

    B, T, S, F = 2, 4, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 1, :] = torch.randn(B, T, F)   # perturb ACB's (index 1, OFF) news only

    with torch.no_grad():
        pred_a = model(x_har, adj, x_news_a)
        pred_b = model(x_har, adj, x_news_b)

    assert torch.equal(pred_a[:, 1], pred_b[:, 1])


def test_on_stock_is_affected_by_its_own_news():
    stock_names = ["HDB", "ACB"]
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = AblationDerivedGateBaseline(config, n_feat=8, stock_names=stock_names,
                                        d_news=8, dropout=0.0).eval()

    B, T, S, F = 2, 4, 2, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news_a = torch.randn(B, T, S, F)
    x_news_b = x_news_a.clone()
    x_news_b[:, :, 0, :] = torch.randn(B, T, F)   # perturb HDB's (index 0, ON) news only

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
