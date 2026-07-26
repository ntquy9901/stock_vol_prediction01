"""Tests for PerTickerGatedNewsBaseline (design.md §1) — including the CORE claim this baseline
exists to test: gate_logits[i]'s gradient is isolated to ticker i's own prediction error, unlike
gated_crossattn's shared gate_mlp.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import pytest

from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_per_ticker_gate import PerTickerGatedNewsBaseline

pytestmark = pytest.mark.smoke


def _make_model(num_stocks=4, n_feat=16, d_news=8, seed=0):
    torch.manual_seed(seed)
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = PerTickerGatedNewsBaseline(config, n_feat=n_feat, num_stocks=num_stocks,
                                       d_news=d_news, dropout=0.0)
    return model


def test_forward_shape_and_no_nan():
    model = _make_model()
    model.train()
    B, T, S, n_feat = 2, 5, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.randn(B, T, S, n_feat)
    pred = model(x_har, adj, x_news)
    assert pred.shape == (B, S)
    assert not torch.isnan(pred).any()


def test_gate_init_is_neutral_point_five():
    model = _make_model(num_stocks=6)
    gates = model.gate_values()
    assert gates.shape == (6,)
    assert torch.allclose(gates, torch.full((6,), 0.5), atol=1e-6)


def test_gate_values_in_open_unit_interval():
    model = _make_model()
    with torch.no_grad():
        model.gate_logits.copy_(torch.tensor([-10.0, 0.0, 10.0, 3.0]))
    gates = model.gate_values()
    assert torch.all(gates > 0.0) and torch.all(gates < 1.0)
    assert gates[0] < 0.01   # very negative logit -> gate near 0
    assert gates[2] > 0.99   # very positive logit -> gate near 1


def test_gate_gradient_isolated_per_ticker():
    """THE core property test for this baseline (design.md §1 claim). Two forward+backward
    passes with IDENTICAL inputs and IDENTICAL targets except stock j's target -- gate_logits[i]
    (i != j)'s gradient must be UNCHANGED between the two runs. This is exactly the property
    that gated_crossattn's shared gate_mlp does NOT have (its shared weights receive gradient
    contributions from every ticker's loss every step)."""
    num_stocks = 5
    i, j = 1, 3   # arbitrary distinct receiver/perturbed indices
    B, T, n_feat, d_news = 3, 6, 10, 8

    torch.manual_seed(0)
    x_har = torch.randn(B, T, num_stocks, 3)
    adj = torch.rand(B, num_stocks, num_stocks)
    x_news = torch.randn(B, T, num_stocks, n_feat)
    y_base = torch.randn(B, num_stocks)

    def _grad_at(y):
        model = _make_model(num_stocks=num_stocks, n_feat=n_feat, d_news=d_news, seed=42)
        pred = model(x_har, adj, x_news)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        return model.gate_logits.grad[i].item()

    y_a = y_base.clone()
    y_b = y_base.clone()
    y_b[:, j] = y_b[:, j] + 100.0   # large perturbation on a DIFFERENT ticker's target only

    grad_a = _grad_at(y_a)
    grad_b = _grad_at(y_b)

    assert grad_a == pytest.approx(grad_b, abs=1e-6), (
        f"gate_logits[{i}]'s gradient changed ({grad_a} -> {grad_b}) when only ticker {j}'s "
        "target changed -- gradient is NOT isolated per ticker (violates design.md §1's core claim)")


def test_gate_gradient_isolated_per_ticker_also_holds_for_feature_perturbation():
    """Same isolation property, but perturbing ticker j's NEWS FEATURES (not just its target) --
    gate_logits[i]'s gradient (w.r.t. a fixed target) must still be unaffected."""
    num_stocks = 5
    i, j = 0, 2
    B, T, n_feat, d_news = 2, 6, 10, 8

    torch.manual_seed(1)
    x_har = torch.randn(B, T, num_stocks, 3)
    adj = torch.rand(B, num_stocks, num_stocks)
    x_news_base = torch.randn(B, T, num_stocks, n_feat)
    y = torch.randn(B, num_stocks)

    def _grad_at(x_news):
        model = _make_model(num_stocks=num_stocks, n_feat=n_feat, d_news=d_news, seed=7)
        pred = model(x_har, adj, x_news)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        return model.gate_logits.grad[i].item()

    x_news_a = x_news_base.clone()
    x_news_b = x_news_base.clone()
    x_news_b[:, :, j, :] = x_news_b[:, :, j, :] + 50.0

    assert _grad_at(x_news_a) == pytest.approx(_grad_at(x_news_b), abs=1e-6)


def test_gate_gradient_does_change_for_the_affected_ticker():
    """Sanity counterpart to the isolation tests above: gate_logits[j]'s OWN gradient DOES
    change when ticker j's target changes -- proves the perturbation used above is meaningful
    (i.e. the isolation tests aren't passing trivially because nothing propagates gradient at
    all)."""
    num_stocks = 5
    j = 3
    B, T, n_feat, d_news = 3, 6, 10, 8

    torch.manual_seed(0)
    x_har = torch.randn(B, T, num_stocks, 3)
    adj = torch.rand(B, num_stocks, num_stocks)
    x_news = torch.randn(B, T, num_stocks, n_feat)
    y_base = torch.randn(B, num_stocks)

    def _grad_at(y):
        model = _make_model(num_stocks=num_stocks, n_feat=n_feat, d_news=d_news, seed=42)
        pred = model(x_har, adj, x_news)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        return model.gate_logits.grad[j].item()

    y_a = y_base.clone()
    y_b = y_base.clone()
    y_b[:, j] = y_b[:, j] + 100.0

    assert _grad_at(y_a) != pytest.approx(_grad_at(y_b), abs=1e-6)


def test_all_zero_news_does_not_break_forward():
    model = _make_model()
    model.eval()
    B, T, S, n_feat = 1, 4, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.zeros(B, T, S, n_feat)
    pred = model(x_har, adj, x_news)
    assert not torch.isnan(pred).any()
