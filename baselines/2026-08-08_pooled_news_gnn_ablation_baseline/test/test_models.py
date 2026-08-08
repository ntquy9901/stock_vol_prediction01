"""Behavior contracts for pooled P1-P3 sequence models."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models import PooledPriceLSTM, PooledPriceNewsLSTM  # noqa: E402


def _price(batch_size: int = 2) -> torch.Tensor:
    return torch.arange(batch_size * 22 * 3, dtype=torch.float32).reshape(batch_size, 22, 3) / 100


def _same_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _price(), torch.ones(2, 22, 2), torch.ones(2, 22, dtype=torch.bool)


def _deterministic_gated_model() -> PooledPriceNewsLSTM:
    torch.manual_seed(7)
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        model.gate_logits.copy_(torch.tensor([-10.0, 10.0]))
        for parameter in model.parameters():
            if parameter is not model.gate_logits:
                parameter.fill_(0.1)
    model.eval()
    return model


def test_pooled_models_return_one_prediction_per_independent_sample() -> None:
    price_model = PooledPriceLSTM(3)
    news_model = PooledPriceNewsLSTM(3, 146, 33, use_gate=True)
    price = torch.randn(4, 22, 3)
    news = torch.randn(4, 22, 146)
    mask = torch.ones(4, 22, dtype=torch.bool)
    ticker_ids = torch.tensor([0, 5, 5, 32])

    assert price_model(price).shape == (4,)
    assert news_model(price, news, mask, ticker_ids).shape == (4,)


def test_gate_selection_uses_ticker_id_not_batch_position() -> None:
    model = _deterministic_gated_model()
    price, news, mask = _same_inputs()

    first = model(price, news, mask, ticker_ids=torch.tensor([1, 0]))
    second = model(price, news, mask, ticker_ids=torch.tensor([0, 1]))

    assert not torch.equal(first, second)


def test_all_missing_news_is_finite_and_input_independent() -> None:
    model = PooledPriceNewsLSTM(3, 4, 2, use_gate=False)
    model.eval()
    mask = torch.zeros(2, 22, dtype=torch.bool)

    first = model(_price(), torch.randn(2, 22, 4), mask, torch.tensor([0, 1]))
    second = model(_price(), torch.randn(2, 22, 4), mask, torch.tensor([0, 1]))

    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_news_encoder_compacts_internal_and_trailing_missing_timesteps() -> None:
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=False, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    model.eval()
    price = _price(1)
    valid_news = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    compact = torch.zeros(1, 22, 2)
    compact[:, :2] = valid_news
    sparse = torch.randn(1, 22, 2)
    sparse[:, 0] = valid_news[:, 0]
    sparse[:, 21] = valid_news[:, 1]

    compact_output = model(price, compact, torch.tensor([[True, True] + [False] * 20]), torch.tensor([0]))
    sparse_output = model(
        price, sparse, torch.tensor([[True] + [False] * 20 + [True]]), torch.tensor([1])
    )

    torch.testing.assert_close(compact_output, sparse_output)


def test_p2_does_not_use_ticker_ids_as_a_news_gate() -> None:
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=False, dropout=0.0)
    model.eval()
    price, news, mask = _same_inputs()

    first = model(price, news, mask, torch.tensor([0, 1]))
    second = model(price, news, mask, torch.tensor([1, 0]))

    assert model.gate_logits is None
    torch.testing.assert_close(first, second)


def test_gate_gradients_are_isolated_by_explicit_ticker_id() -> None:
    model = _deterministic_gated_model()
    model.train()
    price, news, mask = _same_inputs()

    first_loss = model(price[:1], news[:1], mask[:1], torch.tensor([1])).square().sum()
    first_loss.backward()
    first_gradient = model.gate_logits.grad.detach().clone()

    model.zero_grad()
    changed_loss = model(price[:1], news[:1] * 3, mask[:1], torch.tensor([1])).square().sum()
    changed_loss.backward()
    changed_gradient = model.gate_logits.grad.detach().clone()

    assert first_gradient[0].item() == 0.0
    assert changed_gradient[0].item() == 0.0
    assert first_gradient[1].item() != changed_gradient[1].item()


def test_forward_has_no_hidden_state_from_preceding_batch() -> None:
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, dropout=0.0)
    model.eval()
    price, news, mask = _same_inputs()
    ticker_ids = torch.tensor([0, 1])

    expected = model(price, news, mask, ticker_ids)
    model(torch.randn(3, 22, 3), torch.randn(3, 22, 2), torch.ones(3, 22, dtype=torch.bool), torch.tensor([1, 0, 1]))
    actual = model(price, news, mask, ticker_ids)

    torch.testing.assert_close(actual, expected)
