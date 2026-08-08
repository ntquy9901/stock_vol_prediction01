"""Behavior contracts for pooled P1-P3 sequence models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import PooledSample, SampleKey  # noqa: E402
from models import GraphAblationModel, PooledPriceLSTM, PooledPriceNewsLSTM  # noqa: E402


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


def test_task3_int8_news_mask_is_compacted_as_boolean_mask() -> None:
    sample = PooledSample(
        key=SampleKey(0, "AAA", "2020-01-31"),
        x_price_raw=np.ones((22, 3)),
        x_news=np.ones((22, 2)),
        news_mask=np.array([1, 0] * 11, dtype=np.int8),
        y_raw=1.0,
    )
    model = PooledPriceNewsLSTM(3, 2, 1, use_gate=True, dropout=0.0)

    result = model(
        torch.from_numpy(sample.x_price_raw.copy()).unsqueeze(0),
        torch.from_numpy(sample.x_news.copy()).unsqueeze(0),
        torch.from_numpy(sample.news_mask.copy()).unsqueeze(0),
        torch.tensor([0]),
    )

    assert result.shape == (1,)


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


@pytest.mark.parametrize(
    "ticker_ids",
    [
        torch.tensor([True, False]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0, 1]]),
        torch.tensor([0]),
        torch.tensor([0, 2]),
    ],
)
def test_gated_model_rejects_invalid_ticker_ids(ticker_ids: torch.Tensor) -> None:
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, dropout=0.0)
    price, news, mask = _same_inputs()

    with pytest.raises(ValueError, match="ticker_ids"):
        model(price, news, mask, ticker_ids)


def test_gated_model_rejects_ticker_ids_on_a_different_device() -> None:
    model = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, dropout=0.0)
    price, news, mask = _same_inputs()

    with pytest.raises(ValueError, match="ticker_ids"):
        model(price, news, mask, torch.empty(2, dtype=torch.long, device="meta"))


def test_gate_gradients_are_isolated_by_explicit_ticker_id() -> None:
    model = _deterministic_gated_model()
    model.train()
    price, news, mask = _same_inputs()

    ticker_ids = torch.tensor([0, 1])
    targets = torch.tensor([0.0, 0.0])
    first_loss = (model(price, news, mask, ticker_ids) - targets).square().sum()
    first_loss.backward()
    first_gradient = model.gate_logits.grad.detach().clone()

    model.zero_grad()
    changed_news = news.clone()
    changed_news[1] *= 3
    changed_targets = torch.tensor([0.0, 1.0])
    changed_loss = (model(price, changed_news, mask, ticker_ids) - changed_targets).square().sum()
    changed_loss.backward()
    changed_gradient = model.gate_logits.grad.detach().clone()

    assert first_gradient[0].item() != 0.0
    assert first_gradient[1].item() != 0.0
    torch.testing.assert_close(first_gradient[0], changed_gradient[0])
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


def test_graph_pair_loads_byte_identical_frozen_p3_encoders_and_head(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p3.pt"
    torch.save({"model_state": p3.state_dict(), "graph_safe": True,
                "max_training_target_date": "2020-03-31"}, checkpoint)

    g0 = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=False)
    g1 = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=True)

    assert _state_bytes(g0.price_encoder) == _state_bytes(g1.price_encoder)
    assert _state_bytes(g0.head) == _state_bytes(g1.head)
    assert all(not parameter.requires_grad for parameter in g0.price_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in g1.news_encoder.parameters())
    assert any(parameter.requires_grad for parameter in g1.message_passing.parameters())


def test_graph_g0_has_no_message_passing_and_g1_gradients_only_touch_gnn(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p3.pt"
    torch.save({"model_state": p3.state_dict(), "graph_safe": True,
                "max_training_target_date": "2020-03-31"}, checkpoint)
    g0 = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=False)
    g1 = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=True)
    price, news, mask = _same_inputs()
    adjacency = torch.ones(2, 2)

    assert torch.isfinite(g0(price, news, mask, torch.tensor([0, 1]), adjacency)).all()
    loss = g1(price, news, mask, torch.tensor([0, 1]), adjacency).square().sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in g1.price_encoder.parameters())
    assert all(parameter.grad is None for parameter in g1.news_encoder.parameters())
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in g1.message_passing.parameters())


def test_graph_safe_checkpoint_excludes_pooled_targets_after_graph_train_boundary(tmp_path: Path) -> None:
    from data import GraphManifest, PooledManifest
    from run_pilot import build_graph_safe_p3_checkpoint

    def sample(date: str) -> PooledSample:
        return PooledSample(SampleKey(0, "AAA", date), np.ones((22, 3)), np.ones((22, 2)),
                            np.ones(22, dtype=np.int8), 1.0)
    pooled = PooledManifest({"train": (sample("2020-01-31"), sample("2020-04-30")), "val": (), "test": ()},
                            {}, {"AAA": 0}, "preprocessing")
    graph = GraphManifest((), {"AAA": 0}, "2020-03-31", "2020-04-30",
                          {"snapshots": "x", "node_vocabulary": "x", "adjacency": "x", "tensors": "x"})

    checkpoint = build_graph_safe_p3_checkpoint(pooled, graph, tmp_path, seed=42)
    payload = torch.load(checkpoint, weights_only=False)

    assert payload["graph_safe"] is True
    assert payload["max_training_target_date"] == "2020-01-31"
    assert payload["training_sample_count"] == 1
    assert (tmp_path / "graph_safe_p3_checkpoint.txt").read_text(encoding="utf-8").strip() == str(checkpoint)


def _state_bytes(module: torch.nn.Module) -> bytes:
    return b"".join(value.detach().cpu().numpy().tobytes() for value in module.state_dict().values())
