"""Behavior contracts for the price-only (P1 backbone) graph ablation model.

``PriceGraphAblationModel`` = 'GAT on the price-only P1 backbone' = P1 + graph: the node
embedding is the frozen price LSTM hidden state alone (no news encoder, no per-ticker gate),
combined with the same k-NN message passing + positivity floor as ``GraphAblationModel``.
"""

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

from models import PooledPriceLSTM, PriceGraphAblationModel  # noqa: E402
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


def _target_preprocessor(mean: float, std: float) -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)),
        ArrayStandardizer(np.array([mean]), np.array([std])),
    )


def _price(batch_size: int = 2) -> torch.Tensor:
    return torch.arange(batch_size * 22 * 3, dtype=torch.float32).reshape(batch_size, 22, 3) / 100


def _random_price_model(use_gnn: bool, num_tickers: int = 2) -> PriceGraphAblationModel:
    torch.manual_seed(0)
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for parameter in p1.parameters():
            parameter.uniform_(-0.5, 0.5)
    model = PriceGraphAblationModel(p1, num_tickers=num_tickers, use_gnn=use_gnn)
    if use_gnn:
        with torch.no_grad():
            for parameter in model.message_passing.parameters():
                parameter.uniform_(-0.5, 0.5)
    return model.eval()


def test_price_encoder_is_frozen_and_head_message_passing_are_trainable() -> None:
    model = _random_price_model(use_gnn=True)
    assert all(not parameter.requires_grad for parameter in model.price_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert all(parameter.requires_grad for parameter in model.message_passing.parameters())


def test_encode_base_uses_price_only_and_ignores_news_inputs() -> None:
    model = _random_price_model(use_gnn=True)
    price = _price().unsqueeze(0)  # [batch=1, nodes=2, time, feat]
    ticker_ids = torch.tensor([[0, 1]])
    presence = torch.ones(1, 2, dtype=torch.bool)
    news_a = torch.zeros(1, 2, 22, 2)
    news_b = torch.randn(1, 2, 22, 2)
    mask = torch.ones(1, 2, 22, dtype=torch.bool)

    base_a = model.encode_base(price, news_a, mask, ticker_ids, presence)
    base_b = model.encode_base(price, news_b, mask, ticker_ids, presence)

    assert base_a.shape == (1, 2, 4)
    # Price-only backbone: the node embedding cannot depend on any news input.
    torch.testing.assert_close(base_a, base_b, rtol=0.0, atol=0.0)


def test_absent_nodes_do_not_influence_present_node_embeddings() -> None:
    model = _random_price_model(use_gnn=True)
    price = _price().unsqueeze(0)
    ticker_ids = torch.tensor([[0, 1]])
    news = torch.zeros(1, 2, 22, 2)
    mask = torch.ones(1, 2, 22, dtype=torch.bool)

    both_present = model.encode_base(price, news, mask, ticker_ids, torch.ones(1, 2, dtype=torch.bool))
    node_zero_only = model.encode_base(
        price, news, mask, ticker_ids, torch.tensor([[True, False]]))

    # Present node 0's embedding is unaffected by node 1's presence (up to fp batch-reduction
    # noise from the LSTM's batched vs single-row encode), and absent node 1 emits an exact zero.
    torch.testing.assert_close(both_present[0, 0], node_zero_only[0, 0], rtol=0.0, atol=1e-6)
    assert torch.count_nonzero(node_zero_only[0, 1]) == 0


def test_message_passing_residual_changes_predictions() -> None:
    model = _random_price_model(use_gnn=True)
    price = _price().unsqueeze(0)
    ticker_ids = torch.tensor([[0, 1]])
    presence = torch.ones(1, 2, dtype=torch.bool)
    news = torch.zeros(1, 2, 22, 2)
    mask = torch.ones(1, 2, 22, dtype=torch.bool)
    adjacency = torch.ones(1, 2, 2)
    base = model.encode_base(price, news, mask, ticker_ids, presence)

    graph_off = model.apply_graph_head(base, adjacency, ticker_ids, presence, apply_message_passing=False)
    graph_on = model.apply_graph_head(base, adjacency, ticker_ids, presence, apply_message_passing=True)

    assert not torch.allclose(graph_on, graph_off)


def test_positivity_floor_forces_strictly_positive_denormalized_predictions() -> None:
    model = _random_price_model(use_gnn=True)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(-50.0)
    store = PreprocessorStore({0: _target_preprocessor(0.5, 0.1), 1: _target_preprocessor(0.4, 0.2)})
    price = _price().unsqueeze(0)
    ticker_ids = torch.tensor([[0, 1]])
    presence = torch.ones(1, 2, dtype=torch.bool)
    news = torch.zeros(1, 2, 22, 2)
    mask = torch.ones(1, 2, 22, dtype=torch.bool)
    adjacency = torch.ones(1, 2, 2)
    base = model.encode_base(price, news, mask, ticker_ids, presence)

    raw_without = store.inverse_targets(
        ticker_ids.reshape(-1).numpy(),
        model.apply_graph_head(base, adjacency, ticker_ids, presence).reshape(-1).detach().numpy())
    assert (raw_without <= 0.0).any()

    model.configure_positivity(store)
    raw_with = store.inverse_targets(
        ticker_ids.reshape(-1).numpy(),
        model.apply_graph_head(base, adjacency, ticker_ids, presence).reshape(-1).detach().numpy())
    assert (raw_with > 0.0).all()


def test_from_p1_checkpoint_roundtrips_weights_and_provenance(tmp_path: Path) -> None:
    torch.manual_seed(1)
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p1.pt"
    torch.save({
        "model_state": p1.state_dict(),
        "graph_safe": True,
        "max_training_target_date": "2020-03-31",
        "graph_train_end_date": "2020-03-31",
        "training_sample_hash": "hash",
        "graph_manifest_hash": "manifest",
    }, checkpoint)

    model = PriceGraphAblationModel.from_p1_checkpoint(
        checkpoint, use_gnn=True, num_tickers=2,
        graph_train_end_date="2020-03-31", graph_manifest_hash="manifest")

    assert model.graph_train_end_date == "2020-03-31"
    assert model.graph_manifest_hash == "manifest"
    # Price encoder + head weights match the checkpoint exactly. The checkpoint keys the encoder
    # as ``price_lstm.*`` (PooledPriceLSTM); the graph model exposes it as ``price_encoder.*``.
    got = model.state_dict()
    for key, value in p1.state_dict().items():
        mapped = key.replace("price_lstm.", "price_encoder.", 1) if key.startswith("price_lstm.") else key
        torch.testing.assert_close(got[mapped], value, rtol=0.0, atol=0.0)


def test_from_p1_checkpoint_rejects_non_graph_safe_source(tmp_path: Path) -> None:
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"model_state": p1.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="not graph-safe"):
        PriceGraphAblationModel.from_p1_checkpoint(checkpoint, use_gnn=True, num_tickers=2)


def test_num_tickers_must_be_positive() -> None:
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    with pytest.raises(ValueError, match="num_tickers"):
        PriceGraphAblationModel(p1, num_tickers=0, use_gnn=True)
