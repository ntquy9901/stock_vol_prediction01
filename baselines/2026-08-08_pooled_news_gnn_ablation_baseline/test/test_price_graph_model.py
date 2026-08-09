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


def _p1_safe_payload(p1: PooledPriceLSTM, **overrides: object) -> dict:
    payload = {
        "model_state": p1.state_dict(),
        "graph_safe": True,
        "max_training_target_date": "2020-03-31",
        "graph_train_end_date": "2020-03-31",
        "training_sample_hash": "hash",
        "graph_manifest_hash": "manifest",
    }
    payload.update(overrides)
    return payload


def _single_snapshot_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    price = _price()  # [nodes=2, time=22, feat=3]
    news = torch.zeros(2, 22, 2)
    mask = torch.ones(2, 22, dtype=torch.bool)
    ticker_ids = torch.tensor([0, 1])
    presence = torch.ones(2, dtype=torch.bool)
    return price, news, mask, ticker_ids, presence


def test_forward_drop_in_matches_encode_then_apply_and_ignores_news() -> None:
    model = _random_price_model(use_gnn=True)
    price, news, mask, ticker_ids, presence = _single_snapshot_inputs()
    adjacency = torch.ones(2, 2)

    forward_out = model(price, news, mask, ticker_ids, adjacency, presence)
    base = model.encode_base(price, news, mask, ticker_ids, presence)
    manual = model.apply_graph_head(base, adjacency, ticker_ids, presence)
    torch.testing.assert_close(forward_out, manual, rtol=0.0, atol=0.0)

    # News is ignored: a different news tensor yields the same forward output.
    other = model(price, torch.randn(2, 22, 2), mask, ticker_ids, adjacency, presence)
    torch.testing.assert_close(forward_out, other, rtol=0.0, atol=0.0)


def test_single_snapshot_apply_graph_head_message_passing_changes_output() -> None:
    model = _random_price_model(use_gnn=True)
    price, news, mask, ticker_ids, presence = _single_snapshot_inputs()
    adjacency = torch.ones(2, 2)
    base = model.encode_base(price, news, mask, ticker_ids, presence)  # 2-D [nodes, hidden]
    assert base.ndim == 2

    graph_off = model.apply_graph_head(base, adjacency, ticker_ids, presence, apply_message_passing=False)
    graph_on = model.apply_graph_head(base, adjacency, ticker_ids, presence, apply_message_passing=True)
    assert not torch.allclose(graph_on, graph_off)


def test_encode_base_rejects_wrong_shaped_presence() -> None:
    model = _random_price_model(use_gnn=True)
    price, news, mask, ticker_ids, _ = _single_snapshot_inputs()
    with pytest.raises(ValueError, match="1-D vector over nodes"):
        model.encode_base(price, news, mask, ticker_ids, torch.ones(2, 2, dtype=torch.bool))
    batched = price.unsqueeze(0)
    with pytest.raises(ValueError, match="batched presence_mask"):
        model.encode_base(batched, news.unsqueeze(0), mask.unsqueeze(0), ticker_ids.unsqueeze(0),
                          torch.ones(3, dtype=torch.bool))


def test_all_absent_nodes_encode_to_zero() -> None:
    model = _random_price_model(use_gnn=True)
    price, news, mask, ticker_ids, _ = _single_snapshot_inputs()
    base = model.encode_base(price, news, mask, ticker_ids, torch.zeros(2, dtype=torch.bool))
    assert torch.count_nonzero(base) == 0


def test_encode_base_without_presence_encodes_all_nodes() -> None:
    model = _random_price_model(use_gnn=True)
    price, news, mask, ticker_ids, _ = _single_snapshot_inputs()
    base = model.encode_base(price, news, mask, ticker_ids, None)
    assert base.shape == (2, 4)
    assert torch.count_nonzero(base) > 0


def test_configure_positivity_rejects_nonpositive_epsilon_and_missing_preprocessor() -> None:
    model = _random_price_model(use_gnn=True)
    store = PreprocessorStore({0: _target_preprocessor(0.5, 0.1), 1: _target_preprocessor(0.4, 0.2)})
    with pytest.raises(ValueError, match="positivity epsilon"):
        model.configure_positivity(store, epsilon=0.0)
    short_store = PreprocessorStore({0: _target_preprocessor(0.5, 0.1)})
    with pytest.raises(ValueError, match="missing preprocessor"):
        model.configure_positivity(short_store)


def test_train_mode_keeps_price_encoder_frozen_in_eval() -> None:
    model = _random_price_model(use_gnn=True)
    model.train()
    assert model.training
    assert not model.price_encoder.training


@pytest.mark.parametrize("overrides, match", [
    ({"max_training_target_date": None}, "invalid training provenance"),
    ({"max_training_target_date": "2099-01-01"}, "invalid training provenance"),
])
def test_from_p1_checkpoint_rejects_invalid_training_provenance(
    tmp_path: Path, overrides: dict, match: str) -> None:
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "p1.pt"
    torch.save(_p1_safe_payload(p1, **overrides), checkpoint)
    with pytest.raises(ValueError, match=match):
        PriceGraphAblationModel.from_p1_checkpoint(checkpoint, use_gnn=True, num_tickers=2)


def test_from_p1_checkpoint_rejects_boundary_and_manifest_mismatch(tmp_path: Path) -> None:
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "p1.pt"
    torch.save(_p1_safe_payload(p1), checkpoint)
    with pytest.raises(ValueError, match="train boundary differs"):
        PriceGraphAblationModel.from_p1_checkpoint(
            checkpoint, use_gnn=True, num_tickers=2, graph_train_end_date="2021-01-01")
    with pytest.raises(ValueError, match="manifest hash differs"):
        PriceGraphAblationModel.from_p1_checkpoint(
            checkpoint, use_gnn=True, num_tickers=2, graph_manifest_hash="other")


def test_from_p1_checkpoint_rejects_missing_or_incompatible_state(tmp_path: Path) -> None:
    p1 = PooledPriceLSTM(3, hidden_dim=4, dropout=0.0)
    no_state = tmp_path / "no_state.pt"
    torch.save(_p1_safe_payload(p1, model_state="not-a-dict"), no_state)
    with pytest.raises(ValueError, match="no model_state"):
        PriceGraphAblationModel.from_p1_checkpoint(no_state, use_gnn=True, num_tickers=2)

    incompatible = tmp_path / "incompatible.pt"
    torch.save(_p1_safe_payload(p1, model_state={"unexpected.weight": torch.zeros(1)}), incompatible)
    with pytest.raises(ValueError, match="compatible price model"):
        PriceGraphAblationModel.from_p1_checkpoint(incompatible, use_gnn=True, num_tickers=2)


def test_graph_off_model_has_no_message_passing_layer() -> None:
    model = _random_price_model(use_gnn=False)
    assert model.message_passing is None
    price, news, mask, ticker_ids, presence = _single_snapshot_inputs()
    adjacency = torch.ones(2, 2)
    base = model.encode_base(price, news, mask, ticker_ids, presence)
    # With no message-passing layer, apply_message_passing has no effect.
    out = model.apply_graph_head(base, adjacency, ticker_ids, presence, apply_message_passing=True)
    assert out.shape == (2,)
