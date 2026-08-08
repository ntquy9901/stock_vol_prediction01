"""Behavior contracts for pooled P1-P3 sequence models."""

from __future__ import annotations

import sys
import json
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
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


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
    torch.save(_graph_safe_payload(p3), checkpoint)

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
    torch.save(_graph_safe_payload(p3), checkpoint)
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


def test_graph_batched_forward_matches_independent_snapshots(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p3.pt"
    torch.save(_graph_safe_payload(p3), checkpoint)
    model = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=True).eval()
    price, news, mask = _same_inputs()
    ticker_ids = torch.tensor([0, 1])
    adjacency = torch.ones(2, 2)
    with torch.no_grad():
        expected = torch.stack([model(price, news, mask, ticker_ids, adjacency) for _ in range(2)])
        actual = model(price.unsqueeze(0).repeat(2, 1, 1, 1),
                      news.unsqueeze(0).repeat(2, 1, 1, 1),
                      mask.unsqueeze(0).repeat(2, 1, 1),
                      ticker_ids.unsqueeze(0).repeat(2, 1),
                      adjacency.unsqueeze(0).repeat(2, 1, 1))
    torch.testing.assert_close(actual, expected)


def test_batched_validation_loss_weights_non_divisible_batches_equally() -> None:
    from run_pilot import _mean_snapshot_mse

    predictions = torch.zeros(3, 2)
    targets = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    first_batch = _mean_snapshot_mse(predictions[:2], targets[:2])
    second_batch = _mean_snapshot_mse(predictions[2:], targets[2:])
    weighted = (first_batch * 2 + second_batch) / 3

    torch.testing.assert_close(_mean_snapshot_mse(predictions, targets), weighted)


def test_graph_safe_checkpoint_excludes_pooled_targets_after_graph_train_boundary(tmp_path: Path) -> None:
    from data import GraphManifest, PooledManifest
    from run_pilot import _canonical_sample_hash, build_graph_safe_p3_checkpoint

    def sample(date: str) -> PooledSample:
        return PooledSample(SampleKey(0, "AAA", date), np.ones((22, 3)), np.ones((22, 2)),
                            np.ones(22, dtype=np.int8), 1.0)
    pooled = PooledManifest({"train": (sample("2020-01-31"), sample("2020-04-30")), "val": (), "test": ()},
                            {}, {"AAA": 0}, "preprocessing")
    graph = GraphManifest((), {"AAA": 0}, "2020-03-31", "2020-04-30",
                          {"snapshots": "x", "node_vocabulary": "x", "adjacency": "x", "tensors": "x"})

    warm_start = tmp_path / "p3.pt"
    p3 = PooledPriceNewsLSTM(3, 2, 1, use_gate=True, dropout=0.0)
    torch.save({"config_name": "P3", "model_state": p3.state_dict(), "graph_bound_warm_start": True,
                "max_training_target_date": "2020-01-31", "graph_train_end_date": graph.train_end_date,
                "training_sample_hash": _canonical_sample_hash((pooled.samples["train"][0],)),
                "graph_manifest_hash": graph.content_hash("train")}, warm_start)
    preprocessor = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([1.0]), np.array([1.0])),
    )
    checkpoint = build_graph_safe_p3_checkpoint(pooled, graph, tmp_path, seed=42,
                                                 warm_start_checkpoint=warm_start,
                                                 store=PreprocessorStore({0: preprocessor}))
    payload = torch.load(checkpoint, weights_only=False)

    assert payload["graph_safe"] is True
    assert payload["max_training_target_date"] == "2020-01-31"
    assert payload["training_sample_count"] == 1
    assert (tmp_path / "graph_safe_p3_checkpoint.txt").read_text(encoding="utf-8").strip() == str(checkpoint)


def test_unrestricted_task6_style_p3_checkpoint_is_rejected_not_relabelled(tmp_path: Path) -> None:
    from data import GraphManifest, PooledManifest
    from run_pilot import build_graph_safe_p3_checkpoint

    sample = PooledSample(SampleKey(0, "AAA", "2020-01-31"), np.ones((22, 3)), np.ones((22, 2)),
                          np.ones(22, dtype=np.int8), 1.0)
    pooled = PooledManifest({"train": (sample,), "val": (), "test": ()}, {}, {"AAA": 0}, "preprocessing")
    graph = GraphManifest((), {"AAA": 0}, "2020-03-31", "2020-04-30",
                          {"snapshots": "x", "node_vocabulary": "x", "adjacency": "x", "tensors": "x"})
    p3 = PooledPriceNewsLSTM(3, 2, 1, use_gate=True, dropout=0.0)
    task6_best = tmp_path / "best.pt"
    torch.save({"config_name": "P3", "seed": 42, "model_state": p3.state_dict()}, task6_best)
    processor = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([1.0]), np.array([1.0])),
    )

    with pytest.raises(ValueError, match="verified graph-bound"):
        build_graph_safe_p3_checkpoint(pooled, graph, tmp_path, 42, task6_best,
                                       PreprocessorStore({0: processor}))
    assert not (tmp_path / "graph_bound_p3_warm_start.pt").exists()


def test_fresh_graph_bound_p3_producer_attests_restricted_samples(tmp_path: Path) -> None:
    from data import GraphManifest, PooledManifest
    from run_pilot import build_graph_bound_p3_warm_start

    sample = PooledSample(SampleKey(0, "AAA", "2020-01-31"), np.ones((22, 3)), np.ones((22, 2)),
                          np.ones(22, dtype=np.int8), 1.0)
    pooled = PooledManifest({"train": (sample,), "val": (), "test": ()}, {}, {"AAA": 0}, "preprocessing")
    graph = GraphManifest((), {"AAA": 0}, "2020-03-31", "2020-04-30",
                          {"snapshots": "x", "node_vocabulary": "x", "adjacency": "x", "tensors": "x"})
    processor = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([1.0]), np.array([1.0])),
    )

    path = build_graph_bound_p3_warm_start(pooled, graph, tmp_path, 42, PreprocessorStore({0: processor}))

    assert torch.load(path, weights_only=False)["graph_bound_warm_start"] is True


def test_graph_cli_parser_and_one_batch_runner_emit_paired_artifact(tmp_path: Path) -> None:
    from data import GraphManifest, GraphNode, GraphSnapshot
    from run_pilot import _run_one_graph_model, parse_args

    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    def snapshot(split: str, date: str) -> GraphSnapshot:
        return GraphSnapshot(
            date, split, tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
            (GraphNode(0, "AAA", split, 1.0), GraphNode(1, "BBB", split, 1.1)),
            np.ones((2, 22, 3)), np.ones((2, 22, 2)), np.ones((2, 22)), np.eye(2),
        )
    graph = GraphManifest((snapshot("train", "2020-01-27"), snapshot("val", "2020-02-27"),
                           snapshot("val", "2020-02-28")),
                          {"AAA": 0, "BBB": 1}, "2020-01-31", "2020-02-28",
                          {"snapshots": "s", "node_vocabulary": "n", "adjacency": "a", "tensors": "t"})
    checkpoint = tmp_path / "safe.pt"
    payload = _graph_safe_payload(p3)
    payload["max_training_target_date"] = graph.train_end_date
    payload["graph_train_end_date"] = graph.train_end_date
    payload["graph_manifest_hash"] = graph.content_hash("train")
    torch.save(payload, checkpoint)
    args = parse_args(["--phase", "graph", "--p3-checkpoint", str(checkpoint), "--epochs", "1"])
    model = GraphAblationModel.from_p3_checkpoint(
        checkpoint, True, graph.train_end_date, graph.content_hash("train"),
    )

    preprocessor = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([1.0]), np.array([1.0])),
    )
    result = _run_one_graph_model(model, graph, PreprocessorStore({0: preprocessor, 1: preprocessor}),
                                  "G1", args.epochs, args.seed, tmp_path / "G1")

    assert args.phase == "graph"
    assert result["graph_hash"] == graph.content_hash("val")
    assert (tmp_path / "G1" / "results.json").exists()


def test_graph_device_cpu_path_records_runtime_metadata(tmp_path: Path) -> None:
    from data import GraphManifest, GraphNode, GraphSnapshot
    from run_pilot import _run_one_graph_model, parse_args

    p3 = PooledPriceNewsLSTM(3, 2, 1, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    snapshot = GraphSnapshot(
        "2020-01-27", "train", tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
        (GraphNode(0, "AAA", "train", 1.0),), np.ones((1, 22, 3)), np.ones((1, 22, 2)),
        np.ones((1, 22)), np.ones((1, 1)),
    )
    validation = GraphSnapshot(
        "2020-02-27", "val", snapshot.input_dates, (GraphNode(0, "AAA", "val", 1.1),),
        np.ones((1, 22, 3)), np.ones((1, 22, 2)), np.ones((1, 22)), np.ones((1, 1)),
    )
    validation_next = GraphSnapshot(
        "2020-02-28", "val", snapshot.input_dates, (GraphNode(0, "AAA", "val", 1.2),),
        np.ones((1, 22, 3)), np.ones((1, 22, 2)), np.ones((1, 22)), np.ones((1, 1)),
    )
    graph = GraphManifest((snapshot, validation, validation_next), {"AAA": 0}, "2020-01-31", "2020-02-28",
                          {"snapshots": "s", "node_vocabulary": "n", "adjacency": "a", "tensors": "t"})
    checkpoint = tmp_path / "safe.pt"
    payload = _graph_safe_payload(p3)
    payload["max_training_target_date"] = graph.train_end_date
    payload["graph_train_end_date"] = graph.train_end_date
    payload["graph_manifest_hash"] = graph.content_hash("train")
    torch.save(payload, checkpoint)
    model = GraphAblationModel.from_p3_checkpoint(
        checkpoint, True, graph.train_end_date, graph.content_hash("train"),
    )
    processor = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([1.0]), np.array([1.0])),
    )
    args = parse_args(["--phase", "graph", "--epochs", "1", "--device", "cpu"])

    _run_one_graph_model(model, graph, PreprocessorStore({0: processor}), "G1", args.epochs,
                         args.seed, tmp_path / "G1", args.device)

    metadata = json.loads((tmp_path / "G1" / "results.json").read_text(encoding="utf-8"))["runtime"]
    assert args.device == "cpu"
    assert metadata["selected"] == "cpu"
    assert metadata["torch_version"] == torch.__version__


def test_graph_device_rejects_explicit_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from run_pilot import resolve_graph_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_graph_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_graph_device("cuda")


def test_graph_runner_rejects_provenance_before_model_device_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from data import GraphManifest
    from run_pilot import _run_one_graph_model

    model = GraphAblationModel(PooledPriceNewsLSTM(3, 2, 1, use_gate=True, dropout=0.0), use_gnn=True)
    model.graph_train_end_date = "2020-01-31"
    model.graph_manifest_hash = "wrong"
    graph = GraphManifest((), {"AAA": 0}, "2020-01-31", "2020-02-28",
                          {"snapshots": "s", "node_vocabulary": "n", "adjacency": "a", "tensors": "t"})
    transfers: list[torch.device] = []
    monkeypatch.setattr(model, "to", lambda device: transfers.append(device))

    with pytest.raises(ValueError, match="manifest hash"):
        _run_one_graph_model(model, graph, PreprocessorStore({}), "G1", 1, 42, tmp_path, "cpu")

    assert not transfers


def test_message_passing_masks_non_neighbors_for_an_isolated_node() -> None:
    from models import _ResidualMessagePassing

    layer = _ResidualMessagePassing(1)
    with torch.no_grad():
        layer.projection.weight.fill_(1.0)
    output = layer(torch.tensor([[[2.0], [100.0]]]), torch.eye(2))

    torch.testing.assert_close(output, torch.tensor([[[2.0], [100.0]]]))


def _target_preprocessor(mean: float, std: float) -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)),
        ArrayStandardizer(np.array([mean]), np.array([std])),
    )


def _negative_prone_graph_model(use_gnn: bool) -> GraphAblationModel:
    """A graph model whose trainable head emits a strongly negative prediction."""

    torch.manual_seed(0)
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    model = GraphAblationModel(p3, use_gnn=use_gnn)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(-50.0)
    return model.eval()


def test_graph_positivity_floor_forces_strictly_positive_denormalized_predictions() -> None:
    model = _negative_prone_graph_model(use_gnn=True)
    store = PreprocessorStore({0: _target_preprocessor(0.5, 0.1), 1: _target_preprocessor(0.4, 0.2)})
    price, news, mask = _same_inputs()
    ticker_ids = torch.tensor([0, 1])
    adjacency = torch.ones(2, 2)

    raw_without_floor = store.inverse_targets(
        ticker_ids.numpy(), model(price, news, mask, ticker_ids, adjacency).detach().numpy(),
    )
    assert (raw_without_floor <= 0.0).any()

    model.configure_positivity(store)
    raw_with_floor = store.inverse_targets(
        ticker_ids.numpy(), model(price, news, mask, ticker_ids, adjacency).detach().numpy(),
    )
    assert (raw_with_floor > 0.0).all()


def test_graph_positivity_floor_preserves_prediction_spread_no_collapse() -> None:
    torch.manual_seed(1)
    p3 = PooledPriceNewsLSTM(3, 2, 3, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for parameter in p3.parameters():
            parameter.uniform_(-0.5, 0.5)
    model = GraphAblationModel(p3, use_gnn=True).eval()
    store = PreprocessorStore({index: _target_preprocessor(0.02 + 0.01 * index, 0.01) for index in range(3)})
    price = torch.randn(3, 22, 3)
    news = torch.randn(3, 22, 2)
    mask = torch.ones(3, 22, dtype=torch.bool)
    ticker_ids = torch.tensor([0, 1, 2])
    adjacency = torch.ones(3, 3)

    before = model(price, news, mask, ticker_ids, adjacency).detach()
    model.configure_positivity(store)
    after = model(price, news, mask, ticker_ids, adjacency).detach()

    raw_after = store.inverse_targets(ticker_ids.numpy(), after.numpy())
    assert (raw_after > 0.0).all()
    # The bulk of positive-denormalizing predictions must survive unchanged, so
    # the spread cannot collapse toward a single value (the Softplus-collapse mode).
    assert float(after.std()) > 0.5 * float(before.std())
    assert float(after.std()) > 1e-3


def test_configure_positivity_does_not_mutate_scaler_or_provenance(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p3.pt"
    torch.save(_graph_safe_payload(p3), checkpoint)
    model = GraphAblationModel.from_p3_checkpoint(
        checkpoint, use_gnn=True, graph_train_end_date="2020-03-31", graph_manifest_hash="manifest",
    )
    store = PreprocessorStore({0: _target_preprocessor(0.5, 0.1), 1: _target_preprocessor(0.4, 0.2)})
    scaler_snapshot = store.to_dict()
    boundary, manifest_hash = model.graph_train_end_date, model.graph_manifest_hash

    model.configure_positivity(store)

    assert store.to_dict() == scaler_snapshot
    assert model.graph_train_end_date == boundary
    assert model.graph_manifest_hash == manifest_hash
    assert all(not parameter.requires_grad for parameter in model.price_encoder.parameters())


def test_positivity_buffers_do_not_break_p3_checkpoint_loading(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "graph_safe_p3.pt"
    torch.save(_graph_safe_payload(p3), checkpoint)

    state = torch.load(checkpoint, weights_only=False)["model_state"]
    assert "target_mean" not in state and "target_std" not in state

    model = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=False)
    reference = GraphAblationModel.from_p3_checkpoint(checkpoint, use_gnn=False)
    price, news, mask = _same_inputs()
    ticker_ids = torch.tensor([0, 1])
    adjacency = torch.ones(2, 2)

    # An unconfigured model applies no floor: forward is backward compatible.
    torch.testing.assert_close(
        model(price, news, mask, ticker_ids, adjacency),
        reference(price, news, mask, ticker_ids, adjacency),
    )


def test_run_one_graph_model_positivity_gate_passes_for_negative_prone_head(tmp_path: Path) -> None:
    from data import GraphManifest, GraphNode, GraphSnapshot
    from run_pilot import _run_one_graph_model

    p3 = PooledPriceNewsLSTM(3, 2, 1, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for parameter in p3.head.parameters():
            parameter.zero_()
        p3.head[-1].bias.fill_(-50.0)
    train = GraphSnapshot(
        "2020-01-27", "train", tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
        (GraphNode(0, "AAA", "train", 1.0),), np.ones((1, 22, 3)), np.ones((1, 22, 2)),
        np.ones((1, 22)), np.ones((1, 1)),
    )
    val = GraphSnapshot(
        "2020-02-27", "val", train.input_dates, (GraphNode(0, "AAA", "val", 1.1),),
        np.ones((1, 22, 3)), np.ones((1, 22, 2)), np.ones((1, 22)), np.ones((1, 1)),
    )
    val_next = GraphSnapshot(
        "2020-02-28", "val", train.input_dates, (GraphNode(0, "AAA", "val", 1.2),),
        np.ones((1, 22, 3)), np.ones((1, 22, 2)), np.ones((1, 22)), np.ones((1, 1)),
    )
    graph = GraphManifest((train, val, val_next), {"AAA": 0}, "2020-01-31", "2020-02-28",
                          {"snapshots": "s", "node_vocabulary": "n", "adjacency": "a", "tensors": "t"})
    checkpoint = tmp_path / "safe.pt"
    payload = _graph_safe_payload(p3)
    payload["max_training_target_date"] = graph.train_end_date
    payload["graph_train_end_date"] = graph.train_end_date
    payload["graph_manifest_hash"] = graph.content_hash("train")
    torch.save(payload, checkpoint)
    model = GraphAblationModel.from_p3_checkpoint(
        checkpoint, True, graph.train_end_date, graph.content_hash("train"),
    )

    # Pre-fix this raised "nonpositive prediction rate 100% exceeds 1%"; the positivity
    # floor now lets the safety gate pass and yields finite metrics.
    result = _run_one_graph_model(model, graph, PreprocessorStore({0: _target_preprocessor(1e-3, 1e-3)}),
                                  "G1", 1, 42, tmp_path / "G1", "cpu")

    metrics = result["validation_metrics"]
    assert all(np.isfinite(value) for value in metrics.values())


def _state_bytes(module: torch.nn.Module) -> bytes:
    return b"".join(value.detach().cpu().numpy().tobytes() for value in module.state_dict().values())


def _graph_safe_payload(p3: PooledPriceNewsLSTM) -> dict[str, object]:
    return {"model_state": p3.state_dict(), "graph_safe": True, "training_sample_hash": "samples",
            "max_training_target_date": "2020-03-31", "graph_train_end_date": "2020-03-31",
            "graph_manifest_hash": "manifest"}
