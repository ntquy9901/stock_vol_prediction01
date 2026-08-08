"""Behavior contracts for availability-aware MASKED graph message passing.

The masked path builds one graph per trading date over only the tickers PRESENT that
day (variable node set + presence mask), so the GNN trains on the full ~4,900-date
union rather than the 26% (~1,296) synchronized-date intersection.  Absent tickers on
a date are masked, never imputed.
"""

from __future__ import annotations

import json
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

from data import (  # noqa: E402
    GraphManifest,
    PooledManifest,
    PooledSample,
    SampleKey,
    build_graph_manifest,
    build_masked_graph_manifest,
)
from models import GraphAblationModel, PooledPriceNewsLSTM, _ResidualMessagePassing  # noqa: E402
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


def _target_preprocessor(mean: float = 1e-2, std: float = 1e-2) -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)),
        ArrayStandardizer(np.array([mean]), np.array([std])),
    )


def _pooled_sample(ticker_id: int, ticker: str, date: str, value: float,
                   seq: int = 22, price_w: int = 3, news_w: int = 2) -> PooledSample:
    # A per-date-varying value gives non-degenerate correlations across present nodes.
    ramp = (value + np.arange(seq, dtype=np.float32))[:, None]
    return PooledSample(
        SampleKey(ticker_id, ticker, date),
        np.repeat(ramp, price_w, axis=1).astype(np.float32),
        np.full((seq, news_w), value, dtype=np.float32),
        np.ones(seq, dtype=np.int8),
        float(value), float(value), float(value),
        tuple(f"2020-01-{day:02d}" for day in range(1, seq + 1)),
    )


def _staggered_pooled_manifest() -> PooledManifest:
    """Ticker A on train dates 01..10, ticker B on 06..15: union 15, intersection 5."""

    def date(day: int) -> str:
        return f"2021-{(day - 1) // 28 + 1:02d}-{(day - 1) % 28 + 1:02d}"

    train = [
        _pooled_sample(0, "AAA", date(day), 0.10 + 0.01 * day) for day in range(1, 11)
    ] + [
        _pooled_sample(1, "BBB", date(day), 0.20 + 0.01 * day) for day in range(6, 16)
    ]
    val = [
        _pooled_sample(tid, ticker, date(day), 0.5 + 0.1 * tid + 0.01 * day)
        for day in (30, 31) for tid, ticker in ((0, "AAA"), (1, "BBB"))
    ]
    test = [
        _pooled_sample(tid, ticker, date(day), 0.5 + 0.1 * tid + 0.01 * day)
        for day in (40, 41) for tid, ticker in ((0, "AAA"), (1, "BBB"))
    ]
    return PooledManifest(
        {"train": tuple(train), "val": tuple(val), "test": tuple(test)},
        {}, {"AAA": 0, "BBB": 1}, "preprocessing",
    )


def _store_for(manifest: PooledManifest) -> PreprocessorStore:
    return PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})


# --- (a) absent-node perturbation invariance -------------------------------------


def test_masked_message_passing_ignores_absent_node_features() -> None:
    layer = _ResidualMessagePassing(1)
    with torch.no_grad():
        layer.projection.weight.fill_(1.0)
    node_features = torch.tensor([[[2.0], [3.0], [99.0]]])
    adjacency = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    presence = torch.tensor([[True, True, False]])

    baseline = layer(node_features, adjacency, presence)
    perturbed_features = node_features.clone()
    perturbed_features[0, 2, 0] = -1234.0
    perturbed = layer(perturbed_features, adjacency, presence)

    torch.testing.assert_close(baseline[:, :2], perturbed[:, :2])


def test_graph_model_present_output_is_independent_of_absent_node(tmp_path: Path) -> None:
    p3 = PooledPriceNewsLSTM(3, 2, 3, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    model = GraphAblationModel(p3, use_gnn=True).eval()
    price = torch.randn(3, 22, 3)
    news = torch.randn(3, 22, 2)
    mask = torch.ones(3, 22, dtype=torch.bool)
    ticker_ids = torch.tensor([0, 1, 2])
    adjacency = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    presence = torch.tensor([True, True, False])

    with torch.no_grad():
        baseline = model(price, news, mask, ticker_ids, adjacency, presence)
        perturbed_price = price.clone()
        perturbed_price[2] += 500.0
        perturbed = model(perturbed_price, news, mask, ticker_ids, adjacency, presence)

    torch.testing.assert_close(baseline[:2], perturbed[:2])


# --- (b) variable node count per snapshot ----------------------------------------


def test_masked_snapshots_have_variable_present_node_counts() -> None:
    manifest = _staggered_pooled_manifest()
    graph = build_masked_graph_manifest(manifest, _store_for(manifest))

    train = [snapshot for snapshot in graph.snapshots if snapshot.split == "train"]
    present_counts = {int(snapshot.presence_mask.sum()) for snapshot in train}
    assert present_counts == {1, 2}
    # Every snapshot pads to the full 2-node vocabulary but masks absentees.
    assert all(snapshot.presence_mask.shape == (2,) for snapshot in train)
    assert all(len(snapshot.nodes) == 2 for snapshot in train)


# --- (c) loss / metrics aggregate over present nodes only ------------------------


def test_mean_snapshot_mse_aggregates_over_present_nodes_only() -> None:
    from run_pilot import _mean_snapshot_mse

    predictions = torch.tensor([[0.0, 5.0]])
    targets = torch.tensor([[0.0, 0.0]])
    presence = torch.tensor([[True, False]])

    torch.testing.assert_close(_mean_snapshot_mse(predictions, targets, presence),
                               torch.tensor(0.0))
    torch.testing.assert_close(_mean_snapshot_mse(predictions, targets),
                               torch.tensor(12.5))


def test_masked_run_evaluates_only_present_nodes(tmp_path: Path) -> None:
    from run_pilot import _run_one_graph_model

    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = _masked_graph_safe_checkpoint(graph, tmp_path)
    model = GraphAblationModel.from_p3_checkpoint(
        checkpoint, True, graph.train_end_date, graph.content_hash("train"),
    )

    result = _run_one_graph_model(model, graph, store, "G1", epochs=2, seed=42,
                                  output=tmp_path / "G1", device="cpu", train_batch_size=4)

    payload = json.loads((tmp_path / "G1" / "results.json").read_text(encoding="utf-8"))
    assert all(np.isfinite(value) for value in result["validation_metrics"].values())
    assert payload["nonpositive_prediction_rate"] <= 0.01
    # Validation snapshots have exactly 2 present nodes each (2 val dates x AAA/BBB).
    assert payload["present_validation_node_count"] == 4


# --- (d) masked manifest uses far MORE dates than the intersection ---------------


def test_masked_manifest_recovers_union_dates_beyond_intersection() -> None:
    manifest = _staggered_pooled_manifest()
    graph = build_masked_graph_manifest(manifest, _store_for(manifest))

    train_dates = {snapshot.target_date for snapshot in graph.snapshots if snapshot.split == "train"}
    # Union of AAA(01..10) and BBB(06..15) target dates is 15 distinct days; the
    # fixed-node intersection would keep only the 5 shared days.
    assert len(train_dates) == 15
    assert len(train_dates) > 5


@pytest.mark.smoke
def test_masked_manifest_on_real_data_far_exceeds_intersection() -> None:
    """Real-data smoke: the union manifest must use many more dates than the 1,296 intersection."""

    from run_pilot import (
        _fit_graph_preprocessors,
        _select_tickers,
        _train_news_cutoffs,
        load_runner_news_panel,
        np_concat_frames,
    )
    from data import attach_news, build_pooled_manifest, load_and_split_price_data

    raw = load_and_split_price_data(_ROOT / "data" / "processed")
    selected = sorted(raw.ticker_to_id)[:6]
    raw = _select_tickers(raw, selected)
    full_frames = {
        ticker: np_concat_frames(raw.frames[ticker][name] for name in ("train", "val", "test"))
        for ticker in selected
    }
    store = _fit_graph_preprocessors(full_frames)
    pooled = build_pooled_manifest(raw, store, horizon=5)
    panel = load_runner_news_panel(
        _ROOT / "data" / "features" / "dual_group_news_panel.parquet", selected,
        _train_news_cutoffs(pooled),
    )
    pooled = PooledManifest(
        {split: tuple(attach_news(pooled.samples[split], panel, panel.feature_cols))
         for split in ("train", "val", "test")},
        pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash,
    )

    masked = build_masked_graph_manifest(pooled, store)
    intersection = build_graph_manifest(full_frames, panel, store, horizon=5)

    masked_dates = {snapshot.target_date for snapshot in masked.snapshots}
    intersection_dates = {snapshot.target_date for snapshot in intersection.snapshots}
    # The masked union recovers the full timeline (near the longest ticker's ~4,800 usable
    # dates), far beyond the intersection this 6-ticker slice is capped to by its newest listing.
    longest_history = max(len(frame) for frame in full_frames.values())
    assert len(masked_dates) > 2 * len(intersection_dates)
    assert len(masked_dates) > 0.9 * (longest_history - 26)


# --- regression: intersection path is byte-unchanged (presence stays None) -------


def test_intersection_snapshots_carry_no_presence_mask() -> None:
    import pandas as pd
    from data import NewsPanel

    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    frames = {
        ticker: pd.DataFrame({"date": dates,
                              "parkinson_volatility": np.arange(120, dtype=float) + offset + 1})
        for ticker, offset in (("AAA", 0), ("BBB", 10))
    }
    store = PreprocessorStore({
        index: TickerPreprocessor.fit(frame, ["parkinson_volatility"], "parkinson_volatility")
        for index, frame in enumerate(frames.values())
    })
    graph = build_graph_manifest(frames, NewsPanel({}, (), {}), store, seq_length=22, horizon=5)

    assert all(snapshot.presence_mask is None for snapshot in graph.snapshots)


def test_masked_graph_manifest_marks_mode() -> None:
    manifest = _staggered_pooled_manifest()
    graph = build_masked_graph_manifest(manifest, _store_for(manifest))

    assert graph.hashes.get("graph_mode") == "masked"
    # Present-node adjacency keeps a self-loop for isolated single-ticker snapshots.
    single = next(snapshot for snapshot in graph.snapshots
                  if int(snapshot.presence_mask.sum()) == 1)
    present_index = int(np.flatnonzero(single.presence_mask)[0])
    assert single.adjacency[present_index, present_index] == pytest.approx(1.0)


def _masked_graph_safe_checkpoint(graph: GraphManifest, tmp_path: Path,
                                  negative_head: bool = False) -> Path:
    p3 = PooledPriceNewsLSTM(3, 2, len(graph.ticker_to_id), use_gate=True,
                             hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    if negative_head:
        with torch.no_grad():
            for parameter in p3.head.parameters():
                parameter.zero_()
            p3.head[-1].bias.fill_(-50.0)
    checkpoint = tmp_path / "safe.pt"
    torch.save({
        "model_state": p3.state_dict(), "graph_safe": True, "training_sample_hash": "samples",
        "max_training_target_date": graph.train_end_date,
        "graph_train_end_date": graph.train_end_date,
        "graph_manifest_hash": graph.content_hash("train"),
    }, checkpoint)
    return checkpoint
