"""Behavior contracts for GAT-hybrid-style SPARSE adjacency on the masked graph.

The dense masked path builds a full signed-correlation adjacency over PRESENT tickers.
These tests pin the additive sparsification options (``knn`` top-k, ``threshold`` |corr|>tau)
that trim edges on the present-node subgraph while leaving the dense default byte-identical
and keeping absent nodes fully masked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import (  # noqa: E402
    PooledManifest,
    PooledSample,
    SampleKey,
    _correlation_adjacency,
    _masked_correlation_adjacency,
    _sparsify_correlation,
    _validate_adjacency_config,
    build_masked_graph_manifest,
)
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


def _distinct_corr(n: int, seed: int = 0) -> np.ndarray:
    """A symmetric Pearson correlation matrix with distinct off-diagonal magnitudes."""

    rng = np.random.default_rng(seed)
    series = rng.standard_normal((200, n))
    correlation = np.corrcoef(series, rowvar=False).astype(np.float32)
    np.fill_diagonal(correlation, 1.0)
    return correlation


# --- (a) k-NN: bounded degree + symmetric ----------------------------------------


def test_knn_sparsify_bounded_degree_and_symmetric() -> None:
    correlation = _distinct_corr(6)
    result = _sparsify_correlation(correlation, "knn", top_k=2, corr_threshold=0.7)

    assert np.array_equal(result, result.T)  # undirected
    off_diagonal = result.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    per_row_nonzeros = np.count_nonzero(off_diagonal, axis=1)
    assert (per_row_nonzeros <= 2).all()  # <= k off-diagonal neighbours per row
    assert np.array_equal(np.diag(result), np.diag(correlation))  # self-loop kept


def test_knn_retains_signed_correlation_weight() -> None:
    correlation = np.array(
        [[1.0, -0.9, 0.1], [-0.9, 1.0, 0.2], [0.1, 0.2, 1.0]], dtype=np.float32
    )
    result = _sparsify_correlation(correlation, "knn", top_k=1, corr_threshold=0.7)
    # Nodes 0 and 1 mutually rank each other first (|-0.9| strongest); the edge keeps
    # its NEGATIVE sign, not the absolute value used only for ranking.
    assert result[0, 1] == pytest.approx(-0.9)
    assert result[1, 0] == pytest.approx(-0.9)


# --- (b) threshold: zeros all |corr| <= tau --------------------------------------


def test_threshold_sparsify_zeros_below_tau() -> None:
    correlation = np.array(
        [[1.0, 0.8, 0.5], [0.8, 1.0, -0.75], [0.5, -0.75, 1.0]], dtype=np.float32
    )
    result = _sparsify_correlation(correlation, "threshold", top_k=8, corr_threshold=0.7)

    # |0.5| <= 0.7 -> zeroed; |0.8|, |-0.75| > 0.7 -> kept with sign.
    assert result[0, 2] == 0.0 and result[2, 0] == 0.0
    assert result[0, 1] == pytest.approx(0.8)
    assert result[1, 2] == pytest.approx(-0.75)
    off_diagonal = np.abs(result.copy())
    np.fill_diagonal(off_diagonal, 0.0)
    assert not ((off_diagonal > 0.0) & (off_diagonal <= 0.7)).any()
    assert np.array_equal(np.diag(result), np.diag(correlation))


# --- (c) presence: sparse masked adjacency keeps absent nodes edgeless -----------


def _price_with_absent(n: int = 6, seq: int = 40, feat: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, seq)).astype(np.float32)
    return np.repeat(base[:, :, None], feat, axis=2)


def test_sparse_masked_adjacency_respects_presence() -> None:
    price = _price_with_absent()
    presence = np.array([1, 1, 1, 1, 0, 0], dtype=np.int8)  # nodes 4,5 absent
    adjacency = _masked_correlation_adjacency(price, presence, mode="knn", top_k=2)

    absent = np.flatnonzero(presence == 0)
    assert np.all(adjacency[absent, :] == 0.0)
    assert np.all(adjacency[:, absent] == 0.0)
    present = np.flatnonzero(presence)
    assert np.array_equal(adjacency, adjacency.T)
    for index in present:
        assert adjacency[index, index] == pytest.approx(1.0)
    # k=2 genuinely sparsifies the 4-present subgraph (dense would give 3 neighbours).
    off_diagonal = adjacency.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert (np.count_nonzero(off_diagonal[present], axis=1) <= 2).all()


# --- (d) dense default is byte-identical to the pre-sparsification behaviour ------


def test_dense_default_masked_adjacency_byte_identical() -> None:
    price = _price_with_absent()
    presence = np.array([1, 1, 1, 1, 0, 0], dtype=np.int8)

    produced = _masked_correlation_adjacency(price, presence)  # default mode="dense"

    # Reference: the exact pre-change computation (corrcoef over present, self-loop=1).
    reference = np.zeros((6, 6), dtype=np.float32)
    present = np.flatnonzero(presence)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(price[present, :, 0])
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    reference[np.ix_(present, present)] = correlation
    for index in present:
        reference[index, index] = 1.0

    assert np.array_equal(produced, reference)
    assert produced.dtype == reference.dtype


def test_dense_default_intersection_adjacency_byte_identical() -> None:
    rng = np.random.default_rng(3)
    values = rng.standard_normal((5, 60)).astype(np.float32)

    produced = _correlation_adjacency(values)  # default mode="dense"

    correlation = np.corrcoef(values)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(correlation, 1.0)
    assert np.array_equal(produced, correlation)


# --- (e) manifest hash differs across adjacency modes ----------------------------


def _target_preprocessor() -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)),
        ArrayStandardizer(np.array([1e-2]), np.array([1e-2])),
    )


def _pooled_sample(ticker_id: int, ticker: str, date: str, value: float) -> PooledSample:
    seq, price_w, news_w = 22, 3, 2
    ramp = (value + np.arange(seq, dtype=np.float32))[:, None]
    return PooledSample(
        SampleKey(ticker_id, ticker, date),
        np.repeat(ramp, price_w, axis=1).astype(np.float32),
        np.full((seq, news_w), value, dtype=np.float32),
        np.ones(seq, dtype=np.int8),
        float(value), float(value), float(value),
        tuple(f"2020-01-{day:02d}" for day in range(1, seq + 1)),
    )


def _staggered_manifest() -> PooledManifest:
    def date(day: int) -> str:
        return f"2021-{(day - 1) // 28 + 1:02d}-{(day - 1) % 28 + 1:02d}"

    train = [_pooled_sample(0, "AAA", date(day), 0.10 + 0.01 * day) for day in range(1, 11)] + [
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


def test_manifest_hash_differs_across_adjacency_modes() -> None:
    manifest = _staggered_manifest()
    store = PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})

    dense = build_masked_graph_manifest(manifest, store)
    knn = build_masked_graph_manifest(manifest, store, adjacency="knn", top_k=8)
    threshold = build_masked_graph_manifest(manifest, store, adjacency="threshold", corr_threshold=0.7)

    hashes = {
        "dense": dense.content_hash("train"),
        "knn": knn.content_hash("train"),
        "threshold": threshold.content_hash("train"),
    }
    assert len(set(hashes.values())) == 3  # no cross-usable collision
    # The dense hash is unchanged (mode field only added for non-dense), so prior dense
    # runs stay comparable; the mode label lives in the manifest hashes dict.
    assert "adjacency_mode" not in dense.hashes
    assert knn.hashes["adjacency_mode"] == "knn"
    assert threshold.hashes["adjacency_mode"] == "threshold"


def test_invalid_adjacency_config_is_rejected() -> None:
    manifest = _staggered_manifest()
    store = PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})
    with pytest.raises(ValueError):
        build_masked_graph_manifest(manifest, store, adjacency="bogus")
    with pytest.raises(ValueError):
        build_masked_graph_manifest(manifest, store, adjacency="knn", top_k=0)


def test_threshold_config_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        _validate_adjacency_config("threshold", 8, 1.0)  # tau must be in [0, 1)
    with pytest.raises(ValueError):
        _validate_adjacency_config("threshold", 8, -0.1)


def test_sparsify_single_node_returns_unchanged() -> None:
    single = np.array([[1.0]], dtype=np.float32)
    result = _sparsify_correlation(single, "knn", top_k=8, corr_threshold=0.7)
    assert np.array_equal(result, single)


def test_intersection_correlation_adjacency_sparsifies_non_dense() -> None:
    rng = np.random.default_rng(7)
    series = rng.standard_normal((5, 80)).astype(np.float32)
    dense = _correlation_adjacency(series, "dense")
    knn = _correlation_adjacency(series, "knn", top_k=1)

    assert np.array_equal(knn, knn.T)
    off_diagonal = knn.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert (np.count_nonzero(off_diagonal, axis=1) <= 1).all()
    assert np.count_nonzero(knn) < np.count_nonzero(dense)  # genuinely sparser


def test_parse_args_rejects_adjacency_outside_graph_phase() -> None:
    from run_pilot import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--phase", "P1", "--adjacency", "knn"])


def test_plot_learning_curve_survives_missing_matplotlib(tmp_path: Path, monkeypatch) -> None:
    from run_pilot import _plot_learning_curve

    monkeypatch.setitem(sys.modules, "matplotlib", None)  # force the import to fail
    target = tmp_path / "curve.png"
    _plot_learning_curve([1.0, 0.5], [1.1, 0.6], target)
    assert not target.exists()  # best-effort: no plot, no crash


def test_adjacency_config_records_active_hyperparameter() -> None:
    from run_pilot import _adjacency_config

    assert _adjacency_config("dense", 8, 0.7) == {"mode": "dense"}
    assert _adjacency_config("knn", 8, 0.7) == {"mode": "knn", "top_k": 8}
    assert _adjacency_config("threshold", 8, 0.7) == {"mode": "threshold", "corr_threshold": 0.7}


def test_build_graph_manifest_for_mode_masked_branch() -> None:
    from run_pilot import _build_graph_manifest_for_mode

    manifest = _staggered_manifest()
    store = PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})
    graph = _build_graph_manifest_for_mode(
        "masked", manifest, store, {}, None, 5, "knn", 8, 0.7,
    )
    assert graph.hashes["adjacency_mode"] == "knn"
    assert any(snapshot.presence_mask is not None for snapshot in graph.snapshots)


def test_build_graph_manifest_for_mode_intersection_branch() -> None:
    import pandas as pd
    from data import NewsPanel
    from run_pilot import _build_graph_manifest_for_mode

    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    frames = {
        ticker: pd.DataFrame({"date": dates,
                              "parkinson_volatility": np.arange(120, dtype=float) + offset + 1})
        for ticker, offset in (("AAA", 0), ("BBB", 10), ("CCC", 5))
    }
    store = PreprocessorStore({
        index: TickerPreprocessor.fit(frame, ["parkinson_volatility"], "parkinson_volatility")
        for index, frame in enumerate(frames.values())
    })
    graph = _build_graph_manifest_for_mode(
        "intersection", None, store, frames, NewsPanel({}, (), {}), 5, "knn", 1, 0.7,
    )
    assert graph.hashes["adjacency_mode"] == "knn"
    assert all(snapshot.presence_mask is None for snapshot in graph.snapshots)


@pytest.mark.smoke
def test_run_graph_screening_masked_knn_smoke(tmp_path: Path) -> None:
    """Real-data slice: drive the whole masked knn graph screening runner end-to-end."""

    import json
    from run_pilot import parse_args, run_graph_screening

    args = parse_args([
        "--phase", "graph", "--graph", "masked", "--adjacency", "knn", "--top-k", "8",
        "--max-tickers", "6", "--epochs", "1", "--device", "cpu",
        "--output-dir", str(tmp_path / "out"),
    ])
    result_path = run_graph_screening(args)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["adjacency"] == {"mode": "knn", "top_k": 8}
    assert payload["edge_density"]["max_offdiag_nonzeros_per_present_row"] <= 8
    for name in ("G0", "G1"):
        metrics = payload["results"][name]["validation_metrics"]
        assert set(metrics) >= {"mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"}


# --- runner integration: knn manifest trains + emits a learning curve ------------


def _masked_checkpoint(graph, tmp_path: Path) -> Path:
    import torch
    from models import PooledPriceNewsLSTM

    p3 = PooledPriceNewsLSTM(3, 2, len(graph.ticker_to_id), use_gate=True,
                             hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "safe.pt"
    torch.save({
        "model_state": p3.state_dict(), "graph_safe": True, "training_sample_hash": "samples",
        "max_training_target_date": graph.train_end_date,
        "graph_train_end_date": graph.train_end_date,
        "graph_manifest_hash": graph.content_hash("train"),
    }, checkpoint)
    return checkpoint


def test_graph_epochs_allows_convergence_length_runs() -> None:
    from run_pilot import _MAX_GRAPH_EPOCHS, _validate_graph_epochs

    _validate_graph_epochs(15)  # user-approved GAT convergence length (>10) must be accepted
    _validate_graph_epochs(_MAX_GRAPH_EPOCHS)
    with pytest.raises(ValueError):
        _validate_graph_epochs(0)
    with pytest.raises(ValueError):
        _validate_graph_epochs(_MAX_GRAPH_EPOCHS + 1)


def test_knn_masked_run_emits_learning_curve_and_edge_density(tmp_path: Path) -> None:
    from models import GraphAblationModel
    from run_pilot import _edge_density_stats, _run_one_graph_model
    import json

    manifest = _staggered_manifest()
    store = PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})
    graph = build_masked_graph_manifest(manifest, store, adjacency="knn", top_k=8)

    density = _edge_density_stats(graph)
    assert density["present_row_count"] > 0
    assert density["max_offdiag_nonzeros_per_present_row"] <= 8  # bounded by k

    checkpoint = _masked_checkpoint(graph, tmp_path)
    model = GraphAblationModel.from_p3_checkpoint(
        str(checkpoint), True, graph.train_end_date, graph.content_hash("train"),
    )
    _run_one_graph_model(model, graph, store, "G1", epochs=3, seed=42,
                         output=tmp_path / "G1", device="cpu", train_batch_size=4)

    payload = json.loads((tmp_path / "G1" / "results.json").read_text(encoding="utf-8"))
    assert len(payload["validation_losses"]) == 3
    assert all(np.isfinite(value) for value in payload["validation_losses"])
    assert (tmp_path / "G1" / "learning_curve.png").exists()
