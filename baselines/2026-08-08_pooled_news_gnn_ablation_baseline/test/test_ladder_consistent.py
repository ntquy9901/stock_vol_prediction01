"""Consistent-basis ladder driver: helper contracts + one end-to-end graph-rung smoke run."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ladder_consistent as ladder  # noqa: E402
from data import PooledManifest, PooledSample, SampleKey, build_masked_graph_manifest  # noqa: E402
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


def _preprocessor(mean: float, std: float) -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 10.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)), ArrayStandardizer(np.array([mean]), np.array([std])),
    )


def _sample(ticker_id: int, ticker: str, date: str, value: float) -> PooledSample:
    rng = np.random.default_rng(hash((ticker_id, date)) % (2**32))
    return PooledSample(
        SampleKey(ticker_id, ticker, date),
        rng.normal(size=(22, 3)).astype(np.float32),
        rng.normal(size=(22, 2)).astype(np.float32),
        np.ones(22, dtype=np.int8), float(value), float(value), float(value),
        tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
    )


def _tiny_basis():
    """A 2-ticker pooled manifest (news attached) + its masked k-NN graph + matching store."""

    tickers = (("AAA", 0), ("BBB", 1))
    def rows(split_dates):
        return tuple(
            _sample(ticker_id, ticker, date, 0.02 + 0.005 * offset + 0.001 * ticker_id)
            for offset, date in enumerate(split_dates)
            for ticker, ticker_id in tickers
        )
    pooled = PooledManifest(
        {"train": rows(("2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04")),
         "val": rows(("2020-03-01", "2020-03-02", "2020-03-03")),
         "test": rows(("2020-04-01", "2020-04-02", "2020-04-03"))},
        {}, {"AAA": 0, "BBB": 1}, "preprocessing",
    )
    store = PreprocessorStore({0: _preprocessor(0.02, 0.01), 1: _preprocessor(0.03, 0.01)})
    graph = build_masked_graph_manifest(pooled, store, adjacency="knn", top_k=8)
    return pooled, graph, store


def test_manifest_with_swaps_train_and_val_keeps_test() -> None:
    pooled, _graph, _store = _tiny_basis()
    swapped = ladder._manifest_with(pooled, pooled.samples["train"][:2], pooled.samples["test"])
    assert swapped.samples["train"] == pooled.samples["train"][:2]
    assert swapped.samples["val"] == pooled.samples["test"]
    assert swapped.samples["test"] == pooled.samples["test"]


def test_present_obs_matches_pooled_val_samples() -> None:
    pooled, graph, _store = _tiny_basis()
    graph_val = ladder._present_obs([s for s in graph.snapshots if s.split == "val"])
    pooled_val = {(int(s.key.ticker_id), s.key.target_date) for s in pooled.samples["val"]}
    assert graph_val == pooled_val


def test_evaluate_graph_off_equals_use_gnn_false_model_on_shared_base() -> None:
    from models import GraphAblationModel
    from run_pilot import _build_shared_graph_base

    pooled, graph, store = _tiny_basis()
    torch.manual_seed(5)
    from models import PooledPriceNewsLSTM
    p3 = PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    g1 = GraphAblationModel(p3, use_gnn=True).eval()
    with torch.no_grad():
        for parameter in g1.message_passing.parameters():
            parameter.uniform_(-0.5, 0.5)
    g1.graph_train_end_date = graph.train_end_date
    g1.graph_manifest_hash = graph.content_hash("train")
    g1.configure_positivity(store)
    device = torch.device("cpu")
    base = _build_shared_graph_base(g1, graph, device, 32, 32)
    val = [s for s in graph.snapshots if s.split == "val"]
    _loss, records, evaluation = ladder._evaluate_graph_off(g1, val, base["val"], store, device, 32)

    g0 = GraphAblationModel(p3, use_gnn=False).eval()
    g0.configure_positivity(store)
    g0_pred = []
    with torch.no_grad():
        for start in range(0, len(val), 32):
            batch = val[start:start + 32]
            from run_pilot import _stacked_snapshot_inputs
            adjacency, presence_mask, ticker_ids, _targets = _stacked_snapshot_inputs(batch, device)
            out = g0.apply_graph_head(torch.stack(base["val"][start:start + 32]), adjacency, ticker_ids, presence_mask)
            for snapshot, snap_out, snap_present in zip(batch, out, presence_mask, strict=True):
                for index in range(len(snapshot.nodes)):
                    if bool(snap_present[index]):
                        g0_pred.append(float(snap_out[index]))
    np.testing.assert_allclose([r["prediction_norm"] for r in records], g0_pred, rtol=0, atol=0)
    assert all(math.isfinite(v) for v in evaluation["metrics"].values())


def test_har_and_pooled_rungs_score_val_and_test_on_shared_basis(tmp_path: Path) -> None:
    pooled, _graph, store = _tiny_basis()
    allowed = pooled.samples["train"]
    device = torch.device("cpu")

    p0 = ladder.run_har_rung(pooled, allowed, store, tmp_path / "P0")
    p1 = ladder.run_pooled_rung("P1", pooled, allowed, store, tmp_path / "P1", 42, device)
    p2 = ladder.run_pooled_rung("P2", pooled, allowed, store, tmp_path / "P2", 42, device)

    for rung in (p0, p1, p2):
        assert all(math.isfinite(v) for v in rung["validation_metrics"].values())
        assert all(math.isfinite(v) for v in rung["test_metrics"].values())
    assert set(p0["validation_metrics"]) >= {"mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"}


@pytest.mark.smoke
def test_run_graph_rungs_produces_nested_p3_and_g1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ladder, "GRAPH_EPOCHS", 3)
    monkeypatch.setattr(ladder, "BACKBONE_EPOCHS", 2)
    pooled, graph, store = _tiny_basis()
    out = tmp_path / "h5"
    result = ladder.run_graph_rungs(pooled, graph, store, out, seed=42, device=torch.device("cpu"))

    for rung in ("P3", "G1"):
        assert all(math.isfinite(v) for v in result[rung]["validation_metrics"].values())
        assert all(math.isfinite(v) for v in result[rung]["test_metrics"].values())
    # The graph-off readout is deterministic (bit-identical on re-evaluation).
    assert result["nesting_check"]["graph_off_readout_determinism_max_abs_diff"] == 0.0
    assert (out / "P3" / "predictions.json").exists()
    assert (out / "G1" / "predictions.json").exists()
    nesting = json.loads((out / "nesting_check.json").read_text(encoding="utf-8"))
    assert nesting["n_val_obs"] == len(pooled.samples["val"])
