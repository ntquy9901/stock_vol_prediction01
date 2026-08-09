"""Numerical-equivalence contracts for the frozen-encoder ``base`` cache (proposed updates #1/#2).

The masked G0/G1 graph runner recomputes the FROZEN (dropout-free, ``requires_grad_(False)``)
encoder embeddings every epoch and again for the second model — ~96.7% redundant work.  These
tests are the evidence that caching that ``base`` (proposal #1) and encoding only PRESENT nodes
(proposal #2) leave the results bit-identical:

* present-only ``encode_base`` reproduces the full-encode present rows within fp tolerance;
* a cached ``_run_one_graph_model`` reproduces the uncached run's val loss, metrics and
  per-observation predictions exactly;
* a base cache shared across two models (the G0/G1 sharing) reproduces a self-computed run.
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

from data import PooledManifest, PooledSample, SampleKey, build_masked_graph_manifest  # noqa: E402
from models import GraphAblationModel, PooledPriceNewsLSTM  # noqa: E402
from run_pilot import (  # noqa: E402
    _assert_shared_frozen_encoder,
    _build_shared_graph_base,
    _precompute_graph_base,
    _run_one_graph_model,
    build_graph_bound_p3_warm_start,
    parse_args,
)
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402

_ATOL = 1e-6


def _target_preprocessor(mean: float = 1e-2, std: float = 1e-2) -> TickerPreprocessor:
    return TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        ArrayStandardizer(np.zeros(3), np.ones(3)),
        ArrayStandardizer(np.array([mean]), np.array([std])),
    )


def _pooled_sample(ticker_id: int, ticker: str, date: str, value: float,
                   seq: int = 22, price_w: int = 3, news_w: int = 2) -> PooledSample:
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
    """AAA on train dates 01..10, BBB on 06..15 -> union 15 dates, some single-present."""

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
    return PooledManifest({"train": tuple(train), "val": tuple(val), "test": tuple(test)},
                          {}, {"AAA": 0, "BBB": 1}, "preprocessing")


def _store_for(manifest: PooledManifest) -> PreprocessorStore:
    return PreprocessorStore({tid: _target_preprocessor() for tid in manifest.ticker_to_id.values()})


def _masked_graph_safe_checkpoint(graph, tmp_path: Path) -> Path:
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


def _fresh_model(checkpoint: Path, graph, use_gnn: bool = True,
                 init_seed: int | None = None) -> GraphAblationModel:
    # The G1 message-passing projection is randomly initialized at construction, BEFORE
    # ``_run_one_graph_model`` sets its per-run seed.  Equivalence comparisons therefore seed the
    # construction explicitly so both models share the identical projection init; only then does a
    # cached vs uncached difference reflect the cache rather than a different random start.
    if init_seed is not None:
        torch.manual_seed(init_seed)
    return GraphAblationModel.from_p3_checkpoint(
        str(checkpoint), use_gnn, graph.train_end_date, graph.content_hash("train"),
    )


# --- proposal #2: present-only encode == full encode on present rows -------------------


def test_present_only_encode_matches_full_encode_on_present_rows() -> None:
    torch.manual_seed(0)
    p3 = PooledPriceNewsLSTM(3, 2, 3, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0)
    model = GraphAblationModel(p3, use_gnn=True).eval()
    price = torch.randn(1, 3, 22, 3)
    news = torch.randn(1, 3, 22, 2)
    mask = torch.ones(1, 3, 22, dtype=torch.bool)
    ticker_ids = torch.tensor([[0, 1, 2]])
    presence = torch.tensor([[True, True, False]])

    present_only = model.encode_base(price, news, mask, ticker_ids, presence)
    full = model.encode_base(price, news, mask, ticker_ids, None)

    # Present rows are bit-identical (each node's sequence encodes independently of the batch).
    torch.testing.assert_close(present_only[0, :2], full[0, :2], atol=_ATOL, rtol=0.0)
    # The absent node is never encoded: it carries a zero embedding and cannot affect present rows.
    assert torch.count_nonzero(present_only[0, 2]) == 0


def test_precompute_base_matches_forward_present_predictions(tmp_path: Path) -> None:
    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = _masked_graph_safe_checkpoint(graph, tmp_path)
    model = _fresh_model(checkpoint, graph).eval()
    model.configure_positivity(store)
    validation = [snapshot for snapshot in graph.snapshots if snapshot.split == "val"]

    cache = _precompute_graph_base(model, validation, torch.device("cpu"), batch_size=4)
    snapshot = validation[0]
    presence = torch.from_numpy(snapshot.presence_mask.copy()).to(torch.bool)
    ticker_ids = torch.tensor([node.ticker_id for node in snapshot.nodes])
    adjacency = torch.from_numpy(snapshot.adjacency.copy())

    with torch.no_grad():
        from_cache = model.apply_graph_head(cache[0], adjacency, ticker_ids, presence)
        direct = model(
            torch.from_numpy(snapshot.x_price.copy()),
            torch.from_numpy(snapshot.x_news.copy()),
            torch.from_numpy(snapshot.news_mask.copy()),
            ticker_ids, adjacency, presence,
        )
    torch.testing.assert_close(from_cache[presence], direct[presence], atol=_ATOL, rtol=0.0)


# --- proposal #1: cached full run == uncached full run --------------------------------


def _metrics_close(cached: dict, uncached: dict) -> None:
    assert cached["validation_loss"] == pytest.approx(uncached["validation_loss"], abs=_ATOL, rel=0.0)
    assert cached["validation_metrics"].keys() == uncached["validation_metrics"].keys()
    for key, value in cached["validation_metrics"].items():
        assert value == pytest.approx(uncached["validation_metrics"][key], abs=_ATOL, rel=0.0)


def _predictions_close(a_dir: Path, b_dir: Path) -> None:
    a_rows = json.loads((a_dir / "predictions.json").read_text(encoding="utf-8"))
    b_rows = json.loads((b_dir / "predictions.json").read_text(encoding="utf-8"))
    assert len(a_rows) == len(b_rows)
    for a_row, b_row in zip(a_rows, b_rows, strict=True):
        assert (a_row["ticker_id"], a_row["target_date"]) == (b_row["ticker_id"], b_row["target_date"])
        assert a_row["target_raw"] == pytest.approx(b_row["target_raw"], abs=_ATOL, rel=0.0)
        assert a_row["prediction_raw"] == pytest.approx(b_row["prediction_raw"], abs=_ATOL, rel=0.0)


def test_cached_graph_run_equals_uncached_graph_run(tmp_path: Path) -> None:
    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = _masked_graph_safe_checkpoint(graph, tmp_path)

    def run(use_cache: bool, name: str) -> dict:
        return _run_one_graph_model(
            _fresh_model(checkpoint, graph, init_seed=1234), graph, store, "G1", epochs=3, seed=42,
            output=tmp_path / name, device="cpu", train_batch_size=4, use_base_cache=use_cache,
        )

    cached = run(True, "cached")
    uncached = run(False, "uncached")
    _metrics_close(cached, uncached)
    _predictions_close(tmp_path / "cached", tmp_path / "uncached")


def test_g0_cached_graph_run_equals_uncached(tmp_path: Path) -> None:
    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = _masked_graph_safe_checkpoint(graph, tmp_path)

    def run(use_cache: bool, name: str) -> dict:
        return _run_one_graph_model(
            _fresh_model(checkpoint, graph, use_gnn=False, init_seed=7), graph, store, "G0",
            epochs=3, seed=7, output=tmp_path / name, device="cpu", train_batch_size=4,
            use_base_cache=use_cache,
        )

    _metrics_close(run(True, "g0_cached"), run(False, "g0_uncached"))
    _predictions_close(tmp_path / "g0_cached", tmp_path / "g0_uncached")


# --- proposal #1 (cross-model sharing): shared cache == self-computed cache ------------


def test_shared_base_cache_matches_self_computed_run(tmp_path: Path) -> None:
    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = _masked_graph_safe_checkpoint(graph, tmp_path)

    self_computed = _run_one_graph_model(
        _fresh_model(checkpoint, graph, init_seed=99), graph, store, "G1", epochs=3, seed=42,
        output=tmp_path / "self", device="cpu", train_batch_size=4,
    )

    source = _fresh_model(checkpoint, graph, init_seed=5)  # projection unused by the base
    trained = _fresh_model(checkpoint, graph, init_seed=99)  # same init as self_computed
    _assert_shared_frozen_encoder(source, trained)  # must not raise: same checkpoint
    shared = _build_shared_graph_base(source, graph, torch.device("cpu"), 4, 4)
    shared_run = _run_one_graph_model(
        trained, graph, store, "G1", epochs=3, seed=42,
        output=tmp_path / "shared", device="cpu", train_batch_size=4, base_cache=shared,
    )
    _metrics_close(shared_run, self_computed)
    _predictions_close(tmp_path / "shared", tmp_path / "self")


def test_parse_args_exposes_backbone_and_cache_flags() -> None:
    args = parse_args([
        "--phase", "graph", "--graph", "masked", "--adjacency", "knn", "--top-k", "8",
        "--backbone-epochs", "5", "--backbone-dropout", "0.2", "--no-base-cache",
    ])
    assert args.backbone_epochs == 5
    assert args.backbone_dropout == 0.2
    assert args.no_base_cache is True
    # Defaults: screening-config backbone, cache on.
    defaults = parse_args(["--phase", "graph", "--graph", "masked", "--adjacency", "knn"])
    assert (defaults.backbone_epochs, defaults.backbone_dropout, defaults.no_base_cache) == (5, 0.2, False)


def test_graph_bound_warm_start_trains_with_dropout(tmp_path: Path) -> None:
    manifest = _staggered_pooled_manifest()
    store = _store_for(manifest)
    graph = build_masked_graph_manifest(manifest, store)
    checkpoint = build_graph_bound_p3_warm_start(
        manifest, graph, tmp_path, seed=42, store=store, epochs=2,
        device=torch.device("cpu"), train_batch_size=8, dropout=0.2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config_name"] == "P3"
    assert saved["graph_bound_warm_start"] is True
    assert saved["graph_train_end_date"] == graph.train_end_date


def test_assert_shared_frozen_encoder_rejects_divergent_encoders() -> None:
    torch.manual_seed(1)
    first = GraphAblationModel(
        PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0), True)
    torch.manual_seed(2)
    second = GraphAblationModel(
        PooledPriceNewsLSTM(3, 2, 2, use_gate=True, hidden_dim=4, news_hidden_dim=4, dropout=0.0), True)
    with pytest.raises(ValueError, match="identical frozen"):
        _assert_shared_frozen_encoder(first, second)
