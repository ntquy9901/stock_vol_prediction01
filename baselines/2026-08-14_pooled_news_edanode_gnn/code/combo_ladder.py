"""Combo ladder: pooled 5-feature (HAR+MarketPK+vol_z) + PhoBERT news + directed vol->PK Top-5 GNN.

One leakage-safe basis; rungs:
  P0 = pooled HAR linear regression (3 HAR features only, pure baseline).
  P1 = pooled price LSTM, 5 node features (HAR + MarketPK + volume_zscore_20), no news/graph.
  P2 = P1 + PhoBERT news late-fusion (gate off), no graph.
  P3 = P2 + per-ticker gate, graph off (trained G1 read out with message passing disabled).
  G1 = P3 + message passing over the directed volume->PK Top-5 adjacency (the combo).

Reuses Pipeline B (2026-08-08 ladder_consistent) rung logic and Pipeline A (2026-08-11 eda_gnn)
node features + directed edge. Only the basis is new; no model code is written. See
``design/design.md``. P0 uses A's ``run_e0`` (first-3-column HAR slice) because B's
``run_har_reference`` slices the LAST 3 columns, which are not HAR under the 5-feature order.

Run (GPU venv):  python <.../code/combo_ladder.py> <TS> [device]
Writes results/pooled_news_edanode_seed{seed}_<TS>/h5/ladder_metrics.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
PILOT = ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
EDA = ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code"
for _p in (str(CODE), str(PILOT), str(EDA), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from data import (  # noqa: E402
    PooledManifest, attach_news, build_masked_graph_manifest, build_pooled_manifest,
    load_and_split_price_data,
)
from edges import build_vol2pk_adjacency, swap_adjacency  # noqa: E402
from eda_ladder import run_e0  # noqa: E402
from features import EXTRA_FEATURE_COLUMNS, augment_split_frames, build_extended_store  # noqa: E402
from ladder_consistent import (  # noqa: E402
    HORIZON, SEEDS, TOP_K, _log, _present_obs, run_graph_rungs, run_pooled_rung,
)
from run_pilot import (  # noqa: E402
    _edge_density_stats, _train_news_cutoffs, _write_json, load_runner_news_panel,
    resolve_graph_device,
)

SEQ = 22
VOL2PK_TOP_K = 5
_PROCESSED = ROOT / "data" / "processed"
_PRICE_DIR = ROOT / "data" / "raw" / "prices"
_NEWS_PANEL = ROOT / "data" / "features" / "dual_group_news_panel.parquet"
_SPLITS = ("train", "val", "test")


def build_basis(stamp: Path):
    """Seed-independent basis: 5-feature + news manifest, directed vol->PK Top-5 masked graph.

    Node features (nested order): parkinson_volatility, har_weekly, har_monthly, market_pk,
    volume_zscore_20. Per-ticker train-only scalers (fixes prior review H1). News attached with a
    per-ticker causal cutoff. Edge = directed volume->PK Top-5, estimated on train rows and frozen.
    """

    raw = load_and_split_price_data(_PROCESSED)
    augmented = augment_split_frames(raw, _PRICE_DIR)
    store = build_extended_store(augmented, EXTRA_FEATURE_COLUMNS)
    pooled = build_pooled_manifest(augmented, store, seq_length=SEQ, horizon=HORIZON)
    selected = tuple(sorted(augmented.ticker_to_id))
    cutoffs = _train_news_cutoffs(pooled)
    panel = load_runner_news_panel(_NEWS_PANEL, selected, cutoffs)
    pooled = PooledManifest(
        {s: tuple(attach_news(pooled.samples[s], panel, panel.feature_cols)) for s in _SPLITS},
        pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash)
    _log(stamp, "pooled 5-feature + news manifest built; building masked knn-8 scaffold ...")
    graph_corr = build_masked_graph_manifest(pooled, store, adjacency="knn", top_k=TOP_K)
    vol2pk = build_vol2pk_adjacency(augmented, graph_corr.ticker_to_id, top_k=VOL2PK_TOP_K)
    graph = swap_adjacency(graph_corr, vol2pk)
    allowed = tuple(s for s in pooled.samples["train"] if s.key.target_date <= graph.train_end_date)
    if not allowed:
        raise ValueError("graph-bound train set is empty")
    # One-basis invariant: masked graph present-node val/test obs == pooled val/test samples.
    for split in ("val", "test"):
        pooled_obs = {(int(s.key.ticker_id), s.key.target_date) for s in pooled.samples[split]}
        if not pooled_obs:
            raise ValueError(f"{split} split is empty")
        graph_obs = _present_obs([s for s in graph.snapshots if s.split == split])
        if graph_obs != pooled_obs:
            raise ValueError(f"{split} observation set differs between graph and pooled manifests")
    # count only real directed edges (exclude the per-node self-loops on the diagonal)
    adjacency = np.asarray(vol2pk)
    edge_count = int((adjacency[~np.eye(adjacency.shape[0], dtype=bool)] != 0).sum())
    price_dim = int(allowed[0].x_price_raw.shape[1])
    news_dim = int(allowed[0].x_news.shape[1])
    if price_dim != 5 or news_dim <= 0:
        raise ValueError(f"unexpected basis width: price_dim={price_dim} (want 5), news_dim={news_dim}")
    _log(stamp, f"basis ready: snapshots={len(graph.snapshots)} train={len(allowed)} "
                f"val_obs={len(pooled.samples['val'])} test_obs={len(pooled.samples['test'])} "
                f"vol2pk_directed_edges={edge_count} price_dim={price_dim} news_dim={news_dim}")
    return pooled, graph, store, allowed, edge_count


def run_seed(pooled, graph, store, allowed, out: Path, seed: int, device, stamp: Path,
             edge_count: int) -> None:
    """P0 (pure HAR) + P1/P2 (pooled) + P3/G1 (graph off/on) on the shared combo basis."""

    out.mkdir(parents=True, exist_ok=True)
    _log(stamp, f"seed={seed} P0 (HAR)")
    p0 = run_e0(pooled, allowed, store, out / "P0")
    _log(stamp, f"seed={seed} P1 (price5)")
    p1 = run_pooled_rung("P1", pooled, allowed, store, out / "P1", seed, device)
    _log(stamp, f"seed={seed} P2 (price5+news)")
    p2 = run_pooled_rung("P2", pooled, allowed, store, out / "P2", seed, device)
    _log(stamp, f"seed={seed} P3+G1 (graph off/on: directed vol->PK)")
    graph_rungs = run_graph_rungs(pooled, graph, store, out, seed, device)
    ladder = {
        "seed": seed, "horizon": HORIZON, "vol2pk_top_k": VOL2PK_TOP_K, "vol2pk_edges": edge_count,
        "adjacency": {"mode": "vol2pk_dir_top5_frozen", "top_k": VOL2PK_TOP_K},
        "edge_density": _edge_density_stats(graph),
        "rungs": {"P0": p0, "P1": p1, "P2": p2, "P3": graph_rungs["P3"], "G1": graph_rungs["G1"]},
        "nesting_check": graph_rungs["nesting_check"],
        "snapshot_count": graph_rungs["snapshot_count"],
    }
    _write_json(out / "ladder_metrics.json", ladder)
    _log(stamp, f"seed={seed} DONE")


def main(ts: str, device_name: str = "cuda") -> None:
    stamp = ROOT / "temp" / f"combo_ladder_{ts}_h{HORIZON}"
    stamp.parent.mkdir(parents=True, exist_ok=True)  # reused _log does not create it
    device = resolve_graph_device(device_name)
    _log(stamp, f"device={device} horizon={HORIZON} building seed-independent basis ...")
    pooled, graph, store, allowed, edge_count = build_basis(stamp)
    for seed in SEEDS:
        out = ROOT / "results" / f"pooled_news_edanode_seed{seed}_{ts}" / f"h{HORIZON}"
        t0 = time.perf_counter()
        run_seed(pooled, graph, store, allowed, out, seed, device, stamp, edge_count)
        _log(stamp, f"seed={seed} elapsed {time.perf_counter() - t0:.1f}s")
    stamp.with_suffix(".ALL_DONE").write_text("done", encoding="utf-8")
    _log(stamp, "ALL_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cuda")
