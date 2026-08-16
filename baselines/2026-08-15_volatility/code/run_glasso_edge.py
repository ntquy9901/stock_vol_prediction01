"""P5: train the graph model with a graphical-LASSO (partial-correlation) edge and DM it against the
vol->PK edge, HAR, and the graph-removed control — all on the identical combo basis (same 5 features,
QLIKE estimation, 2-hop), so the ONLY difference vs the main run's vol->PK FULL is the adjacency.

Reuses the main QLIKE run's FULL / minus_graph / HAR (P0) dumps for the comparators (the basis is
deterministic, so the vol->PK FULL from that run shares this run's basis except for the edge).

Run: python <.../code/run_glasso_edge.py> <GLASSO_TS> <VOL2PK_TS_qlike> [seeds_csv] [epochs] [horizons...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
           _ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
           _ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import combo_ladder  # noqa: E402
from combo_ladder import build_basis  # noqa: E402
from data import load_and_split_price_data  # noqa: E402
from edges import swap_adjacency  # noqa: E402
from edges_glasso import build_glasso_adjacency  # noqa: E402
from features import augment_split_frames  # noqa: E402
from run_ablation import _train  # noqa: E402
from run_pilot import resolve_graph_device  # noqa: E402
from run_volatility import ROOT, _StoreShim, _evaluate_rung  # noqa: E402


def _snaps_from_graph(graph) -> list[dict[str, Any]]:
    """Convert a GraphManifest's snapshots into the batched-training snap dicts (as build_volatility_basis)."""
    snaps: list[dict[str, Any]] = []
    for s in graph.snapshots:
        presence = (s.presence_mask if s.presence_mask is not None
                    else np.ones(len(s.nodes), dtype=np.int8))
        snaps.append({
            "price": torch.from_numpy(np.asarray(s.x_price, dtype=np.float32)),
            "news": torch.from_numpy(np.asarray(s.x_news, dtype=np.float32)),
            "news_mask": torch.from_numpy(np.asarray(s.news_mask, dtype=np.float32)),
            "ticker_ids": torch.arange(len(s.nodes), dtype=torch.long),
            "adjacency": torch.from_numpy(np.asarray(s.adjacency, dtype=np.float32)),
            "target": torch.tensor([n.y_norm for n in s.nodes], dtype=torch.float32),
            "target_raw": [float(n.y_raw) for n in s.nodes],
            "presence_mask": torch.from_numpy(np.asarray(presence, dtype=np.float32)),
            "split": s.split, "target_date": s.target_date,
        })
    return snaps


def _scaler_tensors(store, num_tickers):
    mean = torch.zeros(num_tickers, dtype=torch.float32)
    std = torch.ones(num_tickers, dtype=torch.float32)
    for tid in range(num_tickers):
        sc = store.get(tid).target_scaler
        mean[tid] = float(sc.mean[0])
        std[tid] = float(sc.std[0])
    return mean, std


def train_glasso_horizon(h: int, seed: int, epochs: int, ts: str, device) -> Path:
    combo_ladder.HORIZON = h
    out = ROOT / "results" / f"volatility_glasso_h{h}_seed{seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    pooled, graph_vol2pk, store, allowed, _ = build_basis(out / "_basis")
    raw = load_and_split_price_data(ROOT / "data" / "processed")
    augmented = augment_split_frames(raw, combo_ladder._PRICE_DIR)
    glasso_adj = build_glasso_adjacency(augmented, graph_vol2pk.ticker_to_id, top_k=combo_ladder.VOL2PK_TOP_K)
    graph_glasso = swap_adjacency(graph_vol2pk, glasso_adj)
    snaps = _snaps_from_graph(graph_glasso)
    num_tickers = max(graph_glasso.ticker_to_id.values()) + 1
    mean, std = _scaler_tensors(store, num_tickers)
    basis = {"price_dim": int(graph_glasso.snapshots[0].x_price.shape[-1]),
             "news_dim": int(graph_glasso.snapshots[0].x_news.shape[-1]),
             "num_tickers": num_tickers, "scaler_mean": mean, "scaler_std": std}
    tr = [s for s in snaps if s["split"] == "train"]
    va = [s for s in snaps if s["split"] == "val"]
    te = [s for s in snaps if s["split"] == "test"]
    model = _train(basis, tr, va, out / "FULL.pt", epochs, seed, device,
                   use_news=True, use_gate=True, use_graph=True, loss="qlike", gat_layers=2)
    shim = _StoreShim(mean, std)
    _evaluate_rung(model, te, shim, device, apply_graph=True, out=out / "FULL", dump=True)
    edges = int((glasso_adj != 0).sum() - num_tickers)   # minus self-loops
    print(f"h={h} seed={seed} glasso FULL dumped ({edges} off-diagonal edges)", flush=True)
    return out


def dm_glasso(glasso_ts: str, vol2pk_ts: str, h: int, seeds) -> list[dict[str, Any]]:
    """DM of the glasso-edge FULL against each comparator (vol->PK FULL, HAR, graph-removed).
    Returns one dict per comparator; negative dm favors the glasso edge."""
    import dm_report as dm
    from diebold_mariano import diebold_mariano
    dm.RESULTS = ROOT / "results"

    def load(pattern_dir, rung_dir):
        per = [dm._load_rows(ROOT / "results" / pattern_dir.format(s=s) / rung_dir / "predictions_test.json")
               for s in seeds]
        keys = sorted(set(per[0]))
        tgt = np.array([per[0][k][0] for k in keys])
        pred = np.array([np.mean([p[k][1] for p in per]) for k in keys])
        return keys, tgt, pred

    kg, tg, pg = load(f"volatility_glasso_h{h}_seed{{s}}_{glasso_ts}", "FULL")
    out: list[dict[str, Any]] = []
    for name, rung_dir in (("vol2pk_FULL", "FULL"), ("HAR", "P0"), ("minus_graph", "minus_graph")):
        kv, tv, pv = load(f"volatility_ablation_h{h}_seed{{s}}_{vol2pk_ts}", rung_dir)
        if kv != kg or not np.allclose(tv, tg):
            out.append({"vs": name, "error": "misaligned"})
            continue
        r = diebold_mariano(dm._qlike(tg, pg), dm._qlike(tg, pv), h=h)
        fav = "glasso" if r.mean_diff < 0 else (name if r.mean_diff > 0 else "tie")
        out.append({"vs": name, "dm_hln": r.dm_hln, "p_value": r.p_value, "favors": fav})
    return out


def main(glasso_ts: str, vol2pk_ts: str, seeds, epochs: int = 15, horizons=(1, 5, 10, 22)) -> None:
    device = resolve_graph_device("cuda")
    for h in horizons:
        for s in seeds:
            t0 = time.perf_counter()
            train_glasso_horizon(h, s, epochs, glasso_ts, device)
            print(f"  (h{h} seed{s} {time.perf_counter() - t0:.0f}s)", flush=True)
    print("\n# DM: graphical-LASSO edge vs vol->PK edge / HAR / graph-removed (QLIKE, seed-ensemble)")
    for h in horizons:
        cells = []
        for r in dm_glasso(glasso_ts, vol2pk_ts, h, seeds):
            if "error" in r:
                cells.append(f"{r['vs']}:{r['error']}")
            else:
                sig = "*" if r["p_value"] < 0.05 else ""
                cells.append(f"{r['vs']}: dm={r['dm_hln']:+.2f}p{r['p_value']:.2f}{sig}(favors {r['favors']})")
        print(f"h{h} glasso-FULL vs | " + " | ".join(cells))


if __name__ == "__main__":  # pragma: no cover
    _gts = sys.argv[1]
    _vts = sys.argv[2]
    _seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42, 123, 2026]
    _epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    _hz = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5, 10, 22)
    main(_gts, _vts, _seeds, _epochs, _hz)
