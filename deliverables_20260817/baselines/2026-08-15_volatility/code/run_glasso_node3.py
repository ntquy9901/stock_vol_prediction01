"""3-feature-node + graphical-LASSO-edge graph model vs HAR(3-feature) — all metrics.

Node features = ONLY the 3 HAR terms (parkinson, har_weekly, har_monthly), no market_pk / volume_z;
edge = train-frozen graphical-LASSO partial-correlation Top-5; no news / no gate; QLIKE, 2-hop. This
asks: does a GNN over the SAME 3 features HAR uses, with a conditional-dependence edge, beat HAR? All
six metrics are reported (3-seed mean) plus DM vs HAR.

Run: python <.../code/run_glasso_node3.py> <TS> [seeds_csv] [epochs] [horizons...]
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

import combo_ladder  # noqa: E402
from combo_ladder import build_basis  # noqa: E402
from data import load_and_split_price_data  # noqa: E402
from edges import swap_adjacency  # noqa: E402
from edges_glasso import build_glasso_adjacency  # noqa: E402
from features import augment_split_frames  # noqa: E402
from run_ablation import _train  # noqa: E402
from run_glasso_edge import _scaler_tensors, _snaps_from_graph  # noqa: E402
from run_pilot import resolve_graph_device  # noqa: E402
from run_volatility import ROOT, _StoreShim, _evaluate_rung  # noqa: E402

_HAR_WIDTH = 3


def slice_price(snaps: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    """Return snaps with each price tensor narrowed to its first ``width`` node features."""
    out = []
    for s in snaps:
        t = dict(s)
        t["price"] = s["price"][:, :, :width]
        out.append(t)
    return out


def run_horizon(h: int, seed: int, epochs: int, ts: str, device) -> dict[str, Any]:
    combo_ladder.HORIZON = h
    out = ROOT / "results" / f"volatility_glasso_node3_h{h}_seed{seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    pooled, graph_vol2pk, store, allowed, _ = build_basis(out / "_basis")
    raw = load_and_split_price_data(ROOT / "data" / "processed")
    augmented = augment_split_frames(raw, combo_ladder._PRICE_DIR)
    glasso_adj = build_glasso_adjacency(augmented, graph_vol2pk.ticker_to_id, top_k=combo_ladder.VOL2PK_TOP_K)
    graph = swap_adjacency(graph_vol2pk, glasso_adj)
    snaps = slice_price(_snaps_from_graph(graph), _HAR_WIDTH)
    num_tickers = max(graph.ticker_to_id.values()) + 1
    mean, std = _scaler_tensors(store, num_tickers)
    basis = {"price_dim": _HAR_WIDTH, "news_dim": int(graph.snapshots[0].x_news.shape[-1]),
             "num_tickers": num_tickers, "scaler_mean": mean, "scaler_std": std}
    tr = [s for s in snaps if s["split"] == "train"]
    va = [s for s in snaps if s["split"] == "val"]
    te = [s for s in snaps if s["split"] == "test"]
    model = _train(basis, tr, va, out / "GLASSO3.pt", epochs, seed, device,
                   use_news=False, use_gate=False, use_graph=True, loss="qlike", gat_layers=2)
    ev = _evaluate_rung(model, te, _StoreShim(mean, std), device, apply_graph=True,
                        out=out / "GLASSO3", dump=True)
    print(f"h={h} seed={seed} glasso-node3 test QLIKE={ev['metrics']['qlike']:.4f}", flush=True)
    return ev["metrics"]


def _har_metrics(h: int, seeds) -> dict[str, float]:
    import json
    keys = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
    vals = {k: [] for k in keys}
    for s in seeds:
        m = json.load(open(ROOT / "results" /
                           f"volatility_ablation_h{h}_seed{s}_2026-08-16_141447_gnnhar_qlike" /
                           "ladder_metrics.json"))["rungs"]["HAR"]["test_metrics"]
        for k in keys:
            vals[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def _dm_vs_har(ts: str, h: int, seeds) -> dict[str, Any]:
    import dm_report as dm
    from diebold_mariano import diebold_mariano
    dm.RESULTS = ROOT / "results"

    def ens(dirpat, rung):
        per = [dm._load_rows(ROOT / "results" / dirpat.format(s=s) / rung / "predictions_test.json")
               for s in seeds]
        keys = sorted(set(per[0]))
        return keys, np.array([per[0][k][0] for k in keys]), np.array([np.mean([p[k][1] for p in per]) for k in keys])

    kg, tg, pg = ens(f"volatility_glasso_node3_h{h}_seed{{s}}_{ts}", "GLASSO3")
    kh, th, ph = ens("volatility_ablation_h{s_h}_seed{s}_2026-08-16_141447_gnnhar_qlike".replace("{s_h}", str(h)), "P0")
    if kg != kh or not np.allclose(tg, th):
        return {"error": "misaligned"}
    r = diebold_mariano(dm._qlike(tg, pg), dm._qlike(tg, ph), h=h)
    return {"dm_hln": r.dm_hln, "p_value": r.p_value,
            "favors": "glasso3" if r.mean_diff < 0 else ("HAR" if r.mean_diff > 0 else "tie")}


def main(ts: str, seeds, epochs: int = 15, horizons=(1, 5, 10, 22)) -> None:
    device = resolve_graph_device("cuda")
    metrics = {}
    for h in horizons:
        rows = []
        for s in seeds:
            t0 = time.perf_counter()
            rows.append(run_horizon(h, s, epochs, ts, device))
            print(f"  (h{h} seed{s} {time.perf_counter() - t0:.0f}s)", flush=True)
        metrics[h] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("\n# 3-feature-node + glasso edge vs HAR(3-feature) — 3-seed mean test metrics")
    print(f"{'h':>3} {'model':10s} {'MSE':>10s} {'RMSE':>9s} {'MAE':>9s} {'R2':>7s} "
          f"{'QLIKE':>8s} {'DirAcc':>7s} | DM vs HAR")
    for h in horizons:
        har = _har_metrics(h, seeds)
        g3 = metrics[h]
        dm = _dm_vs_har(ts, h, seeds)
        dmc = dm.get("error") or (f"dm={dm['dm_hln']:+.2f} p={dm['p_value']:.3f}"
                                  f"{'*' if dm['p_value'] < 0.05 else ''} ({dm['favors']})")
        for name, m in (("HAR", har), ("glasso3", g3)):
            print(f"{h:>3} {name:10s} {m['mse']:10.2e} {m['rmse']:9.6f} {m['mae']:9.6f} {m['r2']:7.3f} "
                  f"{m['qlike']:8.4f} {m['directional_accuracy']:7.2f}" + (f" | {dmc}" if name == "glasso3" else ""))


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 123, 2026]
    _epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    _hz = tuple(int(x) for x in sys.argv[4:]) if len(sys.argv) > 4 else (1, 5, 10, 22)
    main(_ts, _seeds, _epochs, _hz)
