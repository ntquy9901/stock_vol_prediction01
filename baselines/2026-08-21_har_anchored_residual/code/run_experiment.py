"""Orchestrate the HAR-anchored ladder E0-E10 for one (dataset, horizon) and write a full result.

Runs E0 HAR, E1/E2 full neural, E5/E6/E7 additive residual, E8 multiplicative residual (each ensembled
over 5 seeds), E3/E4 frozen-expert convex blends, E9 static gate, E10 dynamic gate; then all 5 metrics,
Diebold-Mariano vs HAR (per-obs + date-clustered), block-bootstrap CI, Model Confidence Set, and the
pre-gating diagnostics. Row-aligned predictions are exported to CSV. Designed to be launched once per
(dataset, horizon) so the grid runs in parallel.

CLI: python run_experiment.py <dataset> <horizon> [--lookback 10] [--smoke] [--data-root DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SUB = HERE.parents[2] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(HERE))

import baselines as B  # noqa: E402
import metrics as M  # noqa: E402
import pipeline_config as pc  # noqa: E402  (single source of truth for tunable constants)
from config import Config, SMOKE  # noqa: E402

import experts  # noqa: E402
import blend as BL  # noqa: E402
import gate as GT  # noqa: E402
import stats as ST  # noqa: E402
import diagnostics as DG  # noqa: E402
import io_preds as IO  # noqa: E402

REPO = HERE.parents[2]
NEURAL = {  # id -> (framing, use_lstm, use_graph)
    "E1": ("full", True, False), "E2": ("full", True, True),
    "E5": ("additive", True, False), "E6": ("additive", False, True),
    "E7": ("additive", True, True), "E8": ("mult", True, True),
}


def _ens(dicts):
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    keys = sorted(keys)
    return {k: (dicts[0][k][0], float(np.mean([d[k][1] for d in dicts]))) for k in keys}


def _train_ensemble(D, cfg, use_lstm, use_graph, framing):
    tes, vas, corrs, nparam = [], [], [], 0
    for s in cfg.seeds:
        te, va, _vq, nparam, corr = experts.train_neural(D, cfg, s, use_lstm, use_graph, framing)
        tes.append(te); vas.append(va); corrs.append(corr)
    corr_mean = {k: np.mean([c[k] for c in corrs], axis=0) for k in ("c_tr", "c_va", "c_te")}
    corr_mean["framing"] = framing
    return _ens(tes), _ens(vas), corr_mean, nparam


def _aligned(preds: dict, keys):
    y = np.array([preds[k][0] for k in keys]); p = np.array([preds[k][1] for k in keys])
    return y, p


def _metrics(y, p, har_p, floor):
    return {"mae": M.mae(y, p), "rmse": M.rmse(y, p), "qlike": M.qlike(y, p, floor),
            "r2": M.r2(y, p), "rel_r2_vs_har": DG.relative_r2_vs_har(y, p, har_p),
            "bias": float(np.mean(p - y)), "n": len(y)}


def run(dataset, files, lookback, horizon, cfg, out_dir, min_common=pc.MIN_COMMON_DATES):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = experts.build_data(files, lookback, horizon, cfg, min_common=min_common)
    print(f"[build] {dataset} h{horizon} N={D['N']} test_dates={len(D['d_te'])}", flush=True)

    preds_te = {"E0_HAR": experts.har_pred_dict(D, "test")}
    preds_va = {"E0_HAR": experts.har_pred_dict(D, "val")}
    corr = {}; params = {}
    for eid, (framing, ul, ug) in NEURAL.items():
        te, va, cbundle, nparam = _train_ensemble(D, cfg, ul, ug, framing)
        preds_te[eid], preds_va[eid], corr[eid], params[eid] = te, va, cbundle, nparam
        print(f"[done] {eid} ({framing}, lstm={ul}, graph={ug}) params={nparam}", flush=True)

    # E3/E4: frozen convex blend of HAR + full neural (E2). E4 = per-horizon alpha (this run's alpha).
    a3, preds_te["E3_blend"] = BL.blend(preds_te["E0_HAR"], preds_te["E2"],
                                        preds_va["E0_HAR"], preds_va["E2"], "qlike", cfg.qlike_floor)
    _, preds_va["E3_blend"] = BL.blend(preds_va["E0_HAR"], preds_va["E2"],
                                       preds_va["E0_HAR"], preds_va["E2"], "qlike", cfg.qlike_floor)
    # E9 static gate + E10 dynamic gate on the combined residual expert (E7)
    scale = D["add_scale"]; pfloor = D["pred_floor"]
    lam9 = GT.fit_lambda_static(D["y_va"], D["harp_va"], corr["E7"]["c_va"], scale, "additive",
                                D["eps"], cfg.qlike_floor, pred_floor=pfloor)
    p9 = GT.reconstruct("additive", D["harp_te"], corr["E7"]["c_te"], scale, D["eps"], lam9, pfloor)
    preds_te["E9_gate_static"] = {(j, D["d_te"][i]): (float(D["y_te"][i, j]), float(p9[i, j]))
                                  for i in range(len(D["d_te"])) for j in range(D["N"])}
    preds_te["E10_gate_dyn"], preds_va["E10_gate_dyn"], lam10 = GT.fit_gate_dynamic(
        D, corr["E7"], cfg, cfg.seeds[0], D["eps"])

    # GARCH baseline (per node, train+val series) for context
    gar = {}
    for j in range(D["N"]):
        series = np.concatenate([D["y_tr"][:, j], D["y_va"][:, j]])
        try:
            fc = B.garch_forecast(series, n_test=len(D["d_te"]), horizon=horizon, floor=cfg.qlike_floor)
        except Exception:
            fc = np.full(len(D["d_te"]), max(float(np.var(series)), cfg.qlike_floor))
        for i in range(len(D["d_te"])):
            gar[(j, D["d_te"][i])] = (float(D["y_te"][i, j]), float(fc[i]))
    preds_te["GARCH"] = gar

    # common test keys across all models
    keys = sorted(set.intersection(*[set(d) for d in preds_te.values()]))
    dates = np.array([k[1] for k in keys])
    y0, har_p = _aligned(preds_te["E0_HAR"], keys)
    metrics, dm, boot = {}, {}, {}
    loss_dict = {}
    for name, d in preds_te.items():
        y, p = _aligned(d, keys)
        metrics[name] = _metrics(y, p, har_p, cfg.qlike_floor)
        lq = M.per_obs_qlike(y, p, cfg.qlike_floor); loss_dict[name] = lq
        if name != "E0_HAR":
            lh = M.per_obs_qlike(y0, har_p, cfg.qlike_floor)
            dm[name] = {"dm_qlike": _safe(lambda: _dm_dict(M.diebold_mariano(lq, lh, h=horizon))),
                        "date_clustered": _safe(lambda: ST.date_clustered_dm(lq, lh, dates, horizon))}
            boot[name] = _safe(lambda: ST.block_bootstrap_ci(lq, lh, dates))
    mcs = _safe(lambda: ST.model_confidence_set(loss_dict, dates, alpha=0.10)) or {"mcs_set": [], "p_values": {}}

    # diagnostics (validation) — error complementarity, disagreement, residual R2_OOS on test
    yv0, harv = _aligned(preds_va["E0_HAR"], sorted(set(preds_va["E0_HAR"]) & set(preds_va["E2"])))
    _, nnv = _aligned(preds_va["E2"], sorted(set(preds_va["E0_HAR"]) & set(preds_va["E2"])))
    diag = {"error_complementarity_har_vs_E2": DG.error_complementarity(yv0, harv, nnv),
            "disagreement_winrate_E2": DG.disagreement_winrate(yv0, harv, nnv),
            "residual_r2_oos": {}}
    for eid in ("E5", "E6", "E7"):
        resid_pred = (scale * corr[eid]["c_te"]).reshape(-1)
        diag["residual_r2_oos"][eid] = DG.residual_r2_oos(D["y_te"].reshape(-1), D["harp_te"].reshape(-1),
                                                          resid_pred)

    # row-aligned export
    cols = {"node": np.array([k[0] for k in keys]), "target_date": dates,
            "horizon": np.full(len(keys), horizon), "y_true": y0}
    for name, d in preds_te.items():
        cols[f"pred_{name}"] = np.array([d[k][1] for k in keys])
    IO.export_predictions(out_dir / "row_predictions.csv", cols)

    result = {
        "dataset": dataset, "lookback": lookback, "horizon": horizon, "design": "snapshot+purge",
        "num_nodes": D["N"], "n_test": len(keys), "seeds": list(cfg.seeds),
        "primary_metric": "qlike", "metrics": metrics, "dm_vs_har": dm, "bootstrap_ci_vs_har": boot,
        "mcs": mcs, "alpha_E3": a3, "lambda_E9_static": lam9, "lambda_E10_mean": lam10,
        "param_counts": params, "diagnostics": diag, "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"[result] {dataset} h{horizon} QLIKE HAR={metrics['E0_HAR']['qlike']:.4f} "
          f"E7={metrics['E7']['qlike']:.4f} E8={metrics['E8']['qlike']:.4f} "
          f"E3={metrics['E3_blend']['qlike']:.4f} alpha={a3:.3f} mcs={mcs['mcs_set']} "
          f"{result['seconds']}s", flush=True)
    return result


def _dm_dict(r):
    return {"dm_hln": r.dm_hln, "p_value": r.p_value, "mean_diff": r.mean_diff, "n": r.n}


def _safe(fn):
    """Run a statistic; on a numerical failure (e.g. degenerate long-run variance) record it as null."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - statistics are best-effort, never fatal to the run
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("horizon", type=int)
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=str(_SUB / "data"))
    ap.add_argument("--batch", type=int, default=None, help="override batch_size (small for large-N graphs)")
    ap.add_argument("--min-common", type=int, default=300,
                    help="min common dates for the snapshot node set (raise for large universes to keep a longer test window)")
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else Config()
    if a.batch:
        from dataclasses import replace
        cfg = replace(cfg, batch_size=a.batch)
    root = Path(a.data_root)
    mp = {"vn30": [root / "vn30" / "*_processed.csv", root / "*_processed.csv"],
          "vn100": [root / "vn100" / "*_processed.csv", root / "vn100_vnstock" / "*_processed.csv"],
          "sp500": [root / "sp500" / "*_processed.csv"]}
    files = next((glob.glob(str(p)) for p in mp[a.dataset] if glob.glob(str(p))), [])
    if not files:
        raise FileNotFoundError(f"no files for {a.dataset} under {a.data_root}")
    out = REPO / "results" / "har_anchored" / f"{a.dataset}_h{a.horizon}"
    run(a.dataset, files, a.lookback, a.horizon, cfg, out, min_common=a.min_common)


if __name__ == "__main__":
    main()
