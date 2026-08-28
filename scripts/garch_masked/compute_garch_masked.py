"""Add a GARCH(1,1) baseline to the masked-panel results ON THE SAME PANEL as the delivered
HAR/LSTM/LSTM+GAT tables (``results/masked_rich_floor1e2/<ds>_h<h>/result.json``).

The panel builder ``masked_rich.build_masked_rich`` is deterministic (no seed in the panel), so
rebuilding it here reproduces the identical test masks / dates / per-node scalers used by the
delivered runner. We (1) recompute HAR-X and assert its QLIKE matches the stored value (basis
guard), then (2) fit a per-node GARCH(1,1) on that node's train Parkinson-variance series and
forecast its test anchors, applying the SAME per-node positivity floor (1e-2*t_mean) and the same
QLIKE floor (cfg.qlike_floor=1e-8). Metrics + date-clustered DM(GARCH vs HAR-X) are written back
into each result.json under ``metrics['GARCH']`` and ``dm_date_clustered['GARCH_vs_HAR']``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
_SUB = REPO / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(_CODE))

import baselines as B  # noqa: E402
import masked_rich as MR  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
from config import Config  # noqa: E402

try:  # provenance for the GARCH benchmark (external review L-03)
    from importlib.metadata import version as _pkg_version
    _ARCH_VERSION = _pkg_version("arch")
except Exception:  # pragma: no cover - arch is installed; guard only for a broken environment
    _ARCH_VERSION = "unavailable"

DATA = _SUB / "data"
FILES = {
    "vn30": [DATA / "vn30" / "*_processed.csv", DATA / "*_processed.csv"],
    "vn100": [DATA / "vn100" / "*_processed.csv", DATA / "vn100_vnstock" / "*_processed.csv"],
}
PRICE = {
    "vn30": REPO / "data" / "raw" / "prices",
    "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
}
HORIZONS = (1, 5, 10, 22)


def _pred_dict(pred, y, tmask, dates, N):
    out = {}
    for i in range(len(dates)):
        for j in range(N):
            if tmask[i, j]:
                out[(j, dates[i])] = (float(y[i, j]), float(pred[i, j]))
    return out


def _metrics(pred, floor):
    ks = sorted(pred)
    y = np.array([pred[k][0] for k in ks])
    p = np.array([pred[k][1] for k in ks])
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor), "r2": M.r2(y, p), "n": len(ks)}


def _dm(a, b, horizon, floor):
    ks = sorted(set(a) & set(b))
    y = np.array([a[k][0] for k in ks])
    pa = np.array([a[k][1] for k in ks])
    pb = np.array([b[k][1] for k in ks])
    dates = np.array([k[1] for k in ks])
    out = {}
    fams = {"qlike": (M.per_obs_qlike(y, pa, floor), M.per_obs_qlike(y, pb, floor)),
            "se": ((y - pa) ** 2, (y - pb) ** 2), "ae": (np.abs(y - pa), np.abs(y - pb))}
    for name, (la, lb) in fams.items():
        r = ST.date_clustered_dm(la, lb, dates, horizon)
        out[name] = {"p_value": r["p_value"], "mean_diff": r["mean_diff"],
                     "favors": "A" if r["mean_diff"] < 0 else "B"}
    return out


def _garch_pred(D, horizon, cfg, status_out=None):
    """Per-node GARCH(1,1): fit on TRAIN-ONLY variance series, forecast the TEST window.

    The masked panel has a validation block between train and test (masked_rich splits train/val/test).
    A frozen-parameter GARCH forecast path runs from the end of train, so the test targets sit ``n_va``
    steps further out. We forecast ``n_va + n_te`` steps and keep the LAST ``n_te`` (the test window),
    skipping the validation interval, so each test target gets a forecast aligned to its distance from
    train-end (Codex review 2026-08-22, finding 2). GARCH mean-reverts, so this uses the mean-reverted
    (test-window) part of the path rather than the transient early steps.

    On the ``n_va`` observation-count offset (external review 2026-08-28, H-02): the GARCH series is the
    node's masked train-target OBSERVATIONS, so the forecast recursion steps in observation units, not
    calendar days. The node's own validation observations are exactly the ``n_va`` steps that precede its
    test observations in that same series, so ``fc_full[n_va:]`` is the observation-consistent alignment.
    A calendar-day offset would instead MIS-match the (irregularly-spaced) observation-indexed path. This
    is an inherent approximation of applying GARCH to a masked panel; GARCH is a dominated benchmark and
    the step-alignment does not change its ranking.
    """
    N = D.N
    pred = np.zeros_like(D.y_te)
    floor_node = 1e-2 * D.t_mean + 1e-12
    for j in range(N):
        tr_mask = D.tmask_tr[:, j].astype(bool)
        te_mask = D.tmask_te[:, j].astype(bool)
        n_te = int(te_mask.sum())
        if n_te == 0:
            continue
        n_va = int(D.tmask_va[:, j].sum())          # validation anchors between train and test for this node
        series = D.y_tr[tr_mask, j]
        fc_full, st = B.garch_forecast(series, n_test=n_va + n_te, horizon=horizon,
                                       floor=cfg.qlike_floor, seed=getattr(cfg, "seed", 42),
                                       return_status=True)   # R-03: use the cfg seed that garch_meta records
        if status_out is not None:
            status_out.append(st)                    # M-08: record per-node GARCH fallback status
        fc = fc_full[n_va:]                          # skip the validation interval -> the test window
        pred[te_mask, j] = np.maximum(fc, floor_node[j])
    return _pred_dict(pred, D.y_te, D.tmask_te, D.d_te, N)


def garch_meta(status_list, horizon, cfg):
    """Aggregate GARCH degradation + provenance for the result artifact (external review M-06/M-08/L-03).

    ``status_list`` are per-node status dicts from ``_garch_pred(status_out=...)``. Reports the fallback
    rate + reason counts, whether the ``arch`` benchmark actually fit, and enough provenance
    (seed/horizon/floor/arch version) to reproduce."""
    n = len(status_list)
    n_fb = sum(1 for s in status_list if s.get("fallback"))
    reasons: dict = {}
    for s in status_list:
        if s.get("fallback"):
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    return {
        "schema": 1,
        "n_nodes": n,
        "n_fallback": n_fb,
        "fallback_rate": (n_fb / n) if n else 0.0,
        "fallback_reasons": reasons,
        "arch_available": bool(status_list and status_list[0].get("arch_available")),
        "degraded": n_fb == n and n > 0,             # every node fell back -> benchmark did not fit
        "horizon": horizon,
        "seed": getattr(cfg, "seed", 42),
        "qlike_floor": cfg.qlike_floor,
        "arch_version": _ARCH_VERSION,
    }


def _harx_pred(D, cfg):
    mtr = D.tmask_tr.astype(bool)
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]
    xte = D.har5_te.reshape(-1, 5)
    hx = (np.column_stack([np.ones(len(xte)), xte]) @ cx).reshape(D.y_te.shape)
    hx = np.maximum(hx, 1e-2 * D.t_mean + 1e-12)
    return _pred_dict(hx, D.y_te, D.tmask_te, D.d_te, D.N)


def main():  # pragma: no cover - CLI entry driver; exercised by real OOS runs, not unit tests
    cfg = Config()
    summary = []
    for ds in ("vn30", "vn100"):
        files = next((glob.glob(str(p)) for p in FILES[ds] if glob.glob(str(p))), [])
        price_dir = str(PRICE[ds])
        for h in HORIZONS:
            rp = REPO / "results" / "masked_rich_floor1e2" / f"{ds}_h{h}" / "result.json"
            res = json.loads(rp.read_text(encoding="utf-8"))
            D = MR.build_masked_rich(files, price_dir, cfg.lookback, h)
            harx = _harx_pred(D, cfg)
            mh = _metrics(harx, cfg.qlike_floor)
            stored = res["metrics"]["HAR-X"]["qlike"]
            assert abs(mh["qlike"] - stored) < 1e-4, f"basis mismatch {ds} h{h}: {mh['qlike']} vs {stored}"
            st_list = []
            g = _garch_pred(D, h, cfg, status_out=st_list)
            mg = _metrics(g, cfg.qlike_floor)
            dmg = _dm(g, harx, h, cfg.qlike_floor)
            res["metrics"]["GARCH"] = mg
            res["dm_date_clustered"]["GARCH_vs_HARX"] = dmg   # compared vs HAR-X (the paper's 5-feature "HAR")
            res["garch_meta"] = garch_meta(st_list, h, cfg)   # M-06/M-08/L-03: degradation + provenance
            rp.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
            row = (ds, h, mg["mse"], mg["rmse"], mg["mae"], mg["qlike"], mg["r2"])
            summary.append(row)
            print(f"[garch] {ds} h{h:<2} N={D.N} n={mg['n']} | MSE={mg['mse']:.6e} RMSE={mg['rmse']:.6e} "
                  f"MAE={mg['mae']:.6e} QLIKE={mg['qlike']:.4f} R2={mg['r2']:.4f} "
                  f"| DM(G vs HAR) qlike p={dmg['qlike']['p_value']:.3f} favors={dmg['qlike']['favors']}", flush=True)
    print("\n=== GARCH summary (raw scale) ===")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
