"""Experiment-B diagnostic: WHY do the topology metrics fail to help the per-stock gamma-GBM?

For each horizon, on the FINAL walk-forward fold, three pieces of evidence are computed and later rendered to
HTML (:mod:`build_topo_diag_html`):

1. **Permutation importance (QLIKE)** of all 16 features of the GBM+topo model. Permuting a feature and
   measuring the QLIKE increase shows how much the model relies on it. Own-history (HAR + realized) features
   carry the signal; the topology metrics move QLIKE by ~0.
2. **Prediction / error correlation** between GBM (own 9) and GBM+topo (own+topology). A correlation near 1 means the
   topology features barely change the forecast; the small delta they add is noise, not signal.
3. **Train vs test QLIKE** for both models. Topology lowers the TRAIN QLIKE (extra capacity fits in-sample
   noise) without lowering the TEST QLIKE, the signature of adding uninformative features.

Run: ``python diag_gbm.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_<market>_diag.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import run_gbm  # noqa: E402  (reuse _merge_topo + its import chain: FM, S1, metrics)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402

FL = FM.FL
PERM_N = 50_000        # cap the test rows used for permutation importance (predict cost bound)
PERM_REPEATS = 3       # shuffles per feature


def _fit(tr, cols, seed=0):
    """Fit one gamma-GBM (same hyperparameters as ``FM.gbm``) on the training rows for ``cols``."""
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(tr[cols].to_numpy(float), np.maximum(tr["y"].to_numpy(float), FL))
    return m


def _qlike_scorer(est, X, y):
    """Higher-is-better scorer for permutation importance: negative mean per-observation QLIKE."""
    p = np.maximum(est.predict(X), FL)
    return -float(np.mean(M.per_obs_qlike(y, p, floor=FL)))


def _last_fold(a, h):
    """The last non-empty (train, test) split from the S1 fold schedule with the horizon embargo, else None."""
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    last = None
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) and len(tr):
            last = (tr, te)
    return last


def _perm_importance(model, te, cols, seed=0):
    """QLIKE permutation importance per feature (mean over ``PERM_REPEATS`` shuffles), test rows capped at
    ``PERM_N``. Value = mean QLIKE increase when the feature is shuffled (>0 means the model uses it)."""
    sub = te if len(te) <= PERM_N else te.sample(PERM_N, random_state=seed)
    X = sub[cols].to_numpy(float)
    y = np.maximum(sub["y"].to_numpy(float), FL)
    r = permutation_importance(model, X, y, scoring=_qlike_scorer, n_repeats=PERM_REPEATS, random_state=seed)
    return {c: float(v) for c, v in zip(cols, r.importances_mean)}


def _train_qlike(model, tr, cols):
    """In-sample (train) mean QLIKE for a fitted model."""
    p = np.maximum(model.predict(tr[cols].to_numpy(float)), FL)
    return float(np.mean(M.per_obs_qlike(np.maximum(tr["y"].to_numpy(float), FL), p, floor=FL)))


def run_diag(market, load_fn=None):
    """Build the per-horizon diagnostic dict for a market (no file write)."""
    load_fn = load_fn or FM.load
    min_rows = 30_000 if market == "sp500" else 3_000
    frames, _, _ = load_fn(market)
    frames = run_gbm._merge_topo(frames, market)
    own, topo = FM.OWN, config.TOPO
    allc = own + topo
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        for c in topo:
            a[c] = a.groupby("ticker")[c].ffill().fillna(0.0) if c in a.columns else 0.0
        fold = _last_fold(a, h)
        if fold is None or len(fold[0]) < min_rows:
            continue
        tr, te = fold
        m_own = _fit(tr, own)
        m_topo = _fit(tr, allc)
        p_own = np.maximum(m_own.predict(te[own].to_numpy(float)), FL)
        p_topo = np.maximum(m_topo.predict(te[allc].to_numpy(float)), FL)
        y = np.maximum(te["y"].to_numpy(float), FL)
        e_own = M.per_obs_qlike(y, p_own, floor=FL)
        e_topo = M.per_obs_qlike(y, p_topo, floor=FL)
        imp = _perm_importance(m_topo, te, allc)
        own_imp = float(sum(imp[c] for c in own))
        topo_imp = float(sum(imp[c] for c in topo))
        denom = own_imp + topo_imp
        out[f"h{h}"] = {
            "n_test": int(len(te)),
            "perm_importance": imp,
            "own_importance_sum": own_imp,
            "topo_importance_sum": topo_imp,
            "topo_importance_frac": (topo_imp / denom) if denom > 0 else 0.0,
            "pred_corr": float(np.corrcoef(p_own, p_topo)[0, 1]),
            "qlike_err_corr": float(np.corrcoef(e_own, e_topo)[0, 1]),
            "train_qlike": {"GBM": _train_qlike(m_own, tr, own), "GBM+topo": _train_qlike(m_topo, tr, allc)},
            "test_qlike": {"GBM": float(np.mean(e_own)), "GBM+topo": float(np.mean(e_topo))},
        }
    return out


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_diag(market)
    for h, r in out.items():
        print(f"{market} {h}: topo_importance_frac={r['topo_importance_frac']:.3f} "
              f"pred_corr={r['pred_corr']:.4f} "
              f"train dQLIKE={r['train_qlike']['GBM'] - r['train_qlike']['GBM+topo']:+.4f} "
              f"test dQLIKE={r['test_qlike']['GBM'] - r['test_qlike']['GBM+topo']:+.4f}", flush=True)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_diag.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"saved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
