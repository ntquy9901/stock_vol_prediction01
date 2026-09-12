"""Causal per-fold feature-importance screen for the per-stock volatility target (Experiment B).

For a continuous target (``y`` = future Parkinson variance) the screen ranks every candidate feature --
the 3 HAR features, the 6 other own-history features, and the 6 network-topology ("graph") metrics -- with
the progression requested for volatility work::

    Variance  ->  Pearson / Spearman  ->  Mutual Information  ->  VIF

* **Pearson**  -- linear association with the target.
* **Spearman** -- monotone (rank) association.
* **Mutual information** -- arbitrary non-linear association (``mutual_info_regression``, kNN estimator).
* **VIF**      -- whether a feature's information is merely duplicated by the other features.

CRITICAL (no look-ahead): every statistic is estimated **separately on each walk-forward training fold**
(rows ``TRAIN_START <= date < fold_start - embargo``), never on the pooled 2015-2026 history. Aggregates are
the mean/std across folds. This mirrors exactly the causal train window used by the GBM so the screen speaks
to the features the model actually sees.

Run: ``python feature_screen.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_<market>_screen.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import run_gbm  # noqa: E402  (reuse _merge_topo + its import chain: FM, S1)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402

# feature -> group label (own-history HAR / own-history other / topology "graph")
GROUP = {**{c: "har" for c in FM.HAR},
         **{c: "own" for c in FM.OWN if c not in FM.HAR},
         **{c: "graph" for c in config.TOPO}}
FEATURES = list(GROUP)                                   # 3 HAR + 6 own + 6 topology = 15 candidates


def _vif(X):
    """Variance-inflation factor per column of standardized ``X`` (n x p).

    ``VIF_j = 1 / (1 - R^2_j)`` where ``R^2_j`` is the fit of feature ``j`` regressed on all others via least
    squares (intercept included). Columns with ~0 variance return ``inf`` (degenerate); a perfect fit is
    clipped so VIF stays finite.
    """
    n, p = X.shape
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
    out = np.empty(p)
    for j in range(p):
        if X[:, j].std() == 0:
            out[j] = np.inf
            continue
        others = np.column_stack([np.ones(n), np.delete(Z, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((Z[:, j] - Z[:, j].mean()) ** 2)
        out[j] = 1.0 / max(1.0 - r2, 1e-6)
    return out


def _fold_stats(tr):
    """Per-feature ``(pearson, spearman, mi, vif)`` on one training fold (``tr`` already filtered, causal).

    MI is estimated on at most ``config.SCREEN_MI_CAP`` rows (fixed-seed subsample) to bound the kNN cost.
    """
    X = tr[FEATURES].to_numpy(float)
    y = tr["y"].to_numpy(float)
    pear = np.array([pearsonr(X[:, j], y)[0] if X[:, j].std() else 0.0 for j in range(len(FEATURES))])
    spear = np.array([spearmanr(X[:, j], y)[0] if X[:, j].std() else 0.0 for j in range(len(FEATURES))])
    subsampled = len(y) > config.SCREEN_MI_CAP
    if subsampled:
        sel = np.random.default_rng(config.SCREEN_MI_SEED).choice(len(y), config.SCREEN_MI_CAP, replace=False)
        Xmi, ymi = X[sel], y[sel]
    else:
        Xmi, ymi = X, y
    mi = mutual_info_regression(Xmi, ymi, random_state=config.SCREEN_MI_SEED)
    return pear, spear, mi, _vif(X), subsampled


def _verdict(pear, spear, mi, vif):
    """Keep / drop recommendation from the aggregated (across-fold mean) statistics.

    ``drop (no signal)``  -- negligible under ALL of Pearson, Spearman and MI (feature is uninformative).
    ``review (redundant)``-- has signal but mean VIF exceeds the threshold (duplicated by other features).
    ``keep``              -- informative and not redundant.
    """
    no_signal = (abs(pear) < config.SCREEN_CORR_LO and abs(spear) < config.SCREEN_CORR_LO
                 and mi < config.SCREEN_MI_LO)
    if no_signal:
        return "drop (no signal)"
    if vif > config.SCREEN_VIF_HI:
        return "review (redundant)"
    return "keep"


def screen_horizon(a, horizon, min_rows):
    """Causal per-fold screen for one horizon. ``a`` is the Experiment-B panel (topology merged, target ``y``).

    Folds with fewer than ``min_rows`` causal training rows are skipped (``min_rows`` mirrors run_gbm's fold
    gate so the screened windows are exactly the windows the GBM fits). Returns
    ``{"n_folds", "mi_subsampled_folds", "features": {feat: {group, pearson, spearman, mi, vif (mean+std),
    verdict}}}``.
    """
    embargo = pd.Timedelta(days=int(horizon * 1.6) + 5)
    acc = {f: {"pear": [], "spear": [], "mi": [], "vif": []} for f in FEATURES}
    n_folds = 0
    n_subsampled = 0
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_rows:   # same gate as run_gbm: only score real GBM test folds
            continue
        pear, spear, mi, vif, subsampled = _fold_stats(tr)
        n_subsampled += int(subsampled)
        for i, f in enumerate(FEATURES):
            acc[f]["pear"].append(pear[i]); acc[f]["spear"].append(spear[i])
            acc[f]["mi"].append(mi[i]); acc[f]["vif"].append(vif[i])
        n_folds += 1
    feats = {}
    for f in FEATURES:
        m = {k: float(np.mean(v)) for k, v in acc[f].items()}
        s = {k: float(np.std(v)) for k, v in acc[f].items()}
        feats[f] = {"group": GROUP[f],
                    "pearson": m["pear"], "pearson_std": s["pear"],
                    "spearman": m["spear"], "spearman_std": s["spear"],
                    "mi": m["mi"], "mi_std": s["mi"],
                    "vif": m["vif"], "vif_std": s["vif"],
                    "verdict": _verdict(m["pear"], m["spear"], m["mi"], m["vif"])}
    return {"n_folds": n_folds, "mi_subsampled_folds": n_subsampled, "features": feats}


def run_screen(market, load_fn=None):
    """Full causal feature screen across all horizons; returns the JSON-serialisable result dict."""
    load_fn = load_fn or FM.load
    frames, _, _ = load_fn(market)
    frames = run_gbm._merge_topo(frames, market)
    min_rows = config.SCREEN_MIN_TRAIN_ROWS.get(market, config.SCREEN_MIN_TRAIN_ROWS["default"])
    out = {"market": market, "train_start": S1.TRAIN_START, "n_folds_max": len(S1.FOLDS) - 1,
           "min_train_rows": min_rows, "mi_subsample_cap": config.SCREEN_MI_CAP, "thresholds": {
               "corr_lo": config.SCREEN_CORR_LO, "mi_lo": config.SCREEN_MI_LO, "vif_hi": config.SCREEN_VIF_HI},
           "horizons": {}}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        for c in config.TOPO:
            a[c] = a.groupby("ticker")[c].ffill().fillna(0.0) if c in a.columns else 0.0
        out["horizons"][f"h{h}"] = screen_horizon(a, h, min_rows)
    return out


def _print(out):  # pragma: no cover - console formatting only
    for h, r in out["horizons"].items():
        print(f"\n=== feature screen {out['market']} {h} (folds={r['n_folds']}) ===", flush=True)
        print(f"{'feature':14s} {'grp':6s} {'pearson':>9s} {'spearman':>9s} {'MI':>8s} {'VIF':>8s}  verdict",
              flush=True)
        for f, v in sorted(r["features"].items(), key=lambda kv: -kv[1]["mi"]):
            print(f"{f:14s} {v['group']:6s} {v['pearson']:+9.4f} {v['spearman']:+9.4f} "
                  f"{v['mi']:8.4f} {v['vif']:8.2f}  {v['verdict']}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_screen(market)
    _print(out)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_screen.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
