"""Experiment A (PRIMARY): predict the FUTURE market-index volatility from the network-topology metrics.

Faithful replication of the paper's Section 2.4 / Figure 1 on our data: features = the topology metrics of
a rolling combined return+volume network; targets = the future L-day index volatility (headline), average
return and average log volume; models = LinearRegression + RandomForest; score = OOS R^2 and RMSE under a
causal expanding walk-forward with an L-day embargo between train targets and test features. Reports the
per-window ticker count N (Refinement 2) and per-metric feature importance (Refinement 4).

Run: ``python run_index.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_index_<market>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
# local modules first: FM's import chain prepends submission/soict_lstm_gat (which has its OWN config.py)
# to sys.path, so importing these before FM binds `config` to THIS baseline's config.py.
import config  # noqa: E402
import topology  # noqa: E402
import market_index  # noqa: E402
import full_matrix as FM  # noqa: E402  (also preloads `metrics`/`stats` into sys.modules via its import chain)

TARGETS = ["idx_vol", "idx_ret", "idx_lnvol"]
MODELS = {"LinearRegression": LinearRegression,
          "RandomForest": lambda: RandomForestRegressor(**config.RF_KW)}


def build_sample(frames, market, idx, alpha, win):
    """Assemble the supervised sample S and the per-window ticker count.

    Returns ``(S, n_by_window)``: ``S`` has one row per window with the topology features joined to the
    strictly-future index targets, NaN-dropped and sorted by window-end date ``d0``; ``n_by_window`` is the
    common-ticker count Series from :func:`topology.build_topo_windows`.
    """
    F, n_by_window = topology.build_topo_windows(frames, market, alpha, win)
    if F.empty:
        empty = pd.DataFrame(columns=["d0"] + config.TOPO + TARGETS + ["target_end"])
        return empty, n_by_window
    T = market_index.future_targets(idx, F.index, config.L)
    S = F.join(T, how="inner").dropna(subset=config.TOPO + TARGETS)
    return S.reset_index(names="d0").sort_values("d0").reset_index(drop=True), n_by_window


def _feature_importance(model):
    """Per-feature importance vector: RF ``feature_importances_`` (non-negative) or the standardized Linear
    Regression coefficient (signed). ``model`` is fit on the StandardScaler-transformed training matrix, so
    ``coef_`` is already the per-1-SD (standardized) effect; no extra rescaling is applied."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return np.asarray(model.coef_, dtype=float)


def walk_forward(S, feat_cols, target_col, model_ctor):
    """Causal expanding walk-forward. For each test window ``j`` (ordered by ``d0``), train only on windows
    whose target is fully observed by ``d0_j`` (``target_end <= d0_j`` — the L-day embargo); skip until at
    least ``MIN_TRAIN_WINDOWS`` such windows exist. Scaler fit on train only. Returns aligned
    ``(y, yhat, train_r2, n_train_final, imp_mean)`` where ``imp_mean`` is the mean per-feature importance over
    the scored folds (length ``len(feat_cols)``; all-NaN when no fold scored)."""
    d0 = S["d0"].to_numpy()
    tend = S["target_end"].to_numpy()
    X = S[feat_cols].to_numpy(float)
    y = S[target_col].to_numpy(float)
    ys, yhats, train_r2s, imps = [], [], [], []
    n_train_final = 0
    for j in range(len(S)):
        mask = tend <= d0[j]                                   # fully-observed past targets only (embargo)
        n_tr = int(mask.sum())
        if n_tr < config.MIN_TRAIN_WINDOWS:
            continue
        sc = StandardScaler().fit(X[mask])
        x_tr = sc.transform(X[mask])
        model = model_ctor()
        model.fit(x_tr, y[mask])
        yhats.append(float(model.predict(sc.transform(X[j:j + 1]))[0]))
        train_r2s.append(float(model.score(x_tr, y[mask])))
        imps.append(_feature_importance(model))
        ys.append(float(y[j]))
        n_train_final = n_tr
    imp_mean = np.mean(imps, axis=0) if imps else np.full(len(feat_cols), np.nan)
    return np.array(ys), np.array(yhats), np.array(train_r2s), n_train_final, imp_mean


def _oos(ys, yhats):
    """OOS (R^2, RMSE) for aligned arrays, or (nan, nan) when empty."""
    if len(ys) == 0:
        return float("nan"), float("nan")
    return float(r2_score(ys, yhats)), float(np.sqrt(np.mean((ys - yhats) ** 2)))


def fit_verdict(train_r2, r2_oos):
    """Overfit/underfit verdict for the fit-diagnostics evidence gate."""
    if not (np.isfinite(train_r2) and np.isfinite(r2_oos)):
        verdict = "underfit"                                   # no scored windows -> cannot demonstrate fit
    elif train_r2 - r2_oos > 0.30:
        verdict = "overfit"
    elif train_r2 < 0.1 and r2_oos < 0.1:
        verdict = "underfit"
    else:
        verdict = "ok"
    return {"verdict": verdict, "train_r2": train_r2, "test_r2": r2_oos}


def evaluate_sample(S):
    """Headline metrics (per model x target, incl. per-metric feature importance) and fit-diagnostics."""
    headline = {m: {} for m in MODELS}
    diags = {}
    for mname, ctor in MODELS.items():
        for tgt in TARGETS:
            ys, yhats, tr2, n_tr, imp = walk_forward(S, config.TOPO, tgt, ctor)
            r2o, rmse = _oos(ys, yhats)
            tr2m = float(np.mean(tr2)) if len(tr2) else float("nan")
            fi = {c: float(v) for c, v in zip(config.TOPO, imp)}
            headline[mname][tgt] = {"r2_oos": r2o, "rmse_oos": rmse, "train_r2": tr2m,
                                    "n_test": int(len(ys)), "n_train_final": int(n_tr),
                                    "feature_importance": fi}
            diags[f"{mname}_{tgt}"] = fit_verdict(tr2m, r2o)
    return headline, diags


def _rob_eval(S):
    """RF+LR OOS r2/rmse on the headline target (idx_vol) for a robustness variant."""
    res = {}
    for mname, ctor in MODELS.items():
        ys, yhats, _, _, _ = walk_forward(S, config.TOPO, "idx_vol", ctor)
        r2o, rmse = _oos(ys, yhats)
        res[mname] = {"r2_oos": r2o, "rmse_oos": rmse}
    return res


def robustness(frames, market, idx):
    """Alpha-grid (at WIN) and 6-month-window (WIN_ROBUST at ALPHA) robustness on the headline target."""
    out = {"alpha_grid": {}, "win132": {}}
    for alpha in config.ALPHA_GRID:
        S, _ = build_sample(frames, market, idx, alpha, config.WIN)
        out["alpha_grid"][str(alpha)] = _rob_eval(S)
    S132, _ = build_sample(frames, market, idx, config.ALPHA, config.WIN_ROBUST)
    out["win132"] = _rob_eval(S132)
    return out


def _n_stats(n_by_window):
    """min/median/max of the per-window ticker count (None-filled when no windows)."""
    vals = np.asarray(pd.Series(n_by_window, dtype=float).dropna(), dtype=float)
    if vals.size == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": int(vals.min()), "median": float(np.median(vals)), "max": int(vals.max())}


def run_index(market, load_fn=None, index_fn=None):
    """Full Experiment A for a market; returns the JSON-serialisable result dict (no file write)."""
    load_fn = load_fn or FM.load
    index_fn = index_fn or market_index.load_index
    frames, _, _ = load_fn(market)
    idx = index_fn(market)
    S, n_by_window = build_sample(frames, market, idx, config.ALPHA, config.WIN)
    headline, diags = evaluate_sample(S)
    rob = robustness(frames, market, idx)
    n_used = n_by_window.reindex(S["d0"]) if len(S) else n_by_window
    return {"market": market, "n_windows": int(len(S)),
            "n_tickers_per_window": _n_stats(n_used),
            "headline": {"win": config.WIN, "alpha": config.ALPHA, "L": config.L, **headline},
            "fit_diagnostics": diags, "robustness": rob}


def _top_importances(fi, k=3):  # pragma: no cover - console formatting only
    return ", ".join(f"{m}={v:+.3f}" for m, v in sorted(fi.items(), key=lambda kv: -abs(kv[1]))[:k])


def _print_table(result):  # pragma: no cover - console formatting only
    npw = result["n_tickers_per_window"]
    print(f"\n=== Experiment A: {result['market']} (n_windows={result['n_windows']}, "
          f"N per window min/med/max={npw['min']}/{npw['median']}/{npw['max']}) ===", flush=True)
    print(f"{'model':18s} {'target':10s} {'r2_oos':>9s} {'rmse_oos':>11s} {'train_r2':>9s} {'n_test':>7s}",
          flush=True)
    for mname in MODELS:
        for tgt in TARGETS:
            r = result["headline"][mname][tgt]
            print(f"{mname:18s} {tgt:10s} {r['r2_oos']:9.4f} {r['rmse_oos']:11.6f} "
                  f"{r['train_r2']:9.4f} {r['n_test']:7d}", flush=True)
        r = result["headline"][mname]["idx_vol"]
        print(f"    idx_vol top features [{mname}]: {_top_importances(r['feature_importance'])}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    result = run_index(market)
    _print_table(result)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_index_{market}.json"
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
