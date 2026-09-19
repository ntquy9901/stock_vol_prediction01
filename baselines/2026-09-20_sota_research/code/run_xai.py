"""Exp 1 --- Explainable AI (TreeSHAP) for the VolTree gamma-XGBoost.

Reuses the exact VolTree pipeline (walk-forward folds, embargo, OWN-8 + earnings features, reg:gamma
booster) from ``baselines/2026-09-18_leaf_graph_paper`` and adds exact TreeSHAP attribution on the test
rows of every fold. Produces, per (market, horizon): global mean|SHAP| per feature, mean signed SHAP,
and the share of total attribution from HAR lags vs momentum vs earnings --- a rigorous, model-faithful
alternative to permutation importance (which had a known placebo pitfall in this project).

SHAP is exact (tree_path_dependent), deterministic (seed 0), leak-safe (fit on train-minus-validation,
explain the held-out test window). Attribution is in the reg:gamma margin (log) space; only relative
magnitudes are interpreted. Output: results/gamma_gbm/xai_shap_<market>.json + returns the dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
# Reuse the VolTree runner's module (path bootstrap + FM/S1/LG/C + OWN-8 + resolve_cols live there).
sys.path.insert(0, str(REPO / "baselines" / "2026-09-18_leaf_graph_paper" / "code"))
import run_leaf_graph_paper as RLG  # noqa: E402

FM, S1, LG, C = RLG.FM, RLG.S1, RLG.LG, RLG.C
FL = FM.FL
SHAP_SAMPLE = 6000          # test rows sampled per fold for global mean|SHAP| (stable; exact SHAP is O(n))


def _feature_group(name: str) -> str:
    """Bucket a feature name into HAR / momentum / earnings for the attribution-share summary."""
    n = name.lower()
    if n.startswith("har") or "har_" in n:
        return "HAR"
    if n.startswith("earn"):
        return "earnings"
    return "momentum"


def shap_global(bst, X: np.ndarray, cols: list[str]) -> dict:
    """Exact TreeSHAP global attribution for one fitted booster on rows X, via XGBoost's native
    ``pred_contribs=True`` (exact tree_path_dependent SHAP; avoids the shap-lib base_score parsing bug on
    XGBoost 3.x). Returns per-feature mean|SHAP| and mean signed SHAP (margin space) + group shares."""
    import xgboost as xgb
    contribs = bst.predict(xgb.DMatrix(np.asarray(X, float)), pred_contribs=True)   # (n, d+1); last col = bias
    sv = contribs[:, :-1]                                 # (n, d) exact per-feature SHAP
    mabs = np.abs(sv).mean(axis=0)                         # global importance per feature
    signed = sv.mean(axis=0)
    total = float(mabs.sum()) or 1.0
    per_feat = {c: {"mean_abs": float(mabs[j]), "mean_signed": float(signed[j]),
                    "share": float(mabs[j] / total)} for j, c in enumerate(cols)}
    groups: dict[str, float] = {}
    for j, c in enumerate(cols):
        groups[_feature_group(c)] = groups.get(_feature_group(c), 0.0) + float(mabs[j] / total)
    return {"per_feature": per_feat, "group_share": groups, "n_rows": int(len(X))}


def run_market(market: str, horizons=None):  # pragma: no cover - data-driven driver over the walk-forward
    """Fit VolTree per fold, TreeSHAP on each test window, aggregate global attribution per horizon."""
    horizons = horizons or C.HORIZONS
    frames, _sect, edates = FM.load(market)
    edates = RLG._load_earn(market, edates)
    cols, use_earn = RLG.resolve_cols("full", bool(edates))
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out = {"market": market, "features": cols, "use_earn": bool(use_earn), "horizons": {}}
    for h in horizons:
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        acc_abs = np.zeros(len(cols)); acc_signed = np.zeros(len(cols)); n_tot = 0
        group_acc: dict[str, float] = {}
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            trf_e = trf[~trf["date"].isin(val_dates)]
            bst = LG.fit_booster(trf_e, cols, seed=0)
            Xte = tef[cols].to_numpy(float)
            if len(Xte) > SHAP_SAMPLE:                    # sample test rows: global mean|SHAP| is stable
                idx = np.random.default_rng(0).choice(len(Xte), SHAP_SAMPLE, replace=False)
                Xte = Xte[idx]
            g = shap_global(bst, Xte, cols)
            w = g["n_rows"]; n_tot += w
            for j, c in enumerate(cols):
                acc_abs[j] += g["per_feature"][c]["mean_abs"] * w
                acc_signed[j] += g["per_feature"][c]["mean_signed"] * w
            for grp, s in g["group_share"].items():
                group_acc[grp] = group_acc.get(grp, 0.0) + s * w
        if n_tot == 0:
            continue
        mabs = acc_abs / n_tot
        tot = float(mabs.sum()) or 1.0
        ranked = sorted(((c, float(mabs[j])) for j, c in enumerate(cols)), key=lambda t: -t[1])
        out["horizons"][str(h)] = {
            "n_test_rows": int(n_tot),
            "importance": {c: {"mean_abs": float(mabs[j]), "share": float(mabs[j] / tot),
                               "mean_signed": float(acc_signed[j] / n_tot)} for j, c in enumerate(cols)},
            "ranking": [c for c, _ in ranked],
            "group_share": {g: float(v / n_tot) for g, v in group_acc.items()},
        }
    return out


def main():  # pragma: no cover - entry driver: runs both markets, writes JSON
    outdir = REPO / "results" / "gamma_gbm"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for market in ("sp500", "hose"):
        res = run_market(market)
        (outdir / f"xai_shap_{market}.json").write_text(json.dumps(res, indent=1))
        summary[market] = res
        print(f"[xai] {market}: horizons {list(res['horizons'])}")
        for h, hd in res["horizons"].items():
            print(f"  h{h}: top3 {hd['ranking'][:3]} | groups {hd['group_share']}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
