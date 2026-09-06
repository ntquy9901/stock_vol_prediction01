"""Build the summary / ablation / limit-lock robustness tables (Markdown) and a permutation-importance
figure from the ``results/xgboost_residual/xgb_<market>_h<h>.json`` artifacts.

Table builders are pure functions over loaded result dicts (unit-tested); the figure + CLI reload data
and draw with matplotlib (thin glue). Importance is computed on the VALIDATION split of one fold only
(explanation, not feature selection) per the guide.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

MODELS = ("HAR", "HAR-X", "XGB_direct", "XGB_resid_base", "XGB_resid_market", "XGB_resid_graph")


def load_results(results_dir, markets=("vn30", "vn100"), horizons=(1, 5, 10, 22)):
    """Load available ``xgb_<market>_h<h>.json`` into ``{(market, horizon): dict}`` (skips missing)."""
    out = {}
    for m in markets:
        for h in horizons:
            p = Path(results_dir) / f"xgb_{m}_h{h}.json"
            if p.exists():
                out[(m, h)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _dmp(res, key):
    dd = res.get("dm_date_clustered", {}).get(key, {}).get("qlike", {})
    return dd.get("p_value"), dd.get("favors")


def summary_table(results):
    """Markdown QLIKE summary: HAR / HAR-X / direct / residual ladder + VolGA(ref) + DM(base vs HAR-X)."""
    rows = ["| Panel | h | HAR-X | XGB-direct | XGB-resid-base | +market | +graph | VolGA(ref) | "
            "DM base vs HAR-X (p, favors) | base ticker/date win |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, h) in sorted(results):
        r = results[(m, h)]
        mt = r["metrics"]
        p, fav = _dmp(r, "XGB_resid_base_vs_HAR-X")
        win = r.get("win_rates_vs_harx", {}).get("XGB_resid_base", {})
        rows.append(
            f"| {m} | {h} | {_fmt(mt['HAR-X']['qlike'])} | {_fmt(mt['XGB_direct']['qlike'])} | "
            f"{_fmt(mt['XGB_resid_base']['qlike'])} | {_fmt(mt['XGB_resid_market']['qlike'])} | "
            f"{_fmt(mt['XGB_resid_graph']['qlike'])} | {_fmt(r.get('reference_models', {}).get('VolGA_qlike'))} | "
            f"{_fmt(p, 3)}, {fav} | {_fmt(win.get('ticker_win_rate'), 2)}/{_fmt(win.get('date_win_rate'), 2)} |")
    return "\n".join(rows)


def ablation_table(results):
    """Markdown feature-group ablation: base/market/graph QLIKE + DM(market vs base), DM(graph vs market)."""
    rows = ["| Panel | h | base | +market | +graph | DM market vs base (p,fav) | DM graph vs market (p,fav) |",
            "|---|---|---|---|---|---|---|"]
    for (m, h) in sorted(results):
        r = results[(m, h)]
        mt = r["metrics"]
        pm, fm = _dmp(r, "XGB_resid_market_vs_base")
        pg, fg = _dmp(r, "XGB_resid_graph_vs_market")
        rows.append(f"| {m} | {h} | {_fmt(mt['XGB_resid_base']['qlike'])} | {_fmt(mt['XGB_resid_market']['qlike'])} | "
                    f"{_fmt(mt['XGB_resid_graph']['qlike'])} | {_fmt(pm, 3)}, {fm} | {_fmt(pg, 3)}, {fg} |")
    return "\n".join(rows)


def lock_table(results, model="XGB_resid_base"):
    """Markdown limit-lock robustness for one model: standard / non-lock / lock-only QLIKE + share + counts."""
    rows = [f"| Panel | h | QLIKE (all) | non-lock | lock-only | lock QLIKE share | n_lock | "
            f"QLIKE excl top-1% dates | (model={model}) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (m, h) in sorted(results):
        mt = results[(m, h)]["metrics"][model]
        rows.append(f"| {m} | {h} | {_fmt(mt['qlike'])} | {_fmt(mt['qlike_nonlock'])} | {_fmt(mt['qlike_lockonly'], 3)} | "
                    f"{_fmt(mt['lock_qlike_share'], 3)} | {mt['n_lock']} | {_fmt(mt['qlike_excl_top_dates'])} | |")
    return "\n".join(rows)


def go_no_go(results):
    """Mechanical GO/NO-GO tally for HAR-X+XGB residual-base (guide Stage 12): count panel/horizons where
    residual-base has strictly lower standard QLIKE than HAR-X AND is DM-significant (p<0.05, favors A)."""
    lower = sig = total = 0
    for r in results.values():
        total += 1
        mt = r["metrics"]
        if mt["XGB_resid_base"]["qlike"] < mt["HAR-X"]["qlike"]:
            lower += 1
        p, fav = _dmp(r, "XGB_resid_base_vs_HAR-X")
        if p is not None and p < 0.05 and fav == "A":
            sig += 1
    verdict = "GO" if (lower > 1 and sig >= 1) else "NO-GO"
    return {"n_panel_horizon": total, "n_lower_qlike_than_harx": lower,
            "n_dm_significant_better": sig, "verdict": verdict}


def build_report(results, date_str):  # pragma: no cover - assembles tested tables into a markdown string
    parts = [f"# XGBoost residual-ratio volatility study — results ({date_str})", "",
             "## Summary (standard QLIKE, Parkinson variance)", "", summary_table(results), "",
             "## Feature-group ablation", "", ablation_table(results), "",
             "## Limit-lock robustness (XGB_resid_base)", "", lock_table(results), "",
             "## GO/NO-GO tally", "", "```", json.dumps(go_no_go(results), indent=2), "```"]
    return "\n".join(parts)


def importance_figure(market, horizon, out_png):  # pragma: no cover - retrains one fold + matplotlib I/O
    """Permutation importance of the +graph residual model on ONE fold's validation split (explanation
    only). Saved as a grouped bar figure. Retrains a single XGBoost (CPU)."""
    import matplotlib
    matplotlib.use("Agg")
    import glob as _glob

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.inspection import permutation_importance

    import run_xgboost_residual as R
    files = _glob.glob(R.enriched_glob(market))
    keep = R.frozen_universe(files, R.C.LOOKBACK, horizon)
    panel = R.build_enriched_panel(files, R.C.LOOKBACK, horizon, keep)
    aligned = R.read_aligned_columns(files, panel, ["daily_return", "log_range", "zero_range_flag"])
    Fsm, names, groups = R.build_sm_features(panel, aligned["daily_return"], aligned["log_range"],
                                             panel.feats[:, :, 4])
    wf = R.VolgaWFConfig(lookback=R.C.LOOKBACK, horizon=horizon, folds_target=R.C.FOLDS_TARGET)
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    fold = R.make_folds(n, ts, K, wf.val, wf.horizon)[-1]
    fl = R.training_config().qlike_floor
    D = R.pack_fold(panel, fold, wf.lookback, wf.horizon)
    nfloor = R.pc.POS_FLOOR_FRAC * D.t_mean + R.pc.POS_FLOOR_EPS
    _, harx = R._har_ols_preds(D, fl, nfloor)
    last_tr = int(panel.anchors[fold.train][-1]) + horizon
    adj = R.directed_vol2pk_hmatched(panel.feats[:, :, 4], np.sqrt(panel.pk), last_tr, horizon, R.C.EDGE_TOP_K)
    G, gnames = R.build_graph_features(panel, aligned["daily_return"], adj)
    Fall = np.concatenate([Fsm, G], axis=2)
    all_names = names + gnames
    cols = R.feature_column_indices(groups, G.shape[2])["XGB_resid_graph"]
    mtr, mva = D.tmask_tr.astype(bool), D.tmask_va.astype(bool)
    oof = R.oof_harx(D.har5_tr, D.y_tr, mtr, fl, nfloor)
    z = R.residual_target(D.y_tr, oof)
    rmask = mtr & np.isfinite(oof) & np.isfinite(z)
    xtr = R.design_rows(Fall, panel.anchors[fold.train], rmask, cols)
    xva = R.design_rows(Fall, panel.anchors[fold.val], mva, cols)
    zva = R.residual_target(D.y_va, harx["va"])[mva]
    br = R.tune_residual(xtr, z[rmask], xva, zva, harx["va"][mva], D.y_va[mva],
                         np.broadcast_to(nfloor, mva.shape)[mva], fl)
    imp = permutation_importance(br["model"], np.nan_to_num(xva), zva, n_repeats=5,
                                 random_state=R.C.XGB_SEED, n_jobs=R.C.XGB_N_JOBS)
    sel = [all_names[i] for i in cols]
    order = np.argsort(imp.importances_mean)[-20:]
    plt.figure(figsize=(7, 8))
    plt.barh([sel[i] for i in order], imp.importances_mean[order])
    plt.xlabel("permutation importance (val, residual model)")
    plt.title(f"XGB_resid_graph feature importance — {market} h{horizon} (fold {fold.idx}, val split)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()
    return out_png


def main():  # pragma: no cover - CLI driver
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results" / "xgboost_residual"))
    ap.add_argument("--date", default="unset")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--figure", action="store_true", help="also draw the permutation-importance figure")
    a = ap.parse_args()
    results = load_results(a.results_dir)
    if not results:
        raise SystemExit(f"no xgb result JSONs under {a.results_dir}")
    report = build_report(results, a.date)
    out = Path(a.report_out) if a.report_out else REPO / "docs" / "reports" / f"{a.date}_xgboost_residual_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[summarize] wrote {out}")
    if a.figure:
        png = importance_figure("vn100", 1, REPO / "results" / "xgboost_residual" / "importance_vn100_h1.png")
        print(f"[summarize] wrote {png}")


if __name__ == "__main__":  # pragma: no cover
    main()
