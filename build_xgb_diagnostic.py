"""Diagnostic panels for the PAPER model — XGBoost ``reg:gamma`` on the OWN-8 own-history feature set — on the
S&P 500 TEST set: WHERE does the model still lose, and is own-history exhausted? Self-contained diagnostic on a
single chronological split (fast; qualitative, not the headline walk-forward).

This replaces the stale ``build_gbm_diagnostic.py`` (HistGradientBoostingRegressor, 11-feature) so the three
paper motivation figures come from the ACTUAL paper model:
  * model  = XGBoost gamma booster (params single-sourced from ``leaf_graph_config``, fit via ``leaf_graph.fit_booster``);
  * features = OWN-8 own-history only (``full_matrix.OWN`` minus rq via the paper ``own_set``) — NO market_pk / volume / rq;
  * HAR reference = OLS on the three HAR lags (matches the paper's 3-lag HAR).

Writes THREE PNG panels directly into ``docs/paper/figures/`` under the existing names:
  * fig_remaining_error_by_decile.png  (A per-decile QLIKE XGB vs HAR + B where the remaining error is)
  * fig_forecast_bias_by_decile.png    (C median forecast/realized by decile)
  * fig_feature_importance.png         (E manual gamma-deviance permutation importance over the 8 OWN-8 features)

Run: .venv_gpu_encode/Scripts/python.exe build_xgb_diagnostic.py
"""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parent
for _p in (str(_REPO / "scripts" / "eda"),
           str(_REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(_REPO / "baselines" / "2026-09-18_gbm_leaf_graph" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import vn_gbm_graph_stage1 as S1  # noqa: E402  (canonical own-history feature builder _feat)
import full_matrix as FM  # noqa: E402          (canonical OWN list + FL floor, single source)
import leaf_graph_config as C  # noqa: E402      (XGB gamma params, single source)
import leaf_graph as LG  # noqa: E402            (fit_booster / predict_booster — same train pattern as the paper)


def _own8():
    """OWN-8 own-history feature list, single-sourced from the paper_models config (loaded by path so it does
    NOT register a second bare ``config`` module)."""
    cfg_path = _REPO / "baselines" / "2026-09-13_paper_models" / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("paper_models_config", cfg_path)
    pmc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmc)
    return pmc.own_set(FM.OWN)


OWN = _own8()                                  # 8 own-history features (paper model input)
HAR3 = ["har_daily", "har_weekly", "har_monthly"]
FL = FM.FL                                      # QLIKE / positivity floor (single source)
TRAIN_START, TEST_START = "2018-01-01", "2024-10-18"   # match the original diagnostic split
FIGDIR = _REPO / "docs" / "paper" / "figures"
XGB_LABEL, HAR_LABEL = "XGBoost (gamma)", "HAR"


def qlike(y, f):
    y = np.maximum(y, FL); f = np.maximum(f, FL); r = y / f; return r - np.log(r) - 1.0


def load(market="sp500_clean", h=5):  # pragma: no cover - data loader (globs the enriched CSV panel from disk)
    rows = []
    for p in glob.glob(str(_REPO / "data" / "processed_enriched" / market / "*.csv")):
        if p.endswith("_rejections.csv"):
            continue
        d = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        d = S1._feat(d); d["y"] = d["parkinson_variance"].shift(-h); d["ticker"] = Path(p).stem
        rows.append(d[OWN + ["date", "y", "ticker", "parkinson_variance"]])
    return pd.concat(rows, ignore_index=True).dropna(subset=OWN + ["y"])


def fit(market, h):
    d = load(market, h)
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)          # mirror the headline walk-forward embargo: the last
    tr = d[(d["date"] >= TRAIN_START) & (d["date"] < pd.Timestamp(TEST_START) - embargo)].copy()  # ~h train tails
    te = d[d["date"] >= TEST_START].copy()                 # carry test-window targets -> gap them out (no leak)
    # HAR reference: OLS on the 3 HAR lags (matches the paper's 3-lag HAR baseline).
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c] for c in HAR3])
    Xte = np.column_stack([np.ones(len(te))] + [te[c] for c in HAR3])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].to_numpy(float), rcond=None)
    te["har"] = np.maximum(Xte @ beta, FL)
    # Paper model: XGBoost gamma booster (same fit pattern / params as leaf_graph.fit_booster).
    bst = LG.fit_booster(tr, OWN, seed=0)
    te["xgb"] = LG.predict_booster(bst, te[OWN].to_numpy(float))
    te["dec"] = pd.qcut(te["y"].to_numpy(float), 10, labels=False, duplicates="drop")
    assert te["dec"].nunique() == 10, "expected 10 equal-count deciles (continuous realized variance)"
    return te, bst


def fig_decile(te, h):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    decs = range(10)
    qh = [qlike(te[te.dec == d].y.values, te[te.dec == d].har.values).mean() for d in decs]
    qg = [qlike(te[te.dec == d].y.values, te[te.dec == d].xgb.values).mean() for d in decs]
    axs[0].bar([x - 0.2 for x in decs], qh, 0.4, label=HAR_LABEL, color="tab:gray")
    axs[0].bar([x + 0.2 for x in decs], qg, 0.4, label=XGB_LABEL, color="tab:red")
    axs[0].set_xticks(list(decs)); axs[0].set_xlabel("realized-vol decile (D0 calm .. D9 storm)")
    axs[0].set_ylabel("QLIKE"); axs[0].legend()
    axs[0].set_title(f"A. h{h} per-decile QLIKE ({XGB_LABEL} vs {HAR_LABEL})")
    contrib = [q / sum(qg) * 100 for q in qg]                      # each decile's share of the XGB QLIKE total
    axs[1].bar(list(decs), contrib, color=["tab:red" if c > 12 else "tab:blue" for c in contrib])
    axs[1].set_xticks(list(decs)); axs[1].set_xlabel("decile")
    axs[1].set_ylabel(f"% of {XGB_LABEL} total QLIKE")
    axs[1].set_title("B. Where the REMAINING error is")
    fig.savefig(FIGDIR / "fig_remaining_error_by_decile.png", dpi=110, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_remaining_error_by_decile.pdf", bbox_inches="tight"); plt.close(fig)
    return qh, qg, contrib


def fig_bias(te, h):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    decs = range(10)
    bh = [np.median(te[te.dec == d].har.values / np.maximum(te[te.dec == d].y.values, FL)) for d in decs]
    bg = [np.median(te[te.dec == d].xgb.values / np.maximum(te[te.dec == d].y.values, FL)) for d in decs]
    ax.plot(list(decs), bh, "o-", color="tab:gray", label=HAR_LABEL)
    ax.plot(list(decs), bg, "o-", color="tab:red", label=XGB_LABEL)
    ax.axhline(1, color="k", lw=.8); ax.set_yscale("log"); ax.set_xticks(list(decs))
    ax.set_xlabel("decile"); ax.set_ylabel("median forecast / realized (>1 over, <1 under)")
    ax.legend(); ax.set_title(f"C. h{h} forecast bias by decile — over-forecast calm, under-forecast storm?")
    fig.savefig(FIGDIR / "fig_forecast_bias_by_decile.png", dpi=110, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_forecast_bias_by_decile.pdf", bbox_inches="tight"); plt.close(fig)
    return bg


def _gamma_deviance(y, pred):
    """Mean gamma deviance (same shape sklearn's neg_mean_gamma_deviance scores), floored predictions."""
    pred = np.clip(pred, FL, C.PRED_CAP); r = y / pred
    return 2.0 * np.mean(r - np.log(r) - 1.0)


def fig_importance(te, bst, h):
    sub = te.sample(min(20000, len(te)), random_state=0)
    X = sub[OWN].to_numpy(float); y = np.maximum(sub["y"].to_numpy(float), FL)
    base = _gamma_deviance(y, LG.predict_booster(bst, X))
    rng = np.random.RandomState(0)
    imp = np.zeros(len(OWN))
    for j in range(len(OWN)):
        reps = []
        for _ in range(3):                                        # 3 permutation repeats, averaged
            Xs = X.copy(); Xs[:, j] = rng.permutation(Xs[:, j])
            reps.append(_gamma_deviance(y, LG.predict_booster(bst, Xs)) - base)
        imp[j] = float(np.mean(reps))
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([OWN[i] for i in order], imp[order], color="tab:green")
    ax.set_xlabel("permutation importance (increase in gamma deviance)")
    ax.set_title(f"E. h{h} feature importance — what {XGB_LABEL} relies on")
    fig.savefig(FIGDIR / "fig_feature_importance.png", dpi=110, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_feature_importance.pdf", bbox_inches="tight"); plt.close(fig)
    return imp


def main():  # pragma: no cover - entry driver: fits on the full S&P 500 panel and writes the three figures
    h = 5
    te, bst = fit("sp500_clean", h)
    ov_h = qlike(te.y.values, te.har.values).mean(); ov_g = qlike(te.y.values, te.xgb.values).mean()
    print(f"h{h} overall QLIKE: {HAR_LABEL} {ov_h:.4f} vs {XGB_LABEL} {ov_g:.4f} "
          f"({(ov_h - ov_g) / ov_h * 100:+.2f}%)  n_test={len(te)}")
    qh, qg, contrib = fig_decile(te, h)
    bg = fig_bias(te, h)
    imp = fig_importance(te, bst, h)

    print("\nPer-decile QLIKE (D0 calm .. D9 storm):")
    print("  decile :", " ".join(f"{d:>8d}" for d in range(10)))
    print(f"  {HAR_LABEL:<7}:", " ".join(f"{v:8.4f}" for v in qh))
    print(f"  {'XGB':<7}:", " ".join(f"{v:8.4f}" for v in qg))
    print(f"  XGB<HAR:", " ".join(f"{'Y' if qg[d] < qh[d] else 'n':>8}" for d in range(10)))
    print("  %oftotal:", " ".join(f"{v:8.2f}" for v in contrib))

    print("\nStorm-decile forecast bias (median XGB forecast / realized; <1 = under-forecast):")
    print("  decile :", " ".join(f"{d:>8d}" for d in range(10)))
    print("  ratio  :", " ".join(f"{v:8.3f}" for v in bg))

    print("\nPermutation importance ranked (largest first):")
    for rank, i in enumerate(np.argsort(imp)[::-1], 1):
        print(f"  {rank:>2}. {OWN[i]:<12} {imp[i]:.6f}")
    print("\nWrote:", ", ".join(n for n in ("fig_remaining_error_by_decile.png",
          "fig_forecast_bias_by_decile.png", "fig_feature_importance.png")))


if __name__ == "__main__":  # pragma: no cover
    main()
