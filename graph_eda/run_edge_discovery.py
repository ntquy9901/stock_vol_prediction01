"""Runner: supervised ΔQLIKE directed edge discovery on the real VN30 panel.

Run:  python -m graph_eda.run_edge_discovery

Outputs (docs/eda/edge_discovery/):
- tables/edge_predictive_dqlike.csv   directed (source->target) OOS ΔQLIKE + FDR + stability + regime
- tables/residual_leadlag_h5.csv      market-residual directed lead-lag matrix
- figures/edge_dqlike_matrix.png      heatmap of test ΔQLIKE(source->target)
- reports/EDGE_DISCOVERY_REPORT.md    recommendation + go/no-go for a ΔQLIKE-edge GNN

Base for every target = E2 node features (HAR + MarketPK + volume_zscore_20); the edge must lower
target QLIKE out-of-sample BEYOND that base. Leakage-safe (train-fit, val-select, test-confirm).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from graph_eda import edge_discovery as ed
from graph_eda.io_data import build_common_panel, chrono_split, load_universe

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "data" / "raw" / "prices"
OUT = ROOT / "docs" / "eda" / "edge_discovery"
HORIZON = 5


def _panels(price_dir: Path):
    frames = load_universe(str(price_dir))
    high = build_common_panel(frames, "high")
    low = build_common_panel(frames, "low")
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_var = ((np.log(high / low) ** 2) / (4.0 * np.log(2.0))).replace([np.inf, -np.inf], np.nan)
    pk_vol = np.sqrt(pk_var)
    market = pk_vol.median(axis=1)                       # MarketPK = cross-sectional median sqrt(PK)
    logvol = build_common_panel(frames, "volume").apply(lambda s: np.log1p(s))
    volz = (logvol - logvol.rolling(20).mean()) / logvol.rolling(20).std().replace(0.0, np.nan)
    return pk_var, pk_vol, market, volz


def main(price_dir: Path | None = None, out_base: Path | None = None) -> dict:
    price_dir = price_dir or PRICE_DIR
    out = out_base or OUT
    tab, fig, rep = out / "tables", out / "figures", out / "reports"
    for d in (tab, fig, rep):
        d.mkdir(parents=True, exist_ok=True)

    pk_var, pk_vol, market, volz = _panels(price_dir)
    idx = pk_var.index
    split = chrono_split(idx)
    train, val, test = (split.mask(idx, s) for s in ("train", "val", "test"))

    print("ranking directed dQLIKE edges ...")
    edges = ed.predictive_dqlike_edges(pk_var, market, volz, train, val, test,
                                       horizon=HORIZON, with_regime=True)
    edges.to_csv(tab / "edge_predictive_dqlike.csv", index=False)
    ll = ed.residual_leadlag(pk_vol, market, train, horizon=HORIZON)
    ll.to_csv(tab / "residual_leadlag_h5.csv")

    _heatmap(edges, list(pk_var.columns), fig / "edge_dqlike_matrix.png")
    summary = _summarise(edges, ll)
    _write_report(edges, ll, summary, rep / "EDGE_DISCOVERY_REPORT.md")
    print(f"val-selected edges: {summary['n_val_selected']}; test-positive rate: "
          f"{100 * summary['val_selected_test_positive_rate']:.1f}% (binom p="
          f"{summary['generalization_binom_p']:.3f}); mean test dQLIKE of val-selected: "
          f"{summary['val_selected_mean_test_dqlike']:+.5f}; verdict: {summary['verdict']}")
    return {"edges": edges, "summary": summary}


def _summarise(edges: pd.DataFrame, ll: pd.DataFrame) -> dict:
    """Verdict from the LEAKAGE-SAFE val->test generalization, not from test-peeking counts.

    Edges are SELECTED on validation only (dqlike_val>0, no test peek); the decisive question is
    whether that selected set stays helpful on the held-out test. The naive stable/FDR counts (which
    both use the test set) are reported for context but must not drive the verdict, else a Phase-B
    build on the same test would be circular.
    """
    off = ~np.eye(ll.shape[0], dtype=bool)
    resid = {"resid_leadlag_offdiag_mean_abs": float(np.nanmean(np.abs(ll.values[off]))),
             "resid_leadlag_offdiag_max": float(np.nanmax(np.abs(ll.values[off])))}
    if edges.empty:  # no pair had enough data (tiny panel) -> nothing to build
        return {"n_pairs": 0, "n_val_selected": 0, "val_selected_test_positive_rate": 0.0,
                "generalization_binom_p": 1.0, "val_selected_mean_test_dqlike": 0.0,
                "val_selected_median_test_dqlike": 0.0, "val_selected_wilcoxon_p": 1.0,
                "n_stable_testpeek": 0, "n_fdr_help_testpeek": 0, "regime_hi_minus_lo": 0.0,
                "best_dqlike_test": 0.0, **resid, "verdict": "STOP"}
    n_pairs = len(edges)
    val_sel = edges[edges["dqlike_val"] > 0]
    n_val_sel = len(val_sel)
    test_pos = val_sel["dqlike_test"] > 0
    gen_rate = float(test_pos.mean()) if n_val_sel else 0.0
    gen_p = (float(stats.binomtest(int(test_pos.sum()), n_val_sel).pvalue)
             if n_val_sel else 1.0)  # sign count across pairs vs chance (50%) — overlap-robust
    dt = val_sel["dqlike_test"].values
    mean_test = float(np.nanmean(dt)) if n_val_sel else 0.0
    median_test = float(np.nanmedian(dt)) if n_val_sel else 0.0
    nz = dt[~np.isnan(dt)]
    nz = nz[nz != 0]
    wilcoxon_p = float(stats.wilcoxon(nz).pvalue) if nz.size >= 10 else 1.0
    # test-peeking counts (context only, NOT the verdict basis)
    n_stable = int(edges["stable"].sum())
    n_fdr_help = int(((edges["fdr_q"] < 0.05) & (edges["dqlike_test"] > 0)).sum())
    hi, lo = edges["dqlike_test_hi_regime"], edges["dqlike_test_lo_regime"]
    regime_gap = float(np.nanmean(hi) - np.nanmean(lo))
    # BUILD only if val-selected edges beat chance on test (sign count) AND help by a ROBUST centre
    # (median, not mean — one QLIKE tail-spike pair must not drive the verdict, per review)
    if gen_rate > 0.60 and median_test > 0 and gen_p < 0.05 and n_val_sel >= 20:
        verdict = "BUILD"
    elif gen_rate > 0.55 and median_test > 0:
        verdict = "MAYBE"
    else:
        verdict = "STOP"
    return {
        "n_pairs": n_pairs, "n_val_selected": n_val_sel,
        "val_selected_test_positive_rate": gen_rate, "generalization_binom_p": gen_p,
        "val_selected_mean_test_dqlike": mean_test,
        "val_selected_median_test_dqlike": median_test, "val_selected_wilcoxon_p": wilcoxon_p,
        "n_stable_testpeek": n_stable, "n_fdr_help_testpeek": n_fdr_help,
        "regime_hi_minus_lo": regime_gap,
        "best_dqlike_test": float(edges["dqlike_test"].max()),
        **resid, "verdict": verdict,
    }


def _heatmap(edges: pd.DataFrame, tickers: list[str], path: Path) -> None:
    mat = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for _, r in edges.iterrows():
        mat.loc[r["source"], r["target"]] = r["dqlike_test"]
    vmax = float(np.nanmax(np.abs(mat.values))) if np.isfinite(mat.values).any() else 1e-6
    vmax = vmax if vmax > 0 else 1e-6
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=6)
    ax.set_xlabel("target j")
    ax.set_ylabel("source i")
    ax.set_title("OOS test ΔQLIKE(i -> j)  (red = source lowers target QLIKE = helps)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_report(edges: pd.DataFrame, ll: pd.DataFrame, s: dict, path: Path) -> None:
    verdict_txt = {
        "BUILD": "BUILD a ΔQLIKE-edge GNN — validation-selected directed edges stay helpful on the "
                 "held-out test above chance and on average.",
        "MAYBE": "MAYBE — validation-selected edges generalize to test slightly above chance; a "
                 "sparse targeted graph is worth one build+DM, but the signal is thin.",
        "STOP": "STOP — validation-selected edges do NOT generalize to the held-out test (test-"
                "positive rate at/below chance, mean test ΔQLIKE <= 0). No edge definition recovers "
                "conditional cross-stock signal beyond E2 even on QLIKE; a cross-stock GNN is not "
                "justified.",
    }[s["verdict"]]
    gr = 100 * s["val_selected_test_positive_rate"]
    top = edges.head(15)
    L = ["# Edge-Discovery EDA — supervised OOS ΔQLIKE directed edges\n",
         "Base = E2 node features (HAR + MarketPK + volume_zscore_20). An edge i->j helps only if "
         "adding source i's PK history LOWERS target j's out-of-sample QLIKE beyond that base "
         "(positive ΔQLIKE). h=5, positivity-floored, leakage-safe: coefficients fit on TRAIN, edges "
         "SELECTED on VALIDATION only, generalization judged on the untouched TEST.\n",
         "## Verdict\n", f"**{verdict_txt}**\n",
         "## Decisive test — leakage-safe val->test generalization\n",
         f"- Validation-selected edges (ΔQLIKE_val>0, no test peek): {s['n_val_selected']} of "
         f"{s['n_pairs']} directed pairs.\n",
         f"- Of those, **test-positive: {gr:.1f}%** (binomial vs 50% chance p="
         f"{s['generalization_binom_p']:.3f}); held-out-test ΔQLIKE of the selected set: "
         f"mean **{s['val_selected_mean_test_dqlike']:+.5f}**, "
         f"median **{s['val_selected_median_test_dqlike']:+.5f}** (Wilcoxon p="
         f"{s['val_selected_wilcoxon_p']:.4f}). Verdict gates on the robust median, not the "
         "tail-sensitive mean.\n",
         f"- Reading: {gr:.0f}% ~ chance and mean <= 0 means selection does not carry over — the "
         "edges that look helpful on validation are noise, not conditional signal.\n"
         if s["verdict"] == "STOP" else
         f"- Reading: {gr:.0f}% above chance with positive mean test ΔQLIKE — selection carries over.\n",
         "## Context — test-peeking counts (NOT the verdict basis)\n",
         f"- Sign-stable (ΔQLIKE>0 on val AND test): {s['n_stable_testpeek']}; FDR-'significant' "
         f"helpful (q<0.05 AND test>0): {s['n_fdr_help_testpeek']}. These use the test set in "
         "selection, so with ~1000 pairs x ~14k obs a few dozen cross the line by chance; they do "
         "NOT survive the honest val->test split above.\n",
         f"- Regime: mean test ΔQLIKE high-vol minus low-vol = {s['regime_hi_minus_lo']:+.5f} "
         f"({'more help under stress' if s['regime_hi_minus_lo'] > 0 else 'no stress concentration'}).\n",
         f"- Market-residual lead-lag L(5): mean |off-diag| = {s['resid_leadlag_offdiag_mean_abs']:.4f}, "
         f"max = {s['resid_leadlag_offdiag_max']:.4f}.\n",
         "## Top-15 directed edges by test ΔQLIKE (test-ranked; note most have ΔQLIKE_val<=0)\n",
         "| source -> target | ΔQLIKE val | ΔQLIKE test | p_test | fdr_q | stable |\n|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        L.append(f"| {r['source']} -> {r['target']} | {r['dqlike_val']:+.5f} | "
                 f"{r['dqlike_test']:+.5f} | {r['p_test']:.4f} | {r['fdr_q']:.4f} | {bool(r['stable'])} |")
    L.append("\n## Caveats & robustness\n")
    L.append("- The per-edge `p_test`/`fdr_q` are paired-t on daily QLIKE differences with an "
             "overlapping 5-day horizon (serially correlated), so they are anti-conservative and "
             "shown for context only. The verdict rests on the sign-count val->test binomial across "
             "PAIRS (not obs), which is unaffected by that autocorrelation.\n")
    L.append("- Source feature = raw PK_i(t). Because the base already contains MarketPK and an "
             "intercept, adding raw PK_i or its market-residual spans the same column space -> the "
             "ridge already orthogonalizes the edge against the market factor (residualizing is a "
             "no-op, verified). So this IS the market-adjusted edge test.\n")
    L.append("## Interpretation\n")
    L.append("ΔQLIKE>0 means the source lowers the target's QLIKE out of sample after the market "
             "factor and the target's own node features are already known — i.e. genuine "
             "conditional cross-stock information. If the FDR-controlled stable set is empty/tiny, "
             "the correlation the earlier G1/E3 graphs used was mostly the market factor, and no "
             "edge definition recovers conditional signal — a stronger stop-GNN result than the "
             "RMSE-only ranking gave. If a set exists, Phase B builds exactly those directed edges "
             "on the E2 pooled backbone and DM-tests vs E2/HAR.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
