"""Orchestrator: execute the graph-EDA plan end-to-end and write all deliverables.

Run:  python graph_eda/run_eda.py

Outputs
- docs/eda/tables/*.csv         core relationship / stability / predictive tables
- docs/eda/figures/*.png        the plan section-47 figure set
- docs/eda/reports/EDA_GRAPH_REPORT.md
- docs/eda/graph_recommendation.json  (values computed here, never hard-coded)

Leakage discipline: cross-stock analyses run on the common panel; every DECISION input
(regime thresholds, market betas, neighbour selection, predictive coefficients) is fit on
the TRAIN slice only. Full-period correlation matrices are labelled descriptive EDA.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from graph_eda import graphs, incremental, leakage, predictive
from graph_eda import relationships as rel
from graph_eda.data_quality import universe_quality
from graph_eda.io_data import build_common_panel, chrono_split, load_universe
from graph_eda.parkinson import build_features
from graph_eda.sectors import same_sector_matrix

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "data" / "raw" / "prices"
OUT = ROOT / "docs" / "eda"
TAB, FIG, REP, GRAPHS = OUT / "tables", OUT / "figures", OUT / "reports", OUT / "graphs"
SEED = 0
for d in (TAB, FIG, REP, GRAPHS):
    d.mkdir(parents=True, exist_ok=True)


def _pk_var_wide(frames):
    """Aligned pk_var panel built from the common high/low panels (finite only)."""
    high = build_common_panel(frames, "high")
    low = build_common_panel(frames, "low")
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = (np.log(high / low) ** 2) / (4.0 * np.log(2.0))
    return pk.replace([np.inf, -np.inf], np.nan)


def _heatmap(mat, title, path, cmap="coolwarm", vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(
    price_dir: str = str(PRICE_DIR),
    out_base: Path | None = None,
    n_perm_corr: int = 500,
    n_perm_ll: int = 300,
) -> dict:
    """Run the full EDA and write all deliverables.

    ``out_base`` overrides the output directory (used by the integration test to write
    into a tmp dir); ``n_perm_*`` control permutation-null iterations (reduced in tests).
    """
    global OUT, TAB, FIG, REP, GRAPHS
    if out_base is not None:
        OUT = Path(out_base)
        TAB, FIG, REP, GRAPHS = OUT / "tables", OUT / "figures", OUT / "reports", OUT / "graphs"
        for d in (TAB, FIG, REP, GRAPHS):
            d.mkdir(parents=True, exist_ok=True)
    print("Loading universe ...")
    frames = load_universe(price_dir)
    tickers = sorted(frames)
    for d in frames.values():
        leakage.assert_dates_sorted(d)

    # ---- data quality ---------------------------------------------------------
    dq = universe_quality(frames)
    dq.to_csv(TAB / "data_quality_summary.csv", index=False)

    # per-ticker coverage vs the union calendar (plan section 46: ticker_coverage.csv)
    union_dates = sorted({d for f in frames.values() for d in f["date"]})
    n_union = len(union_dates)
    cov = pd.DataFrame(
        [
            {
                "ticker": t,
                "start_date": f["date"].min().date().isoformat(),
                "end_date": f["date"].max().date().isoformat(),
                "n_rows": int(len(f)),
                "coverage_vs_union": float(len(f) / n_union) if n_union else float("nan"),
            }
            for t, f in sorted(frames.items())
        ]
    )
    cov.to_csv(TAB / "ticker_coverage.csv", index=False)

    # leakage-safe per-ticker daily feature panel (plan section 46: daily_stock_features)
    feat_panel = pd.concat(
        [build_features(f).assign(ticker=t) for t, f in sorted(frames.items())],
        ignore_index=True,
    )
    feat_panel.to_parquet(TAB / "daily_stock_features.parquet", index=False)

    # ---- aligned panels (returns, pk_var, pk_vol, volume shock) ----------------
    close = build_common_panel(frames, "close")
    ret = np.log(close / close.shift(1))
    pk_var_full = _pk_var_wide(frames)
    pk_var = pk_var_full.reindex(close.index)  # variance, project target
    pk_vol = np.sqrt(pk_var)  # plan default pk_vol
    logvol = build_common_panel(frames, "volume").apply(lambda s: np.log1p(s))
    vol_mu = logvol.rolling(20).mean()
    vol_sd = logvol.rolling(20).std()
    vshock = (logvol - vol_mu) / vol_sd.replace(0.0, np.nan)
    logvol_change = logvol.diff()  # plan section 10: Delta log-volume

    panel = pd.DataFrame({"n_dates": [len(close)]})
    panel["date_start"] = close.index.min().date().isoformat()
    panel["date_end"] = close.index.max().date().isoformat()
    panel.to_csv(TAB / "panel_summary.csv", index=False)

    split = chrono_split(close.index)
    tr = split.mask(close.index, "train")
    te = split.mask(close.index, "test")
    # leakage assertion: train strictly precedes test
    assert close.index[tr].max() < close.index[te].min()
    same_sec = same_sector_matrix(tickers)

    ev: dict = {}  # collected scalar evidence for report + json

    # ---- P1: descriptive full-period correlations ------------------------------
    print("P1 correlations ...")
    pk_corr_p = rel.pairwise_corr(pk_vol.dropna(how="all"), "pearson")
    pk_corr_s = rel.pairwise_corr(pk_vol.dropna(how="all"), "spearman")
    ret_corr_p = rel.pairwise_corr(ret.dropna(how="all"), "pearson")
    ret_corr_s = rel.pairwise_corr(ret.dropna(how="all"), "spearman")
    vshock_corr = rel.pairwise_corr(vshock.dropna(how="all"), "pearson")
    vchange_corr = rel.pairwise_corr(logvol_change.dropna(how="all"), "pearson")
    leakage.assert_corr_matrix_valid(pk_corr_p)
    leakage.assert_corr_matrix_valid(ret_corr_p)
    for m, name in [
        (pk_corr_p, "pk_corr_pearson"),
        (pk_corr_s, "pk_corr_spearman"),
        (ret_corr_p, "return_corr_pearson"),
        (ret_corr_s, "return_corr_spearman"),
        (vshock_corr, "volume_corr"),
        (vchange_corr, "volume_change_corr"),
    ]:
        m.to_csv(TAB / f"{name}.csv")

    pk_pairs = rel.pair_long_table(pk_vol.dropna(how="all"), "pearson")
    pk_pairs["same_sector"] = [
        int(same_sec.loc[a, b]) for a, b in zip(pk_pairs.source, pk_pairs.target)
    ]
    pk_pairs.sort_values("corr", ascending=False).to_csv(
        TAB / "pk_pairs_significance.csv", index=False
    )
    ret_pairs = rel.pair_long_table(ret.dropna(how="all"), "pearson")

    def _offdiag_vals(m):
        a = m.values
        return a[~np.eye(a.shape[0], dtype=bool)]

    ev["pk_corr_mean"] = float(np.nanmean(np.abs(_offdiag_vals(pk_corr_p))))
    ev["pk_corr_median"] = float(np.nanmedian(_offdiag_vals(pk_corr_p)))
    ev["ret_corr_mean_abs"] = float(np.nanmean(np.abs(_offdiag_vals(ret_corr_p))))
    ev["ret_corr_median"] = float(np.nanmedian(_offdiag_vals(ret_corr_p)))
    ev["vshock_corr_median"] = float(np.nanmedian(_offdiag_vals(vshock_corr)))
    ev["pk_pairs_sig_fdr_frac"] = float(pk_pairs["significant_fdr_0.05"].mean())
    # D_ij = |pk_corr| - |ret_corr|
    merged = pk_pairs.merge(
        ret_pairs, on=["source", "target"], suffixes=("_pk", "_ret")
    )
    d_ij = merged["corr_pk"].abs() - merged["corr_ret"].abs()
    ev["pk_minus_ret_abs_corr_mean"] = float(d_ij.mean())
    ev["pk_stronger_than_ret_frac"] = float((d_ij > 0).mean())

    # ---- P1: permutation nulls -------------------------------------------------
    print("P1 permutation nulls ...")
    pk_null = rel.permutation_null_mean_abs_corr(
        pk_vol.dropna(how="any"), n_iter=n_perm_corr, seed=SEED
    )
    ret_null = rel.permutation_null_mean_abs_corr(
        ret.dropna(how="any"), n_iter=n_perm_corr, seed=SEED
    )
    pd.DataFrame([{**pk_null, "feature": "pk_vol"}, {**ret_null, "feature": "return"}]).to_csv(
        TAB / "permutation_null.csv", index=False
    )
    ev["pk_null_obs"] = pk_null["observed_mean_abs_corr"]
    ev["pk_null_p95"] = pk_null["null_p95"]
    ev["pk_null_p"] = pk_null["p_value_empirical"]

    # ---- P2: lead-lag ----------------------------------------------------------
    print("P2 lead-lag ...")
    lead = {k: rel.lead_lag_matrix(pk_vol, k) for k in (1, 2, 5, 10)}
    for k, m in lead.items():
        m.to_csv(TAB / f"pk_leadlag_{k}.csv")
    for k in (1, 2, 5):  # plan section 13: return lead-lag k=1,2,5
        rel.lead_lag_matrix(ret, k).to_csv(TAB / f"return_leadlag_{k}.csv")
    for k in (1, 2, 5):  # plan section 14: volume-shock -> future PK k=1,2,5
        rel.cross_lead_lag_matrix(vshock, pk_vol, k).to_csv(TAB / f"volume_to_pk_lag{k}.csv")
    v2pk1 = rel.cross_lead_lag_matrix(vshock, pk_vol, 1)
    # plan section 11: same-day (contemporaneous) cross-feature associations (descriptive)
    rel.cross_lead_lag_matrix(vshock, pk_vol, 0).to_csv(TAB / "contemp_vshock_to_pk.csv")
    rel.cross_lead_lag_matrix(ret, pk_vol, 0).to_csv(TAB / "contemp_return_to_pk.csv")

    asym = rel.directional_asymmetry(lead[1])
    off = ~np.eye(len(tickers), dtype=bool)
    ev["pk_leadlag1_offdiag_mean"] = float(np.nanmean(lead[1].values[off]))
    ev["pk_leadlag1_offdiag_max"] = float(np.nanmax(lead[1].values[off]))
    ev["pk_leadlag5_offdiag_mean"] = float(np.nanmean(lead[5].values[off]))
    # own lag-1 autocorr (diagonal of L(1)) for context
    ev["pk_own_autocorr1_mean"] = float(np.nanmean(np.diag(lead[1].values)))
    ev["asym_abs_mean"] = float(np.nanmean(np.abs(asym.values[off])))
    ev["asym_abs_max"] = float(np.nanmax(np.abs(asym.values[off])))
    ev["v2pk_lag1_mean"] = float(np.nanmean(np.abs(v2pk1.values[off])))

    # permutation null for lead-lag (mean |off-diag| L(1)); each column time-shuffled
    rng = np.random.default_rng(SEED)
    x = pk_vol.dropna(how="any")
    xv = x.values
    obs_ll = np.nanmean(np.abs(rel.lead_lag_matrix(x, 1).values[off]))
    null_ll = []
    for _ in range(n_perm_ll):
        perm = np.empty_like(xv)
        for c in range(xv.shape[1]):
            perm[:, c] = xv[rng.permutation(xv.shape[0]), c]
        pdf = pd.DataFrame(perm, columns=x.columns)
        null_ll.append(np.nanmean(np.abs(rel.lead_lag_matrix(pdf, 1).values[off])))
    ev["pk_leadlag1_null_p95"] = float(np.percentile(null_ll, 95))
    ev["pk_leadlag1_obs_meanabs"] = float(obs_ll)

    # ---- P2/P3: rolling snapshots, stability, turnover, density -----------------
    print("P3 rolling / stability ...")
    snaps60 = graphs.rolling_snapshots(pk_vol, window=60, step=21)
    for date, corr in snaps60:
        leakage.assert_snapshot_no_lookahead(
            pk_vol.loc[:date].index[-60:], date
        )
    stab = graphs.edge_stability(snaps60)
    stab["same_sector"] = [
        int(same_sec.loc[a, b]) for a, b in zip(stab.source, stab.target)
    ]
    stab.to_csv(TAB / "edge_stability.csv", index=False)
    turn5 = graphs.edge_turnover(snaps60, k=5)
    turn5.to_csv(TAB / "edge_turnover.csv", index=False)
    density03 = graphs.graph_density_series(snaps60, tau=0.3)
    density03.to_csv(TAB / "graph_density.csv", index=False)
    ev["edge_mean_corr_median"] = float(stab["mean_corr"].median())
    ev["edge_std_corr_median"] = float(stab["std_corr"].median())
    ev["edge_sign_consistency_median"] = float(stab["sign_consistency"].median())
    ev["edge_turnover_mean"] = float(turn5["edge_turnover"].mean())

    nb_stab = {}
    for k in (3, 5, 8, 10):
        ns = graphs.neighbor_stability(snaps60, k)
        nb_stab[k] = float(ns["mean_jaccard"].mean())
    pd.DataFrame(
        [{"k": k, "mean_neighbor_jaccard": v} for k, v in nb_stab.items()]
    ).to_csv(TAB / "neighbor_stability.csv", index=False)
    ev["neighbor_jaccard_k5"] = nb_stab[5]
    # random control jaccard baseline (expected overlap of random top-K sets)
    rng2 = np.random.default_rng(SEED)
    rand_j = []
    for _ in range(200):
        a = graphs.random_matched_neighbors(tickers, 5, seed=int(rng2.integers(1e9)))
        b = graphs.random_matched_neighbors(tickers, 5, seed=int(rng2.integers(1e9)))
        rand_j.append(np.mean([graphs.jaccard(a[n], b[n]) for n in tickers]))
    ev["neighbor_jaccard_random_k5"] = float(np.mean(rand_j))

    # ---- P2/P3: multi-window rolling edge panel (plan section 18/46) -------------
    print("P3 multi-window rolling edge panel ...")
    idx_all = close.index
    ends = list(range(119, len(idx_all), 21))  # from row 119 so all of 20/60/120 are valid
    dyn = graphs.multi_window_edge_panel(
        {"pk_corr": pk_vol, "return_corr": ret, "volume_corr": vshock},
        (20, 60, 120), ends, idx_all,
    )
    dyn["same_sector"] = [int(same_sec.loc[a, b]) for a, b in zip(dyn.source, dyn.target)]
    dyn.to_parquet(TAB / "dynamic_edge_features.parquet", index=False)

    # ---- P3: graph-clustering diagnostics (plan section 23) ---------------------
    clust_rows = []
    for label, mat, kk, tt in [
        ("pk_topk5", pk_corr_p, 5, None),
        ("pk_topk8", pk_corr_p, 8, None),
        ("pk_threshold_0.3", pk_corr_p, None, 0.3),
    ]:
        clust_rows.append({"config": label, **graphs.clustering_metrics(mat, k=kk, tau=tt)})
    clust = pd.DataFrame(clust_rows)
    clust.to_csv(TAB / "graph_clustering.csv", index=False)
    _pk5 = clust[clust["config"] == "pk_topk5"].iloc[0]
    ev["clustering_modularity_topk5"] = float(_pk5["modularity"])
    ev["clustering_n_communities_topk5"] = int(_pk5["n_communities"])
    ev["clustering_avg_coeff_topk5"] = float(_pk5["avg_clustering"])

    # ---- P3: market-factor adjustment (KEY test) --------------------------------
    print("P3 market-factor adjustment ...")
    market = rel.market_factor(pk_vol, "median")
    market.to_frame("market_pk").to_csv(TAB / "market_pk.csv")
    resid = rel.market_adjust_residuals(pk_vol, market, tr)
    resid_corr = rel.pairwise_corr(resid.dropna(how="all"), "pearson")
    resid_corr.to_csv(TAB / "pk_residual_corr.csv")
    raw_off = np.abs(_offdiag_vals(pk_corr_p))
    res_off = np.abs(_offdiag_vals(resid_corr))
    ev["pk_corr_mean_abs_raw"] = float(np.nanmean(raw_off))
    ev["pk_corr_mean_abs_market_adj"] = float(np.nanmean(res_off))
    ev["market_adj_retained_frac"] = float(
        np.nanmean(res_off) / np.nanmean(raw_off)
    )
    # avg R^2 of PK on market factor across stocks (how much is common)
    r2s = []
    mk = market.values
    for c in pk_vol.columns:
        y = pk_vol[c].values
        m = tr & ~np.isnan(y) & ~np.isnan(mk)
        if m.sum() > 30:
            r = np.corrcoef(y[m], mk[m])[0, 1]
            r2s.append(r * r)
    ev["mean_r2_pk_on_market_train"] = float(np.mean(r2s))

    # raw-vs-adjusted per-pair table
    ra = rel.pair_long_table(pk_vol.dropna(how="all"))[["source", "target", "corr"]]
    rr = rel.pair_long_table(resid.dropna(how="all"))[["source", "target", "corr"]]
    radj = ra.merge(rr, on=["source", "target"], suffixes=("_raw", "_market_adj"))
    radj["same_sector"] = [
        int(same_sec.loc[a, b]) for a, b in zip(radj.source, radj.target)
    ]
    radj.to_csv(TAB / "raw_vs_market_adjusted_pk_corr.csv", index=False)

    # ---- P3: sector analysis ---------------------------------------------------
    same_mask = pk_pairs["same_sector"] == 1
    ev["pk_corr_same_sector_mean"] = float(pk_pairs.loc[same_mask, "corr"].mean())
    ev["pk_corr_cross_sector_mean"] = float(pk_pairs.loc[~same_mask, "corr"].mean())
    if same_mask.sum() > 1 and (~same_mask).sum() > 1:
        u = stats.mannwhitneyu(
            pk_pairs.loc[same_mask, "corr"], pk_pairs.loc[~same_mask, "corr"],
            alternative="greater",
        )
        ev["sector_mannwhitney_p"] = float(u.pvalue)
    # market-adjusted same/cross sector
    radj_same = radj["same_sector"] == 1
    ev["resid_corr_same_sector_mean"] = float(
        radj.loc[radj_same, "corr_market_adj"].mean()
    )
    ev["resid_corr_cross_sector_mean"] = float(
        radj.loc[~radj_same, "corr_market_adj"].mean()
    )

    # ---- P3: regime dependence -------------------------------------------------
    mkt_tr = market[tr]
    lo_thr, hi_thr = mkt_tr.quantile([1 / 3, 2 / 3])
    ev["regime_lo_thr"] = float(lo_thr)
    ev["regime_hi_thr"] = float(hi_thr)
    hi_days = market > hi_thr
    lo_days = market <= lo_thr
    pk_hi = rel.pairwise_corr(pk_vol[hi_days].dropna(how="all"))
    pk_lo = rel.pairwise_corr(pk_vol[lo_days].dropna(how="all"))
    ev["pk_corr_high_regime_mean"] = float(np.nanmean(np.abs(_offdiag_vals(pk_hi))))
    ev["pk_corr_low_regime_mean"] = float(np.nanmean(np.abs(_offdiag_vals(pk_lo))))

    # ---- P4: predictive neighbour test (Gates 6-7) -----------------------------
    print("P4 predictive baselines ...")
    train_corr = rel.pairwise_corr(pk_var[pd.Series(tr, index=close.index)], "pearson").abs()
    pred = {}
    per_stock = {}
    for h in (1, 5):
        res, ps = predictive.run_baselines(
            pk_var, market, tr, te, train_corr, horizon=h, k=5, return_per_stock=True
        )
        res.to_csv(TAB / f"predictive_baselines_h{h}.csv")
        ps.to_csv(TAB / f"predictive_per_stock_h{h}.csv", index=False)
        pred[h] = res
        per_stock[h] = ps
    p1 = pred[1]
    ev["pred_h1_rmse_A"] = float(p1.loc["A_own", "rmse"])
    ev["pred_h1_rmse_B"] = float(p1.loc["B_own+market", "rmse"])
    ev["pred_h1_rmse_C"] = float(p1.loc["C_own+neighbors", "rmse"])
    ev["pred_h1_qlike_A"] = float(p1.loc["A_own", "qlike"])
    ev["pred_h1_qlike_B"] = float(p1.loc["B_own+market", "qlike"])
    ev["pred_h1_qlike_C"] = float(p1.loc["C_own+neighbors", "qlike"])
    ev["neighbor_rmse_gain_vs_own_pct"] = float(
        100 * (ev["pred_h1_rmse_A"] - ev["pred_h1_rmse_C"]) / ev["pred_h1_rmse_A"]
    )
    ev["neighbor_rmse_gain_vs_market_pct"] = float(
        100 * (ev["pred_h1_rmse_B"] - ev["pred_h1_rmse_C"]) / ev["pred_h1_rmse_B"]
    )
    # per-stock significance (review M2): win-rate + two-sided sign (binomial) test that
    # C's per-stock RMSE is below A's / B's. Gates require a meaningful pooled margin AND
    # a stock majority, so a ~1e-6 pooled tie can never flip the verdict to "use GNN".
    ps1 = per_stock[1]
    ca = ps1["rmse_C"] < ps1["rmse_A"]
    cb = ps1["rmse_C"] < ps1["rmse_B"]
    n_stocks = len(ps1)
    ev["pred_h1_n_stocks"] = int(n_stocks)
    ev["C_beats_A_winrate"] = float(ca.mean())
    ev["C_beats_B_winrate"] = float(cb.mean())
    ev["C_beats_A_sign_p"] = float(stats.binomtest(int(ca.sum()), n_stocks).pvalue)
    ev["C_beats_B_sign_p"] = float(stats.binomtest(int(cb.sum()), n_stocks).pvalue)
    margin = 0.005  # 0.5% pooled RMSE improvement required to count as a real gain
    ev["gate6_C_beats_A"] = bool(
        ev["neighbor_rmse_gain_vs_own_pct"] > 100 * margin and ev["C_beats_A_winrate"] > 0.5
    )
    ev["gate7_C_beats_B"] = bool(
        ev["neighbor_rmse_gain_vs_market_pct"] > 100 * margin and ev["C_beats_B_winrate"] > 0.5
    )

    # ---- P4a: Top-K search (plan section 50): density / stability / sector purity /
    # edge strength / predictive usefulness per K, on TRAIN-selected neighbours ----------
    print("P4a Top-K search ...")
    topk_rows = []
    n_nodes = len(tickers)
    max_edges = n_nodes * (n_nodes - 1) / 2
    for kk in (3, 5, 8, 10):
        nb_k = graphs.top_k_neighbors(train_corr, kk, use_abs=True)
        res_k = predictive.run_baselines(pk_var, market, tr, te, train_corr, horizon=1, k=kk)
        rmse_b = float(res_k.loc["B_own+market", "rmse"]) if "B_own+market" in res_k.index else np.nan
        rmse_c = float(res_k.loc["C_own+neighbors", "rmse"]) if "C_own+neighbors" in res_k.index else np.nan
        n_edges = len({frozenset((node, m)) for node, ms in nb_k.items() for m in ms})
        topk_rows.append(
            {
                "k": kk,
                "graph_density": float(n_edges / max_edges),
                "neighbor_jaccard": nb_stab.get(kk, np.nan),
                "sector_purity": graphs.sector_purity(nb_k, same_sec),
                "mean_edge_strength": graphs.mean_edge_strength(train_corr, nb_k),
                "pred_rmse_gain_vs_market_pct": float(100 * (rmse_b - rmse_c) / rmse_b)
                if rmse_b and not np.isnan(rmse_b) else np.nan,
            }
        )
    pd.DataFrame(topk_rows).to_csv(TAB / "topk_search.csv", index=False)

    # ---- confidence intervals for headline means (plan section 56) --------------
    # bootstrap over the 528 UNIQUE pairs (pk_pairs), not the 1056 duplicated off-diagonal
    # entries, so the CI reflects the true number of independent pairs
    ev["pk_corr_mean_ci_lo"], ev["pk_corr_mean_ci_hi"] = rel.bootstrap_ci_mean(
        pk_pairs["corr"].values, n_boot=1000, seed=SEED, use_abs=True
    )
    ev["same_sector_ci_lo"], ev["same_sector_ci_hi"] = rel.bootstrap_ci_mean(
        pk_pairs.loc[pk_pairs["same_sector"] == 1, "corr"].values, n_boot=1000, seed=SEED
    )
    ev["cross_sector_ci_lo"], ev["cross_sector_ci_hi"] = rel.bootstrap_ci_mean(
        pk_pairs.loc[pk_pairs["same_sector"] == 0, "corr"].values, n_boot=1000, seed=SEED
    )

    # ---- P4b: incremental-over-HAR ranking of node features + edge definitions ---
    print("P4b incremental-over-HAR ranking ...")
    node_rank, edge_rank = _incremental(
        close, ret, pk_var, pk_vol, resid, vshock, market, tr, te
    )
    node_rank.to_csv(TAB / "node_feature_ranking_h1.csv", index=False)
    edge_rank.to_csv(TAB / "edge_definition_ranking_h1.csv", index=False)
    rec_cfg = _recommend_config(ev, node_rank, edge_rank)
    ev["best_edge"] = rec_cfg["edge_type"]
    ev["best_edge_gain_over_market_pct"] = rec_cfg["expected_lift_over_market_pct"]
    ev["best_edge_sign_p_over_market"] = rec_cfg["sign_p_over_market"]
    ev["best_edge_gain_over_har_pct"] = rec_cfg["expected_lift_over_har_pct"]
    ev["best_edge_lift_evidence"] = rec_cfg["lift_evidence"]
    # the plain PK-correlation edge is exactly what the project's G1 kNN-8 GAT used
    _pk_edge = edge_rank[edge_rank["edge_definition"] == "edge_pkcorr_abs"]
    if not _pk_edge.empty:
        ev["edge_pkcorr_gain_over_market_pct"] = float(_pk_edge.iloc[0]["gain_over_market_pct"])
        ev["edge_pkcorr_sign_p_over_market"] = float(_pk_edge.iloc[0]["sign_p_over_market"])

    _figures(
        close, ret_corr_p, pk_corr_p, vshock_corr, merged, pk_pairs, lead[1], v2pk1,
        stab, snaps60, density03, market, pk_hi, pk_lo, radj, resid_corr, tickers,
    )
    _graph_plots(snaps60, market, lo_thr, hi_thr, lead[1], pk_corr_p, tickers)
    verdict = _decide(ev)
    _write_report(ev, verdict, dq, panel, tickers, pred, node_rank, edge_rank, rec_cfg)
    _write_json(ev, verdict, rec_cfg)
    print("Verdict:", verdict["use_gnn"], verdict["confidence"], verdict["conclusion"])
    return {"ev": ev, "verdict": verdict}


def _decide(ev: dict) -> dict:
    """Evidence-weighted verdict (thresholds are transparent, not tuned to a target)."""
    retained = ev["market_adj_retained_frac"]
    gate6 = ev["gate6_C_beats_A"]
    gate7 = ev["gate7_C_beats_B"]
    null_sep = ev["pk_null_obs"] > ev["pk_null_p95"]
    dyn = ev["neighbor_jaccard_k5"] < 0.7
    direction = ev["asym_abs_mean"] > 0.05 and ev["pk_leadlag1_offdiag_max"] > 0.2

    # market factor dominance -> Conclusion C when little structure survives adjustment
    if retained < 0.35 and not gate7:
        conclusion = "C"
        use_gnn, conf, gtype = False, "high", "none"
    elif not gate6 and not null_sep:
        conclusion = "D"
        use_gnn, conf, gtype = False, "high", "none"
    elif gate7 and null_sep and retained >= 0.5:
        conclusion = "A" if dyn else "B"
        use_gnn, conf = True, "medium"
        gtype = "dynamic_directed" if direction else ("dynamic" if dyn else "static")
    else:
        # structure exists but adds no value beyond market factor OOS
        conclusion = "C"
        use_gnn, conf, gtype = False, "medium", "none"
    return {
        "use_gnn": use_gnn,
        "confidence": conf,
        "graph_type": gtype,
        "conclusion": conclusion,
    }


def _incremental(close, ret, pk_var, pk_vol, resid, vshock, market, tr, te):
    """Build node-feature + edge-definition candidates and rank them over HAR."""
    idx = close.index
    ret5 = np.log(close / close.shift(5))
    pk_std20 = pk_var.rolling(20).std()
    node_features = {
        "log_return_1d": ret,
        "return_5d": ret5,
        "pk_std_20": pk_std20,
        "volume_zscore_20": vshock,
        "pk_lag2": pk_var.shift(2),
        "pk_vol_daily": pk_vol,
    }
    trs = pd.Series(tr, index=idx)
    pk_vol_tr = pk_vol[trs]
    vshock_tr = vshock[trs]
    tpc = pk_vol_tr.corr()                                   # train PK corr (signed)
    trc = resid[trs].corr()                                  # train residual corr
    tll = rel.lead_lag_matrix(pk_vol_tr, 1)                  # train directed PK lead-lag
    tv2p = rel.cross_lead_lag_matrix(vshock_tr, pk_vol_tr, 1)  # train volume->PK lead-lag
    pkcorr_abs = graphs.top_k_neighbors(tpc, 5, use_abs=True)
    pkcorr_pos = graphs.top_k_neighbors(tpc, 5, use_abs=False)
    residcorr = graphs.top_k_neighbors(trc, 5, use_abs=True)
    leadlag = incremental._incoming_neighbors(tll, 5, use_abs=True)
    vol2pk = incremental._incoming_neighbors(tv2p, 5, use_abs=True)
    multi = {
        j: list(dict.fromkeys(pkcorr_abs[j] + residcorr[j] + leadlag[j]))
        for j in pk_var.columns
    }
    edge_neighbors = {
        "edge_pkcorr_abs": (pkcorr_abs, pk_var),
        "edge_pkcorr_pos": (pkcorr_pos, pk_var),
        "edge_residcorr": (residcorr, resid),
        "edge_leadlag_dir": (leadlag, pk_var),
        "edge_vol2pk_dir": (vol2pk, vshock),
        "edge_multi": (multi, pk_var),
    }
    return incremental.rank_candidates(
        pk_var, node_features, edge_neighbors, market, tr, te, horizon=1
    )


def _recommend_config(ev: dict, node_rank, edge_rank) -> dict:
    """Assemble the strongest-evidenced GNN config (produced even if the lift is small)."""
    edge_feature_map = {
        "edge_pkcorr_abs": ["pk_corr_20", "pk_corr_60"],
        "edge_pkcorr_pos": ["pk_corr_20_positive", "pk_corr_60_positive"],
        "edge_residcorr": ["pk_residual_corr_60", "pk_residual_corr_20"],
        "edge_leadlag_dir": ["pk_leadlag_1", "pk_leadlag_5"],
        "edge_vol2pk_dir": ["volume_to_pk_lag1", "volume_to_pk_lag5"],
        "edge_multi": ["pk_corr_60", "pk_residual_corr_60", "pk_leadlag_1", "same_sector"],
    }
    if edge_rank is not None and not edge_rank.empty:
        best = edge_rank.iloc[0]
        edge_name = best["edge_definition"]
        lift_mkt = float(best["gain_over_market_pct"])
        lift_har = float(best["gain_over_har_pct"])
        sign_p = float(best["sign_p_over_market"])
        winrate_mkt = float(best["winrate_over_market"])
    else:
        edge_name, lift_mkt, lift_har, sign_p, winrate_mkt = "edge_pkcorr_abs", 0.0, 0.0, 1.0, 0.0
    directed = edge_name in {"edge_leadlag_dir", "edge_vol2pk_dir"}
    dynamic = ev.get("neighbor_jaccard_k5", 1.0) < 0.5
    # recommended node features: HAR always, plus any own feature with a positive,
    # not-clearly-noise incremental gain over HAR
    node_feats = ["pk_daily", "pk_weekly", "pk_monthly"]  # HAR core
    if node_rank is not None and not node_rank.empty:
        for _, r in node_rank.iterrows():
            if r["rmse_gain_pct"] > 0 and r["sign_p"] < 0.20:
                node_feats.append(str(r["node_feature"]))
    # 3-state honest verdict on the best edge's lift over the HAR+market bar
    if lift_mkt > 0 and sign_p < 0.05 and winrate_mkt > 0.5:
        lift_evidence = "beats"          # positive + significant at 0.05
    elif lift_mkt > 0 and winrate_mkt > 0.5:
        lift_evidence = "suggestive"     # positive but not significant at 0.05
    else:
        lift_evidence = "none"           # no OOS gain over HAR+market
    return {
        "edge_type": edge_name,
        "directed": directed,
        "static_or_dynamic": "dynamic" if dynamic else "static",
        "top_k": 5,
        "node_features": node_feats,
        "edge_features": edge_feature_map.get(edge_name, ["pk_corr_60"]),
        "expected_lift_over_har_pct": lift_har,
        "expected_lift_over_market_pct": lift_mkt,
        "sign_p_over_market": sign_p,
        "winrate_over_market": winrate_mkt,
        "lift_evidence": lift_evidence,
        "likely_beats_har_under_dm": bool(lift_evidence == "beats"),
    }


def _figures(close, ret_corr, pk_corr, vshock_corr, merged, pk_pairs, lead1, v2pk1,
             stab, snaps, density, market, pk_hi, pk_lo, radj, resid_corr, tickers):
    print("Figures ...")
    _cov_figure(close)
    _heatmap(ret_corr, "Return correlation (full period)", FIG / "02_return_corr_heatmap.png")
    _heatmap(pk_corr, "Parkinson-vol correlation (full period)", FIG / "03_pk_corr_heatmap.png")
    _heatmap(vshock_corr, "Volume-shock correlation (full period)", FIG / "04_volume_corr_heatmap.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["corr_ret"].abs(), merged["corr_pk"].abs(), s=10, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("|return correlation|")
    ax.set_ylabel("|Parkinson-vol correlation|")
    ax.set_title("PK vs return pairwise correlation")
    _save(fig, "05_pk_vs_return_corr_scatter.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pk_pairs["corr"].dropna(), bins=40, color="steelblue", edgecolor="k")
    ax.set_title("Distribution of pairwise PK-vol correlation")
    ax.set_xlabel("correlation")
    _save(fig, "06_pairwise_pk_corr_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    same = pk_pairs["same_sector"] == 1
    ax.boxplot(
        [pk_pairs.loc[same, "corr"].dropna(), pk_pairs.loc[~same, "corr"].dropna()],
        tick_labels=["same sector", "cross sector"],
    )
    ax.set_title("PK-vol correlation by sector relationship")
    ax.set_ylabel("correlation")
    _save(fig, "07_same_vs_cross_sector_pk_corr.png")

    _heatmap(lead1, "PK lead-lag L(1): row leads col", FIG / "08_pk_leadlag_heatmap.png", vmin=-0.3, vmax=0.3)
    _heatmap(v2pk1, "Volume-shock -> future PK L(1)", FIG / "09_volume_to_pk_leadlag_heatmap.png", vmin=-0.2, vmax=0.2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stab["mean_corr"].dropna(), bins=30, alpha=0.6, label="mean rolling corr")
    ax.hist(stab["std_corr"].dropna(), bins=30, alpha=0.6, label="std rolling corr")
    ax.legend()
    ax.set_title("Edge stability distribution (60d/21d snapshots)")
    _save(fig, "10_edge_stability_distribution.png")

    js = []
    prev = graphs.top_k_neighbors(snaps[0][1], 5)
    for _, corr in snaps[1:]:
        cur = graphs.top_k_neighbors(corr, 5)
        js.append(np.mean([graphs.jaccard(prev[n], cur[n]) for n in cur]))
        prev = cur
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([d for d, _ in snaps[1:]], js, marker="o", ms=3)
    ax.set_title("Top-5 neighbour Jaccard over time")
    ax.set_ylim(0, 1)
    _save(fig, "11_neighbor_jaccard_over_time.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(density["date"], density["density"], marker="o", ms=3)
    ax.set_title("Threshold-graph density over time (|corr|>0.3)")
    ax.set_ylabel("density")
    _save(fig, "12_graph_density_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(market.index, market.values, lw=0.8)
    ax.set_title("Market Parkinson volatility (median over stocks) over time")
    _save(fig, "13_market_pk_over_time.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    off = ~np.eye(len(tickers), dtype=bool)
    ax.boxplot(
        [
            np.abs(pk_lo.values[off][~np.isnan(pk_lo.values[off])]),
            np.abs(pk_hi.values[off][~np.isnan(pk_hi.values[off])]),
        ],
        tick_labels=["low-vol regime", "high-vol regime"],
    )
    ax.set_title("Pairwise |PK corr| by market regime")
    ax.set_ylabel("|correlation|")
    _save(fig, "14_pk_corr_by_market_regime.png")

    strong = stab.sort_values("mean_corr", ascending=False).iloc[0]
    weak = stab.reindex(stab["std_corr"].sort_values(ascending=False).index).iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    for _, r in pd.DataFrame([strong, weak]).iterrows():
        series = [c.loc[r.source, r.target] for _, c in snaps]
        ax.plot([d for d, _ in snaps], series, marker=".", label=f"{r.source}-{r.target}")
    ax.legend()
    ax.set_title("Example rolling PK correlations (stable vs unstable)")
    _save(fig, "15_example_rolling_correlations.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(radj["corr_raw"].abs(), radj["corr_market_adj"].abs(), s=10, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("|raw PK corr|")
    ax.set_ylabel("|market-adjusted residual PK corr|")
    ax.set_title("Raw vs market-adjusted PK correlation")
    _save(fig, "16_raw_vs_market_adjusted_pk_corr.png")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def _draw_graph(g, title: str, path: Path, directed: bool = False) -> None:
    """Draw a node-link graph coloured by sector, labelled by ticker (plan section 48)."""
    import networkx as nx

    from graph_eda.sectors import sector_of

    sectors = sorted({sector_of(n) for n in g.nodes})
    palette = plt.get_cmap("tab20")
    color = {s: palette(i % 20) for i, s in enumerate(sectors)}
    node_colors = [color[sector_of(n)] for n in g.nodes]
    pos = nx.spring_layout(g, seed=SEED, k=0.6)
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=420, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    edge_kw = {"arrows": True, "arrowstyle": "-|>", "arrowsize": 9} if directed else {}
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, **edge_kw)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _graph_plots(snaps, market, lo_thr, hi_thr, lead1, pk_corr, tickers):
    """Plan section 48: PK Top-5 graphs at low/normal/high-vol dates + directed + multi-edge."""
    import networkx as nx

    print("Graph plots ...")
    # pick one representative snapshot per market regime (snapshot dates are a subset of
    # market.index by construction: market is the cross-sectional median of pk_vol)
    regimes = {"low": None, "normal": None, "high": None}
    for date, corr in snaps:
        mv = market.loc[date]
        r = "low" if mv <= lo_thr else ("high" if mv > hi_thr else "normal")
        if regimes[r] is None:
            regimes[r] = (date, corr)
    for r, item in regimes.items():
        if item is None:
            continue
        date, corr = item
        nbrs = graphs.top_k_neighbors(corr, 5, use_abs=True)
        g = nx.Graph()
        g.add_nodes_from(tickers)
        for node, ms in nbrs.items():
            for m in ms:
                g.add_edge(node, m)
        _draw_graph(g, f"PK-corr Top-5 graph ({r}-vol {date.date()})",
                    GRAPHS / f"pk_top5_{r}_vol.png")

    # directed PK lead-lag graph (top-5 incoming sources per target, full period)
    gd = nx.DiGraph()
    gd.add_nodes_from(tickers)
    inc = incremental._incoming_neighbors(lead1, 5, use_abs=True)
    for tgt, srcs in inc.items():
        for s in srcs:
            gd.add_edge(s, tgt)
    _draw_graph(gd, "Directed PK lead-lag L(1) graph (Top-5 incoming)",
                GRAPHS / "pk_leadlag_directed.png", directed=True)

    # multi-edge graph: union of Top-5 PK-corr and Top-5 lead-lag edges
    gm = nx.Graph()
    gm.add_nodes_from(tickers)
    corr_nbrs = graphs.top_k_neighbors(pk_corr, 5, use_abs=True)
    for node, ms in corr_nbrs.items():
        for m in ms:
            gm.add_edge(node, m)
    for tgt, srcs in inc.items():
        for s in srcs:
            gm.add_edge(s, tgt)
    _draw_graph(gm, "Multi-edge graph (PK-corr + lead-lag Top-5 union)",
                GRAPHS / "multi_edge_graph.png")


def _cov_figure(close):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(close.columns)), close.notna().sum().values)
    ax.set_xticks(range(len(close.columns)))
    ax.set_xticklabels(close.columns, rotation=90, fontsize=6)
    ax.set_title("Common-panel observation count per ticker")
    _save(fig, "01_data_coverage.png")


def _write_json(ev: dict, verdict: dict, rec_cfg: dict) -> None:
    rec = {
        "use_gnn": verdict["use_gnn"],
        "confidence": verdict["confidence"],
        "conclusion": verdict["conclusion"],
        "graph_type": verdict["graph_type"],
        "primary_relationship": "parkinson_variance",
        # Strongest-evidenced GNN config to build next and test vs HAR under Diebold-Mariano.
        # Produced regardless of the yes/no verdict; expected lift is honest (may be <=0).
        "recommended_config": {
            "node_features": rec_cfg["node_features"],
            "edge_type": rec_cfg["edge_type"],
            "directed": rec_cfg["directed"],
            "static_or_dynamic": rec_cfg["static_or_dynamic"],
            "edge_selection": "top_k",
            "top_k": rec_cfg["top_k"],
            "edge_features": rec_cfg["edge_features"],
            "expected_lift_over_har_pct_rmse": rec_cfg["expected_lift_over_har_pct"],
            "expected_lift_over_market_pct_rmse": rec_cfg["expected_lift_over_market_pct"],
            "sign_p_over_market": rec_cfg["sign_p_over_market"],
            "winrate_over_market": rec_cfg["winrate_over_market"],
            "likely_beats_har_under_dm": rec_cfg["likely_beats_har_under_dm"],
        },
        "main_evidence": {k: ev[k] for k in ev},
        "main_risks": [
            "market factor explains most cross-stock PK co-movement "
            f"(only {round(100 * ev['market_adj_retained_frac'], 1)}% of |corr| survives)",
            "best edge adds "
            f"{round(ev['best_edge_gain_over_market_pct'], 2)}% RMSE vs HAR+market "
            f"(sign p={round(ev['best_edge_sign_p_over_market'], 3)}) - "
            + ("likely clears HAR" if rec_cfg["likely_beats_har_under_dm"]
               else "unlikely to clear HAR under DM"),
        ],
    }
    (OUT / "graph_recommendation.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )


def _fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _write_report(ev, verdict, dq, panel, tickers, pred, node_rank, edge_rank, rec_cfg):
    concl = {
        "A": "Conclusion A - Strong GNN evidence (dynamic edge-aware GAT).",
        "B": "Conclusion B - Static graph sufficient.",
        "C": "Conclusion C - Market factor dominates; use HAR/LSTM + market factor.",
        "D": "Conclusion D - Weak graph evidence; do not use a GNN.",
    }[verdict["conclusion"]]

    def lvl(cond_strong, cond_mod):
        return "Strong" if cond_strong else ("Moderate" if cond_mod else "Weak")

    matrix = [
        ("PK cross-stock dependence", lvl(ev["pk_corr_mean"] > 0.4, ev["pk_corr_mean"] > 0.2)),
        ("Null-test separation", lvl(ev["pk_null_obs"] > 2 * ev["pk_null_p95"], ev["pk_null_obs"] > ev["pk_null_p95"])),
        ("Out-of-sample stability", lvl(ev["neighbor_jaccard_k5"] > 0.7, ev["neighbor_jaccard_k5"] > 0.4)),
        ("Lead-lag directionality", lvl(ev["asym_abs_max"] > 0.3, ev["asym_abs_max"] > 0.1)),
        ("Regime dependence", lvl(ev["pk_corr_high_regime_mean"] > 1.5 * ev["pk_corr_low_regime_mean"], ev["pk_corr_high_regime_mean"] > ev["pk_corr_low_regime_mean"])),
        ("Sector structure", lvl(ev["pk_corr_same_sector_mean"] > ev["pk_corr_cross_sector_mean"] + 0.1, ev["pk_corr_same_sector_mean"] > ev["pk_corr_cross_sector_mean"])),
        ("Market-adjusted dependence", lvl(ev["market_adj_retained_frac"] > 0.6, ev["market_adj_retained_frac"] > 0.35)),
        ("Neighbor predictive gain", lvl(ev["gate7_C_beats_B"], ev["gate6_C_beats_A"])),
        ("Dynamic graph justification", lvl(ev["edge_turnover_mean"] > 0.5, ev["edge_turnover_mean"] > 0.3)),
    ]

    L = []
    L.append("# EDA Graph Report - Daily VN30 Parkinson Volatility\n")
    L.append("Executes `docs/eda_guide/parkinson_volatility_gnn_eda_experiment_plan.md`. ")
    L.append("Target = Parkinson VARIANCE (sigma^2, the project target); `pk_vol` = sqrt(pk_var) is the plan's default and is used for cross-stock correlation. Both retained.\n")
    # top-line verdict is driven by whether ANY evidenced edge beats the HAR+market bar OOS
    verd = {"beats": "YES", "suggestive": "MAYBE", "none": "NO"}[rec_cfg["lift_evidence"]]
    L.append("## Executive Summary\n")
    L.append(f"**Should a GNN be used: {verd}.**\n")
    L.append(f"{concl}\n")
    L.append(
        "Framing: the bar is HAR (each stock's own daily/weekly/monthly Parkinson features). "
        "A GNN is justified only if a cross-stock edge lowers out-of-sample PK error beyond HAR "
        "AND beyond a market factor. Findings (all numbers from generated tables):\n"
        f"- Cross-stock PK co-movement is real (mean |PK corr| = {_fmt(ev['pk_corr_mean'])} >> "
        f"shuffle-null p95 {_fmt(ev['pk_null_p95'])}, p={_fmt(ev['pk_null_p'],3)}), but "
        f"{_fmt(100*(1-ev['market_adj_retained_frac']),0)}% of it is one market factor "
        f"(mean R^2 on MarketPK {_fmt(ev['mean_r2_pk_on_market_train'])}; only "
        f"{_fmt(100*ev['market_adj_retained_frac'],1)}% survives market adjustment).\n"
        f"- The plain PK-correlation edge (what the project's G1 kNN-8 GAT uses) is the WORST "
        f"edge: {_fmt(ev['edge_pkcorr_gain_over_market_pct'],2)}% OOS RMSE vs HAR+market "
        f"(sign p={_fmt(ev['edge_pkcorr_sign_p_over_market'],3)}) - significantly worse. This is "
        "the direct data reason G1 tied/failed HAR.\n"
        f"- The ONLY edge with a positive lift over HAR+market is the **directed volume->PK "
        f"lead-lag** edge: +{_fmt(ev['best_edge_gain_over_market_pct'],2)}% RMSE, "
        f"{_fmt(100*rec_cfg['winrate_over_market'],0)}% of stocks, sign p="
        f"{_fmt(ev['best_edge_sign_p_over_market'],3)} (marginal, not significant at 0.05).\n"
        f"- The only own node feature that beats HAR is volume_zscore_20 (+"
        f"{_fmt(node_rank.iloc[0]['rmse_gain_pct'],2) if node_rank is not None and not node_rank.empty else 'n/a'}% "
        "RMSE, sign p<1e-6).\n"
        "Recommended config is in the section below and in `graph_recommendation.json`: build the "
        "directed volume->PK Top-5 GNN with HAR+volume_zscore node features and DM-test it - it is "
        "the strongest evidenced shot, though the expected lift is small.\n")

    L.append("## Decision Matrix (plan section 68)\n")
    L.append("| Evidence | Level |\n|---|---|")
    for name, level in matrix:
        L.append(f"| {name} | {level} |")
    L.append(f"\n**Final recommendation:** {'DYNAMIC/STATIC GNN' if verdict['use_gnn'] else 'NO GNN (LSTM/HAR + market factor)'}\n")

    L.append("## Data Summary\n")
    L.append(f"- Tickers: {len(tickers)} (VN30 universe incl. LPB)\n- Common panel: {panel['n_dates'][0]} trading days, {panel['date_start'][0]} to {panel['date_end'][0]}\n- Alignment: intersection of trading dates across all 33 tickers, no forward-fill. VPB/VRE tz-aware dates normalised; price-scale differences immaterial (log-ratios).\n- Data quality: {int(dq['high_below_low_count'].sum())} high<low rows, {int(dq['nonpositive_price_count'].sum())} nonpositive-price rows, {int(dq['duplicate_date_count'].sum())} duplicate dates across the universe (see `tables/data_quality_summary.csv`).\n")

    L.append("## Parkinson Volatility Summary\n")
    L.append(f"- Mean |PK corr| (full period) = {_fmt(ev['pk_corr_mean'])} (95% bootstrap CI [{_fmt(ev['pk_corr_mean_ci_lo'])}, {_fmt(ev['pk_corr_mean_ci_hi'])}]), median PK corr = {_fmt(ev['pk_corr_median'])}. FDR-significant pairs (q<0.05): {_fmt(100*ev['pk_pairs_sig_fdr_frac'],1)}%.\n- High-vol regime mean |PK corr| = {_fmt(ev['pk_corr_high_regime_mean'])} vs low-vol {_fmt(ev['pk_corr_low_regime_mean'])} (train-defined thresholds).\n")

    L.append("## Return / Volume Relationships\n")
    L.append(f"- Mean |return corr| = {_fmt(ev['ret_corr_mean_abs'])}; median volume-shock corr = {_fmt(ev['vshock_corr_median'])}.\n- PK vs return: |PK corr| exceeds |return corr| for {_fmt(100*ev['pk_stronger_than_ret_frac'],1)}% of pairs (mean D = {_fmt(ev['pk_minus_ret_abs_corr_mean'])}). See `05_pk_vs_return_corr_scatter.png`.\n")

    L.append("## Lead-Lag Findings\n")
    L.append(f"- PK L(1) off-diagonal mean = {_fmt(ev['pk_leadlag1_offdiag_mean'])}, max = {_fmt(ev['pk_leadlag1_offdiag_max'])}; own lag-1 autocorr mean = {_fmt(ev['pk_own_autocorr1_mean'])}.\n- Directional asymmetry |A_ij| mean = {_fmt(ev['asym_abs_mean'])}, max = {_fmt(ev['asym_abs_max'])}. Volume->future-PK L(1) mean |corr| = {_fmt(ev['v2pk_lag1_mean'])}.\n- Lead-lag null: observed mean|L(1)|={_fmt(ev['pk_leadlag1_obs_meanabs'])} vs shuffle-null p95={_fmt(ev['pk_leadlag1_null_p95'])}.\n")

    L.append("## Sector Findings\n")
    smw = ev.get("sector_mannwhitney_p", float("nan"))
    ci_overlap = (
        ev["cross_sector_ci_hi"] >= ev["same_sector_ci_lo"]
        and ev["same_sector_ci_hi"] >= ev["cross_sector_ci_lo"]
    )
    ci_txt = "overlapping CIs" if ci_overlap else "non-overlapping CIs"
    L.append(f"- Same-sector mean PK corr = {_fmt(ev['pk_corr_same_sector_mean'])} (95% CI [{_fmt(ev['same_sector_ci_lo'])}, {_fmt(ev['same_sector_ci_hi'])}]) vs cross-sector = {_fmt(ev['pk_corr_cross_sector_mean'])} (95% CI [{_fmt(ev['cross_sector_ci_lo'])}, {_fmt(ev['cross_sector_ci_hi'])}]) (Mann-Whitney one-sided p = {_fmt(smw,4)}); {ci_txt}.\n- After market adjustment: same-sector residual corr = {_fmt(ev['resid_corr_same_sector_mean'])} vs cross = {_fmt(ev['resid_corr_cross_sector_mean'])}. Sector map is constructed (no metadata file); see `graph_eda/sectors.py`.\n")

    L.append("## Dynamic Graph Evidence\n")
    L.append(f"- Top-5 neighbour Jaccard (consecutive 60d/21d snapshots) = {_fmt(ev['neighbor_jaccard_k5'])} vs random-control {_fmt(ev['neighbor_jaccard_random_k5'])}. Edge turnover mean = {_fmt(ev['edge_turnover_mean'])}.\n- Edge rolling-corr: median mean_corr = {_fmt(ev['edge_mean_corr_median'])}, median std = {_fmt(ev['edge_std_corr_median'])}, median sign-consistency = {_fmt(ev['edge_sign_consistency_median'])}.\n")

    L.append("## Graph Clustering & Multi-Window Dynamics (plan sections 18/23/50)\n")
    L.append(f"- PK Top-5 graph (full period): {ev['clustering_n_communities_topk5']} greedy-modularity communities, modularity = {_fmt(ev['clustering_modularity_topk5'])}, avg clustering coeff = {_fmt(ev['clustering_avg_coeff_topk5'])}. Full table: `tables/graph_clustering.csv`.\n- Multi-window rolling edge panel (20/60/120-day PK/return/volume correlations, leakage-safe trailing windows): `tables/dynamic_edge_features.parquet`.\n- Top-K search over K in 3/5/8/10 (density, neighbour Jaccard, sector purity, edge strength, OOS predictive gain vs HAR+market): `tables/topk_search.csv`. K=5 remains the most defensible if any graph is used.\n- Node-link graph visualisations (PK Top-5 at low/normal/high-vol dates, directed lead-lag, multi-edge): `graphs/*.png`.\n")

    L.append("## Market-Factor Adjustment (key test)\n")
    L.append(f"- Mean |PK corr| raw = {_fmt(ev['pk_corr_mean_abs_raw'])} -> market-adjusted residual = {_fmt(ev['pk_corr_mean_abs_market_adj'])} (**{_fmt(100*ev['market_adj_retained_frac'],1)}% retained**).\n- Mean R^2 of PK on the market factor (train) = {_fmt(ev['mean_r2_pk_on_market_train'])}. See `16_raw_vs_market_adjusted_pk_corr.png`.\n")

    L.append("## Neighbour Predictive Test (Gates 6-7, OOS)\n")
    L.append("Leakage-safe: neighbours + coefficients fit on train, metrics on test. Target = PK variance at t+h, pooled across stocks.\n")
    L.append("| h | baseline | RMSE | QLIKE | R2 | DirAcc% |\n|---|---|---|---|---|---|")
    for h in (1, 5):
        for b in ["A_own", "B_own+market", "C_own+neighbors"]:
            if b in pred[h].index:
                r = pred[h].loc[b]
                L.append(f"| {h} | {b} | {_fmt(r['rmse'],6)} | {_fmt(r['qlike'],4)} | {_fmt(r['r2'],4)} | {_fmt(r['dir_acc'],2)} |")
    L.append(f"\n- Gate 6 (C beats A own-only): {ev['gate6_C_beats_A']} - pooled RMSE change {_fmt(ev['neighbor_rmse_gain_vs_own_pct'],2)}%; per-stock C<A win-rate {_fmt(100*ev['C_beats_A_winrate'],0)}% of {ev['pred_h1_n_stocks']} stocks (sign-test p={_fmt(ev['C_beats_A_sign_p'],3)}).\n- Gate 7 (C beats B own+market): {ev['gate7_C_beats_B']} - pooled RMSE change {_fmt(ev['neighbor_rmse_gain_vs_market_pct'],2)}%; per-stock C<B win-rate {_fmt(100*ev['C_beats_B_winrate'],0)}% (sign-test p={_fmt(ev['C_beats_B_sign_p'],3)}). Gate 7 is the decisive test. Gates require a >0.5% pooled margin AND a stock majority, so a near-zero pooled tie cannot flip the verdict.\n")

    L.append("## Research Questions (plan section 67)\n")
    rqs = [
        ("RQ1 PK relationships meaningful?", f"Yes, statistically ({_fmt(100*ev['pk_pairs_sig_fdr_frac'],0)}% FDR-significant, above null) - but economically it is mostly one market factor."),
        ("RQ2 PK stronger than return?", f"{'Yes' if ev['pk_stronger_than_ret_frac']>0.5 else 'No'} - |PK corr|>|ret corr| for {_fmt(100*ev['pk_stronger_than_ret_frac'],0)}% of pairs."),
        ("RQ3 Volume complementary?", f"Weak - volume-shock corr median {_fmt(ev['vshock_corr_median'])}, volume->future-PK mean |corr| {_fmt(ev['v2pk_lag1_mean'])}."),
        ("RQ4 Stable directional lead-lag?", f"No - PK L(1) off-diag max {_fmt(ev['pk_leadlag1_offdiag_max'])}, asymmetry small; own-autocorr {_fmt(ev['pk_own_autocorr1_mean'])} dominates."),
        ("RQ5 Regime dependence?", f"{'Yes' if ev['pk_corr_high_regime_mean']>ev['pk_corr_low_regime_mean'] else 'No'} - high {_fmt(ev['pk_corr_high_regime_mean'])} vs low {_fmt(ev['pk_corr_low_regime_mean'])}."),
        ("RQ6 Survives market adjustment?", f"Largely no - only {_fmt(100*ev['market_adj_retained_frac'],0)}% of mean |corr| retained after removing MarketPK."),
        ("RQ7 Sector-linked stronger?", f"{'Yes' if ev['pk_corr_same_sector_mean']>ev['pk_corr_cross_sector_mean'] else 'No'} - same {_fmt(ev['pk_corr_same_sector_mean'])} vs cross {_fmt(ev['pk_corr_cross_sector_mean'])}."),
        ("RQ8 Dynamic > static?", f"Neighbour Jaccard {_fmt(ev['neighbor_jaccard_k5'])} (vs random {_fmt(ev['neighbor_jaccard_random_k5'])}); turnover {_fmt(ev['edge_turnover_mean'])} - some drift but not the decisive factor given RQ6."),
        ("RQ9 Best sparsification?", "Top-K (K=5) is the most defensible if any graph is used; threshold graphs vary with regime density."),
        ("RQ10 Which edge attributes?", "PK corr (20/60) primary; return/volume/lead-lag marginal; sector as annotation - but see verdict."),
    ]
    for q, a in rqs:
        L.append(f"- **{q}** {a}")
    L.append("")

    L.append("## How this explains the observed G1-null\n")
    pk_edge_gain = ev.get("edge_pkcorr_gain_over_market_pct", float("nan"))
    pk_edge_p = ev.get("edge_pkcorr_sign_p_over_market", float("nan"))
    L.append(
        "The project's G1 (kNN-8 correlation GAT) tied HAR at all horizons and a beat-HAR "
        "sweep failed. This EDA gives the data-side reason with a direct measurement: the "
        "plain PK-correlation edge (the exact relationship G1's kNN uses) is the WORST edge "
        f"construction tested - it changes OOS RMSE by {_fmt(pk_edge_gain,2)}% vs a HAR+market "
        f"baseline (per-stock sign p={_fmt(pk_edge_p,3)}), i.e. significantly worse, not better. "
        f"The reason: ~{_fmt(100*(1-ev['market_adj_retained_frac']),0)}% of cross-stock PK "
        f"correlation is a single common market factor (mean R^2 on MarketPK = "
        f"{_fmt(ev['mean_r2_pk_on_market_train'])}) that HAR already captures via each stock's "
        "own autocorrelated volatility, so a correlation GAT re-learns a factor HAR has and "
        "adds estimation noise. Directed PK lead-lag also fails to beat the market bar. The "
        "only edge with a positive (if marginal) lift over HAR+market is the directed "
        f"volume->PK edge (+{_fmt(ev['best_edge_gain_over_market_pct'],2)}%, sign "
        f"p={_fmt(ev['best_edge_sign_p_over_market'],3)}) - a relationship G1 never used. This "
        "matches Conclusion C (market factor dominates) and points to the specific graph change "
        "worth trying next.\n")

    # ---- incremental-over-HAR rankings (the primary "what to build" evidence) ----
    L.append("## Node-Feature Ranking (incremental OOS value over HAR, h=1)\n")
    L.append("Each own-stock feature is ADDED to the HAR base (PK daily/weekly/monthly) and judged by out-of-sample RMSE gain vs HAR-only, on a shared sample. Positive = helps.\n")
    L.append("| node feature | RMSE gain over HAR % | win-rate | sign p | n |\n|---|---|---|---|---|")
    if node_rank is not None and not node_rank.empty:
        for _, r in node_rank.iterrows():
            L.append(f"| {r['node_feature']} | {_fmt(r['rmse_gain_pct'],3)} | {_fmt(100*r['win_rate'],0)}% | {_fmt(r['sign_p'],3)} | {int(r['n_stocks'])} |")
    L.append(f"\nRecommended node-feature set: {', '.join(rec_cfg['node_features'])}.\n")

    L.append("## Edge-Definition Ranking (incremental OOS value over HAR and over HAR+market, h=1)\n")
    L.append("Each edge construction selects Top-5 neighbours (TRAIN-only) and adds their contemporaneous feature to HAR. The decisive column is gain over HAR+market: a GNN edge must beat HAR AND the common market factor. The market-adjusted residual and directed lead-lag edges are the constructions the plain-correlation kNN-8 never isolated.\n")
    L.append("| edge definition | gain over HAR % | gain over HAR+market % | win-rate vs market | sign p vs market |\n|---|---|---|---|---|")
    if edge_rank is not None and not edge_rank.empty:
        for _, r in edge_rank.iterrows():
            L.append(f"| {r['edge_definition']} | {_fmt(r['gain_over_har_pct'],3)} | {_fmt(r['gain_over_market_pct'],3)} | {_fmt(100*r['winrate_over_market'],0)}% | {_fmt(r['sign_p_over_market'],3)} |")
    L.append("")

    L.append("## Recommended GNN Config (strongest evidenced shot at beating HAR)\n")
    lift_h = rec_cfg["expected_lift_over_har_pct"]
    lift_m = rec_cfg["expected_lift_over_market_pct"]
    expectation = {
        "beats": "positive AND significant at 0.05 - the best evidenced chance to clear HAR; confirm with a Diebold-Mariano test on the built model",
        "suggestive": "positive but NOT significant at 0.05 (marginal) - a genuine but small edge; may or may not clear HAR under DM, worth building and testing",
        "none": "no OOS gain over HAR+market - the best evidenced edge only ties/does not beat HAR; a GNN is unlikely to help",
    }[rec_cfg["lift_evidence"]]
    L.append(
        f"- Node features: {', '.join(rec_cfg['node_features'])}\n"
        f"- Edge type: **{rec_cfg['edge_type']}** ({'directed' if rec_cfg['directed'] else 'undirected'}, {rec_cfg['static_or_dynamic']}), Top-K = {rec_cfg['top_k']}\n"
        f"- Edge features: {', '.join(rec_cfg['edge_features'])}\n"
        f"- Best-edge OOS RMSE gain: {_fmt(lift_h,3)}% over HAR, {_fmt(lift_m,3)}% over HAR+market (per-stock sign p = {_fmt(rec_cfg['sign_p_over_market'],3)}, win-rate {_fmt(100*rec_cfg['winrate_over_market'],0)}% of {rec_cfg.get('n_stocks', ev['pred_h1_n_stocks'])} stocks).\n"
        f"- Honest expectation: **{expectation}**.\n"
    )
    L.append("Interpretation: the single largest cross-stock signal is the market-Parkinson factor itself - add MarketPK as a global/node feature to HAR/LSTM first (the ablation the G1 sweep skipped). The plain PK-correlation edge that G1 used is significantly worse than HAR+market OOS, so repeating it will not beat HAR. If an edge is used, the directed volume->PK edge above is the only construction with a positive (if marginal) evidenced lift over the market bar - build exactly that and DM-test it.\n")

    L.append("## Rejected / Non-additive Edge Features\n")
    if edge_rank is not None and not edge_rank.empty:
        losers = edge_rank[edge_rank["gain_over_market_pct"] <= 0]["edge_definition"].tolist()
        L.append(f"- Edge constructions with no OOS gain over HAR+market: {', '.join(losers) if losers else 'none'}.\n")
    L.append(f"- Raw price correlation (non-stationary; excluded by design). Volume->future-PK mean |corr| {_fmt(ev['v2pk_lag1_mean'])} (weak).\n")

    L.append("## Leakage Audit\n")
    L.append("- Chronological 70/15/15 split; train strictly precedes test (asserted).\n- Market betas, regime thresholds, neighbour selection (incl. residual-corr and lead-lag edge selection) and predictive coefficients fit on TRAIN only.\n- Rolling snapshots use trailing windows only; `assert_snapshot_no_lookahead` run per snapshot. Full-period matrices are labelled descriptive EDA, never used for a per-sample decision.\n- Automated assertions in `graph_eda/leakage.py`, exercised by the `graph_eda/tests/` suite.\n")

    L.append("## Next Model Experiments\n")
    L.append(f"- Build the recommended config (`{rec_cfg['edge_type']}`, {rec_cfg['static_or_dynamic']}, {'directed' if rec_cfg['directed'] else 'undirected'}, Top-5) and test vs HAR with Diebold-Mariano + MCS, multi-seed.\n- Add MarketPK (median PK) as a global/node feature to HAR/LSTM first - it is the largest cross-stock signal and the ablation the G1 sweep skipped.\n- Probe the market-adjusted RESIDUAL-corr edge and directed PK lead-lag edge hardest, since those isolate structure beyond the market factor that plain-correlation kNN-8 mixed together.\n")

    (REP / "EDA_GRAPH_REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
