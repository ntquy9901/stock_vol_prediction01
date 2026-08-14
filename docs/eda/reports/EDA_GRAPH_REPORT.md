# EDA Graph Report - Daily VN30 Parkinson Volatility

Executes `docs/eda_guide/parkinson_volatility_gnn_eda_experiment_plan.md`. 
Target = Parkinson VARIANCE (sigma^2, the project target); `pk_vol` = sqrt(pk_var) is the plan's default and is used for cross-stock correlation. Both retained.

## Executive Summary

**Should a GNN be used: MAYBE.**

Conclusion C - Market factor dominates; use HAR/LSTM + market factor.

Framing: the bar is HAR (each stock's own daily/weekly/monthly Parkinson features). A GNN is justified only if a cross-stock edge lowers out-of-sample PK error beyond HAR AND beyond a market factor. Findings (all numbers from generated tables):
- Cross-stock PK co-movement is real (mean |PK corr| = 0.3778 >> shuffle-null p95 0.0233, p=0.002), but 77% of it is one market factor (mean R^2 on MarketPK 0.4241; only 23.1% survives market adjustment).
- The plain PK-correlation edge (what the project's G1 kNN-8 GAT uses) is the WORST edge: -2.20% OOS RMSE vs HAR+market (sign p=0.014) - significantly worse. This is the direct data reason G1 tied/failed HAR.
- The ONLY edge with a positive lift over HAR+market is the **directed volume->PK lead-lag** edge: +0.17% RMSE, 67% of stocks, sign p=0.080 (marginal, not significant at 0.05).
- The only own node feature that beats HAR is volume_zscore_20 (+0.55% RMSE, sign p<1e-6).
Recommended config is in the section below and in `graph_recommendation.json`: build the directed volume->PK Top-5 GNN with HAR+volume_zscore node features and DM-test it - it is the strongest evidenced shot, though the expected lift is small.

## Decision Matrix (plan section 68)

| Evidence | Level |
|---|---|
| PK cross-stock dependence | Moderate |
| Null-test separation | Strong |
| Out-of-sample stability | Weak |
| Lead-lag directionality | Weak |
| Regime dependence | Strong |
| Sector structure | Strong |
| Market-adjusted dependence | Weak |
| Neighbor predictive gain | Weak |
| Dynamic graph justification | Strong |

**Final recommendation:** NO GNN (LSTM/HAR + market factor)

## Data Summary

- Tickers: 33 (VN30 universe incl. LPB)
- Common panel: 1296 trading days, 2021-03-24 to 2026-06-09
- Alignment: intersection of trading dates across all 33 tickers, no forward-fill. VPB/VRE tz-aware dates normalised; price-scale differences immaterial (log-ratios).
- Data quality: 0 high<low rows, 1 nonpositive-price rows, 0 duplicate dates across the universe (see `tables/data_quality_summary.csv`).

## Parkinson Volatility Summary

- Mean |PK corr| (full period) = 0.3778 (95% bootstrap CI [0.3655, 0.3903]), median PK corr = 0.3707. FDR-significant pairs (q<0.05): 99.4%.
- High-vol regime mean |PK corr| = 0.2686 vs low-vol 0.0789 (train-defined thresholds).

## Return / Volume Relationships

- Mean |return corr| = 0.3563; median volume-shock corr = 0.2215.
- PK vs return: |PK corr| exceeds |return corr| for 66.3% of pairs (mean D = 0.0215). See `05_pk_vs_return_corr_scatter.png`.

## Lead-Lag Findings

- PK L(1) off-diagonal mean = 0.2466, max = 0.4868; own lag-1 autocorr mean = 0.4712.
- Directional asymmetry |A_ij| mean = 0.0245, max = 0.0857. Volume->future-PK L(1) mean |corr| = 0.0469.
- Lead-lag null: observed mean|L(1)|=0.2472 vs shuffle-null p95=0.0230.

## Sector Findings

- Same-sector mean PK corr = 0.5003 (95% CI [0.4738, 0.5285]) vs cross-sector = 0.3450 (95% CI [0.3342, 0.3564]) (Mann-Whitney one-sided p = 0.0000); non-overlapping CIs.
- After market adjustment: same-sector residual corr = 0.0912 vs cross = -0.0298. Sector map is constructed (no metadata file); see `graph_eda/sectors.py`.

## Dynamic Graph Evidence

- Top-5 neighbour Jaccard (consecutive 60d/21d snapshots) = 0.3900 vs random-control 0.0917. Edge turnover mean = 0.5982.
- Edge rolling-corr: median mean_corr = 0.3015, median std = 0.1978, median sign-consistency = 0.9322.

## Graph Clustering & Multi-Window Dynamics (plan sections 18/23/50)

- PK Top-5 graph (full period): 4 greedy-modularity communities, modularity = 0.3059, avg clustering coeff = 0.5477. Full table: `tables/graph_clustering.csv`.
- Multi-window rolling edge panel (20/60/120-day PK/return/volume correlations, leakage-safe trailing windows): `tables/dynamic_edge_features.parquet`.
- Top-K search over K in 3/5/8/10 (density, neighbour Jaccard, sector purity, edge strength, OOS predictive gain vs HAR+market): `tables/topk_search.csv`. K=5 remains the most defensible if any graph is used.
- Node-link graph visualisations (PK Top-5 at low/normal/high-vol dates, directed lead-lag, multi-edge): `graphs/*.png`.

## Market-Factor Adjustment (key test)

- Mean |PK corr| raw = 0.3778 -> market-adjusted residual = 0.0874 (**23.1% retained**).
- Mean R^2 of PK on the market factor (train) = 0.4241. See `16_raw_vs_market_adjusted_pk_corr.png`.

## Neighbour Predictive Test (Gates 6-7, OOS)

Leakage-safe: neighbours + coefficients fit on train, metrics on test. Target = PK variance at t+h, pooled across stocks.

| h | baseline | RMSE | QLIKE | R2 | DirAcc% |
|---|---|---|---|---|---|
| 1 | A_own | 0.000501 | 0.3801 | 0.2963 | 33.41 |
| 1 | B_own+market | 0.000513 | 0.4018 | 0.2605 | 39.43 |
| 1 | C_own+neighbors | 0.000517 | 0.4185 | 0.2489 | 40.83 |
| 5 | A_own | 0.000540 | 0.4557 | 0.1874 | 48.60 |
| 5 | B_own+market | 0.000555 | 0.4903 | 0.1424 | 48.73 |
| 5 | C_own+neighbors | 0.000553 | 0.5096 | 0.1491 | 49.23 |

- Gate 6 (C beats A own-only): False - pooled RMSE change -3.31%; per-stock C<A win-rate 24% of 33 stocks (sign-test p=0.005).
- Gate 7 (C beats B own+market): False - pooled RMSE change -0.78%; per-stock C<B win-rate 42% (sign-test p=0.487). Gate 7 is the decisive test. Gates require a >0.5% pooled margin AND a stock majority, so a near-zero pooled tie cannot flip the verdict.

## Research Questions (plan section 67)

- **RQ1 PK relationships meaningful?** Yes, statistically (99% FDR-significant, above null) - but economically it is mostly one market factor.
- **RQ2 PK stronger than return?** Yes - |PK corr|>|ret corr| for 66% of pairs.
- **RQ3 Volume complementary?** Weak - volume-shock corr median 0.2215, volume->future-PK mean |corr| 0.0469.
- **RQ4 Stable directional lead-lag?** No - PK L(1) off-diag max 0.4868, asymmetry small; own-autocorr 0.4712 dominates.
- **RQ5 Regime dependence?** Yes - high 0.2686 vs low 0.0789.
- **RQ6 Survives market adjustment?** Largely no - only 23% of mean |corr| retained after removing MarketPK.
- **RQ7 Sector-linked stronger?** Yes - same 0.5003 vs cross 0.3450.
- **RQ8 Dynamic > static?** Neighbour Jaccard 0.3900 (vs random 0.0917); turnover 0.5982 - some drift but not the decisive factor given RQ6.
- **RQ9 Best sparsification?** Top-K (K=5) is the most defensible if any graph is used; threshold graphs vary with regime density.
- **RQ10 Which edge attributes?** PK corr (20/60) primary; return/volume/lead-lag marginal; sector as annotation - but see verdict.

## How this explains the observed G1-null

The project's G1 (kNN-8 correlation GAT) tied HAR at all horizons and a beat-HAR sweep failed. This EDA gives the data-side reason with a direct measurement: the plain PK-correlation edge (the exact relationship G1's kNN uses) is the WORST edge construction tested - it changes OOS RMSE by -2.20% vs a HAR+market baseline (per-stock sign p=0.014), i.e. significantly worse, not better. The reason: ~77% of cross-stock PK correlation is a single common market factor (mean R^2 on MarketPK = 0.4241) that HAR already captures via each stock's own autocorrelated volatility, so a correlation GAT re-learns a factor HAR has and adds estimation noise. Directed PK lead-lag also fails to beat the market bar. The only edge with a positive (if marginal) lift over HAR+market is the directed volume->PK edge (+0.17%, sign p=0.080) - a relationship G1 never used. This matches Conclusion C (market factor dominates) and points to the specific graph change worth trying next.

## Node-Feature Ranking (incremental OOS value over HAR, h=1)

Each own-stock feature is ADDED to the HAR base (PK daily/weekly/monthly) and judged by out-of-sample RMSE gain vs HAR-only, on a shared sample. Positive = helps.

| node feature | RMSE gain over HAR % | win-rate | sign p | n |
|---|---|---|---|---|
| volume_zscore_20 | 0.548 | 94% | 0.000 | 33 |
| pk_lag2 | -0.087 | 45% | 0.728 | 33 |
| pk_std_20 | -0.278 | 42% | 0.487 | 33 |
| pk_vol_daily | -0.309 | 55% | 0.728 | 33 |
| log_return_1d | -1.200 | 30% | 0.035 | 33 |
| return_5d | -2.019 | 27% | 0.014 | 33 |

Recommended node-feature set: pk_daily, pk_weekly, pk_monthly, volume_zscore_20.

## Edge-Definition Ranking (incremental OOS value over HAR and over HAR+market, h=1)

Each edge construction selects Top-5 neighbours (TRAIN-only) and adds their contemporaneous feature to HAR. The decisive column is gain over HAR+market: a GNN edge must beat HAR AND the common market factor. The market-adjusted residual and directed lead-lag edges are the constructions the plain-correlation kNN-8 never isolated.

| edge definition | gain over HAR % | gain over HAR+market % | win-rate vs market | sign p vs market |
|---|---|---|---|---|
| edge_vol2pk_dir | 0.236 | 0.175 | 67% | 0.080 |
| edge_residcorr | -0.761 | -0.925 | 30% | 0.035 |
| edge_leadlag_dir | -2.723 | -1.149 | 33% | 0.080 |
| edge_pkcorr_abs | -3.480 | -2.200 | 27% | 0.014 |
| edge_pkcorr_pos | -3.480 | -2.200 | 27% | 0.014 |
| edge_multi | -3.614 | -2.213 | 24% | 0.005 |

## Recommended GNN Config (strongest evidenced shot at beating HAR)

- Node features: pk_daily, pk_weekly, pk_monthly, volume_zscore_20
- Edge type: **edge_vol2pk_dir** (directed, dynamic), Top-K = 5
- Edge features: volume_to_pk_lag1, volume_to_pk_lag5
- Best-edge OOS RMSE gain: 0.236% over HAR, 0.175% over HAR+market (per-stock sign p = 0.080, win-rate 67% of 33 stocks).
- Honest expectation: **positive but NOT significant at 0.05 (marginal) - a genuine but small edge; may or may not clear HAR under DM, worth building and testing**.

Interpretation: the single largest cross-stock signal is the market-Parkinson factor itself - add MarketPK as a global/node feature to HAR/LSTM first (the ablation the G1 sweep skipped). The plain PK-correlation edge that G1 used is significantly worse than HAR+market OOS, so repeating it will not beat HAR. If an edge is used, the directed volume->PK edge above is the only construction with a positive (if marginal) evidenced lift over the market bar - build exactly that and DM-test it.

## Rejected / Non-additive Edge Features

- Edge constructions with no OOS gain over HAR+market: edge_residcorr, edge_leadlag_dir, edge_pkcorr_abs, edge_pkcorr_pos, edge_multi.

- Raw price correlation (non-stationary; excluded by design). Volume->future-PK mean |corr| 0.0469 (weak).

## Leakage Audit

- Chronological 70/15/15 split; train strictly precedes test (asserted).
- Market betas, regime thresholds, neighbour selection (incl. residual-corr and lead-lag edge selection) and predictive coefficients fit on TRAIN only.
- Rolling snapshots use trailing windows only; `assert_snapshot_no_lookahead` run per snapshot. Full-period matrices are labelled descriptive EDA, never used for a per-sample decision.
- Automated assertions in `graph_eda/leakage.py`, exercised by the `graph_eda/tests/` suite.

## Next Model Experiments

- Build the recommended config (`edge_vol2pk_dir`, dynamic, directed, Top-5) and test vs HAR with Diebold-Mariano + MCS, multi-seed.
- Add MarketPK (median PK) as a global/node feature to HAR/LSTM first - it is the largest cross-stock signal and the ablation the G1 sweep skipped.
- Probe the market-adjusted RESIDUAL-corr edge and directed PK lead-lag edge hardest, since those isolate structure beyond the market factor that plain-correlation kNN-8 mixed together.
