# Overnight consolidated report — 2026-09-11

Autonomous overnight run covering: promotion of the SP500 refined champion to a committed baseline + its transfer
to VN, a cross-market feature probe, and faithful replication of 6 papers + 1 survey (5 requested by the user)
via parallel agents. Every experiment evaluated on OUR terms: QLIKE + date-clustered Diebold-Mariano vs HAR-X,
walk-forward, with random-edge placebos where a graph is involved.

## Headline
1. **NEW COMMITTED CHAMPION (SP500): `GBM + asymmetric-earnings + estimators + spike-propensity`** — beats the
   previously committed GBM+earn at ALL 4 horizons (+1.7 to +2.0% QLIKE, DM p<0.05 at all 4) and HAR-X by
   +12–21%, with no overfitting. Promoted to `baselines/2026-09-11_gbm_refined_champion`.
2. **The refined champion does NOT transfer to Vietnam** — honest negative (details below).
3. **All 7 external-method papers replicate to NO-GO / nothing-new on volatility-QLIKE.** No graph / cross-market /
   news / multi-source / technical-indicator method beats own-history HAR on QLIKE out-of-sample with DM
   significance. This overwhelmingly confirms the project thesis.

## 1. SP500 refined champion — GO (committed baseline)
Canonical walk-forward, vs HAR-X and vs the committed GBM+earn:

| h | HAR-X | GBM+earn | **GBM+refined** | refined vs HAR-X | refined vs GBM+earn | overfit gap |
|---|-------|----------|-----------------|------------------|---------------------|-------------|
| 1 | 0.4061 | 0.3256 | **0.3192** | +21.4% (p=.000) | +1.96% (p=.000) | −2.2% |
| 5 | 0.4550 | 0.4008 | **0.3930** | +13.6% (p=.000) | +1.94% (p=.000) | +3.3% |
| 10 | 0.4797 | 0.4278 | **0.4206** | +12.3% (p=.000) | +1.68% (p=.020) | +3.1% |
| 22 | 0.4959 | 0.4294 | **0.4221** | +14.9% (p=.000) | +1.70% (p=.009) | −10.0% |

The three refinements (asymmetric earnings pre/post, GK/RS/YZ estimators, spike-propensity) — each a small
significant probe earlier — combine into a coherent, non-overfitting +1.7–2.0% over GBM+earn. Independent
adversarial code review: **no CRITICAL/MAJOR leakage or correctness bug**; the one MAJOR (a test-adequacy gap on
the dual index-space) was fixed with a value-level alignment test (pytest 8/8). Full DoD in
`baselines/2026-09-11_gbm_refined_champion/`.

## 2. Refined champion on VN30 / VN100 / HOSE — honest NEGATIVE
On VN the earnings features are inert (vnstock has no per-firm earnings dates), so the model reduces to
`GBM + estimators + spike`. Results:
- **VN30**: refined ≈ HAR-X (+0.7/+0.3/−1.3/−5.1%, all n.s.) and slightly WORSE than a plain own-GBM (−0.3 to −1.1%).
- **VN100**: NUMERICALLY UNSTABLE — the 22-feature model overfits the thinner VN panel and blows up on some folds
  (h1 QLIKE = 21.3, h10 = 0.92 vs a ~0.6 baseline; near-floor gamma-prediction explosions).
- **HOSE**: partial (h5 completed ~0.5 region; h10/h22 runs did not produce output — same thin-market instability).

Conclusion: **the SP500 refined champion is SP500-specific.** Its power comes from earnings (absent on VN); its
extra features add variance, not signal, on the thinner VN markets. VN's committed winners remain the simpler
own-stock models (HARQ, deep+HARQ stack). This matches the technical-indicator probe (below), which also found
extra features degrade thin VN.

## 3. Paper replications (all requested by the user) — QLIKE / DM verdicts

| Paper | Model | What it is | Verdict on QLIKE (our terms) |
|---|---|---|---|
| fractalfract 2025 | **H-ETE-GNN** | country-ETF, Effective-Transfer-Entropy lag-1 (time-zone), Hurst regime | **NO-GO** — ETE ≈ no-graph (p=.71 h1, .21 h5); graph adds nothing; beats HAR only via node-encoder, not graph |
| 2409.15320 (DCRNN-HAR) | **DCRNN-HAR** | 8 global indices, dynamic Diebold-Yilmaz graph, all-trading-days | **NO-GO** — HAR beats it (h1/h5 p<.001); dynamic ≤ static; DY never beats no-graph. **Found a LEAK in the official repo (early-stop on TEST set)** → the paper's gains are inflated |
| Gong 2025 JEF | **ASTGCN** | 17 markets, DY-spillover, attention ST-GCN, VRP features | **NO-GO** — ties HAR h1/h22, sig worse h5; DY graph = placebo. Matches the paper's own QLIKE table (~1% avg; gains are MSE-driven) |
| Zhang 2025 ESWA | influence-weighted news | GCN account-influence × sentiment | **NO-GO** — news doesn't beat no-news; GCN influence-weighting NEVER beats equal-weighting (the paper's core claim); placebo mixed |
| Li 2022 MTA | **MSub-GNN** | multi-source fusion (trading+news+candlestick), metapath attention, fluctuation classification | **NO-GO** — fusion never beats trading-only; metapath ≈ concat; VN news covers only 17% of days |
| ACM Comput. Surv. (10.1145/3696411) | survey | systematic review of GNN-for-stock | **nothing-new** — volatility isn't even in its scope; every feasible lever maps to our already-NO-GO set; the one vol primary needs intraday LOB |
| thuvienchungkhoan/anfin | technical indicators | 12 TA indicators as GBM features | **NO-GO** — SP500 ≤0.63% (marginal h1/h10); VN significantly HURT; only ATR matters but is redundant with Parkinson-variance; volume/momentum ≈0 |

## 4. Cross-market feature probe (VIX + lagged global-index vol → SP500 GBM) — MIXED
Unlike the graph replications, this one shows genuine but two-sided signal: it strongly HELPS calm days (d1: +6–15%,
VIX being low on calm days corrects the champion's calm over-forecast) but HURTS elevated/storm days (d8–d10:
−7 to −22%). Net is horizon-dependent (h22 +4.63% p=.0016; h10 −3.37%). Not a clean adopt; VIX-only (dropping the
noisier lagged global indices) may be cleaner and is a documented follow-up.

## 5. Overarching thesis (now very strongly supported)
The single most robust finding across ~10 experiments tonight and the whole project: **cross-firm / cross-market /
external-information graphs add no out-of-sample QLIKE value over a market-factor + own-history HAR/GBM baseline,
because the cross-sectional volatility signal is contemporaneous (already in the market factor) and the residual
storm tail is exogenous / unforecastable at the origin.** The only levers that work are FORWARD-LOOKING,
SCHEDULED, per-firm own signals — earnings on SP500 (dominant), realized-quarticity / spike-propensity as small
refinements. Even the genuinely non-contemporaneous time-zone / cross-market angle (ETE lag-1, DCRNN dynamic
graph, ASTGCN, VIX) does not survive a leakage-free QLIKE + DM test.

## 6. Artifacts
- Committed baseline: `baselines/2026-09-11_gbm_refined_champion/` (+ results `results/gbm_refined_champion/gbmR_*.json`).
- Cross-market: `scripts/eda/sp500_crossmarket.py`, `results/gamma_gbm/crossmarket.json`.
- Paper probes (uncommitted throwaway, on disk): `scripts/eda/{hetegnn_probe,dcrnn_har_*,astgcn_*,vn_news_influence_gnn,msubgnn_*,techind_gbm_probe}.py`; per-agent summary reports under `docs/reports/2026-09-11_*`.
- Prior same-cycle: storm root-cause, feature-importance, M4 GATv2, overfit-edge, graph-spike (all NO-GO / diagnostic), in earlier reports + memory.

## 7. Git / commit status
Three agents auto-committed their throwaway eda probes (ASTGCN, MSub-GNN, DCRNN-HAR) locally; none reached origin
(all blocked by the pre-push coverage gate on untested probes). Per convention (scripts/eda/ probes are
throwaway/uncommitted), these were reset to the working tree; the ONLY commit pushed is the SP500 refined-champion
baseline (tested, gate-clean) + its results + this report. No force-push; no QG_SKIP.
