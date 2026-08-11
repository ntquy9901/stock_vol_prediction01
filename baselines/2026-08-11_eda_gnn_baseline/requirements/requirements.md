# EDA-recommended GNN — Specify (spec.md)

## Goal
Build and honestly test the GNN configuration the graph EDA recommended
(`docs/eda/graph_recommendation.json`, `docs/eda/reports/EDA_GRAPH_REPORT.md`), to see whether it
can beat the HAR baseline out-of-sample. The user wants a GNN in the final model; this baseline
identifies the strongest evidenced GNN config and measures its real lift over HAR — no fabricated
win, no leakage.

## The EDA verdict being acted on (Conclusion C)
- ~77% of cross-stock Parkinson (PK) correlation is one common market factor HAR already captures.
- The plain PK-correlation kNN-8 edge (the current G1) is significantly WORSE than HAR+market OOS
  (−2.20%, sign p=0.014).
- Only components with positive/neutral incremental value over HAR+market:
  - node feature `volume_zscore_20` (+0.55% RMSE, sign p≈1.3e-7),
  - edge `edge_vol2pk_dir` = directed volume→PK lead-lag Top-5 (+0.17% over HAR+market, sign p=0.08,
    marginal),
  - and the largest single untried lever: **MarketPK** (cross-sectional median PK) as a global node
    feature (the market factor itself, which the G1 sweep never fed the model).
- EDA honest expectation: `likely_beats_har_under_dm = false` (marginal, small edge).

## Config (the recommended GNN)
- Node features: HAR(3 scales: pk_daily, pk_weekly, pk_monthly) + `volume_zscore_20` + `MarketPK`
  (global/broadcast node feature).
- Graph edge: directed volume→PK lead-lag, Top-K=5, neighbour sets estimated on the TRAIN window
  only and frozen (leakage-safe); the correlation edge is dropped.
- MarketPK at t = cross-sectional median PK at t (contemporaneous feature, allowed).
- Reuse the pilot pipeline: same leakage-safe basis + same val/test observations as
  `ladder_consistent` (h5) so HAR is directly comparable.

## Ablation ladder (isolate what helps vs HAR)
Same basis, seeds 42/123/2026, 20 epochs, horizon 5. All 6 metrics (MSE/RMSE/MAE/R²/QLIKE/DirAcc)
val+test, plus Diebold–Mariano (per-observation, test) vs HAR (E0) and vs a controlled current-G1
edge (correlation kNN-8 on identical features/backbone).
- E0 = HAR (reference; 3 HAR features, pooled linear regression).
- E1 = price-LSTM backbone + MarketPK (4 features, no graph) — the biggest single lever alone.
- E2 = E1 + volume_zscore_20 (5 features, no graph).
- E3 = E2 + directed volume→PK Top-5 graph (the recommended full config).
- Controls: E3off = E3 with message-passing disabled (same weights, nested "remove the graph");
  G1corr = E3's backbone/features with the correlation kNN-8 edge (controlled "current-G1" edge).
- (optional) E3 trained with a QLIKE loss if cheap.

## Acceptance criteria (go/no-go)
1. **Leakage-safe (hard gate):** every edge/market/volume statistic used for a per-sample decision
   is estimated on TRAIN dates only and frozen; volume_zscore_20 is a trailing (causal) rolling
   z-score; MarketPK is contemporaneous only. Automated tests assert: (a) the vol2pk neighbour
   matrix uses no date > train_end; (b) the frozen adjacency is identical across all snapshots;
   (c) trailing z-score does not respond to a future spike; (d) MarketPK at t uses only column t;
   (e) present-node masking never imports an absent node's value.
2. **Same observation set:** adding the two features does not change which windows are eligible
   (monthly-HAR rolling-22 remains the binding warm-up), so E0 reproduces the existing HAR P0 and
   the val/test (id,date) set matches `ladder_consistent`. Asserted numerically.
3. **Positivity / scalers preserved:** per-ticker target scalers and the positivity floor are
   unchanged; nonpositive-prediction rate ≤ 1%.
4. **Every number from a real run:** 3 seeds × {E0,E1,E2,E3,E3off,G1corr}; DM verdicts computed on
   real per-observation dumps.
5. **Honest verdict:** state for each rung whether it beats HAR on any metric (esp. QLIKE) and
   whether the difference is DM-significant. A partial win (QLIKE, DM-significant) counts. If
   nothing beats HAR, report the null result; the recommended config remains the proposed strongest
   GNN, but no win is claimed that the DM test does not support.

## Out of scope
- Retuning k, hidden size, or horizon; multi-horizon; news features (the EDA config is price-only).
- Modifying other baselines' code (`data.py`, `models.py`, `train.py`, `scaling.py`,
  `run_pilot.py`) — imported read-only.

## LPB (no OHLCV volume)
LPB is in the 33-ticker basis but has no volume series. To preserve the observation set,
volume_zscore_20 for LPB is set to 0.0 (neutral z-score) and LPB is excluded as a vol2pk *source*
(it can still be a target with a self-loop). Documented as a data limitation.
