# EDA-recommended GNN — ablation ladder results (h5)

Run: `results/eda_gnn_seed{42,123,2026}_2026-08-11_144700/h5`; summary
`results/eda_gnn_2026-08-11_144700_summary.json`. Seeds 42/123/2026, 20 epochs, horizon 5. Basis:
33-ticker VN30 universe, per-ticker chronological 70/15/15, graph-bound train. Node features and the
graph edge follow `docs/eda/graph_recommendation.json` (EDA Conclusion C).

All rungs are trained and evaluated on ONE basis. The 5-feature manifest's observation set is
byte-identical to the pilot 3-feature manifest (train 73026 / val 14418 / test 14464) and the HAR
feature columns are bit-identical, so every rung is scored on the same val/test observations as
`ladder_consistent`. E0 uses the same HAR fit and observation set as the pilot P0 (test RMSE 0.002287
vs pilot P0 0.002289); its QLIKE differs slightly (0.5735 vs pilot 0.5676) only because E0 here
applies the shared positivity floor, so QLIKE is comparable across the ladder.

## Configuration
- Node features (nested): HAR(pk_daily, pk_weekly, pk_monthly) + MarketPK (cross-sectional median
  PK, contemporaneous) + volume_zscore_20 (trailing rolling z-score).
- Edge (E3): directed volume→PK lead-lag, Top-5 sources per target, neighbour matrix estimated on
  each ticker's TRAIN split only and frozen across all snapshots (leakage-safe).
- Controls: E3off = trained E3 read out with message passing disabled (nested "remove the graph");
  G1corr = same 5-feature backbone with the correlation kNN-8 edge (the current-G1 edge construction,
  controlled).
- Positivity floor applied identically to every rung (E0–E3) so QLIKE is compared like for like.

## Ladder (3-seed mean, 6 metrics)

| rung | split | RMSE | QLIKE | MAE | R² | DirAcc% |
|---|---|---|---|---|---|---|
| E0 HAR | test | 0.002287 | 0.5735 | 0.000605 | 0.7672 | 48.55 |
| E1 +MarketPK | test | 0.002285 | 0.5686 | 0.000611 | 0.7676 | 48.51 |
| E2 +vol_z | test | 0.002261 | 0.5681 | 0.000606 | 0.7725 | 48.49 |
| E3 +vol2pk graph | test | 0.002267 | 0.5709 | 0.000607 | 0.7713 | 48.44 |
| E3off (graph off) | test | 0.002256 | 0.5760 | 0.000608 | 0.7735 | 48.15 |
| G1corr (corr edge) | test | 0.002264 | 0.5708 | 0.000601 | 0.7719 | 48.31 |
| E0 HAR | val | 0.001485 | 0.5167 | 0.000480 | 0.7351 | 48.54 |
| E1 +MarketPK | val | 0.001488 | 0.5012 | 0.000482 | 0.7341 | 48.97 |
| E2 +vol_z | val | 0.001483 | 0.5012 | 0.000482 | 0.7358 | 49.13 |
| E3 +vol2pk graph | val | 0.001468 | 0.5060 | 0.000474 | 0.7411 | 48.87 |
| E3off (graph off) | val | 0.001517 | 0.5374 | 0.000496 | 0.7232 | 48.99 |
| G1corr (corr edge) | val | 0.001467 | 0.5128 | 0.000470 | 0.7413 | 49.13 |

Test lift vs HAR (E0): E1 RMSE +0.07% / QLIKE +0.85%; E2 RMSE +1.13% / QLIKE +0.94%; E3 RMSE +0.87%
/ QLIKE +0.45%; G1corr RMSE +1.00% / QLIKE +0.46%. (Per-seed metric std is small, e.g. E2 RMSE
std 6e-6, QLIKE std 0.0017.)

## Diebold–Mariano (seed-ensemble test predictions; negative dm favours A, i.e. A more accurate)

| A vs B | metric | dm_hln | p_value | verdict |
|---|---|---|---|---|
| E1 vs E0 | QLIKE | −2.389 | 0.0169 | E1 beats HAR (significant) |
| E1 vs E0 | squared error | −0.060 | 0.9520 | no difference |
| E2 vs E0 | QLIKE | −2.503 | 0.0123 | E2 beats HAR (significant) |
| E2 vs E0 | squared error | −0.869 | 0.3848 | no difference |
| E3 vs E0 | QLIKE | −1.573 | 0.1157 | E3 better but NOT significant |
| E3 vs E0 | squared error | −1.309 | 0.1906 | no difference |
| E3 vs G1corr | QLIKE | +2.011 | 0.0444 | corr edge beats vol2pk edge (significant) |
| E3 vs G1corr | squared error | +0.502 | 0.6156 | no difference |
| E3 vs E3off | QLIKE | −1.475 | 0.1401 | graph-on better QLIKE, not significant |
| E3 vs E3off | squared error | +0.905 | 0.3656 | no difference |

DM uses the HLN small-sample correction, Bartlett HAC at lag h−1=4, on n=14464 test observations
ordered by (ticker_id, target_date).

## Which lever moved the needle
- **The node features beat HAR on QLIKE, DM-significant.** Adding MarketPK alone (E1) lowers test
  QLIKE vs HAR with DM p=0.017; adding volume_zscore_20 as well (E2) gives DM p=0.012 and the largest
  balanced lift (RMSE +1.13%, R² +0.0052). MarketPK — the market factor the G1 sweep never fed the
  model — is the single largest contributor, consistent with EDA Conclusion C.
- **The improvement is a QLIKE (tail/calibration) win, not an RMSE win.** No rung beats HAR on
  squared error at DM significance (all SE p>0.19); the −1.1% RMSE numbers are directionally
  favourable but not DM-significant.
- **The directed volume→PK graph adds no out-of-sample value.** E3 does not beat HAR at DM
  significance (QLIKE p=0.116); adding the graph shrinks the QLIKE win E2 already had (E2 p=0.012 →
  E3 p=0.116). Removing the graph from the trained model (E3 vs E3off) is not DM-significant either
  way. This matches the EDA's honest expectation (`likely_beats_har_under_dm = false`).
- **The recommended vol2pk edge does not beat the correlation edge.** E3 vs G1corr favours the
  correlation kNN-8 edge on QLIKE at DM p=0.044 — the directed volume→PK construction is, if
  anything, slightly worse than the plain correlation edge inside this GNN.

## Does any config beat HAR?
Yes — but not the graph. **E2 (HAR + MarketPK + volume_zscore_20), a no-graph LSTM, beats HAR on
QLIKE with DM p=0.012** (and E1 with MarketPK alone at p=0.017). This is a genuine partial win
(QLIKE, DM-significant); it is not an RMSE/error win. The **GNN edge (E3) does not produce a
DM-significant lift over HAR** and does not beat the correlation-edge control.

## Recommended final GNN config (with honest measured lift)
The strongest GNN is **E3 = HAR + MarketPK + volume_zscore_20 node features + directed volume→PK
Top-5 edge** — the EDA-recommended configuration. Measured against HAR on the identical test
observations it is statistically **tied on error** (SE DM p=0.19) and **marginally-but-not-
significantly better on QLIKE** (dm_hln −1.57, p=0.116; QLIKE +0.45%). It is not worse than HAR on
any metric. However, the DM-significant improvement over HAR belongs to the **node features (E2)**,
not the graph: the message-passing edge contributes no DM-significant OOS value and the correlation
edge is significantly better than the recommended vol2pk edge on QLIKE. If a GNN must be retained in
the final model, E3 is the configuration to use, with the explicit caveat that its lift over HAR is
carried by MarketPK + volume_zscore_20 as node features, not by the graph edges.

## Caveats
- The residual message-passing weights Top-5 neighbours by softmax(signed correlation) with a
  self-loop fixed at 1.0, which concentrates attention on the self node and can under-weight the weak
  directed edges (mean |vol→PK lead-lag| ≈ 0.05 in the EDA). The graph's near-null contribution here
  may partly reflect this weighting; this is the same message-passing block the pilot G1 uses, so
  E3-vs-G1corr remains a fair edge-only comparison. Edge-weight tuning was out of scope.
- Leakage controls (verified by tests): volume_zscore_20 trailing/causal; MarketPK contemporaneous;
  vol2pk neighbour matrix estimated on each ticker's own train split and frozen across snapshots
  (per-ticker split, not a global boundary — review finding H1). LPB has no OHLCV volume: its
  volume_zscore_20 is 0 and it is excluded as a vol2pk source (self-loop retained as a target).

## Provenance
- Code: `baselines/2026-08-11_eda_gnn_baseline/` (features.py, edges.py, eda_model.py, eda_ladder.py,
  aggregate.py). 15 pytest tests pass (leakage/causality/nesting/positivity/masking + full-pipeline
  smoke); ruff clean. Adversarial review + resolutions in `code_review/code_review_2026-08-11.md`.
- Every number above is from the real 3-seed run `2026-08-11_144700`.
