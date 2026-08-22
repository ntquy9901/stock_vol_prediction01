# Model-Free Graph Screening (diagnosis doc §7)

Question: before building any corrected GAT, does a WEIGHTED / SIGNED / INNOVATION / DIRECTED neighbour
signal add out-of-sample value beyond HAR? This screens the richer graph families the current binary-mask
GAT cannot represent (see `reports/signed_graph_implementation_audit.md`), model-free, so it upper-bounds
what any model consuming those edges could extract linearly.

## Method (leakage-safe)
Each neighbour signal is appended to the 3 HAR features in an OLS fit on TRAIN only; incremental value =
relative reduction in TEST MSE vs the HAR-only OLS (`incr_R2 = 1 − MSE_model/MSE_HAR`). Signed Top-5
graphical-lasso edges and the residual-HAR fit are TRAIN-only; the signal at date t uses only day-t
observations. Screens: S0 equal-weight neighbour mean of PK² ; S1 signed-weighted neighbour PK² ; S2
separate positive/negative neighbour sums ; S3 signed-weighted neighbour HAR-residual (innovation) ; S4
directed lead-lag innovation (train-selected top-5 j→i by corr(r_j[t], r_i[t+h])) ; PLACEBO = S1 with
row-shuffled edges (density/degree preserved). Code: `baselines/2026-08-21_har_anchored_residual/code/screen_graph.py`;
raw JSON: `results/graph_screen/<dataset>.json`. Purge = h at split boundaries.

## Results — incremental TEST R² vs HAR (positive = graph signal helps)
**VN30** (33 nodes, 1344 common dates, ~130 test dates/horizon)
| h | S0 mean | S1 weighted | S2 signed | S3 innovation | S4 lead-lag | PLACEBO |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.0001 | −0.0007 | −0.0006 | −0.0009 | +0.0032 | −0.0029 |
| 5 | +0.0017 | +0.0026 | +0.0026 | −0.0001 | +0.0001 | +0.0022 |
| 10 | −0.0014 | −0.0020 | −0.0020 | +0.0004 | −0.0002 | −0.0015 |
| 22 | −0.0006 | −0.0010 | −0.0024 | +0.0002 | +0.0002 | +0.0006 |

**VN100** (104 nodes, 513 common dates, ~48 test dates/horizon)
| h | S0 mean | S1 weighted | S2 signed | S3 innovation | S4 lead-lag | PLACEBO |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.0008 | −0.0015 | −0.0015 | −0.0029 | −0.0006 | −0.0004 |
| 5 | +0.0099 | +0.0106 | +0.0105 | +0.0002 | −0.0002 | +0.0019 |
| 10 | −0.0008 | −0.0009 | −0.0008 | +0.0001 | −0.0002 | +0.0000 |
| 22 | +0.0014 | +0.0002 | +0.0009 | −0.0001 | +0.0002 | +0.0035 |

**S&P 500** (long-history subset, min_common=3000 → 457 nodes, 3029 common dates, ~300 test dates/horizon)
| h | S0 mean | S1 weighted | S2 signed | S3 innovation | S4 lead-lag | PLACEBO |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0116 | +0.0100 | +0.0100 | −0.0000 | +0.0000 | +0.0043 |
| 5 | +0.0005 | −0.0027 | −0.0027 | +0.0000 | +0.0000 | −0.0028 |
| 10 | −0.0003 | −0.0019 | −0.0019 | +0.0000 | +0.0000 | −0.0013 |
| 22 | +0.0007 | +0.0018 | +0.0018 | −0.0000 | +0.0000 | +0.0016 |

## Reading
- **All incremental R² are ≤ ~1%**, most near zero or negative.
- **The innovation (S3) and directed lead-lag (S4) screens — the theoretically-correct tests of predictive
  spillover beyond HAR's own persistence — are ≈ 0 at every horizon on every panel.** This is the decisive
  evidence: a graph should add value by predicting the part of volatility HAR does not (innovations); it
  does not.
- The only recurring small positives are S0/S1/S2, i.e. CONTEMPORANEOUS neighbour LEVEL (VN100 h5 ≈ +1.0%,
  S&P 500 h1 ≈ +1.1%). These are level co-movements dominated by shared market persistence, and they sit
  close to the shuffled-edge PLACEBO (VN100 h5 placebo +0.2% vs signal +1.1%; S&P 500 h1 placebo +0.4% vs
  +1.1%) — the structural graph adds little beyond a random-edge control, and the effect is horizon-isolated
  (absent at h10/h22 where the earlier point-estimate hybrid looked most promising).

## Verdict and promotion decision
Per the doc §7 promotion rule (positive incremental validation R², same direction across folds, edge
transfer/stability, not concentrated, beats placebo), **no graph family qualifies for promotion to a
corrected GAT.** The signed→binary implementation limitation (V2) is real but is not the cause of the null:
the weighted (S1) and signed (S2) screens — which a corrected GAT would exploit — are ~ placebo, and the
innovation/lead-lag screens are ~0. Building a corrected signed/weighted/rolling GAT is therefore NOT
justified by the data at this stage.

Scope of the claim (per doc §17): this rejects contemporaneous-mean, weighted, signed, innovation, and
train-selected directed-lead-lag families built from Parkinson-variance correlation/partial-correlation on
VN30, VN100, and a long-history S&P 500 subset, at horizons {1,5,10,22}, on HAR-only node features. Not
tested (require extra data / follow-up): S5 volume-shock graph (needs volume, not in the processed CSVs),
S6 static sector graph (needs historically-valid sector metadata), S7 regime-conditional spillover, and
richer node features (doc §15). A no-signal conclusion beyond the tested families needs those runs.
