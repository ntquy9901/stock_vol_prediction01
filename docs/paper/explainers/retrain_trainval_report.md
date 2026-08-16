# Retrain-on-(train+val) variant — held-out test results and comparison to the train-only regime

## Purpose

The reported models use a per-ticker chronological 70/15/15 split: train (fit), validation (early
stopping / model selection), test (held-out final evaluation). This document reports a variant that
merges train and validation into a single training set, keeps the same held-out test, and re-scores
all rungs on that test. It answers two questions raised during review:

1. If validation is folded into training (more training data, no held-out early-stopping set), does
   the parsimony finding — no single configuration dominates HAR across horizons on the standard
   loss families — change?
2. Can the test set be used to check overfitting when there is no validation set? No: scoring the
   test to pick epochs/models is selection on test and leaks. The variant below avoids this by using
   a fixed epoch budget (no early stopping, no test-based selection); test is read exactly once.

## Setup

- Rungs (all retrained on train+val, evaluated on test): HAR (pooled linear, refit on train+val),
  FULL, minus_graph (GAT branch removed), minus_gate, minus_news, LSTM_only (price-only backbone).
- Fixed budget: 9 epochs, patience=0 (no early stopping). HAR is refit on train+val.
- Seed 42, single seed. The primary reported study uses a 5-seed ensemble; the numbers here are one
  seed and carry more sampling noise (see Caveats).
- Volume z-score window 22 (unified), directed vol→PK Top-5 edge frozen on the training portion.
- Code: `baselines/2026-08-15_volatility/code/run_retrain_trainval.py` (training),
  `dm_retrain.py` (Diebold-Mariano). Runs: `results/volatility_retrain_h{h}_seed42_2026-08-15_182005_retrain/`.
- No leakage: test is never used for selection; every model uses the same held-out test observations
  per horizon (n reported below). Scalers and the vol→PK edge are estimated on training data only.

## Test metrics (held-out test, seed 42)

RMSE and MAE are shown ×10³ for readability; MSE = RMSE². Lower is better for RMSE/MAE/QLIKE; higher
is better for R².

### h = 1  (n = 14596)
| Rung | RMSE ×10³ | MAE ×10³ | R² | QLIKE |
|---|---|---|---|---|
| HAR | 2.01 | 0.54 | 0.820 | 0.4789 |
| FULL | 1.98 | 0.53 | 0.825 | 0.4702 |
| minus_graph | 2.00 | 0.54 | 0.822 | 0.4731 |
| minus_gate | 1.98 | 0.53 | 0.825 | 0.4714 |
| minus_news | 1.99 | 0.53 | 0.824 | 0.4747 |
| LSTM_only | 2.00 | 0.54 | 0.822 | 0.4735 |

### h = 5  (n = 14464)
| Rung | RMSE ×10³ | MAE ×10³ | R² | QLIKE |
|---|---|---|---|---|
| HAR | 2.28 | 0.60 | 0.768 | 0.5699 |
| FULL | 2.31 | 0.60 | 0.763 | 0.5726 |
| minus_graph | 2.28 | 0.59 | 0.768 | 0.5703 |
| minus_gate | 2.30 | 0.60 | 0.764 | 0.5737 |
| minus_news | 2.29 | 0.60 | 0.767 | 0.5718 |
| LSTM_only | 2.28 | 0.59 | 0.769 | 0.5673 |

### h = 10  (n = 14299)
| Rung | RMSE ×10³ | MAE ×10³ | R² | QLIKE |
|---|---|---|---|---|
| HAR | 2.35 | 0.63 | 0.754 | 0.6103 |
| FULL | 2.36 | 0.63 | 0.753 | 0.6205 |
| minus_graph | 2.39 | 0.62 | 0.746 | 0.6151 |
| minus_gate | 2.30 | 0.62 | 0.765 | 0.6099 |
| minus_news | 2.31 | 0.62 | 0.764 | 0.6115 |
| LSTM_only | 2.33 | 0.62 | 0.760 | 0.6129 |

### h = 22  (n = 13903)
| Rung | RMSE ×10³ | MAE ×10³ | R² | QLIKE |
|---|---|---|---|---|
| HAR | 2.45 | 0.66 | 0.731 | 0.6710 |
| FULL | 2.50 | 0.64 | 0.720 | 0.6990 |
| minus_graph | 2.56 | 0.65 | 0.707 | 0.6930 |
| minus_gate | 2.57 | 0.64 | 0.705 | 0.7301 |
| minus_news | 2.49 | 0.65 | 0.722 | 0.6971 |
| LSTM_only | 2.49 | 0.65 | 0.721 | 0.6895 |

## Diebold-Mariano (HLN, HAC lag h−1) — FULL vs each rung, seed 42

Cells are `dm(p)`; the statistic is signed so a negative value favors FULL and a positive value favors
the comparator; `*` marks p < 0.05. Three loss families: QLIKE, squared error (SE; the MSE/RMSE/R²
family), absolute error (AE; the MAE family).

### FULL vs HAR
| Horizon | QLIKE | SE | AE |
|---|---|---|---|
| h1 | −5.40 (0.00)* FULL | −2.26 (0.02)* FULL | −2.81 (0.00)* FULL |
| h5 | +0.92 (0.36) tie | +1.54 (0.12) tie | −2.42 (0.02)* FULL |
| h10 | +3.85 (0.00)* HAR | +0.33 (0.74) tie | −0.78 (0.44) tie |
| h22 | +3.18 (0.00)* HAR | +1.42 (0.15) tie | −2.28 (0.02)* FULL |

### FULL vs other rungs (per horizon; sign convention as above)
| Horizon | vs minus_graph | vs minus_gate | vs minus_news | vs LSTM_only |
|---|---|---|---|---|
| h1 (QLIKE) | −4.79* FULL | −1.84 tie | −5.16* FULL | −5.22* FULL |
| h5 (QLIKE) | +2.58* worse | −1.54 tie | +0.67 tie | +5.04* worse |
| h10 (QLIKE) | +1.02 tie | +3.52* worse | +2.80* worse | +3.95* worse |
| h22 (QLIKE) | +1.14 tie | −1.85 tie | +0.23 tie | +1.08 tie |

(Full SE/AE cells per horizon are in the console dump; QLIKE shown here as the headline volatility loss.)

## Reading

- **h1 changes under retrain.** With validation folded into training, FULL beats HAR at the one-day
  horizon on all three loss families (QLIKE, SE, AE), each at p < 0.05. In the primary train-only
  5-seed study the FULL−HAR difference at h1 was not significant (bootstrap CI included zero). The
  extra training data and the removed early-stopping constraint help the deep model most at the
  shortest horizon.
- **h5 is mixed.** FULL beats HAR on AE (p = 0.02) and ties on QLIKE and SE. LSTM_only has the lowest
  QLIKE, and FULL is significantly worse than LSTM_only on QLIKE — the price-only backbone is the
  strongest configuration at this horizon.
- **h10 and h22: HAR retains the QLIKE edge.** HAR has significantly lower QLIKE than FULL at both
  h10 (dm +3.85) and h22 (dm +3.18); the SE family ties at both, and AE ties at h10 / favors FULL at
  h22. Adding the graph, gate, or news branch does not recover a QLIKE advantage over HAR at the long
  horizons.
- **Overall.** The retrain regime strengthens the deep model at h1 (a genuine, significant win over
  HAR there) but does not overturn HAR's QLIKE advantage at h10/h22, and the price-only LSTM remains
  the best deep configuration at h5. No single model dominates HAR across all horizons and all loss
  families — the parsimony reading of the primary study holds, with the qualification that at h1 the
  full model does beat HAR once validation is added to training.

## Caveats

- **Single seed.** These are seed-42 results; DM p-values do not average over seed variance the way
  the primary 5-seed ensemble does. The h1 FULL-over-HAR result is large (dm ≈ −5.4) and unlikely to
  be seed noise, but a multi-seed retrain (seeds 42/123/2026/7/2024) is the appropriate follow-up
  before citing the h1 reversal as a headline result.
- **Fixed 9-epoch budget.** With no held-out validation set, epochs cannot be selected without
  touching test. A fixed budget avoids leakage but is not tuned; a different budget could shift the
  deep-model numbers. HAR (linear, closed-form) is unaffected by the epoch budget.
- **Not a replacement for the reported split.** This variant removes the held-out early-stopping set,
  so it cannot itself detect overfitting during training. It is a robustness check on the final
  ranking, not the primary evaluation protocol.

## Files

- Training: `baselines/2026-08-15_volatility/code/run_retrain_trainval.py`
  (+ test `test/test_retrain_trainval.py`).
- DM: `baselines/2026-08-15_volatility/code/dm_retrain.py`
  (+ test `test/test_dm_retrain.py`).
- Raw metrics + per-observation test dumps:
  `results/volatility_retrain_h{1,5,10,22}_seed42_2026-08-15_182005_retrain/`.
