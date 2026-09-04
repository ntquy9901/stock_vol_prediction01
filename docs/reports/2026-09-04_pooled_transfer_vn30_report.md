# Pooled/transfer ablation for VN30 — results

Single VN100 panel; Arm 0 trains VN30 (31), Arm 1 trains VN100 (102); both score the 31 VN30
stocks on the identical OOS grid. Headline = paired DM Arm1-vs-Arm0 (favours A = pooling helps).
Prior Track B A1 (2026-08-08) found pooling did not help deep beat HAR; this is the clean-data,
walk-forward, cross-universe re-test. Objective report — verdict stated regardless of sign.

**Interpretation caveat:** Arm 0 here is VN30 trained on the shared VN100 panel (VN100 fold
calendar, OOS grid and `market_pk` factor), NOT the previously delivered standalone-VN30 run
(which used the VN30-only panel: fewer OOS obs, K=16, a VN30 `market_pk`). The two are on
different OOS grids and a different market factor, so their absolute numbers are NOT directly
comparable. The only valid comparison is Arm 0 vs Arm 1 (both on the VN100 grid, differing only
in the training node set) — that is what the paired DM below measures.

## Horizon h1  (22 folds, 31 VN30 nodes, 102 panel nodes, 3.6h)

| Model | QLIKE Arm0 (VN30) | QLIKE Arm1 (VN100) | Δ (pooled−base) |
|---|---|---|---|
| HAR | 0.4705 | 0.4811 | +0.0105 |
| HAR-X | 0.4796 | 0.4766 | -0.0030 |
| LSTM | 0.4712 | 0.4810 | +0.0098 |
| LSTM_wGAT_vol2pk | 0.4690 | 0.4789 | +0.0098 |

**Headline — paired DM (Arm1 vs Arm0), favours A = pooling helps the deep model:**

| Deep model | QLIKE p (favors) | SE p (favors) | AE p (favors) |
|---|---|---|---|
| LSTM | 0.136 (B) | 0.592 (A) | 0.000 (A) |
| VolGA | 0.147 (B) | 0.172 (A) | 0.000 (A) |

**Secondary — diff-in-diff gap(deep − HAR):**  
- LSTM: gap Arm0 = +0.0007, Arm1 = -0.0001, Δgap = -0.0008 (negative Δ = pooling narrows the deep−HAR gap)
- VolGA: gap Arm0 = -0.0015, Arm1 = -0.0022, Δgap = -0.0007 (negative Δ = pooling narrows the deep−HAR gap)

## Horizon h5  (22 folds, 31 VN30 nodes, 102 panel nodes, 3.5h)

| Model | QLIKE Arm0 (VN30) | QLIKE Arm1 (VN100) | Δ (pooled−base) |
|---|---|---|---|
| HAR | 0.5574 | 0.5648 | +0.0073 |
| HAR-X | 0.5549 | 0.5593 | +0.0044 |
| LSTM | 0.6102 | 0.5895 | -0.0207 |
| LSTM_wGAT_vol2pk | 0.5779 | 0.5809 | +0.0029 |

**Headline — paired DM (Arm1 vs Arm0), favours A = pooling helps the deep model:**

| Deep model | QLIKE p (favors) | SE p (favors) | AE p (favors) |
|---|---|---|---|
| LSTM | 0.081 (A) | 0.910 (B) | 0.006 (A) |
| VolGA | 0.666 (B) | 0.306 (A) | 0.000 (A) |

**Secondary — diff-in-diff gap(deep − HAR):**  
- LSTM: gap Arm0 = +0.0528, Arm1 = +0.0247, Δgap = -0.0280 (negative Δ = pooling narrows the deep−HAR gap)
- VolGA: gap Arm0 = +0.0205, Arm1 = +0.0161, Δgap = -0.0044 (negative Δ = pooling narrows the deep−HAR gap)

## Horizon h10  (22 folds, 31 VN30 nodes, 102 panel nodes, 3.5h)

| Model | QLIKE Arm0 (VN30) | QLIKE Arm1 (VN100) | Δ (pooled−base) |
|---|---|---|---|
| HAR | 0.6007 | 0.6069 | +0.0062 |
| HAR-X | 0.6012 | 0.6053 | +0.0041 |
| LSTM | 0.6272 | 0.6199 | -0.0073 |
| LSTM_wGAT_vol2pk | 0.6231 | 0.6308 | +0.0077 |

**Headline — paired DM (Arm1 vs Arm0), favours A = pooling helps the deep model:**

| Deep model | QLIKE p (favors) | SE p (favors) | AE p (favors) |
|---|---|---|---|
| LSTM | 0.033 (A) | 0.569 (A) | 0.008 (A) |
| VolGA | 0.405 (B) | 0.506 (A) | 0.001 (A) |

**Secondary — diff-in-diff gap(deep − HAR):**  
- LSTM: gap Arm0 = +0.0265, Arm1 = +0.0131, Δgap = -0.0134 (negative Δ = pooling narrows the deep−HAR gap)
- VolGA: gap Arm0 = +0.0224, Arm1 = +0.0239, Δgap = +0.0015 (negative Δ = pooling narrows the deep−HAR gap)

## Horizon h22  (22 folds, 31 VN30 nodes, 102 panel nodes, 3.2h)

| Model | QLIKE Arm0 (VN30) | QLIKE Arm1 (VN100) | Δ (pooled−base) |
|---|---|---|---|
| HAR | 0.6548 | 0.6593 | +0.0045 |
| HAR-X | 0.6523 | 0.6580 | +0.0057 |
| LSTM | 0.6939 | 0.6774 | -0.0165 |
| LSTM_wGAT_vol2pk | 0.6942 | 0.6722 | -0.0220 |

**Headline — paired DM (Arm1 vs Arm0), favours A = pooling helps the deep model:**

| Deep model | QLIKE p (favors) | SE p (favors) | AE p (favors) |
|---|---|---|---|
| LSTM | 0.252 (A) | 0.020 (A) | 0.558 (A) |
| VolGA | 0.263 (A) | 0.021 (A) | 0.117 (A) |

**Secondary — diff-in-diff gap(deep − HAR):**  
- LSTM: gap Arm0 = +0.0391, Arm1 = +0.0181, Δgap = -0.0210 (negative Δ = pooling narrows the deep−HAR gap)
- VolGA: gap Arm0 = +0.0394, Arm1 = +0.0129, Δgap = -0.0265 (negative Δ = pooling narrows the deep−HAR gap)
