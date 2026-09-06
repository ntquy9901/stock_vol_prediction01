# QLIKE-loss / HAR-X-anchor on VN100 — results (GO: QLIKE-loss beats HAR-X at the daily horizon)

Experiment `baselines/2026-09-07_qlike_anchor` (fixed, review-approved code). VN100, lb10, 7 folds, 5 seeds,
shared QLIKE floor. Configs: loss∈{mse(baseline=edge_hmatched), qlike} × anchor∈{none, harx}. Ticker-date
pooled OOS QLIKE; DM date-clustered vs HAR-X. Answers the two questions: does QLIKE-loss let the deep model
beat HAR-X, and does it cost MSE/RMSE/MAE?

## Outcome: GO (conditional)
Training the deep models (LSTM, VolGA) on a **QLIKE loss** makes them **significantly beat HAR-X on QLIKE at
the daily horizon** — reversing the MSE-loss result where the LSTM lost — **with no material MSE/RMSE/MAE cost**
on VN100. At longer horizons QLIKE-loss brings them to parity with HAR-X (no longer losing, not significantly
winning). The HAR-X-anchor is secondary; the loss function is the lever.

## QLIKE by horizon (VN100)
| h | model | HAR-X | MSE-loss | QLIKE-loss (none) | QLIKE-loss (harx) | DM none / harx vs HAR-X |
|---|---|---|---|---|---|---|
| 1 | LSTM | 0.5000 | 0.5155 | **0.4840** | 0.4903 | **p=0.003** / **p<0.001** (both favor deep) |
| 1 | VolGA | 0.5000 | 0.4879 | **0.4833** | 0.4905 | **p=0.004** / **p<0.001** |
| 5 | LSTM | 0.5607 | 0.5759 | 0.5601 | 0.5594 | p=0.94 / 0.79 (n.s.) |
| 5 | VolGA | 0.5607 | 0.5662 | 0.5565 | 0.5590 | p=0.53 / 0.70 (n.s.) |
| 10 | LSTM | 0.5999 | 0.6127 | 0.5973 | 0.6020 | p=0.78 / 0.80 (n.s.) |
| 10 | VolGA | 0.5999 | 0.6127 | 0.5985 | 0.6010 | p=0.87 / 0.89 (n.s.) |
| 22 | LSTM | 0.6385 | 0.6452 | 0.6390 | 0.6512 | n.s. |
| 22 | VolGA | 0.6385 | 0.6476 | 0.6436 | 0.6492 | n.s. |

- **h1 (daily): QLIKE-loss significantly beats HAR-X** for both LSTM (0.4840) and VolGA (0.4833), DM p≈0.003.
  The MSE-loss LSTM lost at h1 (0.5155); QLIKE-loss training fixes that.
- **h5/h10:** QLIKE-loss lands the deep models at or just below HAR-X (not significant) — versus MSE-loss which
  was worse than HAR-X. So QLIKE-loss removes the deep models' QLIKE disadvantage across horizons.
- **h22:** parity with HAR-X.

## MSE / RMSE / MAE tradeoff — negligible on VN100
Switching the deep-model objective MSE→QLIKE improved QLIKE (esp. h1) with **no material change in MSE, RMSE,
or MAE** at every horizon (identical to 3 decimals of MSE×10⁴, RMSE, MAE). The feared multi-objective tradeoff
does not materialise here: on VN100 the deep models were already near-MSE-optimal, and the QLIKE loss mainly
corrects the spike under-forecast (see below) rather than over-forecasting calm days.

## Spike-collapse cured (mechanism confirmed)
On the top-1% realized-variance ticker-days at h1, the QLIKE-loss LSTM's median forecast/realized ratio is
**0.238**, matching HAR-X (0.232) — i.e. it now tracks spikes like HAR-X. This is the direct fix of the
MSE-loss failure mode (on SP500 the MSE-loss LSTM collapsed to ~0.05 of the realized spike). Training on QLIKE
penalises under-forecasting, so the deep model no longer collapses on spikes → its QLIKE improves.

## Fit / provenance
All 8 result JSONs carry train/val/test metrics + fit_diagnostics (fit=ok for LSTM/VolGA at every horizon) +
learning_curves; config records loss/anchor/lookback/floor/seeds. `train_metrics_note` documents that the
anchor train fit-evidence uses an in-sample HAR-X base (not same-basis as val/test).

## Most-justified next step
SP500 (the panel with the largest MSE-loss collapse) is the strongest test of QLIKE-loss — run the same
`(qlike,none)` config on SP500 (Colab notebook `train_qlike_anchor_sp500_colab.ipynb`), where the fix should
help most. For VN, QLIKE-loss is the recommended training objective going forward (beats HAR-X at h1, parity
elsewhere, no squared-error cost).
