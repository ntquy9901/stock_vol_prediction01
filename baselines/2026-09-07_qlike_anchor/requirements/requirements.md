# Requirements — QLIKE-loss / HAR-X-anchor experiment

## Objective
Test whether training the deep models (LSTM, VolGA) with a QLIKE loss and/or a HAR-X-anchored residual lets
them beat HAR-X on QLIKE, and measure the cost to MSE/RMSE/MAE. Motivated by the SP500 issue analysis
(`docs/reports/2026-09-07_sp500_issues_master.md`): MSE-trained deep models collapse-under-forecast on spikes,
which QLIKE punishes; HAR-X wins QLIKE by tracking the level.

## Knobs (4 configs)
- loss ∈ {mse, qlike}: masked MSE vs masked QLIKE on the floored positive forecast.
- anchor ∈ {none, harx}: predict the level (as delivered) vs a bounded residual yhat = HAR-X * exp(clip(z,-c,c)),
  trained on chronological OOF HAR-X (leakage-safe).

## Success criteria (go/no-go)
- GO: some (loss, anchor) gives a deep model with DM-significantly lower QLIKE than HAR-X on >=1 market/horizon,
  without a catastrophic MSE/RMSE regression that makes it useless multi-objectively.
- NO-GO: no variant beats HAR-X on QLIKE, or QLIKE gains come only with unacceptable squared-error loss.
Report ALL five metrics (MSE/RMSE/MAE/QLIKE/R2) train/val/test so the tradeoff is explicit.

## Constraints
Identical folds (lb10, 7 folds), 5 seeds, shared QLIKE positivity floor, per-node train-only scaler, causal
features — same basis as results/edge_hmatched (fairness). VN100 all horizons local; SP500 via Colab notebook.
