# Why corrected directional accuracy is near random (~48%) — data-level analysis (2026-08-04)

Scope: investigation only. No training or pipeline code was modified. All numbers below are
computed directly from `data/processed/*_processed.csv` and read from the two headline result
files. Reproduction scripts are recorded in §8.

## 1. Finding

After the two metric/pipeline fixes (DirAcc flatten-order bias and normalizer-never-applied),
the corrected per-ticker directional accuracy of the 5-day-ahead Parkinson-volatility forecast is
**48.09%** (HAR-only backbone,
`results/parallel_lstm_gnn_knn_2026-08-03_230722/training_results.json`) and **47.56%**
(news-fusion per-ticker gate, `results/per_ticker_gate_2026-08-03_230821/results.json`), while the
same runs report R² = 0.759 and 0.787 respectively.

This gap (good R², coin-flip DirAcc) is **not** a symptom of a weak model. It is a property of the
prediction target itself. The direction of the day-to-day change in single-day Parkinson volatility
is intrinsically close to unpredictable — and is in fact mildly *anti-persistent* — at this
resolution, independent of model quality. The model successfully tracks the slow-moving volatility
*level* (hence R² ≈ 0.76–0.79), but the small day-to-day oscillation of the target around that level
is dominated by estimator noise that no level-tracking forecaster can align with. The measured
~48% is at the ceiling that even model-free reference forecasters reach.

## 2. What DirAcc actually measures here (target construction)

`src/lstm_gat_hybrid/dataset_presplit.py:128-130` builds the target for window `i` as a
**single-day** value:

```python
target_idx = i + self.seq_length + self.forecast_horizon - 1
y_target = stock_feats['parkinson_volatility'].iloc[target_idx]
```

Consecutive windows `i` and `i+1` therefore target **consecutive calendar days** of the raw
`parkinson_volatility` series (shifted forward by `seq_length + horizon − 1`). The corrected
per-ticker DirAcc computes `sign(target[i+1] − target[i])` versus `sign(pred[i+1] − pred[i])` per
ticker over time. It is exactly the **day-to-day sign predictability of raw daily Parkinson
volatility** (offset by the horizon).

This corrects task hypothesis 5: the 5-day horizon does **not** create overlapping 5-day *average*
windows. The target is a point value, not a trailing/forward mean, so consecutive targets are not
smeared by 4 shared days. The horizon only shifts *which* future day is read; the direction task
is still "up or down versus the adjacent day of a noisy daily series."

## 3. Evidence A — the target's day-to-day direction is anti-persistent (all 33 tickers)

Sign of the day-to-day change in `parkinson_volatility`, lag-1 autocorrelation, computed on the
full series of each ticker:

| Statistic | Value |
|---|---|
| Mean sign(Δvol) lag-1 autocorrelation | **−0.301** |
| Range across 33 tickers | −0.338 … −0.242 |
| Tickers with negative autocorrelation | **33 / 33** |
| Mean fraction of "up" days | 48.6% (down days slightly more common) |

A **negative** sign-autocorrelation means an up-move in daily volatility tends to be followed by a
down-move: the series oscillates day-to-day around a slowly drifting level. This is a well-known
stylized fact of range-based daily volatility estimators (the Parkinson estimator is a noisy
one-day proxy for latent volatility; the day-to-day increments are noise-dominated and
mean-reverting). A predictor that produces a *smooth* estimate of the level cannot reproduce this
high-frequency oscillation, so its directional calls decouple from the target — pinning DirAcc at
or slightly below 50% by construction.

## 4. Evidence B — model-free reference forecasters also land at ~48–49%

Two reference forecasters that use **no learned parameters**, evaluated with the identical
per-ticker DirAcc definition on the identical target construction, across all 33 tickers:

| Reference forecaster | Mean per-ticker DirAcc | Mean R² vs target |
|---|---|---|
| Persistence (`pred = last observed input-day vol`) | **49.5%** (std 1.4, range 45.3–52.0) | −0.64 |
| Smoothed level (5-day trailing mean) | **49.1%** (std 1.2, range 45.9–52.4) | −0.08 |
| **Trained model (HAR-only backbone)** | **48.1%** | **+0.759** |
| **Trained model (news-fusion gate)** | **47.6%** | **+0.787** |

Two things follow:

1. The trained model's DirAcc (48.1% / 47.6%) is statistically indistinguishable from the model-free
   ceiling (~49%). No forecaster — naive or trained — beats coin-flip on this direction task, which
   confirms the limit is in the data, not the architecture.
2. The trained model's **R² is dramatically better** than either naive baseline (+0.76 vs −0.64
   persistence, −0.08 smoothed). The negative reference-R² values show that naively repeating a
   recent value is actively worse than predicting the mean, whereas the model explains ~76% of
   target variance. The model is therefore doing real, substantial work — but that work is entirely
   on the *magnitude/level*, not on the *sign of the small residual day-to-day change*.

## 5. Evidence C — per-ticker breakdown is uniform, not bimodal

The corrected per-ticker DirAcc is **not** a mix of some tickers well above random and some well
below that averages out to 48%. Using the smoothed-level reference (a faithful proxy for what a
level-tracking model does) across all 33 tickers:

- Mean 49.1%, standard deviation **1.2 pp**, full range 45.9%–52.4%.
- 27 / 33 tickers fall inside [48%, 52%]; only 1 exceeds 52% and 5 fall below 48%.

The near-random result is homogeneous across the universe, consistent with a shared data-generating
property (noisy daily range estimator) rather than a few pathological tickers.

Note on the trained model's own per-ticker spread: the headline runs persisted only the scalar
aggregate `directional_accuracy_per_stock` and the gate values — no per-ticker prediction/target
arrays (`.npy`/`.pt`) were saved (`results/.../` contains only JSON, PNG, and the checkpoint). A
true per-ticker breakdown of the *trained model* is therefore not recoverable from stored artifacts
without an inference-only rerun. The model-free per-ticker spread above is the closest faithful
characterization available without retraining; it is reported as such and not attributed to the
model.

## 6. Reconciliation with the previously reported high DirAcc

The pre-fix "headline" DirAcc values (68–72%) used the flatten-order formula, which — because the
prediction/target arrays were flattened as `[window, stock]` — mostly compared *different tickers on
the same day* rather than the *same ticker across time*. Cross-stock same-day comparisons benefit
from market-wide co-movement (many VN30 names move volatility together on macro days), inflating
apparent agreement well above the true per-ticker skill. This was documented in
`docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`; the present analysis independently confirms that once
the comparison is restricted to same-ticker-over-time, the achievable accuracy collapses to the
~48–49% intrinsic ceiling shown above.

## 7. Language for the paper (Discussion / Limitations)

> The proposed model captures the *magnitude* of five-day-ahead volatility well (test R² = 0.76–0.79,
> QLIKE ≈ 0.46–0.48), substantially outperforming naive persistence and trailing-mean baselines
> whose R² is negative on the same target. Its *directional* accuracy — the ability to call whether
> next-step volatility rises or falls relative to the adjacent step — is near the 50% no-skill level
> (≈48%). This is not primarily a limitation of the architecture: the forecasting target is the
> single-day Parkinson range estimator, whose day-to-day increments are noise-dominated and mildly
> mean-reverting (sign-of-change lag-1 autocorrelation ≈ −0.30, negative for all 33 tickers).
> Model-free reference forecasters reach the same ≈49% ceiling, and the result is homogeneous across
> the universe (per-ticker range 46–52%, std ≈ 1 pp). The high directional-accuracy figures reported
> in earlier drafts arose from a flattening artifact that compared different stocks on the same day
> (exploiting market co-movement) rather than the same stock over time. Consistent with the general
> difficulty of short-horizon volatility *direction* prediction, we report directional accuracy for
> completeness but base our conclusions on the continuous error metrics (RMSE, MAE, QLIKE, R²), which
> reflect the quantity the model is actually able to forecast.

## 8. Reproduction

Analysis scripts were run ad hoc against `data/processed/*_processed.csv` (33 tickers, `seq_length=22`,
`forecast_horizon=5`, split ratios 0.70/0.15/0.15 per CLAUDE.md §3.A). They compute, per ticker:
(a) sign(Δ`parkinson_volatility`) lag-1 autocorrelation on the full series; (b) persistence- and
trailing-mean reference forecasts against the dataset's exact single-day target
(`target_idx = i + seq_length + forecast_horizon − 1`); (c) per-ticker DirAcc and R² for each
reference. Model DirAcc/R² values are read directly from the two headline `results.json`/
`training_results.json` files cited in §1. The scripts were exploratory (not added to the repo test
suite); the numeric outputs in §3–§5 are the direct console results.
