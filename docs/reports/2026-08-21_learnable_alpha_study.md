# Learnable-alpha HAR+LSTM forecast-combination study (5 seeds x 4 horizons)

Date: 2026-08-21

> **Statistical-inference caveat (added 2026-08-22).** The Diebold-Mariano p-values in this report are the
> **per-observation (row-level)** DM over every (ticker, date) row. On a cross-sectionally dependent panel
> (all tickers share each date) this treats n = N x T_dates and over-states significance by roughly
> sqrt(N); the S&P500 h1 value p~1e-74 is a symptom, not a genuine effect size. Under the panel-correct
> **date-clustered** DM (one loss value per date), the combination's beat-HAR significance is expected to
> weaken substantially, as demonstrated on the same panels by the HAR-anchored study
> (`reports/experiment_results.md`), where the convex combination does NOT significantly beat HAR on
> VN30/VN100 (date-clustered p in 0.14-0.51). Read the QLIKE/MSE improvements below as point estimates
> pending date-clustered re-testing; several alpha_combo QLIKE cells are also inflated by single
> floor-clamped outlier seeds (noted per row).
Scope: full study of `pred = alpha*HAR + (1-alpha)*LSTM` with `alpha = sigmoid(theta)` learned
end-to-end, jointly with the LSTM, on the per-observation pooled design. Panels: VN30 (33 tickers),
VN100 (104 tickers), and S&P500 (501 tickers), lookback 10, horizons {1, 5, 10, 22}, 5 seeds
{42, 123, 2026, 7, 2024}. S&P500 was run from `data/processed/sp500/*_processed.csv` via
`--data-root data/processed` (3.48-3.49M train observations per horizon). Runner:
`submission/soict_lstm_gat/run_alpha.py`; smoke test
`submission/soict_lstm_gat/tests/test_run_alpha.py` (2 passed). Raw results:
`results/soict_alpha/{panel}_lb10_h{h}/result.json`.

## Method

`AlphaCombo` = the exact price-only per-observation LSTM from `run_lstm.py` (2-layer, hidden 64, MSE
loss, val-MSE early stop) plus one global learnable scalar `theta`. Training prediction is combined on
the per-ticker NORMALIZED scale, `pred_norm = alpha*har_norm + (1-alpha)*lstm_norm`, with
`har_norm = (har_raw - t_mean)/t_std` using the same train scaler as the LSTM target. Because the
per-ticker normalization is affine and the weights are convex, this equals raw-scale combination
`alpha*har_raw + (1-alpha)*(lstm_norm*t_std + t_mean)` exactly (feasibility report §2). HAR is a frozen
global OLS fit on TRAIN targets only, fit before the LSTM trains. Deep branch = LSTM without GAT. All
models are evaluated on the RAW variance scale after inverse-transform, with a shared QLIKE floor 1e-8.
Four configurations per run: HAR (frozen), LSTM (standalone), fixed0.5 (`0.5*HAR + 0.5*LSTM`), and
alpha_combo (learnable, LSTM co-trained against the combination loss). QLIKE is the seed-average of the
per-seed test QLIKE; Diebold-Mariano uses the seed-ensemble prediction (mean forecast across seeds),
HLN-corrected, HAC lags = h-1.

`alpha` is the weight on HAR (so `1-alpha` is the weight on the LSTM).

## Table 1 — VN30 (lb10, 5 seeds)

QLIKE (lower better), MSE scaled by 1e7, R2. DM p-value of alpha_combo vs HAR / vs standalone LSTM
(QLIKE, two-sided); "favors" = which side carries the smaller loss.

| h | model | QLIKE | MSE (x1e-7) | R2 | learned alpha (mean, sd) |
|---|---|---|---|---|---|
| 1 | HAR | 0.4675 | 2.2449 | 0.289 | |
| 1 | LSTM | 0.4609 | 2.2411 | 0.290 | |
| 1 | fixed0.5 | 0.4587 | 2.2284 | 0.294 | |
| 1 | **alpha_combo** | **0.4552** | 2.2315 | 0.293 | 0.4915 (0.009) |
| 5 | HAR | 0.5514 | 2.5493 | 0.193 | |
| 5 | LSTM | 0.5486 | 2.5495 | 0.193 | |
| 5 | fixed0.5 | 0.5435 | 2.5353 | 0.196 | |
| 5 | **alpha_combo** | **0.5414** | 2.5370 | 0.197 | 0.4607 (0.011) |
| 10 | HAR | 0.5925 | 2.7048 | 0.144 | |
| 10 | LSTM | 0.5928 | 2.7147 | 0.140 | |
| 10 | fixed0.5 | 0.5872 | 2.6970 | 0.147 | |
| 10 | alpha_combo | 0.5883 | 2.7071 | 0.143 | 0.4873 (0.013) |
| 22 | HAR | 0.6366 | 2.8561 | 0.096 | |
| 22 | LSTM | 0.6456 | 2.9190 | 0.076 | |
| 22 | fixed0.5 | 0.6336 | 2.8683 | 0.083 | |
| 22 | alpha_combo | 0.6412 | 2.9046 | 0.081 | 0.4491 (0.042) |

DM (alpha_combo, QLIKE): h1 vs HAR p=1.4e-12 (combo), vs LSTM p=0.017 (combo); h5 vs HAR p=6.8e-5
(combo), vs LSTM p=0.048 (combo); h10 vs HAR p=0.11 (tie, combo side), vs LSTM p=0.29 (tie); h22 vs HAR
p=0.43 (tie, HAR side), vs LSTM p=0.45 (tie, combo side).

## Table 2 — VN100 (lb10, 5 seeds)

| h | model | QLIKE | MSE (x1e-7) | R2 | learned alpha (mean, sd) |
|---|---|---|---|---|---|
| 1 | HAR | 0.4798 | 2.5131 | 0.228 | |
| 1 | LSTM | 0.4745 | 2.5181 | 0.227 | |
| 1 | fixed0.5 | 0.4746 | 2.5024 | 0.230 | |
| 1 | **alpha_combo** | **0.4711** | 2.4978 | 0.233 | 0.5201 (0.007) |
| 5 | HAR | 0.5441 | 2.7815 | 0.146 | |
| 5 | LSTM | 0.5385 | 2.7773 | 0.147 | |
| 5 | fixed0.5 | 0.5372 | 2.7637 | 0.150 | |
| 5 | **alpha_combo** | **0.5349** | 2.7631 | 0.152 | 0.4471 (0.034) |
| 10 | HAR | 0.5773 | 2.9248 | 0.103 | |
| 10 | LSTM | 0.5757 | 2.9223 | 0.104 | |
| 10 | fixed0.5 | 0.5710 | 2.9064 | 0.108 | |
| 10 | alpha_combo | 0.5720 | 2.9082 | 0.108 | 0.4173 (0.039) |
| 22 | HAR | 0.6112 | 3.0548 | 0.063 | |
| 22 | LSTM | 0.6135 | 3.0832 | 0.055 | |
| 22 | fixed0.5 | 0.6048 | 3.0415 | 0.063 | |
| 22 | alpha_combo | 0.6082 | 3.0603 | 0.062 | 0.2930 (0.064) |

DM (alpha_combo, QLIKE): h1 vs HAR p=9.2e-27 (combo), vs LSTM p=4.6e-5 (combo); h5 vs HAR p=1.9e-11
(combo), vs LSTM p=0.0073 (combo); h10 vs HAR p=1.0e-4 (combo), vs LSTM p=0.012 (combo); h22 vs HAR
p=0.20 (tie, combo side), vs LSTM p=7.0e-5 (combo).

## Table 3 — S&P500 (lb10, 5 seeds, 501 tickers)

QLIKE is the seed-average of the per-seed test QLIKE; MSE scaled by 1e7; R2.

| h | model | QLIKE | MSE (x1e-7) | R2 | learned alpha (mean, sd) |
|---|---|---|---|---|---|
| 1 | HAR | 0.3616 | 2.9797 | 0.197 | |
| 1 | LSTM | 0.5515 | 2.9618 | 0.202 | |
| 1 | fixed0.5 | 0.3633 | 2.9336 | 0.209 | |
| 1 | alpha_combo | 5.4272 | **2.9058** | **0.217** | 0.6087 (0.007) |
| 5 | HAR | 0.4226 | 3.3453 | 0.098 | |
| 5 | LSTM | 0.4252 | 3.3119 | 0.107 | |
| 5 | fixed0.5 | **0.4197** | 3.2875 | 0.114 | |
| 5 | alpha_combo | 0.4377 | **3.2752** | **0.117** | 0.5438 (0.008) |
| 10 | HAR | 0.4387 | 3.4220 | 0.078 | |
| 10 | LSTM | 0.4450 | 3.4604 | 0.067 | |
| 10 | fixed0.5 | **0.4353** | **3.3910** | 0.086 | |
| 10 | alpha_combo | 0.4686 | 3.4445 | 0.072 | 0.4976 (0.025) |
| 22 | HAR | 0.4601 | 3.4602 | 0.067 | |
| 22 | LSTM | 0.4723 | 3.5002 | 0.056 | |
| 22 | fixed0.5 | 0.4603 | 3.4194 | 0.078 | |
| 22 | **alpha_combo** | **0.4560** | **3.3855** | **0.087** | 0.4385 (0.022) |

DM (alpha_combo, QLIKE, seed-ensemble): h1 vs HAR p=1.8e-74 (combo), vs LSTM p=1.7e-217 (combo); h5 vs
HAR p=1.6e-15 (combo), vs LSTM p=3.3e-58 (combo); h10 vs HAR p=1.1e-14 (combo), vs LSTM p=3.0e-29
(combo); h22 vs HAR p=2.0e-4 (combo), vs LSTM p=3.1e-43 (combo). All four horizons favor alpha_combo
over both HAR and the standalone LSTM.

Note on the seed-average QLIKE vs the DM tests. The alpha_combo seed-average QLIKE at h1, h5, and h10 is
inflated by a single high-loss seed (h1 seed 42 QLIKE 25.70; h5 seed 2024 QLIKE 0.5043; h10 seed 42
QLIKE 0.6027); the remaining four seeds are 0.358-0.360 (h1), 0.417-0.424 (h5), and 0.433-0.437 (h10).
The standalone LSTM seed-average is likewise inflated at h1 by seed 2026 (QLIKE 1.267). The
Diebold-Mariano tests operate on the five-seed ensemble forecast (mean prediction across seeds), which
averages out these per-seed spikes; on the ensemble, alpha_combo carries the smaller QLIKE loss than
both HAR and the standalone LSTM at every horizon (all p < 1e-3). On MSE and R2 — which are not affected
by the near-zero-prediction QLIKE blow-ups — alpha_combo attains the lowest MSE and highest R2 at h1, h5,
and h22; at h10 fixed0.5 attains the lowest MSE and R2 and alpha_combo sits just above HAR. At h22, where
no seed produced an outlier, the alpha_combo seed-average QLIKE (0.4560) is itself below HAR (0.4601).

## Learned alpha per seed

| panel | h | per-seed alpha | mean | sd |
|---|---|---|---|---|
| vn30 | 1 | 0.5011, 0.4906, 0.5000, 0.4756, 0.4904 | 0.4915 | 0.009 |
| vn30 | 5 | 0.4477, 0.4527, 0.4644, 0.4601, 0.4785 | 0.4607 | 0.011 |
| vn30 | 10 | 0.4837, 0.5050, 0.4789, 0.4996, 0.4695 | 0.4873 | 0.013 |
| vn30 | 22 | 0.4105, 0.4890, 0.3893, 0.4929, 0.4636 | 0.4491 | 0.042 |
| vn100 | 1 | 0.5205, 0.5118, 0.5330, 0.5141, 0.5210 | 0.5201 | 0.007 |
| vn100 | 5 | 0.4153, 0.4046, 0.4964, 0.4718, 0.4475 | 0.4471 | 0.034 |
| vn100 | 10 | 0.4896, 0.3871, 0.3824, 0.4265, 0.4008 | 0.4173 | 0.039 |
| vn100 | 22 | 0.3083, 0.3072, 0.2615, 0.1961, 0.3921 | 0.2930 | 0.064 |
| sp500 | 1 | 0.6157, 0.6114, 0.5982, 0.6145, 0.6038 | 0.6087 | 0.007 |
| sp500 | 5 | 0.5541, 0.5488, 0.5454, 0.5363, 0.5343 | 0.5438 | 0.008 |
| sp500 | 10 | 0.4778, 0.5173, 0.5345, 0.4892, 0.4691 | 0.4976 | 0.025 |
| sp500 | 22 | 0.4195, 0.4414, 0.4692, 0.4533, 0.4089 | 0.4385 | 0.022 |

## Does the learned alpha rise toward 1 at long horizons?

No. The prior weight-sensitivity grid (feasibility report §1) reported the QLIKE-optimal weight on HAR
rising from roughly 0.0-0.3 at h1 to 1.0 at h22. The end-to-end learned global alpha does the opposite:
it sits near 0.5 at h1 (VN30 0.4915, VN100 0.5201) and DRIFTS DOWN as the horizon grows (VN30 0.449 and
VN100 0.293 at h22), i.e. it puts MORE weight on the deep branch at long horizons. S&P500 shows the same
downward drift but from a higher starting point: 0.609 at h1, 0.544 at h5, 0.498 at h10, 0.439 at h22 —
i.e. more weight on HAR than the VN panels at the short horizons, still declining toward the deep branch
as the horizon grows, and still never approaching the all-HAR corner. Mechanism: alpha is selected by the
co-trained model's validation MSE on the
normalized in-sample val split, not by out-of-sample QLIKE; on that objective the deep branch is not
penalized enough to push alpha toward 1, so the learned scalar does not reproduce the horizon-adaptive
optimum that the OOS sensitivity grid implied. The seed dispersion of alpha also grows with horizon
(sd 0.007-0.013 at h1 up to 0.042-0.064 at h22), consistent with a flatter, less-identified objective at
long horizons.

## Verdict

1. Beats HAR (QLIKE, DM): yes at short horizons on the VN panels — h1 and h5 are significant on VN30 and
   VN100 (p from 1e-4 down to 1e-27), and VN100 h10 is also significant (p=1e-4). At the longest horizon
   the VN combination does NOT beat HAR: VN30 h22 favors HAR (p=0.43, tie) and VN100 h22 is a tie
   (p=0.20). VN30 h10 is a non-significant tie. So on the VN panels the beat-HAR result is a
   short/mid-horizon phenomenon; HAR retains its long-horizon (h22) edge, consistent with prior project
   findings. On S&P500 the seed-ensemble alpha_combo favors the combination over HAR at ALL four horizons
   (h1 p=1.8e-74, h5 p=1.6e-15, h10 p=1.1e-14, h22 p=2.0e-4), including h22 where the alpha_combo
   seed-average QLIKE (0.4560) also sits below HAR (0.4601) — i.e. on the large panel the combination
   extends the beat-HAR result to the longest horizon. Caveat: the S&P500 seed-average QLIKE for
   alpha_combo at h1/h5/h10 is dominated by a single high-loss seed (see the note under Table 3), so the
   ensemble DM — not the outlier-inflated seed-average — is the reliable statistic there.

2. Beats the standalone LSTM (QLIKE, DM): yes at short/mid horizons on the VN panels — significant at h1
   and h5 on both panels, at h10 on VN100, and at h22 on VN100 (p=7e-5); ties at VN30 h10/h22. On S&P500
   the seed-ensemble alpha_combo beats the standalone LSTM at all four horizons (p from 3.3e-58 to
   3.1e-43 to 1.7e-217). The combination is never significantly worse than the standalone LSTM on any
   config across all three panels. This is the convex-combination safety property in practice: mixing HAR
   into the deep forecast does not degrade QLIKE relative to the deep model alone, and improves it
   wherever the two disagree usefully.

3. Automates the horizon-adaptive weight: no. A single global learned scalar does not recover the
   grid's rising w_HAR profile; it stays near or below 0.5 and moves the wrong direction at long
   horizons. The value the learnable form adds over a fixed 0.5 is co-training the deep branch against
   the combination loss (which yields the h1/h5 QLIKE wins), not the learned weight itself — the weight
   lands close to 0.5 where a fixed 0.5 already sits. On the VN panels at h10/h22, fixed0.5 attains
   marginally lower QLIKE than alpha_combo (e.g. VN30 h22 0.6336 vs 0.6412; VN100 h22 0.6048 vs 0.6082),
   so the extra parameter did not help there. On S&P500 the learned alpha moves further from 0.5 (0.609 at
   h1 down to 0.439 at h22) and, on MSE and R2 (the metrics not corrupted by the per-seed QLIKE
   blow-ups), alpha_combo attains the lowest MSE/highest R2 at h1, h5, and h22 — better than fixed0.5 —
   with fixed0.5 lowest only at h10; so on the large panel the co-trained learnable form does add value
   over the fixed 0.5 at three of four horizons.

4. Convex-combination safety property: because `alpha in (0,1)` and the corners `alpha=0` (pure deep)
   and `alpha=1` (pure HAR) are reachable, the combination cannot collapse to an arbitrarily worse
   forecaster than its two branches, and every DM test above confirms alpha_combo is never significantly
   worse than the standalone LSTM. The caveat is that a bounded worst case is not the same as selecting
   the best corner: at h22 the OOS-optimal weight is near the all-HAR corner, but validation selection
   did not drive alpha there, so alpha_combo gives up a little QLIKE to HAR at the longest horizon.

## Recommendation and caveats

- The study confirms the feasibility report's core prediction: the learnable global alpha is a safe,
  differentiable, leakage-safe extension whose measurable benefit is short/mid-horizon (co-training),
  not the horizon-adaptive weight automation that would justify the extra parameter over fixed 0.5.
- To actually automate the horizon-adaptive weight, the weight should be selected on out-of-sample
  (validation) QLIKE per horizon, or made input/horizon-dependent — but the feasibility report already
  flags per-ticker/gated weights as OOS-transfer-fragile on these panels; that remains the open item.
- Caveats: per-observation pooled design, single lookback (10), 5 seeds, 20 max epochs. On S&P500 the
  seed-average QLIKE for alpha_combo (and the standalone LSTM at h1) is inflated by isolated high-loss
  seeds arising from near-zero combined predictions clamped to the QLIKE floor; the seed-ensemble DM and
  the MSE/R2 metrics are the reliable summaries for that panel. All numbers are reported exactly as
  written to `results/soict_alpha/*/result.json`.
