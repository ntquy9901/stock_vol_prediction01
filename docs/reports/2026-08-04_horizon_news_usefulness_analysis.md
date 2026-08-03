# Horizon-dependent usefulness of news features: a data-driven explanation

Date: 2026-08-04
Scope: analysis/investigation only (no training or pipeline code changed). Explains why news-fusion
(per-ticker gated) features appear to help at short forecast horizons (1/5-day) but lose their edge,
or reverse, at longer horizons (10/22-day), using measured decay/autocorrelation statistics and the
actual saved result files.

Inputs used:
- Result files: `results/har_only_h{1,10,22}_2026-08-01_*/results.json`,
  `results/per_ticker_gate_h{1,10,22}_2026-08-01_*/results.json`.
- 5-day (primary) numbers: `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §1.2 Bảng B, and the
  post-fix headline in `docs/reports/2026-08-03_final_paper_readiness_report.md` §1.
- News panel: `data/features/dual_group_news_panel.parquet` (159,648 rows, 146 news features).
- EWMA recipe: `baselines/2026-07-25_dual_group_news_embedding_baseline/code/vendor_data_eda/dual_news_features.py`
  (`_ewma_on_series`, `halflife=30` trading days).
- Volatility / HAR source: `data/processed/*_processed.csv` (33 tickers, `parkinson_volatility`).
- Target construction: `src/lstm_gat_hybrid/dataset.py` lines 314–330 — target is the single-day
  `parkinson_volatility` at index `i + seq_length + forecast_horizon − 1` (a point forecast of
  volatility h days ahead, not an average over the window).

---

## 1. Horizon × architecture table (real numbers)

Test-set metrics, pipeline "b" (batch_size=32, no augmentation), 10 epochs unless noted. Bold = the
winning architecture at that horizon on QLIKE/RMSE/R² (news-fusion vs HAR-only, same epoch mark).

| Horizon | Architecture | R² | QLIKE | RMSE | DirAcc per-ticker (fixed) | DirAcc flatten (biased) |
|---|---|---:|---:|---:|---:|---:|
| 1d  | HAR-only   | 0.7581 | 0.5099 | 0.002428 | 32.26% | 72.35% |
| 1d  | Gated-news | **0.7595** | **0.4834** | **0.002420** | 33.16% | 72.39% |
| 5d  | HAR-only   | 0.7141 | 0.5623 | 0.002643 | — | — |
| 5d  | Gated-news | **0.7158** | **0.5436** | **0.002635** | — | — |
| 10d | HAR-only   | **0.7041** | **0.5732** | **0.002689** | 48.87% | 67.80% |
| 10d | Gated-news | 0.7040 | 0.5767 | 0.002690 | 48.32% | 67.92% |
| 22d | HAR-only   | **0.7051** | **0.5938** | **0.002750** | 50.17% | 66.38% |
| 22d | Gated-news | 0.7032 | 0.5943 | 0.002759 | 42.89% | 67.17% |

Notes on the numbers:
- 5-day row: no `h5` result directory exists; these are the §1.2 Bảng B values. The gated-news 5-day
  row uses epoch 20 (its peak), the HAR-only row uses epoch 10 — **not epoch-matched** (the source
  report flags this). All other rows are epoch-10 vs epoch-10.
- 10-day and 22-day margins are within single-seed noise (R²/QLIKE/RMSE differ in the 4th decimal);
  the direction (HAR-only wins) is consistent across all three continuous metrics at both long
  horizons, but the magnitude is not distinguishable from run-to-run variance without multi-seed.
- DirAcc: the "flatten (biased)" column is the historically-reported figure that compares different
  tickers on the same day (see `docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`); it is not a valid
  same-ticker-through-time directional accuracy and must not be cited. The "per-ticker (fixed)"
  column is the corrected metric — for HAR-only it is the mean of the per-ticker `dir_acc` fields in
  `results.json`; for gated-news it is `directional_accuracy_per_stock`. Both are near or below 50%
  (random) once corrected, consistent with the project-wide finding.

### 1.1 Trustworthiness caveat (must read before citing this table as a paper conclusion)

**These horizon baselines are dated 2026-08-01 and predate the P1.1 and P1.2 fixes** applied on the
night of 2026-08-02→2026-08-03 (`docs/reports/2026-08-03_final_paper_readiness_report.md` §3):

- **P1.1** (`src/lstm_gat_hybrid/dataset.py`): the normalizer was fit on the full dataset before the
  temporal split (leakage) and — more seriously — was **never actually applied** (`.transform()`
  missing), so these models trained on raw, unnormalized HAR/volatility.
- **P1.2** (`dataset_with_graph_method.py`): per-ticker outlier removal + positional stacking
  mis-aligned dates across tickers.

After these fixes, the 5-day headline moved materially: HAR-only QLIKE 0.5623 → **0.4839**,
gated-news 0.5436 → **0.4641** (still news-favourable at 5-day, one seed). The 1/10/22-day baselines
in the table above were **never re-run post-fix** — this is explicitly listed as an open item in that
report (§6, point 6). Therefore the "reversal at 10/22-day" pattern is real in the *pre-fix* numbers
but has **not** been reproduced post-fix. The explanations below are structural (they concern the
data, not the buggy training path) and are expected to survive a re-run, but the horizon table itself
**needs a fresh post-fix, multi-seed re-run before being stated as a settled paper result.** Present
it as a preliminary/pre-fix observation, not a final finding.

---

## 2. Measured evidence for the explanation

All statistics below are computed directly from the raw data (news panel + processed volatility),
independent of the (buggy) training pipeline, so they are unaffected by P1.1/P1.2.

### 2.1 The news signal the model consumes is over-smoothed and horizon-agnostic

The EWMA uses `halflife = 30` trading days (`alpha = 1 − exp(−ln2/30) = 0.0228`). Measured lag-k
autocorrelation (mean across embedding dimensions and 33 tickers):

| lag | EWMA(30d) news feature ACF | raw (un-smoothed) news feature ACF |
|---:|---:|---:|
| 1d  | 0.985 | 0.074 |
| 5d  | 0.926 | 0.054 |
| 10d | 0.857 | 0.047 |
| 22d | 0.722 | 0.038 |
| 40d | 0.572 | 0.027 |

Reading: the *raw* news embedding is spiky and essentially memory-less (ACF ≈ 0.07 at 1 day, ≈ 0.04
at 22 days) — event-driven and short-lived, exactly as expected. But the feature actually fed to the
model is the 30-day EWMA, which is nearly a slow step function over the 1–22-day range (ACF still
0.72 at 22 days). The engineered feature has thrown away the short-lived timing information and
retained a slow "recent news intensity" regime indicator that barely changes across any of the four
horizons. This is the core mismatch: the horizon-specific reactivity of news lives in the raw signal,
which the 30-day half-life smooths out before the model ever sees it.

### 2.2 Volatility's own predictability decays fast; HAR's slow component degrades least

Measured `parkinson_volatility` autocorrelation (mean across 33 tickers): 0.351 (1d) → 0.212 (5d) →
0.184 (10d) → 0.124 (22d) → 0.081 (40d). Volatility mean-reverts quickly; by 22 days most of the
day-t information has washed out toward the unconditional level.

`|corr(feature_t, vol_{t+h})|` pooled across tickers:

| h | HAR-monthly (22d) | HAR-daily | news EWMA-norm (kq) | news EWMA-norm (th) |
|---:|---:|---:|---:|---:|
| 1  | 0.377 | 0.376 | 0.091 | 0.064 |
| 5  | 0.314 | 0.224 | 0.091 | 0.067 |
| 10 | 0.273 | 0.193 | 0.090 | 0.070 |
| 22 | 0.194 | 0.127 | 0.091 | 0.070 |

Reading: the HAR-daily feature's predictive correlation collapses with horizon (0.376 → 0.127),
tracking volatility's own ACF, while the smoothed **HAR-monthly** component degrades far less (0.377 →
0.194). As the horizon lengthens, the forecast increasingly relies on the persistent/mean-reverting
component that HAR's multi-scale design already encodes. The news EWMA-norm correlation is small
(≈ 0.06–0.09) and essentially flat across horizons — because it is so over-smoothed it neither gains
nor loses relevance with h; it simply carries little linear volatility signal at any horizon.

### 2.3 Incremental out-of-sample value of news over HAR shrinks (and turns negative) with horizon

Cleanest model-agnostic test: pool all tickers, temporal 70/30 split by date, standardized Ridge
(alpha=10), predict `vol_{t+h}` from HAR (3 features) vs HAR + the 66 EWMA news features. Test-set R²:

| h | R²(HAR) | R²(HAR+news) | Δ (news increment) | n_train | n_test |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.7913 | 0.7909 | −0.0004 | 68,749 | 29,464 |
| 5  | 0.7383 | 0.7375 | −0.0008 | 68,685 | 29,400 |
| 10 | 0.7219 | 0.7211 | −0.0008 | 68,557 | 29,368 |
| 22 | 0.6855 | 0.6831 | **−0.0023** | 68,301 | 29,240 |

Reading: in a linear out-of-sample sense the news block adds **no** value at any horizon and hurts
progressively more as h grows (−0.0004 at 1-day → −0.0023 at 22-day). The tiny short-horizon benefit
the deep gated model extracts (per §1, pre-fix) is beyond what a linear probe recovers — but the
*trend* the probe reveals (news' marginal contribution decays and turns negative with horizon)
matches the direction of the observed HAR-only-wins-at-long-horizon result exactly.

### 2.4 Sample size is NOT the explanation (competing hypothesis ruled out)

With `seq_length = 22`, windows per ticker = `min_len − seq − h`. Going from h=1 to h=22 loses only
21 of 1,276 windows per ticker (~1.6%). Training-set sizes in §2.3 differ by <0.7% across horizons
(68,749 → 68,301). The "longer horizon = fewer windows to learn the news effect" hypothesis is
quantitatively negligible here and can be dismissed.

---

## 3. Proposed explanation (summary)

Three mutually consistent, measured mechanisms explain the pattern:

1. **Feature-horizon mismatch (dominant).** The event-driven information in news is short-lived (raw
   ACF ≈ 0.07 at 1 day), but the 30-day half-life EWMA the model consumes is near-constant over the
   1–22-day range (ACF 0.72 at 22 days). The engineered signal is a slow "recent-news-intensity"
   regime variable, not a horizon-timed shock indicator; whatever marginal, transient volatility
   elevation news can flag is most relevant in the next few days and is dominated later by mean
   reversion.

2. **Growing HAR baseline predictability at long horizons.** Volatility mean-reverts fast (ACF 0.35 →
   0.12 over 1 → 22 days). HAR's smoothed monthly component degrades much more slowly with horizon
   (predictive |corr| 0.377 → 0.194) than its daily component (0.376 → 0.127). At long horizons the
   target is increasingly explained by the persistent/mean-reverting level HAR already captures, so
   the *relative* room for a news increment shrinks.

3. **Net incremental news value ≈ 0 short, slightly negative long.** An out-of-sample linear probe
   finds news adds −0.0004 R² at 1-day and −0.0023 at 22-day — a monotone decline matching the
   HAR-only-wins-at-long-horizon direction. Sample-size differences across horizons are <2% and do
   not compete as an explanation.

---

## 4. Plain-language paragraph for the paper Discussion

> News features improve (or at least do not degrade) volatility forecasts at the 1- and 5-day horizons
> but lose their advantage — and slightly reverse it — at the 10- and 22-day horizons. This is
> consistent with the temporal structure of the two information sources. The news signal supplied to
> the model is a 30-day half-life exponentially-weighted average of article embeddings; its raw,
> un-smoothed form is almost memory-less (autocorrelation ≈ 0.07 at a one-day lag), meaning news
> carries genuinely short-lived, event-driven information, while the smoothing turns it into a slow,
> highly persistent "recent-news-intensity" regime indicator (autocorrelation still 0.72 at 22 days)
> that is nearly identical whether one forecasts one day or one month ahead. Realized volatility, by
> contrast, mean-reverts quickly (autocorrelation falls from 0.35 at one day to 0.12 at 22 days), and
> as the horizon lengthens the forecast is increasingly governed by this slow mean reversion, which
> the multi-scale HAR features already capture well — the smoothed monthly HAR component loses only a
> third of its predictive correlation from the 1- to the 22-day horizon, versus two-thirds for the
> daily component. An out-of-sample linear probe confirms the net effect: the incremental variance
> explained by news over HAR is essentially zero at one day and becomes slightly negative at 22 days,
> a monotone decline that mirrors the model comparison. In short, news provides transient,
> short-horizon information whose relevance window is shorter than the longer forecast targets, while
> the HAR baseline's own predictability accounts for a growing share of the target as the horizon
> extends; the marginal contribution of news therefore concentrates at short horizons and vanishes at
> long ones. (Reduced-form differences in the number of training windows across horizons are below 2%
> and do not account for the pattern.)

## 5. Caveats and recommended next step

- The horizon×architecture table in §1 is **pre-fix (2026-08-01)** and single-seed. Before citing the
  reversal as a paper-final result, re-run the 1/10/22-day HAR-only and gated-news baselines on the
  post-P1.1/P1.2 pipeline, epoch-matched (and ideally multi-seed), matching what was already done for
  the 5-day headline in `docs/reports/2026-08-03_final_paper_readiness_report.md` §1.
- The structural evidence in §2 (autocorrelation/decay, predictive correlation, incremental OOS R²)
  is computed from raw data outside the training pipeline and is unaffected by P1.1/P1.2; it is the
  reliable, citable part of this analysis and is expected to hold after the re-run.
- The 5-day gated-news row is not epoch-matched to its HAR-only counterpart (epoch 20 vs 10); treat
  the 1-day and (post-fix) 5-day news advantage, not the 5-day epoch-mismatched row, as the primary
  short-horizon evidence.
