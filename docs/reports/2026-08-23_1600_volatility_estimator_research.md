# Would replacing Parkinson with another volatility estimator help? — research + empirics (2026-08-23)

Question: for daily-OHLCV volatility forecasting (current target = daily Parkinson variance
σ²=ln(H/L)²/(4ln2)), would switching to another daily estimator improve accuracy, and which is best?
Answered three ways: (A) deep literature research, (B) a data-level low-range diagnostic, (C) an
empirical HAR-X forecasting ablation on this pipeline's own data.

## A. Literature (deep-research, verified citations)
- **Parkinson is the weakest OHLC estimator:** valid only under zero drift and no opening jumps; it ignores
  overnight/opening gaps (underestimates) and is inflated by drift. [Yang–Zhang 2000, `atmif.com/papers/range.pdf`]
- **Rogers–Satchell (1991)** is drift-independent (unbiased for any drift) but still has no overnight term.
  [`projecteuclid.org/.../10.1214/aoap/1177005835`]
- **Yang–Zhang (2000)** is the most complete single estimator: drift-independent **and** opening-jump-consistent
  with minimum variance — proven that no single-period estimator can be independent of both drift and opening
  jumps (must be multi-period). Peak efficiency ~14× close-to-close (typical biweekly ~7.3×); Parkinson ~5×,
  Garman–Klass ~7–8×. [Yang–Zhang 2000]
- **Emerging/price-limit markets:** on limit-locked / zero-range days the intraday range collapses and
  Parkinson/GK/RS collapse with it; **Yang–Zhang degrades least because its overnight component still captures
  variance when the intraday range vanishes.** [research synthesis; Molnár 2012 `doi.org/10.1016/j.irfa.2012.02.016`]
- **Range vs realized:** all range estimators are far more precise than daily squared returns but remain noisy
  single-path proxies; high-frequency realized variance is more accurate but is unavailable here.
  [Andersen–Bollerslev; Patton 2011 on the proxy-robust QLIKE loss]
- **Bottom line from the literature:** Yang–Zhang is the most defensible *estimator*, **but the expected
  forecast-accuracy gain from switching the estimator alone is modest and not guaranteed; the higher-value
  lever is a robust loss (QLIKE/MSE) and floor handling on zero-range days.**

## B. Data-level low-range diagnostic (`scripts/eda/volatility_estimators.py`)
Computed 5 per-day variance estimators from raw OHLCV per panel. On H≈L (zero intraday range) days:

| rescue on H≈L days | close2close | parkinson | garman_klass | rogers_satchell | rs_overnight |
|---|---|---|---|---|---|
| vn30 | 0.65 | **0.00** | **0.00** | **0.00** | **0.65** |
| hnx | 0.26 | 0.00 | 0.00 | 0.00 | 0.26 |

Every **intraday-only** estimator (Parkinson, GK, RS) collapses to 0 on H≈L days (rescue 0.000) — switching
among them does **not** help the floor/QLIKE problem. Only **overnight-bearing** estimators (close2close,
RS+overnight) stay non-zero, rescuing 26–65% of H≈L days (the gap/limit days) and cutting the floored-day
fraction (hnx 0.450→0.385, hose 0.144→0.131). The remaining H≈L days (worst on illiquid HNX) are genuinely
untraded and unrecoverable by any estimator. This confirms the theory at the data level.

## C. Forecasting ablation on THIS pipeline (`scripts/eda/estimator_forecast_ablation.py`)
Regenerated the target from raw with each estimator on the **same date grid** (floored, not dropped — a first
attempt that dropped zero rows fragmented the panel and was discarded as unfair), refit the deterministic
HAR-X, scored the test fold. QLIKE is scale-invariant, so it is comparable across targets. (vn30 Parkinson QLIKE
0.5159 reproduces the delivered value exactly, validating the setup.)

**Data-quality bug found and fixed (do not trust the first run).** The first ablation gave rs_overnight a QLIKE
of **9.499** on HNX — implausibly high. Investigation: the OVERNIGHT return ln(O_t/C_{t-1}) is corrupted by
**unadjusted splits / zero or missing prior closes** in the raw prices (HNX had 1,100 overnight moves above
±20%, median 41%, up to 540% — impossible under the ±10% price limit, i.e. corporate-action / bad-data
artifacts). The intraday Parkinson estimator is immune (H,L are same-day, split-invariant); overnight-based
estimators are not. Fix (`estimators_from_ohlcv`): require `prev_close>0` and **winsorize the overnight
log-return at ±0.20** (twice the price limit). Numbers below are AFTER the fix.

| panel | estimator | obs | QLIKE | MSE | R² | note |
|---|---|---|---|---|---|---|
| vn30 | **parkinson** | 10,106 | **0.5159** | 1.93e-7 | 0.231 | intraday, split-immune |
| vn30 | garman_klass | 10,106 | 0.5151 | 2.06e-7 | 0.195 | ≈ Parkinson |
| vn30 | rs_overnight (YZ-style) | 10,106 | 0.8906 | 8.51e-7 | 0.212 | worse (clean data: real effect) |
| vn30 | close2close | 10,106 | 2.2106 | 1.00e-6 | 0.105 | worse |
| hnx | **parkinson** | 103,910 | **3.885** | 1.60e-6 | 0.166 | |
| hnx | garman_klass | 103,910 | 3.941 | 1.54e-6 | 0.171 | ≈ Parkinson |
| hnx | rs_overnight | 103,910 | **4.020** | 6.72e-6 | 0.154 | ≈ Parkinson after cleaning (was 9.499) |
| hnx | close2close | 103,910 | 5.037 | 3.43e-6 | 0.143 | worse |

**Corrected reading:**
- **Garman–Klass ≈ Parkinson** on both panels (0.516→0.515 vn30; 3.885→3.941 hnx). Both are intraday and
  split-immune, so this is the clean, reliable comparison: **switching among intraday estimators changes nothing.**
- **HNX rs_overnight fell from 9.499 to 4.020** once the overnight data errors were winsorized — the original
  "much worse" was **mostly a data-quality artifact**, not a property of the estimator. After cleaning it is
  ≈ Parkinson (3.885).
- **On CLEAN blue-chip data (vn30, only 0.001% of overnight days winsorized) rs_overnight is 0.891 vs Parkinson
  0.516 — a genuine effect, not an artifact.** Adding the overnight/gap component makes the per-day target less
  forecastable, because overnight jumps are not persistent and a HAR model forecasts persistence.

## Conclusion / recommendation
- **For forecasting in this pipeline, keep Parkinson** (or Garman–Klass — equivalent, intraday, split-immune,
  no measurable change).
- **Overnight-augmented / Yang–Zhang-style targets are neutral-to-worse for per-day forecasting here:** worse on
  clean vn30 (genuine, overnight not persistent), ≈ equal on hnx after cleaning. The initial dramatic gap (9.5)
  was largely a data-quality artifact and should not be cited as evidence — corrected here.
- **Overnight-based estimators cannot be tested fully fairly without split/dividend-ADJUSTED raw prices**, which
  this project's raw OHLCV is not; the intraday-only comparison (Parkinson vs GK) is clean and shows no change.
- The literature's "Yang–Zhang is the best *estimator*" concerns **estimation** accuracy (Section B, real at the
  data level) but does **not** translate into **forecast** gains for this HAR/LSTM/GAT setup, matching the
  deep-research caveat that estimator-swap gains are "modest and not guaranteed."
- **The low-range/QLIKE problem is better addressed by loss/floor handling or a liquidity screen than by the
  estimator.** Caveat: HAR-X only (deterministic); vn30 (liquid) + hnx (illiquid).

## Files / reuse
- `scripts/eda/volatility_estimators.py` (+ test, + `docs/reports/eda/2026-08-23_volatility_estimators.html`) — data-level estimator/low-range diagnostic.
- `scripts/eda/estimator_forecast_ablation.py` (+ test) — HAR-X forecast ablation across estimator targets.
  Re-run: `python scripts/eda/estimator_forecast_ablation.py --panels vn30 hnx --estimators parkinson rs_overnight garman_klass`.
