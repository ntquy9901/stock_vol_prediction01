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

**Second fix — screened universe must match the delivered pipeline.** A middle run reported HNX Parkinson
QLIKE 3.885 over 291 tickers, ~2× the delivered 1.872 over 154. Cause: flooring the target at 1e-10 (above)
made the liquidity screen (which drops tickers whose Parkinson is `== 0.0` on >50% of days, i.e. H==L) keep
nearly all 291 tickers instead of the delivered 154 — the 137 extra illiquid tickers inflated the pooled
QLIKE. Fix (`screened_tickers`): fix the ticker universe ONCE by screening the delivered processed Parkinson
files and apply that same set to every estimator. HNX Parkinson QLIKE is then **1.8721 over 154 tickers /
62,004 obs — matching the delivered 1.8717**. Numbers below use this canonical universe.

| panel | estimator | nodes | obs | QLIKE | MSE | R² | note |
|---|---|---|---|---|---|---|---|
| vn30 | **parkinson** | 31 | 10,106 | **0.5159** | 1.93e-7 | 0.231 | matches delivered |
| vn30 | garman_klass | 31 | 10,106 | 0.5151 | 2.06e-7 | 0.195 | ≈ Parkinson |
| vn30 | rs_overnight (YZ-style) | 31 | 10,106 | 0.8906 | 8.51e-7 | 0.212 | worse (clean data: real effect) |
| vn30 | close2close | 31 | 10,106 | 2.2106 | 1.00e-6 | 0.105 | worse |
| hnx | **parkinson** | 154 | 62,004 | **1.8721** | 1.40e-6 | 0.215 | matches delivered (1.8717) |
| hnx | garman_klass | 154 | 62,004 | 2.1089 | 1.55e-6 | 0.190 | slightly worse |
| hnx | rs_overnight | 154 | 61,866 | 2.3095 | 5.23e-6 | 0.187 | worse |
| hnx | close2close | 154 | 61,866 | 3.9029 | 2.56e-6 | 0.165 | worse |

(High absolute QLIKE on HNX vs vn30 is the illiquidity/low-range effect from the EDA, not a bug — the delivered
HAR-X QLIKE on HNX is 1.87 by the same panel-correct pipeline.)

**Corrected reading (canonical universe):**
- **Garman–Klass ≈ Parkinson** on vn30 (0.516 vs 0.515) and slightly worse on hnx (1.872→2.109). Both are
  intraday and split-immune — the clean, reliable comparison: **switching among intraday estimators does not
  help, and adds nothing on the liquid panel.**
- **rs_overnight (YZ-style) is worse than Parkinson on both panels** (vn30 0.516→0.891; hnx 1.872→2.309). On
  clean vn30 (only 0.001% of overnight days winsorized) this is a **genuine effect** — the overnight/gap
  component is not persistent, so a HAR model cannot forecast it and the target becomes less forecastable. (Two
  data bugs had to be fixed first before this was trustworthy: the overnight split artifact, and the screened
  universe — see above; the earlier 9.5 was not real.)
- **close2close is worst** (pure return noise, lowest efficiency).

## Conclusion / recommendation
- **For forecasting in this pipeline, keep Parkinson.** Garman–Klass is equivalent on the liquid panel (no
  measurable change) and marginally worse on the illiquid one; rs_overnight (Yang–Zhang-style) and close2close
  are worse on both.
- **Overnight-augmented / Yang–Zhang-style targets do not help per-day forecasting here** because the overnight
  component is not persistent. Note the two data-quality traps that had to be fixed to see this cleanly (the
  raw prices are not split-adjusted, so overnight is corrupted; and the liquidity screen must be pinned to the
  delivered universe) — a fully fair YZ test would still want split/dividend-ADJUSTED prices, which this repo's
  raw OHLCV is not. The intraday comparison (Parkinson vs GK) needs no such caveat and shows no gain.
- The literature's "Yang–Zhang is the best *estimator*" concerns **estimation** accuracy (Section B, real at the
  data level) but does **not** translate into **forecast** gains for this HAR/LSTM/GAT setup, matching the
  deep-research caveat that estimator-swap gains are "modest and not guaranteed."
- **The low-range/QLIKE problem is better addressed by loss/floor handling or a liquidity screen than by the
  estimator.** Caveat: HAR-X only (deterministic); vn30 (liquid) + hnx (illiquid).

## Files / reuse
- `scripts/eda/volatility_estimators.py` (+ test, + `docs/reports/eda/2026-08-23_volatility_estimators.html`) — data-level estimator/low-range diagnostic.
- `scripts/eda/estimator_forecast_ablation.py` (+ test) — HAR-X forecast ablation across estimator targets.
  Re-run: `python scripts/eda/estimator_forecast_ablation.py --panels vn30 hnx --estimators parkinson rs_overnight garman_klass`.
