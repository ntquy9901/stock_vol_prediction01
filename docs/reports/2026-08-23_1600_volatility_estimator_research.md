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

| panel | estimator | obs | QLIKE | MSE | R² |
|---|---|---|---|---|---|
| vn30 | **parkinson** | 10,106 | **0.5159** | 1.93e-7 | 0.231 |
| vn30 | garman_klass | 10,106 | 0.5151 | 2.06e-7 | 0.195 |
| vn30 | rs_overnight (YZ-style) | 10,106 | 0.8906 | 8.51e-7 | 0.212 |
| vn30 | rogers_satchell | 10,106 | 0.9691 | 3.50e-7 | 0.125 |
| vn30 | close2close | 10,106 | 2.2106 | 1.00e-6 | 0.105 |
| hnx | **parkinson** | 103,910 | **3.885** | 1.60e-6 | 0.166 |
| hnx | garman_klass | 103,910 | 3.941 | 1.54e-6 | 0.171 |
| hnx | rs_overnight | 103,910 | 9.499 | 7.89e-6 | 0.009 |
| hnx | rogers_satchell | 103,910 | 5.851 | 2.89e-6 | 0.119 |

**Counterintuitive but clear result:** switching Parkinson→**Garman–Klass** changes nothing (QLIKE
0.516→0.515 vn30, 3.885→3.941 hnx); switching Parkinson→**Yang–Zhang-style (RS+overnight)** makes forecast
QLIKE **substantially worse** (0.516→0.891 vn30, 3.885→9.499 hnx), even though it floors slightly fewer days.
Reason: the overnight/gap component is **high-variance and not persistent**, so a HAR-type model (which
forecasts persistence) cannot predict it — the added estimation robustness does not translate into
forecastability; it hurts. Rogers–Satchell and close-to-close are also worse.

## Conclusion / recommendation
- **For forecasting in this pipeline, keep Parkinson.** Garman–Klass is an equivalent alternative (no change);
  Yang–Zhang / Rogers–Satchell / close-to-close are **worse** as forecast targets here because their overnight
  or noise components are not forecastable by HAR.
- The literature's "Yang–Zhang is the best *estimator*" is about **estimation** accuracy (matching latent
  variance), which is real at the data level (Section B) — but it does **not** carry over to **forecast**
  accuracy for this HAR/LSTM/GAT setup (Section C), matching the deep-research caveat that estimator-swap gains
  are "modest and not guaranteed."
- **The low-range/QLIKE problem is better fixed by loss/floor handling or a liquidity screen, not by changing
  the estimator** — the estimator swap that reduces zero-range days (YZ) simultaneously degrades forecast
  QLIKE. Both the literature and the empirics agree on this.
- Caveat: HAR-X only (deterministic); a deep model might exploit the overnight term differently, but given the
  large QLIKE gap this is unlikely to reverse. Ablation is on vn30 (liquid) + hnx (illiquid); the pattern is
  consistent across both.

## Files / reuse
- `scripts/eda/volatility_estimators.py` (+ test, + `docs/reports/eda/2026-08-23_volatility_estimators.html`) — data-level estimator/low-range diagnostic.
- `scripts/eda/estimator_forecast_ablation.py` (+ test) — HAR-X forecast ablation across estimator targets.
  Re-run: `python scripts/eda/estimator_forecast_ablation.py --panels vn30 hnx --estimators parkinson rs_overnight garman_klass`.
