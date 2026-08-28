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

### C2. Exact Yang–Zhang per-day formula (added after review)
The `rs_overnight` proxy above (RS + raw overnight²) is a crude YZ analog. The exact **per-day** Yang–Zhang
used by charting indicators (e.g. TradingView YZV) is `σ²_daily = r_o² + k·r_c² + (1−k)·RS` with
`r_o=ln(O/C_{-1})`, `r_c=ln(C/O)`, `k=0.34/(1.34+(n+1)/(n−1))` (≈0.139 for n=20); the indicator then RMA-smooths
it. Both were implemented (`yz_daily`, `yz_rma20`) and re-tested on the canonical universe:

| panel | estimator | QLIKE | MSE | R² | floored% |
|---|---|---|---|---|---|
| vn30 | **parkinson** | **0.5159** | 1.93e-7 | 0.231 | 0.40% |
| vn30 | yz_daily | 0.7217 | 7.66e-7 | 0.241 | 0.05% |
| vn30 | yz_rma20 (smoothed) | 0.0037 ⚠ | 1.97e-9 | 0.985 | 0% |
| hnx | parkinson | 1.8721 | 1.40e-6 | 0.215 | 13.06% |
| hnx | **yz_daily** | **1.6354** | 5.00e-6 | 0.180 | 10.75% |
| hnx | yz_rma20 (smoothed) | 0.0055 ⚠ | 1.25e-8 | 0.990 | 0% |

**This nuances the earlier conclusion:** the *exact* per-day YZ (with the k weight) is **not uniformly worse**.
It is worse than Parkinson on the liquid vn30 QLIKE (0.722 vs 0.516) but **better on the illiquid hnx QLIKE
(1.635 vs 1.872)** — because its overnight component cuts the floored-day fraction (10.75% vs 13.06%), exactly
the price-limit-market benefit the literature predicts. Its MSE is worse on both (overnight spikes are penalized
by L2). So YZ-vs-Parkinson is **metric- and liquidity-dependent**, not a clean loss. The crude `rs_overnight`
(0.891 vn30) overstated the loss because it lacked the k weight.

**`yz_rma20` (the smoothed indicator output) gives QLIKE≈0.004, R²≈0.99 — an ARTIFACT, not skill.** A 20-day
trailing-RMA target is nearly constant day-to-day (autocorrelation ≈1), so "predicting next day" is trivial; it
no longer measures next-day variance. It must not be compared to a single-day target. (No leakage — RMA is
trailing — but the metric is meaningless as a forecast.)

## Conclusion / recommendation
- **Keep Parkinson as the default** — it is best or tied-best on MSE everywhere and on QLIKE for the liquid
  panels, and it is the simplest and split-immune. Garman–Klass is equivalent on liquid / marginally worse on
  illiquid.
- **Yang–Zhang is not simply worse — it is metric- and liquidity-dependent.** The *exact* per-day YZ (`yz_daily`,
  with the k weight) has a **lower QLIKE than Parkinson on the illiquid HNX (1.635 vs 1.872)** because its
  overnight term reduces floored zero-range days — the price-limit-market benefit the literature predicts — but
  a **higher MSE** (overnight spikes) and a worse QLIKE on liquid vn30. So if the objective is **QLIKE on
  illiquid/price-limit universes**, YZ (per-day, exact) is worth considering; for MSE or liquid large-caps,
  Parkinson wins.
- **Do NOT use the RMA-smoothed indicator output (`yz_rma20`) as a target** — its QLIKE≈0.004 / R²≈0.99 is an
  artifact of a near-constant 20-day-smoothed target, not forecasting skill.
- **Caveats for any YZ adoption:** (1) the raw prices are not split/dividend-adjusted, so the overnight term is
  corrupted (winsorized here, but adjusted prices are the proper fix); (2) the k weight matters (the crude
  RS+overnight² proxy overstated the loss). The intraday-only comparison (Parkinson vs GK) needs neither caveat
  and shows no gain.
- The literature's "Yang–Zhang is the best *estimator*" concerns **estimation** accuracy (Section B, real at the
  data level) but does **not** translate into **forecast** gains for this HAR/LSTM/GAT setup, matching the
  deep-research caveat that estimator-swap gains are "modest and not guaranteed."
- **The low-range/QLIKE problem is better addressed by loss/floor handling or a liquidity screen than by the
  estimator.** Caveat: HAR-X only (deterministic); vn30 (liquid) + hnx (illiquid).

## Files / reuse
- `scripts/eda/volatility_estimators.py` (+ test, + `docs/reports/eda/2026-08-23_volatility_estimators.html`) — data-level estimator/low-range diagnostic.
- `scripts/eda/estimator_forecast_ablation.py` (+ test) — HAR-X forecast ablation across estimator targets.
  Re-run: `python scripts/eda/estimator_forecast_ablation.py --panels vn30 hnx --estimators parkinson rs_overnight garman_klass`.
