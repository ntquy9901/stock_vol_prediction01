# Window size & number of folds for an LSTM — deep-research synthesis (2026-08-30)

Deep-research workflow (97 agents, verified citations). **Caveat:** safety classifier down during verify —
re-open links before citing. One source (DOI 10.1002/for.2841) did NOT resolve in verification — treat as unconfirmed.

Three distinct "windows" separated:

## 1. LSTM input lookback (sequence length)
- **Best-attested daily-financial precedent = ~240 days (~1 trading year)** — Fischer & Krauss 2018 (EJOR
  270(2):654-669, doi:10.1016/j.ejor.2017.11.054) use 240 timesteps of standardized returns. BUT this is
  next-day return DIRECTION, **not volatility**. **No verified source pins an optimal lookback for
  realized/Parkinson volatility** — short (5-22) vs long (60-252) remains a data-specific tuning choice.
- Implication: our **seq=10 is short** vs the 240 precedent; volatility has short memory (HAR uses 1/5/22),
  so a very long lookback is unlikely to help much, but 22/60 is worth a quick check.

## 2. Training window — expanding vs rolling
- **Neither dominates.** Feng, Zhang & Wang 2024 (J. Forecasting, doi:10.1002/for.3046, the paper 2502.15813
  cites) show an **adaptive "momentum of predictability"** rule (keep whichever window recently forecast
  better) beats BOTH individual windows and their mean-combination for OOS volatility + highest investor utility.
- **Window length is first-order under structural change** (Inoue, Jin & Rossi 2017, J. Econometrics
  196(1):55-67); a data-driven window beats a fixed one (asymptotically valid) — macro, not volatility.
- **Canonical rolling exemplar:** Fischer & Krauss use a fixed **~750-day (~3-year) rolling train window** +
  ~250-day trading window, retrained annually. Our **expanding** window is defensible; a rolling ~750-day is
  the canonical alternative if structural change matters.

## 3. Retrain cadence & number of folds
- Use **rolling-origin / walk-forward** averaging errors over many out-of-sample test sets (Hyndman &
  Athanasopoulos FPP3; Tashman 2000, doi:10.1016/S0169-2070(00)00065-0).
- **Standard K-fold CV is valid ONLY for stationary/autoregressive series with uncorrelated residuals**
  (Bergmeir, Hyndman & Koo 2018, CSDA, doi:10.1016/j.csda.2017.11.003; Bergmeir & Benítez 2012, doi:10.1016/j.ins.2011.12.028).
  For real non-stationary financial series, **temporal-order-preserving OOS gives the most accurate performance
  estimate** (Cerqueira et al. 2020, ML journal, doi:10.1007/s10994-020-05910-7).
- **More folds → more reliable OOS estimate.** Fischer & Krauss: annual retrain, **~23 folds over 25 years**.

## (d) Does window/fold choice flip deep-vs-HAR?
- Combining deep + GARCH (GEW-LSTM, Kim & Won 2018, doi:10.1016/j.eswa.2018.03.002) beats either alone on a
  single index. BUT the strong claim "LSTM/NARX beat HAR/GARCH for realized volatility" was **REFUTED in
  verification.** => window/fold choices affect **estimate reliability**, not reliably the verdict.

## Concrete recommendation for our VN100 walk-forward (~102 stocks, ~4500 anchors)

| knob | current | recommended | why |
|---|---|---|---|
| lookback | seq=10 | 10 (or test 22) | short-memory vol; 240-precedent is return-not-vol; low expected change |
| train window | expanding | expanding OK; optionally test rolling ~750d | neither dominates; expanding fine for our history length |
| retrain cadence | K=66 (quarterly) | **K=21 (monthly) → ~22 folds** | our **7 folds is LOW** vs canonical ~23 → monthly retrain gives a more trustworthy OOS/DM estimate |
| #folds | 7 | ~20-23 | more folds = lower-variance OOS error estimate |

**Single most valuable change:** **increase the number of folds (retrain monthly, K=21 → ~22 folds)** so the
walk-forward DM p-value is statistically robust — 7 folds is on the low side for a trustworthy estimate. This
is a reliability improvement, not expected to flip the LSTM≈HAR-X equivalence. Lookback and rolling-vs-expanding
are second-order here (short-memory volatility; the literature shows no universal winner).
