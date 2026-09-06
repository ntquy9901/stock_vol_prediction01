# Improving the models to detect/forecast volatility shocks — research synthesis (2026-09-06)

Deep-research (6 angles, 25 primary sources fetched, 113 claims → 25 verified, 23 confirmed / 2 refuted).
Motivation: error attribution showed the worst QLIKE on VN100/VN30 comes from real market shocks
(spikes on liquid panels; market-wide limit-lock zero-range days on VN30, e.g. 2025-04-10), not dirty data.

## The one constraint that decides everything
The strongest jump toolchain — **bipower variation, BNS G/H tests, Lee–Mykland, realized quarticity (HARQ)** —
requires **intraday high-frequency returns**. Lee–Mykland (2008, RFS) states misclassification "becomes
negligible" only at high frequency; on **daily Parkinson (high–low) variance these estimators are unreliable**.
So the canonical jump-decomposition literature is theoretically strong but **not plug-and-play** for this
thesis. This filters the options sharply.

## Techniques, evidence, and fit to the daily HAR-X/LSTM/VolGA pipeline

| Technique | Evidence (primary) | Needs intraday? | Fit / effort here |
|---|---|---|---|
| **HAR-CJ / threshold-HAR-CJ** (continuous+jump split) | Corsi–Pirino–Reno 2010 JoE: jumps significantly raise future vol; gains **"especially following a jump"** (S&P500, stocks, bonds) | **Yes** (bipower) | Not directly — needs a daily jump proxy (unvalidated) |
| **HARQ / HARQ-J** (quarticity-scaled AR coef) | Bollerslev–Patton–Quaedvlieg 2016 JoE: OLS upgrade, significant OOS gains on S&P500/DJIA | **Yes** (realized quarticity) | Not directly on daily data |
| **BNS / Lee–Mykland jump tests** | BNS 2004; Lee–Mykland 2008 | **Yes** | Not usable on daily |
| **Regime-switching HAR (soft-cluster)** | arXiv 2510.03236 (preprint): MSE −8–12% pre/COVID/post S&P500; better than Markov at abrupt breaks | **No** | ✅ Fits — regime features from daily data |
| **Markov-switching GARCH** | Marcucci 2005: **beats single-regime GARCH at SHORT horizons** (DM/SPA), crossover ~1 week | **No** | ✅ Relevant to h1/h5 shock responsiveness |
| **Change-point (PELT, BOCPD)** | Killick 2012 (PELT); BOCPD on S&P500/CSI300 daily — but **detection latency** (lags the break) | **No** | ✅ Usable as shock-onset/regime **feature** on daily returns (mind latency; PELT "accurate" claim was REFUTED 0-3) |
| **Asymmetric peak-aware loss (APAL)** | arXiv 2607.14871: reweights extreme-target + under-prediction; reduces to MAE at unit weights | **No** | ✅ Drop-in training loss for LSTM/VolGA (⚠ performance claim REFUTED 1-2 — benefit unproven) |
| **QLIKE-as-training-loss (vs MSE)** | fetched GNN/HAR study: QLIKE training handles heteroskedasticity better than MSE | **No** | ✅ Cheap: train the deep branches on QLIKE, not MSE |
| **Distributional / quantile (TFT)** | Lim et al. 2021 (TFT, multi-quantile) | **No** | Medium effort — quantile head on the LSTM/VolGA branch |
| **Cross-sectional spillover GNN early-warning** | ASTGCN cross-market vol (18 markets): graph predicts spillovers across horizons | **No** | ✅ Aligns with VolGA — graph as contagion early-warning |
| **Robust/winsorized QLIKE + limit-lock regime** | Patton 2011 (robust losses for imperfect proxies) | **No** | ✅ Inferred (no source directly on zero-range); high value for VN30 |

Refuted (do not rely on): PELT "accurately detected real-data change points" (0-3); APAL "Peak F1 > 0.79" (1-2).

## Recommended action plan (highest-leverage × lowest-effort first)

**Tier 1 — cheap, daily-data-native, directly slots in:**
1. **Regime / change-point features.** Run PELT or an HMM (2–3 states) on daily market returns; add the
   regime label + "days-since-last-changepoint" + rolling market-vol z-score as node/exogenous features to
   HAR-X **and** the LSTM/VolGA input. Handles abrupt breaks that single-regime models smooth over
   (Marcucci; regime-switching HAR). Mind BOCPD/PELT latency.
2. **Train the deep branches on QLIKE (not MSE).** QLIKE-as-loss handles heteroskedastic spike days better;
   trivial change in `train_masked_rich`. Keep MSE reported for comparison.
3. **Robust QLIKE evaluation + explicit limit-lock handling.** Report QLIKE with and without zero-range
   (limit-lock) days, or winsorize the per-day loss; on VN30 this removes the 2025-04-10 domination. Model the
   zero-range "no-move" regime separately (a hurdle/two-part step) so a floored target does not enter QLIKE
   as a spurious huge ratio.

**Tier 2 — moderate effort:**
4. **Asymmetric peak-aware loss** for LSTM/VolGA (emphasize extreme targets + penalize under-prediction) —
   try it, but its accuracy benefit is unproven (refuted), so A/B it against QLIKE-loss.
5. **Daily jump/spike proxy.** Since HF estimators are out, engineer a range-based/threshold daily jump flag
   (e.g. Parkinson variance > k × trailing median, or |return| > k × rolling σ) as a feature — but no source
   validates such a proxy recovers HAR-CJ gains, so treat as exploratory.

**Tier 3 — larger:**
6. **Distributional/quantile head** (TFT-style p10/p50/p90) on the deep branch for shock-interval forecasts.
7. **Graph contagion early-warning** — extend VolGA's edge to a spillover early-warning signal (ASTGCN-style).

## Caveats (from the verification pass)
- HF-dependent methods (bipower/BNS/LM/HARQ) are the best-evidenced but NOT applicable to daily Parkinson data.
- The regime-cluster HAR (arXiv 2510.03236) and APAL are single non-peer-reviewed preprints; APAL's performance
  claim was refuted. Treat Tier-2 as experiments, not settled wins.
- Robust-QLIKE and the limit-lock zero-range regime are **inferred needs** — no surviving primary source
  addresses them directly (open question). They remain the most defensible fixes for the VN30 failure mode.
- Verifier note: the safety classifier reviewing the verification agents was intermittently unavailable; the
  cited papers (BNS 2004, Corsi–Pirino–Reno 2010, Bollerslev–Patton–Quaedvlieg 2016, Lee–Mykland 2008,
  Marcucci 2005, Patton 2011) are established peer-reviewed sources and independently checkable.

## Open questions
- A validated daily (range-based) jump proxy that recovers HAR-CJ-style gains without intraday data.
- A principled two-part / hurdle model + evaluation for the limit-lock zero-range regime (VN30).
- Whether S&P500-established jump/regime gains transfer to thin, price-limited VN markets, and whether the
  VolGA graph adds cross-sectional contagion early-warning beyond univariate regime features.
