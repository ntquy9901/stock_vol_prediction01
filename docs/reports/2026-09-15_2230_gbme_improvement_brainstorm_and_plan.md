# GBME volatility-forecast improvement — diagnostic, brainstorm, and pre-registered experiment plan

Date: 2026-09-15. Scope: how to improve the champion gamma-GBM ("GBME" = GBM+earn on SP500, GBM own-history
on VN) out-of-sample volatility forecast (QLIKE), grounded in a per-(stock,day) error diagnostic and a
cross-domain literature sweep. Baseline to beat: own-history HAR/GBM; metric QLIKE + Diebold-Mariano,
walk-forward, spike-robust.

## 1. The diagnosis that constrains everything

Per-(stock,day) error panels (`results/gamma_gbm/gbme_error_panel_{hose,sp500}_h*.parquet`, rendered in
`docs/reports/2026-09-15_gbme_error_panel_*.html`) establish:

- The forecast is heavily under-dispersed: `var(pred)/var(realized) = 0.08–0.24`. It over-forecasts calm
  days and under-forecasts storm days (mean-reversion overshoot).
- Error is spread, not concentrated: worst-1% of observations carry ~7% of total QLIKE on HOSE (17–20% on
  SP500). ~15% of QLIKE falls in COVID/2022/Apr-2025 crisis windows; ~85% is ordinary-day compression.

Two quantitative tests then bound what is recoverable:

1. **Global dispersion recalibration** (`log(pred_cal)=c+b·log(pred)`, b fit on validation, applied to test):
   optimal `b ≈ 1.0–1.15`; test QLIKE gain `+0.11…+0.24%` on HOSE, **negative `−0.36…−3.44%` on SP500**.
2. **Oracle per-decile recalibration** (each predicted-decile scaled by its own realized/pred ratio, fit on
   the TEST set — an unattainable upper bound): maximum recoverable QLIKE = **+0.17% (HOSE h1), +0.33%
   (HOSE h22), +0.32% (SP500 h1), +0.76% (SP500 h22).**

Reliability curves (E[realized|decile]/E[pred|decile]) are near-diagonal (HOSE h1: 0.86→0.99; SP500 h1:
1.02–1.09), with only a mild bow at h22.

**Conclusion (drives the plan):** under QLIKE — a Bregman loss consistent for the conditional mean — the
optimal point forecast of a very noisy variance proxy IS a shrunk, low-dispersion series. The measured
under-dispersion is therefore largely **Bayes-optimal shrinkage, not a fixable calibration error**. Any
post-hoc recalibration / dispersion-restoration method is capped below ~0.8% QLIKE even with oracle
knowledge, and realistically ~0. This is itself a clean, publishable negative result.

## 2. What is therefore ruled out or deprioritized (evidence-based)

- **Recalibration / dispersion-restoration family** (heteroskedastic post-hoc scaling, spread-regression,
  NGBoost conditional scale, quarticity inflation, conformal heteroskedastic scaler): capped <0.8% oracle by
  §1 → deprioritized. They redistribute a fixed conditional mean; they cannot add information.
- **Cross-firm graphs / GNN** (correlation, sector, MTGNN, DY-spillover, GNN-embedding, news co-mention):
  NO-GO ×6 (prior work + `newsrel_gbm_test.json`).
- **Technical indicators** (RSI/MACD/stochastic/ADX/OBV): return-oriented, subsumed by own-history `mr_*`.
- **Chart price patterns** (H&S, triangles, flags, cup-and-handle, VCP): return/direction tools; the only
  volatility-relevant kernel (triangle/squeeze compression) = Bollinger-bandwidth, already subsumed by
  GK/RS/YZ; no OOS evidence; hostile to thin HOSE (limit-lock, unadjusted splits).
- **RRG (Relative Rotation Graph)**: benchmark-relative, z-normalized (magnitude removed), lagging → worsens
  over-smoothing; redundant with `market_pk`; normalizer detonates on HOSE limit-lock days; no OOS evidence.
- **EVT tail overlay / Hawkes intensity / wavelet-CEEMDAN / ESN reservoir**: low ceiling (tail attacks only
  the 15% crisis mass), collinear with existing `mr_*`/`semi_neg`, or leakage-prone (two-sided decomposition).
  Deprioritized; EVT/Hawkes are SP500-only if attempted (HOSE too thin).

## 3. Levers that are NOT capped by §1 (they add information or change the estimator)

Only two families can move QLIKE beyond the calibration ceiling, because they change the conditional mean
being estimated (not just its dispersion):

### Tier 1 — pre-registered experiments

**T1a. QLIKE-native (or entropy) boosting loss.** The champion trains gamma-deviance but is evaluated on
QLIKE — related, not identical, so its trained optimum is off the metric optimum. Published evidence is
unusually consistent that aligning the loss helps QLIKE specifically (qlikeHAR, J. Forecasting 2026; HARNet;
entropy-loss HAR, Entropy 2025). Implementation = custom QLIKE objective in LightGBM/XGBoost (closed-form
`∂L/∂p = 1/y − 1/p`, `∂²L/∂p² = 1/p²`), floor kept in the gradient, everything else (features, seed-ensemble,
walk-forward) unchanged.
- Compare: gamma-GBM vs QLIKE-GBM vs entropy-GBM, all h, both markets.
- Go/no-go: DM-significant QLIKE gain at ≥2 horizons on ≥1 market, no overfit verdict, spike-robust (rerun
  excluding COVID/2022/Apr-2025; sign holds), floor-invariant.
- Prior: modest-but-plausible (gamma deviance already close to QLIKE, so marginal gain may be small). Cheap.
  Genuinely under-tried in this repo.

**T1b. Cross-sectional return dispersion feature (CSV⁻).** A market-level scalar per day = variance across
stocks of that day's returns, restricted to the below-cross-mean (negative-side) names. Peer-reviewed OOS
evidence that CSV (and its asymmetric negative component) predicts aggregate realized volatility under HAR
(Niu et al. 2023, J. Forecasting; two ScienceDirect studies). Distinct from the falsified cross-firm graphs —
it is a single causal scalar, not a graph. This is the genuinely vol-relevant idea the RRG "cloud dispersion"
intuition gropes toward, without RRG's lossy transform.
- Operationalization: `CSV⁻_t = var_i(r_{i,t} | r_{i,t} < mean_i r_{i,t})`, one value per market per day,
  added as an exogenous feature to GBME; optionally `CSV⁺` as a placebo/robustness arm.
- Compare: own(+semi_neg) vs +CSV⁻, all h, both markets, date-clustered DM.
- Go/no-go: DM-sig QLIKE gain at ≥2 horizons; spike-robust; **placebo check** (a date-shifted CSV⁻ must NOT
  reproduce the gain — mandatory per the 2026-09-10 seasonal-artifact lesson).
- Prior: strongest published support of any untried lever; but evidence is for AGGREGATE RV and short h, so
  per-stock gains may be trivial (Hwang & Satchell 2005 found ~0 for individual-stock GARCH-X). Bounded test.

### Tier 2 — modest ceiling, run only if Tier 1 shows life

**T2a. Kalman online-level overlay.** GBM is frozen within a fold and cannot track drift inside the test
block. A one-state Kalman filter on `log(realized/GBM_pred)` (process/obs variance fit on val, frozen for
test) adds a slow-moving level for storm CONTINUATION (not onset — onset is exogenous). Evaluate on the
storm-decile QLIKE (must recover D5–D9 without hurting calm deciles). Overlaps the prior regime-blend
(modest recover); this is the online-filter version.

**T2b. MSM long-horizon feature (h10/h22 only).** Markov-switching multifractal has documented long-horizon
gains over GARCH/FIGARCH via true long-memory; h22 is where HAR/GBM are weakest. Feed MSM's analytic h-step
variance forecast as an extra GBM feature at long h only. Prior: plausible at h22, likely null at h1–5.

## 3b. Tier-1 experiment RESULTS (both resolved — no gain)

- **T1a (QLIKE-native loss) → NO-OP.** Proven: sklearn half-gamma-deviance `= y/p − log(y/p) − 1` is QLIKE
  EXACTLY (numerical diff 3.5e-15). The champion gamma-GBM already minimizes QLIKE. Empirical confirmation
  (xgboost custom-QLIKE objective vs sklearn gamma, HOSE h1 fold6, n=50k): QLIKE 1.3705 vs 1.3680 — the
  QLIKE-objective model does NOT beat the champion (marginally worse, tree-algorithm noise). The literature's
  QLIKE-loss gains (qlikeHAR 2026) apply to MSE/OLS-trained HAR, not to a gamma-GBM. Lever already pulled.
- **T1b (CSV⁻ cross-sectional dispersion) → NO-GO (hurts).** HOSE, base = own+semi_neg. csv_neg (real) hurts
  every horizon: h1 −1.68% (DM p=0.044, significantly worse), h5 −1.53% ns, h10 −0.86% ns, h22 −0.30%
  (p=0.073); ex-spike same sign. Placebos (csv_pos, +45d-shifted csv_neg) also hurt. CSV⁻ is the first
  market-level feature added to the own-history set (OWN carries no market_pk) — and it degrades OOS QLIKE,
  consistent with the project-wide finding that cross-sectional/market information does not improve per-stock
  QLIKE OOS. Niu et al. (2023) applies to AGGREGATE market RV, not per-stock (cf. Hwang & Satchell 2005 ~0 for
  individual stocks). `results/gamma_gbm/csv_dispersion_hose.json`.
- **Range-compression / squeeze (chart-pattern operationalized) → NO-GO.** HOSE, no variant DM-significant at
  any horizon (h1 all hurt −0.07…−0.46%, h5/h22 +0.04…+0.15% ns, h10 hurt); ex-spike same. Confirms the
  chart-pattern verdict empirically: the squeeze ratio ≈ har_weekly/har_monthly the tree already forms.
  `results/gamma_gbm/range_compression_hose.json`.

## 4. Results already in hand (this session)

- **semi_neg (downside semivariance) → does NOT robustly clear the bar on either market.**
  - HOSE: h1 `+0.55%` DM p=0.059 (borderline; an earlier run gave p=2e-4 — unstable around 0.05), h5/h10/h22
    ns. `own+semi` (both signed) helps h5 (p=7e-9) but hurts h1.
  - SP500 (base own+earn): h1 `+0.43%` DM p=0.101 (ns) AND **ex-spike −0.07%** — the h1 gain is spike-driven
    and REVERSES when crisis windows are excluded, failing the mandatory spike-robustness gate. h5/h10/h22 ns.
  - Verdict: the one "least-bad" own-history lever is a marginal, non-robust, borderline-at-best effect; it
    does not clear DM p<0.05-robust-to-spike-exclusion on either market. `results/gamma_gbm/augment_leverage_*.json`.
- **Equity conditional-covariance (companion direction)**: factor-rank-k beats Ledoit-Wolf GMV vol by
  `+1.0–1.8%` on HOSE but NOT DM-significant; GBM-diagonal composition HURTS (per-stock QLIKE-optimal ≠
  portfolio-covariance-optimal). See `results/cov_forecast/cov_hose.json`.

## 5. Outcome — both Tier-1 levers resolved NULL; per the pre-registered plan, stop

The pre-registered sequence ran to completion:
1. **T1a QLIKE-native loss → NO-OP** (champion already trains on QLIKE; gamma-deviance ≡ QLIKE, proven + empirical).
2. **T1b CSV⁻ dispersion → NO-GO** (hurts every horizon on HOSE; first market-level feature, degrades OOS).
3. **Tier-2 (Kalman overlay, MSM)** were pre-gated on Tier-1 showing life. Tier-1 showed none → **not run.**
   They attack narrower, modest-ceiling targets (storm-continuation, h22 long-memory); left as optional
   future work but with a low prior given §1.

## 6. Final verdict — a clean, publishable negative

For per-stock daily-OHLCV equity volatility (HOSE + SP500) under QLIKE, the own-history gamma-GBM sits at the
QLIKE frontier. Every lever explored is NO-GO or no-op:

| Lever | Result |
|---|---|
| Cross-firm graphs / GNN (correlation, sector, MTGNN, DY-spillover, embedding, news co-mention) | NO-GO ×6 |
| Technical indicators (RSI/MACD/…) | subsumed by own-history |
| Chart price patterns + range-compression/squeeze | NO-GO (empirically confirmed) |
| RRG (relative rotation graph) | NO-GO (relative-strength, magnitude removed) |
| Post-hoc recalibration / dispersion restoration | capped <0.8% even with oracle → skip |
| QLIKE-native loss (T1a) | no-op (gamma-GBM already = QLIKE) |
| CSV⁻ cross-sectional dispersion (T1b) | NO-GO (hurts; market info doesn't help per-stock) |
| Downside semivariance (semi_neg) | marginal, not spike-robust, doesn't clear the bar |

Two mechanistic results explain and unify the null set: (a) the residual under-dispersion (var-ratio 0.08–0.24)
is **Bayes-optimal shrinkage under a noisy variance proxy**, not a fixable calibration error (oracle
recalibration recovers <0.8%); and (b) storm ONSET is exogenous, so no history-derived feature (price,
pattern, cross-sectional, relational) carries forward information about it. This is consistent with, and
strengthens, the project's own-history-dominates thesis. The exhaustive NO-GO set plus these two mechanisms
is a legitimate thesis contribution, not a failure to find a positive.

The one honest positive to report remains the **companion covariance direction** (factor-rank-k lowers GMV
portfolio OOS vol ~1–1.8% vs Ledoit-Wolf, though not DM-significant), where cross-stock structure is intrinsic
to the target rather than auxiliary.
