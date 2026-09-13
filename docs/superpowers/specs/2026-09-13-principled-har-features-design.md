# Spec — Principled HAR-family feature set for the per-stock volatility GBM

Date: 2026-09-13. Status: approved design (brainstorming), pending spec review → implementation plan.

## 1. Problem & motivation

The current per-stock gamma-GBM own-history feature set (`FM.OWN`, 9 features) mixes published HAR terms with
**ad-hoc, heuristic** features whose windows and transforms are arbitrary and whose names are misleading:
- `mr_change, mr_slope5, mr_slope10, mr_dev5, mr_z22` — hand-engineered momentum / mean-reversion transforms on
  log-variance; windows (1, 5, 10, 22) and the transform-per-window (raw change / slope / raw-deviation /
  z-score) are inconsistent and not derived from any model or the data.
- `rq` — computed as `sqrt(mean(pk², 5d))`, a 5-day RMS of the daily Parkinson variance, LABELLED after HARQ
  realized quarticity (Bollerslev-Patton-Quaedvlieg 2016) but NOT the published RQ formula (which needs intraday
  returns and is a 1-day quantity).

Goal: replace the heuristic block with a **principled, citation-backed HAR-family feature set** whose lag
windows are **data-driven (causal)**, implemented in an **isolated new baseline** (no change to the shared
`FM.OWN`, which 42 files depend on), and validated by **Diebold–Mariano (DM)** head-to-head against the current
own-history set.

## 2. Success criteria (go/no-go)

**Non-inferiority + legitimacy.** The new principled set is a GO if, under the project's standard walk-forward
/ pooled-QLIKE / date-clustered-DM protocol on both markets and all horizons (h = 1, 5, 10, 22), its QLIKE is
**not significantly worse** than the current `FM.OWN` GBM (i.e., no DM test shows the principled set worse with
p<0.05 AND an economically non-trivial margin ≥ ~0.3%). Beating own-history is a bonus, not required (the
project's standing finding is that nothing reliably beats own-history). The win condition is a feature set the
thesis can defend as principled (every feature cited + formula-verified, lags data-justified), performing on
par with the ad-hoc set.

Reported honestly either way; with n≈400k, a DM-significant but <0.1% QLIKE gap is treated as non-inferior
(economically negligible), consistent with the project's standing lesson.

## 3. Feature catalog (all published, buildable from daily OHLC)

Base daily variance estimator stays Parkinson (project-validated). `pk` = Parkinson variance
σ² = (ln(H/L))² / (4·ln2) (Parkinson 1980). All features are causal (trailing, data ≤ t).

### A. HAR-RV core (Corsi 2009)
Multi-scale realized variance: RV⁽ᵈ⁾ = pk_t; RV⁽ʷ⁾ = mean(pk over w_short days); RV⁽ᵐ⁾ = mean(pk over w_long
days); optional RV⁽ q ⁾ = mean over w_longer. Windows chosen data-drivenly (§4). Captures the long-memory /
heterogeneous-horizon structure of volatility.

### B. Leverage — realized semivariance / SHAR (Barndorff-Nielsen et al. 2010; Patton & Sheppard 2015)
Split realized variance into downside/upside using signed daily log-returns r:
RS⁻ = Σ r²·𝟙(r<0), RS⁺ = Σ r²·𝟙(r>0), aggregated over the HAR windows. The documented "good vs bad volatility"
result: RS⁻ (bad vol) predicts future volatility more strongly (leverage effect). This replaces the ad-hoc
momentum features with a theory-grounded, cited directional-asymmetry signal.

### C. Multi-estimator range volatility (Garman-Klass 1980; Rogers-Satchell 1991)
Alternative published OHLC daily variance estimators as complementary daily inputs:
- Garman-Klass: σ²_GK = 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))².
- Rogers-Satchell: σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) (drift-independent).
Candidate features; leave-one-out (§5) prunes any that are not non-inferior-justified.

### Dropped
All `mr_*` heuristics and the `rq` proxy.

### Explicitly NOT buildable (documented limitation)
True realized quarticity (HARQ), jumps / bipower variation (HAR-CJ), signed jumps — all require **intraday**
data; only daily OHLC is available. State as a limitation; do not ship a mislabelled proxy.

## 4. Data-driven, causal lag selection

Windows are NOT hard-coded to {1,5,22}. They are selected **once** from the training data that precedes the
first walk-forward test fold (causal — no test data used), then **fixed across all folds** (so every fold's
feature matrix has identical semantics; avoids per-fold drift and keeps the DM comparison clean):
1. On the pre-first-test-fold training log-RV, compute the sample ACF/PACF and estimate the long-memory
   fractional-integration parameter d (GPH estimator) to confirm slowly-decaying memory.
2. Choose the HAR component windows (w_short, w_long[, w_longer]) by minimising a HAR in-sample criterion
   (AIC / residual) over a small candidate grid, or by matching the ACF decay knots.
3. Report the chosen windows per market + the ACF/d evidence. If the data confirms {1,5,22}, the previously
   arbitrary choice becomes data-justified; if VN differs, use the VN-implied windows.

All scalers fit on train only; semivariance/estimators computed per-ticker causally.

## 5. Validation protocol

- Build the principled panel; GBM(principled) vs GBM(FM.OWN) head-to-head under the sibling protocol
  (`run_gbm` / `verify_reduced_features` pattern): expanding walk-forward over S1.FOLDS, embargo int(h·1.6)+5,
  seed-averaged gamma-HistGBM, pooled per-observation QLIKE, date-clustered DM, train/test fit diagnostics.
- **Leave-one-out** within the principled set (drop each feature/group, DM vs full principled) to prune to the
  non-inferior minimal set.
- Markets: HOSE local; SP500 via a committed Colab notebook (git-centric: commit code → Colab clone + run →
  push result JSON → pull for analysis). Do not run SP500 heavy jobs locally.

## 6. Deliverables (mandated baseline structure)

`baselines/2026-09-13_principled_har_features/` with the 5 subfolders:
- `requirements/requirements.md`, `design/design.md` (this spec, adapted).
- `code/`: config (all constants — windows, grids, seeds), `estimators.py` (Parkinson/GK/RS/semivariance, each
  formula-exact), `lag_select.py` (causal ACF/GPH + grid), `build_panel.py` (causal feature panel), `run_har.py`
  (GBM + DM runner), `__init__.py`, sys.path bootstrap.
- `test/`: formula-exact tests (independent recompute per estimator vs its published formula — MANDATORY per
  the named-estimator rule), causality/leakage tests, lag-selection test on synthetic long-memory series,
  DM-runner smoke with stub loaders + patched folds. C0 line = 100%, C1 branch ≥ 95% on changed lines.
- `code_review/`: 3-layer adversarial review, fix HIGH/MEDIUM.
- A Colab notebook `notebooks/principled_har_sp500_colab.ipynb` (+ `.gitignore` allowlist).
- Result JSONs under `results/gamma_gbm/`; summary report under `docs/reports/`.

## 7. Constraints / non-goals
- Do NOT modify the shared `FM.OWN` or any of the 42 dependents.
- Do NOT introduce any feature that cannot be tied to a published formula + citation.
- Do NOT use intraday-only constructs (RQ/jumps) — document as a limitation.
- Keep parsimony: prefer the smallest non-inferior set; every surviving feature must be formula-verified and
  either DM-justified or a canonical HAR-RV core term.
