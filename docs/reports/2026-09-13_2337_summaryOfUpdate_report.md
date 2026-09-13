# GARCH volatility baseline — summary of update (2026-09-13)

## What changed
New baseline `baselines/2026-09-13_garch/` — a per-stock GARCH(1,1) and GJR-GARCH(1,1,1)
conditional-variance benchmark that plugs into the project comparison table on the shared walk-forward
folds and pooled per-observation QLIKE basis. Built under the full SDD → TDD → quality-gate → code-review
lifecycle. Run on HOSE locally; SP500 left for a Colab run.

## Files (path → purpose)
- `baselines/2026-09-13_garch/requirements/requirements.md` → spec (GARCH/GJR spec, units handling,
  leakage argument, estimability + convergence handling, go/no-go).
- `baselines/2026-09-13_garch/design/design.md` → plan (data flow, module design, SDD gates).
- `baselines/2026-09-13_garch/code/config.py` → tunable constants (SCALE, MEAN, DIST, HORIZONS,
  MIN_ROWS, MIN_TRAIN_OBS, persistence bounds).
- `baselines/2026-09-13_garch/code/garch_model.py` → `arch`-based fit + causal conditional-variance
  recursion + closed-form multi-step forecast + unconditional-variance fallback.
- `baselines/2026-09-13_garch/code/run_garch.py` → walk-forward orchestration (estimability gate,
  thread-pool dispatch), pooled QLIKE + date-clustered DM vs HAR, fit-diagnostics, JSON.
- `baselines/2026-09-13_garch/test/` → formula-exact, units/scale, causality, estimability, smoke tests.
- `baselines/2026-09-13_garch/code_review/code_review_2026-09-13.md` → adversarial review + findings.
- `results/gamma_gbm/garch_hose.json` → HOSE results.

## Method (as reported in the paper)
- GARCH uses ONLY each stock's own daily log-return history — no exogenous features (the pure
  econometric conditional-variance benchmark). GARCH(1,1) symmetric; GJR-GARCH(1,1,1) adds a leverage
  term. Estimated by ML with `arch` (Kevin Sheppard, 8.0.0) per (ticker, fold, horizon).
- Units: returns fit on `daily_return × 100`; conditional-variance forecasts rescaled `÷ 100²` to the
  `parkinson_variance` scale (unit-tested). Multi-step forecast
  `σ²_{t+h|t} = σ̄² + φ^{h-1}(σ²_{t+1|t} − σ̄²)` (φ = α+β; α+β+γ/2 for GJR), verified against an
  independent iteration of the expectation recursion.
- Walk-forward, embargo `int(h·1.6)+5`, fold gate `min_rows` — identical to the sibling GBM/HAR
  baselines; HAR (`full_matrix._har_ols`), GARCH, GJR scored on the IDENTICAL pooled rows under the
  shared QLIKE floor `FM.FL`; date-clustered Diebold-Mariano vs HAR.
- Estimability gate: a per-stock GARCH is undefined without own history, so a ticker-fold is scored
  only where it has ≥ `MIN_TRAIN_OBS`(=250) returns before the fold; excluded rows are dropped from
  every model (rows stay identical) and reported as `n_excluded` (~5.5k of ~399k, ≈1.4%).

## HOSE results (`results/gamma_gbm/garch_hose.json`)
QLIKE (lower better); DM vs HAR (mean_diff<0 ⇒ model beats HAR):

| h  | n       | GARCH  | GJR-GARCH | HAR    | GARCH vs HAR        | GJR vs HAR          |
|----|---------|--------|-----------|--------|---------------------|---------------------|
| 1  | 393,551 | 1.6539 | 1.6564    | 1.8080 | +8.52%, p=1.3e-159  | +8.39%, p=4.0e-154  |
| 5  | 391,975 | 1.7397 | 1.7405    | 1.8179 | +4.30%, p=2.1e-30   | +4.25%, p=5.5e-29   |
| 10 | 390,005 | 1.7802 | 1.7836    | 1.8236 | +2.38%, p=1.3e-08   | +2.20%, p=2.0e-07   |
| 22 | 384,917 | 1.8376 | 1.8414    | 1.8323 | −0.29%, p=0.589     | −0.50%, p=0.328     |

- GARCH significantly BEATS plain HAR at h1/h5/h10 (the ARCH term reacts to recent shocks faster than
  HAR's fixed rolling means); the edge shrinks with horizon and disappears at h22 (GARCH mean-reverts to
  the unconditional variance, no significant difference).
- GJR-GARCH tracks GARCH, marginally worse (the leverage term adds no QLIKE gain here).
- `n_fallback` (convergence fallback to unconditional variance) small: 56–67 ticker-folds per horizon.
- Note: this is GARCH vs the plain Corsi HAR (not HAR-X / GBM). It positions GARCH as a competitive
  classical benchmark; the project's feature-rich GBM champions are the intended comparison target.

## Verdict
Prior expectation was GARCH ≈/worse-than HAR on QLIKE; the measured result is that a correctly-scaled
GARCH beats plain HAR at short horizons and ties at h22 on HOSE. Reported as measured, no spin. The
deliverable is a correctly-implemented, leakage-safe GARCH benchmark for the paper.

## Tests + coverage
- 24 tests pass (`pytest baselines/2026-09-13_garch/test/`); C0 line = 100%, C1 branch = 100% on the
  changed lines (`config.py`, `garch_model.py`, `run_garch.py`), measured with `--cov-branch`.
- Formula-exact (GARCH/GJR multi-step vs independent iteration; reversion to σ̄²), units/scale recovery,
  causality (future returns do not leak), estimability exclusion, fit-failure fallback, and smoke
  (stub loader + patched folds, serial + threaded dispatch) all covered.

## Code review
Adversarial self-review + a formal 3-layer `/code-review` subagent (Blind Hunter / Edge-Case Hunter /
Acceptance Auditor + performance lens). Load-bearing properties confirmed correct (units/×100 rescale,
causality/no-leakage, multi-step closed form incl. GJR γ/2, row alignment vs HAR, DM direction, shared
QLIKE floor, thread-safety). Findings fixed: H-1 (train-ticker NaN in the diagnostic array), H-2
(`config` module-name collision under process spawn → switched to a thread pool), H-3 (zero-history
fallback blew up pooled QLIKE → estimability gate), M-0 (`arch` import made loud). Details in
`baselines/2026-09-13_garch/code_review/code_review_2026-09-13.md`.

## Performance
Not a GPU/tensor workload — classical per-series ML estimation. The ~10⁴ independent per-ticker `arch`
fits are parallelised across tickers with a thread pool (`arch`'s Cython recursion + SciPy optimiser
release the GIL); no batch=1 main-thread anti-pattern. A process pool was rejected because the generic
`config` module name collides under spawn.

## Data-quality gate
N/A (no data change): this baseline reads the existing enriched HOSE frames read-only and adds no new
raw/processed data, manifest, or pipeline change. No Pandera/Evidently run required.

## Risks / follow-ups
- SP500 not run locally (heavy) — run on Colab with `run_garch.py sp500`.
- Reported comparison is vs plain HAR; extending the JSON to also DM against HARQ / GBM is a follow-up
  if the paper wants GARCH in the full matrix beyond HAR.
- The pure-Python conditional-variance recursion is GIL-bound; a future speed-up (numba / truncating the
  recursion at max test index) would cut the HOSE runtime (~35 min) but is not needed for correctness.

## DoD checklist
- [x] SDD artifacts (requirements + design) written before code
- [x] TDD: tests first, 24 pass, C0=100% / C1=100% on changed lines
- [x] Lint: `ruff --select F` clean on the baseline
- [x] Adversarial code review (self + 3-layer subagent); HIGH/MEDIUM fixed
- [x] Real HOSE run, every number from the run (no fabricated values)
- [x] Summary report (this file)
- [ ] Push: intentionally NOT pushed — committed to the worktree branch for review
