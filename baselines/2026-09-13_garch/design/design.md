# GARCH volatility baseline — design (plan)

> SDD phase 4 (Plan). Realises `requirements/requirements.md`. Mirrors the sibling drivers
> `baselines/2026-09-12_complex_network/code/run_gbm.py` and
> `baselines/2026-09-13_principled_har_features/code/run_har.py` for walk-forward / pooled-QLIKE / DM /
> fit-diagnostics / JSON layout, so this baseline drops into the same comparison table.

## 1. Files
```
baselines/2026-09-13_garch/
  requirements/requirements.md   # spec (done)
  design/design.md               # this file
  code/
    __init__.py
    config.py                    # ALL tunable constants (single source of truth)
    garch_model.py               # fit + causal recursion + multi-step forecast (pure, testable)
    run_garch.py                 # walk-forward orchestration + QLIKE + DM + JSON
  code_review/code_review_2026-09-13.md
  test/
    __init__.py
    conftest.py                  # sys.path bootstrap (copied from siblings)
    test_garch_model.py          # formula-exact + units + causality unit tests
    test_run_garch.py            # smoke with stub loader + patched folds
```

## 2. Data flow
```
FM.load(market) -> per-ticker enriched frames (date, daily_return, parkinson_variance, HAR cols)
        │
        ├── FM.panel(frames, {}, h) ── pooled panel `a` (y = pk.shift(-h)), defines eval rows + HAR basis
        │
        └── per-ticker daily_return series ── GARCH/GJR fit (train window) ── causal recursion
                                                                              ── σ²_{t+h|t} per test date
align GARCH/GJR forecasts to panel test rows by (ticker, date) ── identical rows as HAR
        │
   pooled per-obs QLIKE (floor FL) + date-clustered DM vs HAR ── JSON
```

## 3. `garch_model.py` (pure functions — the unit-tested core)
`arch` is imported at module load so a missing/broken install fails LOUD at import (never a silent
degrade to an all-fallback benchmark).
- `fit_params(train_returns, variant)` → `Params(mu, omega, alpha, gamma, beta, ok, fallback_var)`.
  Fits `arch.arch_model(train_returns*SCALE, mean=Constant, vol=GARCH, p=1[,o=1],q=1, dist=normal)`.
  `ok=False` when the fit raises / params degenerate (ω≤0 or φ∉(0,1)) / too few obs.
- `reversion_persistence(p, variant)` → φ (α+β, or α+β+γ/2 for GJR).
- `one_step_next(returns_scaled, p, variant)` → array σ²_{t+1|t} (scaled). Causal recursion:
  `sig[0]=σ̄²_scaled`; `sig[i]=ω+(α+γ·1(ε_{i-1}<0))ε²_{i-1}+β·sig[i-1]`; then
  `s_next[i]=ω+(α+γ·1(ε_i<0))ε²_i+β·sig[i]`.
- `multistep(s_next_scaled, p, variant, h)` → σ²_{t+h|t} (scaled) closed form.
- `forecast(returns, p, variant, test_idx, h)` / `ticker_forecast(returns, n_train, test_idx, h,
  variant)` → forecasts in ORIGINAL variance scale (÷SCALE²), with the unconditional-variance fallback
  when `ok=False`.

## 4. `run_garch.py`
- `_ticker_task(payload)` (module-level) → for one ticker: all (active fold, horizon) test-window (and
  last-fold train-window) forecasts for both variants.
- `run_garch(market, load_fn=None, n_jobs=1)`:
  1. load frames; build `panel(h)` per horizon; determine active folds (min_rows gate).
  2. dispatch `_ticker_task` across tickers (ThreadPoolExecutor, n_jobs=cpu-1 from main(); serial when
     n_jobs==1). Threads (not a process pool): a process pool re-imports the driver per worker and
     `full_matrix`'s import chain shadows the generic ``config`` module name under spawn; threads share
     the correctly-bound interpreter, and `arch`'s Cython recursion + SciPy optimiser release the GIL so
     the independent per-ticker fits still parallelise.
  3. assemble variant forecasts aligned to panel test rows; HAR via `FM._har_ols` on the same rows.
  4. pooled QLIKE + date-clustered DM + fit-diagnostics (last active fold train rows) → dict.
- `main()` (`# pragma: no cover`, entry driver) → `run_garch(sys.argv)`, print, write JSON.

## 5. Gates (SDD phase 4)
- **Simplicity Gate:** one config module, three code modules, no new abstractions beyond the sibling
  pattern. PASS.
- **Anti-Abstraction Gate:** uses `arch` directly for estimation and the repo's existing
  `metrics`/`stats`/`full_matrix` read-only; no wrapper layer. PASS.
- **Performance / Batching Gate:** GARCH is a per-series scipy-MLE fit (not a GPU/tensor workload, so
  batch=1 GPU rules do not apply). The hot cost is ~10⁴ independent per-ticker `arch` fits; these are
  **parallelised across tickers with a thread pool** (one ticker = one task = all its folds/horizons),
  not a batch=1 main-thread loop. Threads rather than processes: the generic ``config`` module name
  collides under process spawn (``full_matrix`` prepends a dir with its own ``config.py``); `arch`'s
  Cython/SciPy core releases the GIL so threads still parallelise. Serial path retained for
  tests/determinism. PASS (parallel plan stated).

## 6. Key design decisions
- **Fit per (ticker, fold, horizon):** the embargo (hence the train window end) depends on h, so the
  fit is horizon-specific to stay faithful to the sibling protocol. Cost bounded by process-pool
  parallelism; short-history tickers hit the cheap fallback.
- **Recursion over full history, params from train only:** the conditional-variance filter is causal
  regardless of window (uses only past returns); estimating params on train then filtering all history
  is the standard GARCH out-of-sample filtering procedure.
- **Reuse `FM.panel` for the eval rows and `FM._har_ols` for HAR** so the QLIKE basis and DM are
  identical to every other model in the table (same floor, same rows, same fold gate).
- **Estimability gate (added after the first real run):** a per-stock GARCH is undefined without own
  history; a ticker-fold is scored only when it has ≥ `MIN_TRAIN_OBS` returns before the fold, and the
  excluded rows are dropped from EVERY model (identical rows preserved) and reported as `n_excluded`.
  This prevents a zero-history ticker's ~0 fallback variance → floored forecast → astronomical QLIKE
  from dominating the pooled mean (observed on HOSE). HAR is still FIT on the full train window; only
  the SCORED rows are gated.
- **Symmetric-innovation multistep for GJR** (φ = α+β+γ/2) — exact under Normal/Student-t zero-mean
  errors, matching the reported distribution.

## 7. Test plan (TDD — write first, confirm fail, implement)
1. `multistep` closed form == independent iteration of `v_{k+1}=ω+φ·v_k` (GARCH and GJR). verify: equal.
2. `multistep` → σ̄² as h→∞. verify: |f − σ̄²| → 0.
3. units/scale: fallback unconditional variance returns the train sample variance in ORIGINAL scale
   (not ×10⁴); a constant-variance synthetic series recovers ≈ v after the ÷SCALE² rescale. verify.
4. causality: perturbing a future return does not change an earlier test date's forecast. verify: equal.
5. `run_garch` smoke on a stub loader + patched folds → JSON has all keys, all variants, DM present,
   `json.dumps` round-trips; empty-when-no-fold. verify.
