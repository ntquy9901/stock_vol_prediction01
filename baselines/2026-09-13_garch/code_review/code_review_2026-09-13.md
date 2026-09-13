# Code review — GARCH baseline (2026-09-13)

Adversarial self-review (Blind Hunter + Edge-Case Hunter + Acceptance Auditor lenses + a performance
lens, per CLAUDE.md). Scope: `code/config.py`, `code/garch_model.py`, `code/run_garch.py`. The formal
`/code-review` skill result is appended at the end.

## Findings and resolutions

### H-1 (FIXED before any real run) — train-only ticker left NaN in the train-metric array
`_build_jobs` originally created the last fold's train job only for tickers present in that fold's
TEST. A ticker delisted before the last test window (in train, absent from test) then had its train
rows unfilled (`NaN`), and `metrics.per_obs_qlike` fails loud on non-finite input — crashing the whole
run at the `train_metrics` step. Fix: on the last fold, iterate `set(test tickers) | set(train
tickers)`; a test-absent ticker gets an empty test window and a filled train window. Regression test
`test_run_handles_ticker_in_train_but_not_last_test`.

### H-2 (FIXED) — `config` module-name collision under process spawn
`ProcessPoolExecutor` (spawn) re-imports the driver in each worker; `full_matrix`'s import chain
prepends `submission/soict_lstm_gat` (which ships its OWN `config.py`) to `sys.path`, so the worker's
`import config` resolved to the wrong module (`AttributeError: MIN_TRAIN_OBS`). Fix: switched the
parallel backend to `ThreadPoolExecutor` — one interpreter, `config` already correctly bound in the
main thread, no pickling/spawn re-import. GARCH's cost is `arch`'s Cython variance recursion + SciPy
optimiser, both of which release the GIL, so threads still parallelise the fits. Verified by a real
HOSE run completing.

### H-3 (FOUND IN FIRST REAL RUN, FIXED) — degenerate zero-history fallback blew up pooled QLIKE
The first HOSE run gave GARCH QLIKE ≈ 82–150 vs HAR ≈ 1.8 — a 45× blowup inconsistent with the normal
in-sample fit (train QLIKE ≈ 2). Per-ticker diagnosis showed the MEDIAN per-ticker QLIKE was 0.59
(GARCH competitive), but two newly-listed tickers (VVS, NO1) with ZERO pre-fold returns had
`fallback_var = var([]) = 0` → forecast floored to `FL=1e-8` → `y/FL` astronomical → per-ticker QLIKE
≈ 2e5, dominating the pooled mean. This is degenerate garbage (a per-stock GARCH cannot be estimated
for a stock with no own history), exactly what the "no silent garbage" rule forbids. Fix: an
**estimability gate** — a ticker-fold is scored only when it has ≥ `MIN_TRAIN_OBS` returns before the
fold, and those rows are dropped from EVERY model (HAR/GARCH/GJR stay on identical rows), with the count
reported as `n_excluded`. Genuine thin-ticker underperformance (QLIKE ≈ 9, forecast right order of
magnitude) is retained — only un-estimable rows are removed. Regression test
`test_run_excludes_short_history_ticker`; re-ran HOSE after the fix.

### M-0 (FIXED, from the formal /code-review) — missing `arch` would silently degrade to all-fallback
`fit_params` did `from arch import arch_model` inside the `try/except`, so a missing/broken `arch`
install would be swallowed and turn the entire benchmark into a constant unconditional-variance
forecast. Fix: `import arch` at module load (loud `ImportError` on a broken environment); the `try`
now wraps only the fit + param extraction (genuine per-series convergence failures).

### M-1 (accepted, documented) — `except Exception` in `fit_params`
Intentional fail-soft: any per-ticker `arch`/SciPy convergence or numeric failure must not crash a
run over hundreds of tickers. It is NOT silent — the ticker-fold falls back to the unconditional
sample variance and is counted in `n_fallback` (the "log" mandated by the brief). Narrowing the catch
would risk missing an `arch`-internal exception type and re-introduce whole-run crashes. Covered by
`test_fit_params_falls_back_when_arch_raises`.

### M-2 (checked, correct) — units / ×100 rescale
`arch` is fit on `daily_return × SCALE` (SCALE=100); every FITTED forecast is divided by `SCALE**2` to
return to the `parkinson_variance` scale. The FALLBACK path returns the sample variance in ORIGINAL
scale (never scaled). Both are asserted: `test_fitted_iid_series_recovers_variance_scale` (fitted path
recovers the true variance, not off by 1e4) and
`test_fallback_returns_sample_variance_in_original_scale`.

### M-3 (checked, correct) — causality / no look-ahead
Params are estimated only on `returns[:n_train]` (dates `< ts − embargo`); the one-step recursion
`s_next[t]` depends only on returns with index ≤ t. `test_forecast_is_causal_future_returns_do_not_leak`
perturbs a future return and asserts an earlier test-date forecast is unchanged.

### M-4 (checked, correct) — GJR multi-step persistence
Reversion uses φ = α+β+γ/2 (expected leverage under a symmetric zero-mean innovation, matching the
Normal distribution used); the realised one-step recursion uses the actual `1(ε_t<0)` indicator. Both
GARCH and GJR multi-step forms are verified against an independent iteration of the expectation
recursion and the h→∞ reversion to σ̄² (`test_multistep_*`, `test_multistep_reverts_*`).

### M-5 (checked, correct) — identical QLIKE / DM basis vs HAR
GARCH/GJR forecasts are scattered into the SAME reset-indexed `te` used to compute
`FM._har_ols(tr, te)` and the target `y`; the pooled QLIKE uses the shared floor `FM.FL`; the
date-clustered DM runs on the same per-obs loss rows/dates. So GARCH, GJR and HAR are scored on exactly
the same rows under the same fold gate — apples-to-apples.

### L-1 (accepted) — `pos` map `KeyError` on a mid-series `daily_return` gap
Date→index mapping assumes every panel test/train date is present in the (NaN-`daily_return`-dropped)
return series. Enriched frames only have a missing `daily_return` at the first row (already dropped by
the panel's `dropna`), so this holds; a genuine mid-series gap would raise loudly (a data-integrity
signal), which is the desired no-silent-degradation behaviour rather than a masked bad row.

### L-2 (accepted) — recursion is a per-series Python loop
`one_step_next` is an inherently sequential O(len) filter (numeric ops vectorised except the scalar
recurrence). It is not the dominant cost (the `arch` MLE is) and cannot be batched across time without
changing semantics; the fits ARE parallelised across independent tickers. Consistent with the
Performance/Batching gate (no GPU tensor workload; parallel across the independent unit).

## Performance lens
- Not a GPU/tensor workload — classical per-series ML estimation. Parallelised across tickers with a
  thread pool (`n_jobs = cpu−1`); no batch=1 main-thread anti-pattern. The per-(ticker,fold,horizon)
  fit is the unavoidable unit of work (embargo, hence the train window, depends on h). Short-history
  tickers short-circuit to the O(1) fallback.

## Formal `/code-review` (3-layer subagent) — result
Ran an adversarial 3-layer review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor + performance
lens). It confirmed the load-bearing properties correct (units/×100 rescale, causality/no-leakage,
multi-step closed form incl. GJR γ/2, row alignment vs HAR, DM direction, shared QLIKE floor,
thread-safety) with NO CRITICAL/HIGH defect. Actionable findings raised: M-0 (arch import — FIXED),
design/docstring drift after the thread switch (FIXED), and reminders to report `n_fallback`/thin-market
prevalence in the paper (done via `n_fallback` + `n_excluded`). LOW items (KeyError coupling on a
mid-series return gap = fails loud by design; GJR γ/2 tied to the symmetric distribution = documented)
accepted.

## Verdict
No open HIGH/MEDIUM findings. HIGH items (H-1 train-ticker NaN, H-2 config-spawn collision, H-3
zero-history blowup) fixed with regression tests; MEDIUM items fixed or accepted with justification;
LOW items accepted. Tests: 24 pass, C0=100% / C1=100% on changed lines.
