# Code review — Foundation-model feature baseline (2026-09-18)

Adversarial review (3 lenses: Blind Hunter, Edge Case Hunter, Acceptance Auditor) of
`foundation_config.py`, `foundation_forecaster.py`, `run_foundation.py`, `test_foundation.py`, cross-checked
against the vetted sibling `2026-09-17_gbme_glm_anchor` and all reused shared deps. `archive/` out of scope.

## Verdict
**No Critical issues. One Major (test-adequacy) — FIXED. Minors — 3 fixed, 2 accepted as low-risk follow-ups.**

## Verified correct (high-risk items pass)
- **Causality / no look-ahead.** `forecast_series` builds each context as `values[max(0,i-CTX_LEN+1):i+1]`
  (strictly ≤ t, never index i+1). The frozen model never trains. Caching over the full series does not leak —
  each row's context stops at its own t.
- **Step ↔ target alignment.** `step = h-1`; `median[:, step]` is the (t+h) forecast, matched to panel
  `y = parkinson_variance.shift(-h)` at row t. Off-by-one correct.
- **Placebo direction.** `_placebo` cyclic map + inverse relabel hands each ticker ANOTHER ticker's forecast on
  a date-aligned merge (genuine wrong-ticker, still causal in the wrong firm's history).
- **Paired rows for DM.** `_attach` inner-joins fnd then placebo, so GBME/FND/PLAC/HAR/ZS are scored on identical
  `(ticker,date)` rows — valid paired DM.
- **QLIKE floor/cap consistency.** All GBM preds floored at `FL`; ZS clipped `[FL, PRED_CAP]`; fnd clipped at
  source; `per_obs_qlike(floor=FL)` uniform across models.
- **Batching / no batch=1.** `_batched_forecast` chunks by `FORECAST_BATCH=512`; one `.cpu().numpy()` per batch,
  no per-step host sync. Measured ~20k forecasts/s on the RTX 4060.
- **Val slice / embargo** (`VALID_LEN=22`, `embargo=int(h*1.6)+5`, fit on `trf_e`, predict `combo`, slice) is the
  sibling logic verbatim; val is out-of-fit, reporting-only (GBM has no early-stopping) — no selection leakage.
- **Gate-skip claim accurate.** Model names match none of `overfit_check._LEARNED_PATTERNS` and there is no
  `design`/`metrics_per_seed`, so `check_overfit_evidence.check_files` returns `{}` (asserted in a test). The doc
  still carries full train/val/test + `fit_diagnostics`.

## Major (FIXED)
- **M1 — no test locked the step→horizon→target alignment.** The persistence fake returned identical values for
  every step, so `fnd_h1==fnd_h5==...` in every driver test; a silent off-by-one in `step=h-1` would have passed.
  **Fix:** added `_step_fn` (step-varying fake: step k = `context[-1] + k*1e-8`) and two tests —
  `test_forecast_series_step_to_horizon_alignment` (asserts `fnd_h{h}` carries step h) and
  `test_attach_uses_horizon_matched_forecast` (asserts `_attach` picks the horizon-matched `fnd_h{h}` and that
  h1≠h5 at the same row).

## Minors
- **m2 — dead config `KILL_HORIZONS`.** FIXED: removed; `success` uses `KILL_MIN_HORIZONS` (count), not specific
  horizons.
- **m4 — `zeroshot_matches_har` misnomer** (flag was true only when ZS *beats* HAR). FIXED: renamed to
  `zeroshot_beats_har` in the result JSON and the printout.
- **Memory (found during the real run, not by the static review).** The full enriched HOSE panel is ~469 MB × 30
  cols; `_attach` duplicated it in the merge and held the 71 MB cache, and the first run OOM'd at h5. FIXED:
  `_attach` now trims the panel to only model columns (`["ticker","date","y"] + base_cols`), predictions are
  stored as float32, and the wide fold copies are `del`+`gc.collect()`ed each fold. Numerically inert (same
  feature columns, float32 QLIKE delta ~1e-7).
- **m3 (accepted follow-up).** `VALID_LEN`/embargo live in the driver, not `foundation_config.py` — mirrors the
  sibling shared-battery pattern exactly; moving them would diverge from the sibling. Left as-is (§3 Surgical).
- **m5/m6 (accepted follow-up, cosmetic).** Flipping `USE_SPREAD` against an existing spread-less cache would
  `KeyError`; `_placebo` degenerates to self-forecast for a single ticker (unreachable on HOSE/SP500). Low risk;
  noted.

## Tests / coverage
`.venv_gpu_encode` pytest: 23 passed. diff-cover on changed lines: **C0 = 100%, C1 = 100%** (0 partial branches)
across all three code modules. One opt-in test exercises the real Chronos model on a tiny slice; the driver tests
inject a fast fake forecaster.
