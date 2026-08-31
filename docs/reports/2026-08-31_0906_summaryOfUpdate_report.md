# Summary of update — single canonical pipeline config (centralize tunable constants)

Commits: `fef4899` (centralization + enforcement) + `58bb2d6` (test coverage fix). Branch: master.
Pre-push gate: PASS (no QG_SKIP). Pushed `6ab2324..58bb2d6`.

## What changed
Created ONE canonical configuration module holding every tunable pipeline constant and refactored the
pipeline modules to READ from it instead of hardcoding scattered magic-numbers. Root cause addressed:
`volume_zscore` used a hardcoded window=20 in `masked_rich` while the project monthly convention is 22
(`har_monthly`) and `screen_features` already used 22 — a silent drift.

### Canonical config (single source of truth)
`submission/soict_lstm_gat/pipeline_config.py` — grouped module-level constants (already on every
consumer's `sys.path`). `Config` (`config.py`) and `WFConfig` (`run_walkforward.py`) became thin views
whose field defaults source from it; the scattered module constants / build-signature defaults / body
floors import from it. Backward compatible (wide `from config import Config` unchanged).

### Files (path → change)
- `submission/soict_lstm_gat/pipeline_config.py` — NEW canonical config (all constants).
- `submission/soict_lstm_gat/config.py` — `Config` defaults → `pc.*`.
- `submission/soict_lstm_gat/data_utils.py` — HAR windows / drop thresholds / scaler eps → `pc.*`.
- `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py` — constants (incl. volume window) + build defaults + scaler eps → `pc.*`.
- `.../code/masked_snapshots.py` — FIRST_VALID + build defaults + scaler eps → `pc.*`.
- `.../code/run_masked_rich.py` — positivity floors (1e-2 / 1e-12) → `pc.*`.
- `.../code/experts.py` — build defaults (min_common / folds / eps) + pred-floor 1e-3 + scaler eps → `pc.*`.
- `.../code/run_experiment.py` — `run()` `min_common` default → `pc.MIN_COMMON_DATES`.
- `.../code/screen_features.py` — windows / horizons / FIRST_VALID / top_k → `pc.*`; removed pre-existing dead code (unused `import baselines`, unused `deg_shuf`) forced by the fail-closed ruff-F gate on the touched file.
- `baselines/2026-08-30_walkforward_harx_lstm/code/run_walkforward.py` — WFConfig defaults → `pc.*`.
- Tests: `submission/.../tests/test_pipeline_config_freeze.py`, `.../code/tests/test_config_centralization.py` (+conftest), `.../walkforward/.../code/tests/test_wf_config_freeze.py` (+conftest).
- Enforcement: `scripts/quality_gate/check_config_hardcode.py` (+test), wired as pre-push step 6/6.
- Docs: `docs/config_centralization/{requirements,design}.md`, code-review config-hardcode lens in `docs/QUICK_REFERENCE_CHECKLIST.md` (+ local `.claude` reviewer template).

## The one intentional value change
`VOLUME_ZSCORE_WINDOW = 22` (was 20 in `masked_rich`). Now the committed canonical value, matching the
monthly convention (`HAR_MONTHLY_WINDOW=22`, `screen_features` volume shock). It remains a config knob.
The delivered/paper result JSONs (`results/masked_rich_floor1e2/...`) were computed with 20; reproducing
them requires temporarily setting the knob back to 20. The 2-day widening changes the feature slightly,
so delivered JSONs are not reproduced by the new default. Documented in the config entry, `masked_rich`
docstring, and `docs/config_centralization/`.

## Value-freeze + regression
- Value-freeze tests pin EVERY centralized constant to its delivered value (volume window pinned to 22);
  a `test_no_undocumented_extra_constants` also fails if a new public constant is added without freezing.
- No existing test pins a 20-derived number (all `masked_rich` tests are structural: shapes / finiteness
  / invariance / positivity), so the 20→22 change breaks zero existing tests.
- Regression (all under the GPU venv, 0 new failures): changed-scope 119 (C0 line **100%**, C1 branch
  **99%** on 427 changed lines); har_anchored delivered `test/` 58; walkforward `test/` 27; submission
  `tests/` 77; delivered gate baselines (2026-08-15_volatility + 2026-08-11_eda_gnn) 69; data-quality +
  lessons + overfit 329; ablation sanity (lstm_gnn_serial_hybrid) 19.

## Enforcement gate (config-hardcode)
`scripts/quality_gate/check_config_hardcode.py` scans only the ADDED lines of the push diff for changed
pipeline files (excludes the config module itself, tests, `archive/`, vendored). Decision:
- **BLOCK** — clear tunable: `.rolling(<int>)`, `top_k=<int>`, or `NAME=<numeric literal>` where NAME is
  a window/threshold/floor/hyperparameter name.
- **WARN** — bare `1e-N` float literal on a changed pipeline line.
- Exceptions: 0/1, `pc.`/`config`/`cfg.` references, `# noqa`/`# config-ok`, excluded paths.
On this push: 0 BLOCK, 0 WARN (all refactor lines reference `pc.`). Scanner test: 16 cases, 100% line +
100% branch coverage. `main()`/git glue is `# pragma: no cover`; the scan/parse logic is fully tested.

## Duplicate-constant audit (unify vs keep separate)
- Unified at one entry (same concept + value): `top_k=5` across `masked_rich` EDGE_TOP_K,
  `masked_snapshots` default, `Config.top_k`, `experts` `cfg.top_k`, `screen_features` `_vshock_adjacency`
  → `EDGE_TOP_K`; the volume-shock window (`masked_rich` 20 and `screen_features` 22) → the single
  `VOLUME_ZSCORE_WINDOW` (now 22).
- Kept SEPARATE (distinct concept despite equal value): `QLIKE_FLOOR` (1e-8) vs `SCALER_EPS` (1e-8) vs
  `RESIDUAL_EPS` (1e-8); `VOL_OF_VOL_WINDOW` (22, rolling std of pk) vs `VOLUME_ZSCORE_WINDOW` (22).
- NOT unified (out of listed scope / local guard, documented as residual): `screen_features._EPS=1e-12`
  (module-local numerical guard, distinct from the positivity floor `POS_FLOOR_EPS`); `run_walkforward`
  `nfloor = 1e-2*mean + 1e-12` (walk-forward floors were scoped to WFConfig fields only) — pre-existing,
  grandfathered, follow-up.

## Code review
Applied the config-hardcode lens (self-review) + the 3-lens standard (Blind Hunter / Edge Case /
Acceptance). Findings addressed: gate-forced removal of pre-existing dead code in `screen_features`
(unused import + unused var); coverage gap on conditional `sys.path` inserts fixed by making them
unconditional. No critical/major open.

## DoD
- [x] Canonical config created; consumers import it (backward compatible).
- [x] All values byte-identical except the documented volume window 20→22.
- [x] Value-freeze test green; regression 0 new failures.
- [x] Enforcement gate + tests green; wired into pre-push (step 6/6).
- [x] ruff-F clean on changed files; C0 line 100% / C1 branch 99% on changed lines.
- [x] Committed + pushed to origin/master through the gate (no QG_SKIP).
