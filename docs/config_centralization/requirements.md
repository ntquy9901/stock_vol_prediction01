# Requirements — Single canonical pipeline config (centralize tunable constants)

## Objective
Create ONE canonical configuration module that is the single source of truth for every tunable
constant (hyperparameters, windows, thresholds/floors, edge params, horizons) in the
volatility-forecasting pipeline, and refactor the pipeline modules to READ their parameters from it
instead of hardcoding scattered magic-numbers. Root cause being fixed: `volume_zscore` used a
hardcoded window=20 in `masked_rich.py` while the project's monthly convention is 22 (har_monthly=22)
and `screen_features.py` already used 22 — a silent drift that editing one place did not fix.

## In scope (survey result — files that hold the constants)
- `submission/soict_lstm_gat/config.py` — `Config` training-hyperparameter dataclass.
- `submission/soict_lstm_gat/data_utils.py` — HAR windows (5/22), FIRST_VALID, drop thresholds, scaler eps.
- `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py` — volume window, N_FEAT, edge params, coverage caps, build defaults.
- `.../code/masked_snapshots.py` — FIRST_VALID + build defaults (edge_min_overlap, top_k, min_valid, min_train_rows).
- `.../code/run_masked_rich.py` — positivity floors 1e-2 / 1e-12.
- `.../code/experts.py` — pred-floor 1e-3, scaler eps, residual eps, min_common, cross-fit folds.
- `.../code/run_experiment.py` — min_common default.
- `.../code/screen_features.py` — VOL_WIN (22), VOV_WIN (22), HORIZONS, FIRST_VALID, top_k.
- `baselines/2026-08-30_walkforward_harx_lstm/code/run_walkforward.py` — WFConfig (lookback/K/val/test_frac/horizon).

## Canonical config location
`submission/soict_lstm_gat/pipeline_config.py` — already on every consumer's `sys.path` (each module
inserts `submission/soict_lstm_gat`), so consumers import it with no new path plumbing. `config.py`
already lives here and is the de-facto shared config; the new module is its generalization.

## Acceptance criteria
1. **One source of truth:** every listed tunable constant is defined once in `pipeline_config.py`;
   consumers import it. `Config`/`WFConfig` dataclasses become thin views whose field defaults source
   from `pipeline_config` (backward compatible — wide `from config import Config` keeps working).
2. **Values preserved byte-for-byte EXCEPT one intentional change:** `volume_zscore` window
   `VOLUME_ZSCORE_WINDOW = 22` (was 20 in `masked_rich`). All other values identical to current.
3. **Value-freeze test** pins every centralized constant to its expected value (volume window pinned
   to 22) and asserts each consumer re-exports the same object/value it imports.
4. **Regression:** the full existing test suite for every touched file + delivered baselines passes
   with 0 NEW failures (the vol-window change breaks no existing test — all are structural).
5. **Enforcement:** a pre-push gate step + a `/code-review` "config-hardcode" lens flag new scattered
   hardcoded tunable constants in changed pipeline files (WARN default, BLOCK on clear tunables).

## [RESOLVED clarifications]
- volume_window: set to **22 permanently** as the committed canonical value (still a knob). Delivered
  paper result JSONs were computed with 20 (historical) — reproducing them requires temporarily
  setting the knob to 20. Documented in the config entry + this doc + the summary report.
- `HORIZONS = (1, 5, 10, 22)` canonical (matches `screen_features.HORIZONS`); CLI still overrides.
  `Config.horizons = (1, 5)` is a submission-local vestigial default and is left unchanged (frozen).
- top_k literals (masked_rich EDGE_TOP_K, masked_snapshots default, config.top_k, screen_features
  `_vshock_adjacency`, experts `cfg.top_k`) are the SAME concept (Top-K graph neighbours = 5) → one
  entry `EDGE_TOP_K`. The two 1e-8 roles (qlike_floor vs scaler std eps) are DISTINCT → two entries
  (`QLIKE_FLOOR`, `SCALER_EPS`). `VOL_OF_VOL_WINDOW` (rolling std of pk) is distinct from the volume
  window even though both equal 22 → separate entry (not unified).

## Go / no-go
- GO when: value-freeze test green, full regression 0 new failures, enforcement gate + its tests green,
  full pre-push gate green without QG_SKIP, pushed to origin/master.
- NO-GO if any delivered value other than the volume window changes, or any existing test regresses
  for a reason other than the documented vol-window recomputation (none expected).
