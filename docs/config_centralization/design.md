# Design — Single canonical pipeline config

## Architecture (Anti-Abstraction / Simplicity gate)
ONE plain Python module of grouped, frozen module-level constants — NOT a config framework, loader,
YAML, or registry. Consumers `import pipeline_config as pc` and reference `pc.NAME`. The two existing
dataclasses (`Config`, `WFConfig`) keep their identity but source their field DEFAULTS from `pc`
(`lookback: int = pc.LOOKBACK`), so all existing `from config import Config` / `Config()` /
`replace(cfg, ...)` call sites keep working unchanged (backward compatible view over the SoT).

Rationale for a module of constants (not a dataclass-only design): the scattered constants live at
MODULE level (`_VOL_WIN`, `EDGE_TOP_K`, `FIRST_VALID`) and as FUNCTION-SIGNATURE DEFAULTS
(`build_masked(..., top_k=5)`); a flat module of names maps 1:1 onto both without wrapping.

## Coverage-aware refactor rule (keeps the pre-push C0=100/C1>=95 gate green)
The pre-push gate covers CHANGED lines with the changed file's ADJACENT tests only. To keep coverage
trivial and the change surgical, refactor edits are confined to lines evaluated at IMPORT time
wherever possible:
- module-level constant assignments (`_VOL_WIN = pc.VOLUME_ZSCORE_WINDOW`) — covered by importing the module;
- function-signature default expressions (`def build_masked(..., top_k=pc.EDGE_TOP_K)`) — evaluated at
  def-time (import), covered by importing the module.
Body-internal literals (positivity floors in `run_masked_rich`/`experts` function bodies) are covered
by NEW integration tests that execute those code paths on a tiny synthetic panel under SMOKE config.

## Consumer wiring
| module | edits | coverage |
|---|---|---|
| `pipeline_config.py` (NEW) | all constants | value-freeze test imports + asserts every name |
| `config.py` | `Config` field defaults → `pc.*` | freeze test imports `Config` |
| `data_utils.py` | module constants → `pc.*` | freeze test imports |
| `masked_rich.py` | module constants (**VOLUME_ZSCORE_WINDOW=22**) + build defaults + body `+SCALER_EPS` | `code/test_masked_rich.py` calls `build_masked_rich` |
| `masked_snapshots.py` | `FIRST_VALID` + build defaults + body `+SCALER_EPS` | `code/tests` integration calls `build_masked` |
| `run_masked_rich.py` | body floors `POS_FLOOR_FRAC/POS_FLOOR_EPS` | `code/tests` integration runs `run`/`train_masked_rich` |
| `experts.py` | sig defaults (min_common/folds/eps) + body floors (`PRED_FLOOR_FRAC`, `SCALER_EPS`) | `code/tests` integration runs `build_data`/`train_neural` |
| `run_experiment.py` | `run()` `min_common` default → `pc.MIN_COMMON_DATES` | import-time default |
| `screen_features.py` | module constants + `_vshock_adjacency` default → `pc.*` (no body edits) | freeze test imports |
| `run_walkforward.py` | `WFConfig` field defaults → `pc.*` | `code/tests` import |

Not unified (documented residuals, out of listed scope / distinct concept):
- `screen_features.py` `_EPS = 1e-12` (module-local numerical guard in a model-free EDA screen) — a
  distinct role from the prediction positivity floor `POS_FLOOR_EPS`; left local.
- `run_walkforward.run_fold` `nfloor = 1e-2*mean + 1e-12` — walk-forward floors were not in the
  centralization scope (only WFConfig fields); pre-existing, grandfathered, flagged as follow-up.
- CLI `argparse default=` mirrors (e.g. `--min-common default=300`) — CLI layer; left as-is
  (untouched lines, not re-introduced hardcodes).

## Grouping in pipeline_config.py
TRAINING (lr/dropout/epochs/min_epochs/batch/seeds/patience/hidden/heads/grad_clip/weight_decay/lookback),
SEEDS + HORIZONS, DATA WINDOWS (HAR 5/22, FIRST_VALID, volume window, vol-of-vol),
SPLIT/DROP THRESHOLDS (train/val frac, min_rows/anchors/train/val/test, min_valid_nodes, min_train_rows,
min_common), GRAPH/EDGE (top_k, min_overlap, min_pairs_directed, coverage caps, n_node_features),
FLOORS/EPS (qlike_floor, pred_floor_frac, pos_floor_frac, pos_floor_eps, scaler_eps, residual_eps,
crossfit_folds), WALK-FORWARD (K/val_tail/test_frac/horizon).

## Enforcement gate (new)
`scripts/quality_gate/check_config_hardcode.py`:
- Pure `scan_text(path, added_lines)` returns findings; heuristic, low false-positive.
- **BLOCK** patterns (clear tunable pipeline constants): `.rolling(<int>)`, `top_k = <literal>`,
  and `NAME = <numeric literal>` where NAME matches a window/threshold/floor/hyperparameter keyword
  (WIN/WINDOW/TOP_K/FLOOR/THRESH/PATIENCE/EPOCHS/DROPOUT/HIDDEN/HEADS/LR/WEIGHT_DECAY/GRAD_CLIP/
  LOOKBACK/SEQ/HORIZON/MIN_OVERLAP/MIN_PAIRS/MIN_TRAIN/MIN_VALID/BATCH/SEED).
- **WARN**: other bare `1e-N` float literals in changed pipeline lines.
- **Exceptions** (not flagged): literals 0/1, lines referencing `pc.`/`config`, `# noqa` or
  `# config-ok` justified lines, and excluded paths (the canonical config module itself, `test`/
  `tests`, `archive/`, `.agents`, `.claude`, `_bmad`, vendored, `data/`).
- Scans only ADDED lines of the push diff (git), so pre-existing literals are grandfathered and the
  refactor's own `= pc.X` lines are clean.
- `main()` (git-diff driver) is `# pragma: no cover`; the pure scanner is fully unit-tested.
- Wired as pre-push step "6/6"; WARN logs to the gate evidence, BLOCK sets FAIL.

## Three-gate check (SDD)
- Simplicity Gate: one flat module + two thin dataclass views; no new framework. PASS.
- Anti-Abstraction Gate: direct `import pipeline_config as pc`; no wrapper layer. PASS.
- Performance/Batching Gate: refactor changes NO compute path (identical values, identical tensors,
  same batching); the one feature change (window 20→22) keeps the same causal-rolling vectorized op.
  PASS (no perf regression; no batch=1 introduced).
