# Summary of update — VolGA multi-horizon walk-forward (code build, B1)

Date: 2026-08-31. Scope: build the CODE (not the full runs) for the VolGA multi-horizon walk-forward
experiment on the clean enriched VN100 data. New baseline `baselines/2026-08-31_walkforward_volga/`.

## What changed
New, hard-isolated baseline (SDD §3.F: requirements / design / code / code_review / test). Reuses the
2026-08-30 walk-forward machinery + the 2026-08-21 masked-rich trainer read-only; only the enriched
reader and the 3-model fold assembly are new.

| Path | Purpose |
|---|---|
| `requirements/requirements.md` | Objective, inputs, params, leakage safety, success criteria, go/no-go |
| `design/design.md` | Data flow, reuse map, 3 SDD design gates, key decisions |
| `code/wf_enriched_panel.py` | Enriched reader: `build_enriched_panel` (5 features read DIRECTLY from `data/processed_enriched/vn100`), `frozen_universe` (fixed-split node screen → 102 nodes), `pack_fold` (per-fold TRAIN-only vol→PK graph + scalers) |
| `code/run_volga_walkforward.py` | `run_fold` (HAR-X / LSTM / VolGA), `run_walkforward` (pool + evidence + DM + JSON), `default_out_path` (horizon-encoded, fixes delivered `_h1` bug), CLI (`--horizon/--lookback/--epochs/--batch/--folds-target/--smoke/--out/--no-gpu-wait`) |
| `code/tests/*` | 6 test modules (17 tests): enriched reader, no-look-ahead graph/scalers, 3-model run, horizon plumbing, real-data smoke, overfit-evidence schema |
| `code_review/code_review_2026-08-31.md` | 3-lens adversarial review + perf lens |

## Models
`HAR-X` (5-feat OLS, refit per fold) · `LSTM` (no-graph) · `VolGA` = `LSTM_wGAT_vol2pk` (LSTM + 2-hop
weighted-GAT over a per-fold TRAIN-only vol→PK Top-5 graph). Leave-one-out: `VolGA − LSTM` isolates the
graph's marginal value; both vs HAR-X. Deep-model keys named `LSTM` / `LSTM_wGAT_vol2pk` so the pre-push
overfit-evidence gate recognises them.

## Leakage safety
Per fold the vol→PK adjacency AND every per-node feature/target scaler are estimated on the train window
only (`last_tr_row = tr_anchor[-1] + horizon`), frozen for val/test. `assert_no_leakage` (reused) +
`test_no_lookahead` (perturb every post-train row → train artifacts bit-identical) both pass.

## Enriched reader
5 node features read directly: `[parkinson_variance, har_weekly, har_monthly, market_pk,
volume_zscore_{VOLUME_ZSCORE_WINDOW=22}]`; target = `parkinson_variance` at t+h (formed at train time).
Leading har_monthly/volume NaN excluded by anchor start + `win_ok`; interior volume_zscore NaN imputed
to neutral 0 on own dates only. Fail-loud if a whole feature is all-NaN for a valid ticker (no silent
all-zero degradation). Verified `market_pk` is the shared cross-sectional factor (identical across
tickers per date). Universe screen reproduces the delivered 102-node VN100 set.

## Tests + coverage
- `python -m pytest baselines/2026-08-31_walkforward_volga/code/tests -q` → **17 passed** (GPU venv).
- Coverage on changed SOURCE (both modules): **C0 line = 100%, C1 branch = 100%** (`--cov-branch`).
  Residual partials are two env-guard lines in test/conftest (`pytest.skip` when real data absent; stale
  `baselines` namespace drop) — overall changed-line branch coverage well above the 95% gate.
- `ruff check --select F` and full `ruff check` on the baseline: clean.

## Smoke (real enriched data, NOT the full sweep)
`python run_volga_walkforward.py --smoke --no-gpu-wait` (12-ticker slice, 1 fold, 2 epochs, 1 seed, h1):
- nodes=12, folds=1, oos_dates=373, obs=4476, 3.9 s.
- pooled QLIKE: HAR-X=0.5055, LSTM=0.5973, VolGA=0.4987, HAR=0.4995; DM VolGA_vs_LSTM p=0.065 (favours
  VolGA). (Smoke numbers are indicative only — 2 epochs / 1 seed.)

## Performance conclusion (CLAUDE.md ENFORCED)
Reuses the delivered **already-batched** `train_masked_rich` (`[B,N,seq,5]` on GPU, mask-aware loss,
per-node train scalers, LR-sched, early stop, grad-clip; tensors kept on-device). No batch=1 loop
introduced; correctness/no-leakage preserved. Measured one full-size fold (102 nodes, 297,240 masked
train obs, 16 epochs, 1 seed, LSTM+VolGA) = 61.4 s on RTX 4060 → full run ≈ 22 folds × 5 seeds × ~60 s
≈ **~1.9 h per horizon** (~7.5 h for {1,5,10,22}). Feasible for the B2 pass.

## Over/under-fit evidence
`run_walkforward` emits top-level `train_metrics` / `val_metrics` / `metrics` + per-model
`fit_diagnostics` + `learning_curves` for `LSTM` and `LSTM_wGAT_vol2pk`, plus `per_fold` + `fit_summary`.
`test_overfit_evidence_schema` asserts `check_overfit_evidence` recognises the result and never fails
for a missing-block (schema) reason.

## Code review
3-lens adversarial review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor + performance lens):
no CRITICAL/MAJOR findings. One accepted simplification (B3: pooled-train double-counts overlapping
expanding windows — used only for the fit verdict, never the headline metric/DM). See
`code_review/code_review_2026-08-31.md`.

## Data-quality gate
`data/processed_enriched` is READ ONLY (no data written/modified). Pandera/Evidently: the pre-push gate
runs them because a changed filename contains "panel"; they validate the existing processed data
(unchanged by this pass). No raw ingestion; no reprocessing.

## Risks / follow-ups
- B2 (separate pass): launch the full 22-fold × 5-seed × {1,5,10,22} sweep (`--horizon H --lookback 22`),
  one horizon per `result.json`, then assemble the multi-horizon VolGA-vs-HAR-X/LSTM comparison.
- Fail-loud coverage guard aborts the whole `frozen_universe` build if one enriched ticker is
  structurally broken — intended, but B2 should ensure the enriched universe passes P1–P6 first.

## DoD checklist
- [x] Code satisfies request; hard-isolated; no unrelated refactor.
- [x] Tests written + pass (17); C0=100%/C1=100% on changed source.
- [x] Lint (ruff F + full) clean.
- [x] Adversarial code review done + documented; no critical/major open.
- [x] Performance conclusion recorded (batched trainer reused; runtime measured).
- [x] Real-data smoke passes; full sweep intentionally NOT launched (B2).
- [x] Summary report (this file).
