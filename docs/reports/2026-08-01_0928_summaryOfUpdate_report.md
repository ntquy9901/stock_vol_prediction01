# Summary — 10-Day-Ahead Horizon Baseline + Panel-Fix Retrain

**What changed:** added `baselines/2026-08-01_horizon10_baseline/` (2 new train scripts, no
dataset/model changes — `forecast_horizon` was already a pass-through parameter in the shared
pipeline). Ran 2 real 10-epoch trainings at `forecast_horizon=10` (HAR-only, per-ticker gated
news) for comparison against the existing 5-day results. Also re-ran
`per_ticker_news_gate_baseline` (code unchanged) on the news panel that was fixed on 2026-07-27
(VPB/VRE now included, 32 tickers) — this closes the "not yet retrained" item flagged in that
fix's report and in `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §6 item 1.

## Files

| File | Purpose |
|---|---|
| `baselines/2026-08-01_horizon10_baseline/code/train_har_only_reference_h10.py` | copy-modify of `2026-07-25_news_usefulness_ablation/code/train_har_only_reference.py`, +1 CLI arg (`--forecast_horizon`, default 10) |
| `baselines/2026-08-01_horizon10_baseline/code/train_per_ticker_gate_h10.py` | copy-modify of `2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py`, same +1 CLI arg; resume-checkpoint feature dropped (not needed for a single 10-epoch run) |
| `baselines/2026-08-01_horizon10_baseline/test/test_target_shift.py` | proves the target actually shifts to `+10-1` (not `+5-1`), run BEFORE the 2 train scripts existed |
| `baselines/2026-08-01_horizon10_baseline/test/test_train_smoke_h10.py` | train_epoch smoke for both models at horizon=10, plus a real-data-slice window-count check |
| `baselines/2026-08-01_horizon10_baseline/requirements/requirements.md`, `design/design.md` | scope decisions (2 architectures only, 5-day stays primary), Simplicity Gate rationale |
| `baselines/2026-08-01_horizon10_baseline/code_review/code_review_2026-08-01.md` | adversarial review, 0 HIGH |

No new file for the panel-fix retrain — same script
(`baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py`), same default
`--news_panel_path`, which now points at the already-rebuilt 32-ticker parquet (rebuilt
2026-07-27, confirmed 32 tickers including VPB/VRE via direct parquet read before running).

## Tests

**9/9 pytest pass** (`baselines/2026-08-01_horizon10_baseline/test/`). `test_target_shift.py` was
run first, standalone, to confirm the core assumption (existing dataset class already supports
`forecast_horizon=10` correctly) before writing the 2 train scripts.

## Results

### A. 5-day vs 10-day (same pipeline, batch_size=32, no augmentation — apples-to-apples)

| Metric | HAR-only 5d | HAR-only 10d | Diff | Gated-news 5d (new 32-ticker panel) | Gated-news 10d | Diff |
|---|---:|---:|---:|---:|---:|---:|
| DirAcc | 68.42% | 67.80% | -0.62pp | 68.69% | 67.92% | -0.77pp |
| R² | 0.7141 | 0.7041 | -0.0100 | 0.7101 | 0.7040 | -0.0061 |
| QLIKE | 0.5623 | 0.5732 | +0.0109 | 0.5631 | 0.5767 | +0.0136 |
| RMSE | 0.002643 | 0.002689 | +0.000046 | 0.002662 | 0.002690 | +0.000028 |

All 4 metrics move in the same direction (worse) at the 10-day horizon, for both architectures.
News does not change this pattern — gated-news is close to HAR-only at both horizons (DirAcc
+0.27pp at 5-day, +0.12pp at 10-day; QLIKE/R² roughly tied or slightly worse with news at both
horizons).

Sources: `results/har_only_ablation_ref_2026-07-25_110813/results.json` (5d reference, same
pipeline), `results/har_only_h10_2026-08-01_090759/results.json`,
`results/per_ticker_gate_2026-08-01_092309/results.json` (5d, new panel),
`results/per_ticker_gate_h10_2026-08-01_091853/results.json`.

### B. Panel-fix retrain (per-ticker-gate, 5-day, 32-ticker panel vs the earlier 30-ticker run)

| Metric | Old 30-ticker panel (2026-07-26) | New 32-ticker panel (2026-08-01) | Diff |
|---|---:|---:|---:|
| DirAcc | 68.76% | 68.69% | -0.07pp |
| R² | 0.7159 | 0.7101 | -0.0058 |
| QLIKE | 0.5497 | 0.5631 | +0.0134 |
| RMSE | 0.002635 | 0.002662 | +0.000027 |

Adding VPB/VRE to the news panel did not improve the result — QLIKE and R² are both slightly
worse on the fixed panel, DirAcc is essentially unchanged. This does not indicate the fix was
wrong (VPB/VRE genuinely had zero news coverage before and their inclusion is a data-correctness
fix, not a performance lever) — it means the 2026-07-26 "current best" numbers reported for this
architecture should be updated to the 32-ticker figures above going forward, since those are the
ones computed on correct data.

## Code review

`code_review_2026-08-01.md` — 0 HIGH. Main risk checked: whether
`create_dual_news_dataloaders(..., forecast_horizon=args.forecast_horizon)` could silently fall
back to the shared default of 5 if the kwarg were dropped somewhere — ruled out by
`test_target_shift.py` proving the actual target value with real arithmetic, not just by reading
the call site. Two LOW items noted: resume-checkpoint feature removed from the h10 gate script
(deliberate simplification), and a theoretical (unobserved) window-count edge case for
short-history tickers at the longer horizon.

## Impact analysis

`baselines/2026-08-01_horizon10_baseline/` is a new, isolated folder — no sibling files were
edited. The panel-fix retrain used an existing, unmodified script; only new `results/`/`models/`
folders were created (`per_ticker_gate_2026-08-01_092309`). No code changes accompanied the
retrain.

## DoD checklist

- [x] Requirements + design (SDD lifecycle: scope confirmed via user Q&A before implementation)
- [x] Test-first: `test_target_shift.py` written and run before the 2 train scripts existed
- [x] Tests: 9/9 pass
- [x] Code review: 0 HIGH, 2 LOW documented
- [x] Real 10-epoch training ×3 (HAR-only-h10, gate-h10, gate-5d-panel-fix)
- [x] Impact analysis
- [x] Summary report (this file) + will update `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §5/§6
- [ ] Diff-coverage: Not run (known tooling gap)

## Follow-ups

- `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §5 (results table) and §6 item 1 (panel-fix status)
  need updating with the numbers in this report — done as part of this change.
- The 5-day HAR-only "69.98% DirAcc" figure quoted elsewhere in the project uses a different
  pipeline (batch_size=11, augmentation on) than the one used for this horizon comparison
  (batch_size=32, no augmentation, matching `2026-07-25_news_usefulness_ablation`'s convention).
  Both are legitimate but not directly comparable to each other — this report only compares
  numbers produced by the same pipeline.
