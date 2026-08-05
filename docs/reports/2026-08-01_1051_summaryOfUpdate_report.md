# Summary — 1-Day-Ahead Horizon Baseline (completes 1/5/10/22-day set)

**What changed:** added `baselines/2026-08-01_horizon1_baseline/` (2 new train scripts, no
dataset/model changes — same `forecast_horizon` pass-through mechanism used for the 10-day and
22-day baselines). Ran 2 real 10-epoch trainings at `forecast_horizon=1`. This completes the full
"1, 10, 22-day Secondary Targets" set listed in CLAUDE.md, alongside the existing 5-day primary
target.

## Files

| File | Purpose |
|---|---|
| `baselines/2026-08-01_horizon1_baseline/code/train_har_only_reference_h1.py` | copy-modify of the horizon22 sibling, default `--forecast_horizon` 1 |
| `baselines/2026-08-01_horizon1_baseline/code/train_per_ticker_gate_h1.py` | copy-modify of the horizon22 sibling (incl. resume support), default `--forecast_horizon` 1 |
| `baselines/2026-08-01_horizon1_baseline/test/test_target_shift_h1.py` | proves target = `+22` (not `+26`/`+31`/`+43` of the other 3 horizons) |
| `baselines/2026-08-01_horizon1_baseline/test/test_train_smoke_h1.py` | train_epoch smoke + real-data, full-universe window-count check (lowest risk of all 4 horizons: 23-day minimum per split) |
| `baselines/2026-08-01_horizon1_baseline/requirements/requirements.md`, `design/design.md` | scope, mechanism (identical to horizon10/22), lower risk profile |
| `baselines/2026-08-01_horizon1_baseline/code_review/code_review_2026-08-01.md` | adversarial review, 0 HIGH — verified the copy-modify (via `cp`+`sed`) left no stray horizon-22 references in the resulting logic |

## Tests

**7/7 pytest pass.** Real-data window-count check: train 891→868, val/test 191→168 windows per
split — the largest margin of all 4 horizons tried today (vs 847/168/147/147 style reductions at
5/10/22 days).

## Results (test set, 10 epochs, same pipeline as the other horizon comparisons)

| Kiến trúc | DirAcc | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|
| HAR-only | 72.35% | 0.7581 | 0.5099 | 0.002428 |
| Gated-news | 72.39% | 0.7595 | 0.4834 | 0.002420 |

Confirms the pre-registered hypothesis: 1-day-ahead is markedly easier to predict than the other
3 horizons (DirAcc ~4pp higher than 5-day, QLIKE substantially lower). Gated-news beats HAR-only
on all 4 metrics here (not just DirAcc as at the other horizons) — the QLIKE gap (0.4834 vs
0.5099) is the largest architecture-vs-architecture difference observed at any horizon in a first
10-epoch run today.

**Convergence:** val loss drops sharply in the first 2 epochs then oscillates mildly through
epoch 10 for both models — fast convergence, matching the 10-day/22-day pattern, unlike the 5-day
gated-news variant (needed ~20 epochs). No further training planned.

## Code review

`code_review_2026-08-01.md` — 0 HIGH. Files were created via `cp` + `sed` from the horizon22
sibling then hand-verified with `grep` for stray "22"/"month" references; a few docstring lines
that `sed` couldn't match (multi-line) were fixed manually and re-verified.

## Impact analysis

New, isolated baseline folder — no sibling files edited.

## DoD checklist

- [x] Requirements + design
- [x] Test-first: target-shift test written and run before the 2 train scripts existed
- [x] Tests: 7/7 pass, including the real-data window-count check
- [x] Code review: 0 HIGH
- [x] Real 10-epoch training ×2
- [x] Impact analysis
- [x] Summary report (this file) + updated `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` (Bảng B → 4 horizons, new §8.6)
- [ ] Diff-coverage: Not run (known tooling gap)

## Note on this update

Per user request mid-session, all specific ticker-count mentions ("30 mã" / "32 mã") were removed
from `BAO_CAO_TONG_HOP.md` and replaced with neutral phrasing ("panel cũ" / "panel đã fix" / "các
mã" / "N mã" in tensor-shape notation) — the user flagged the exact VN30 universe size as a
separate matter they will verify independently later.
