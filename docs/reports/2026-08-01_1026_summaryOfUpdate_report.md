# Summary — 22-Day-Ahead Horizon Baseline

**What changed:** added `baselines/2026-08-01_horizon22_baseline/` (2 new train scripts, no
dataset/model changes — same `forecast_horizon` pass-through mechanism used for the 10-day
baseline). Ran 2 real 10-epoch trainings at `forecast_horizon=22` (HAR-only, per-ticker gated
news) for comparison against the existing 5-day and 10-day results.

## Files

| File | Purpose |
|---|---|
| `baselines/2026-08-01_horizon22_baseline/code/train_har_only_reference_h22.py` | copy-modify of `2026-08-01_horizon10_baseline/code/train_har_only_reference_h10.py`, default `--forecast_horizon` 22 |
| `baselines/2026-08-01_horizon22_baseline/code/train_per_ticker_gate_h22.py` | copy-modify of the h10 sibling (incl. its resume-checkpoint support), default `--forecast_horizon` 22 |
| `baselines/2026-08-01_horizon22_baseline/test/test_target_shift_h22.py` | proves target = `+43` (not `+26` or `+31`), run before the 2 train scripts existed |
| `baselines/2026-08-01_horizon22_baseline/test/test_train_smoke_h22.py` | train_epoch smoke for both models + **real-data, full-32-ticker window-count check** (new risk at this horizon: 44-day minimum per split) |
| `baselines/2026-08-01_horizon22_baseline/requirements/requirements.md`, `design/design.md` | scope (same 2 architectures, 5/10-day untouched), the new window-count risk and how it's tested |
| `baselines/2026-08-01_horizon22_baseline/code_review/code_review_2026-08-01.md` | adversarial review, 0 HIGH |

## Tests

**7/7 pytest pass.** The key new test (`test_every_common_stock_has_positive_windows_at_horizon_22`)
loads the actual `data/processed/*.csv` for all 32 tickers, replicates
`_load_raw_stock_data`/`_split_raw_data_by_date`/`_generate_har_for_split`, and asserts every
split has a positive window count at the 44-day minimum — measured: train 891→847 windows,
val/test 191→147 windows. No ticker was at risk of 0 windows.

## Results (test set, 10 epochs, same pipeline as the 5-day/10-day comparisons)

| Kiến trúc | DirAcc | R² | QLIKE | RMSE |
|---|---:|---:|---:|---:|
| HAR-only | 66.38% | 0.7051 | 0.5938 | 0.002750 |
| Gated-news | 67.17% | 0.7032 | 0.5943 | 0.002759 |

Combined with the existing 5-day and 10-day numbers, DirAcc and QLIKE degrade monotonically as
horizon increases (HAR-only DirAcc: 68.42% → 67.80% → 66.38%; QLIKE: 0.5623 → 0.5732 → 0.5938).
R² does not follow the same monotonic pattern. Gated-news beats HAR-only on DirAcc at all 3
horizons in the first 10-epoch run (+0.27pp, +0.12pp, +0.79pp respectively), but the gap is small
and this is single-seed.

**Convergence:** val loss for both 22-day models oscillates without a clear trend across all 10
epochs (e.g. HAR-only: 1.1495 → 1.1409 → 1.1436, non-monotonic) — already plateaued, same pattern
as the 10-day horizon. This is unlike the 5-day gated-news variant, which needed ~20 epochs to
reach its peak (see `docs/reports/2026-08-01_0928_summaryOfUpdate_report.md` and today's later
epoch-20/30 extension). No further training is planned for the 22-day variant.

## Code review

`code_review_2026-08-01.md` — 0 HIGH. Diff from the h10 sibling is minimal (default arg + a few
strings), so risk is low; the one genuinely new risk (window-count at a longer horizon) was
verified with real data before training, not just reasoned about.

## Impact analysis

New, isolated baseline folder — no sibling files edited. Uses existing `create_dual_news_dataloaders`
and `PerTickerGatedNewsBaseline`/`ParallelLSTMGNN` unchanged.

## DoD checklist

- [x] Requirements + design
- [x] Test-first: target-shift test written and run before the 2 train scripts existed
- [x] Tests: 7/7 pass, including the real-data window-count check
- [x] Code review: 0 HIGH
- [x] Real 10-epoch training ×2
- [x] Impact analysis
- [x] Summary report (this file) + updated `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §1.2 Bảng B, §8.5
- [ ] Diff-coverage: Not run (known tooling gap)
