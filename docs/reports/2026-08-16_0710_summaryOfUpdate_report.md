# Summary of update — reusable OHLC cleaner + clean VN100 raw prices

## What changed

Added a reusable positive-aware OHLC sanitizer and applied it to the crawled VN100 raw prices.

- `src/data/clean_ohlc.py` — for each row: `high = max(positive OHLC)`, `low = min(positive OHLC)`,
  nonpositive `open`/`close` clamped into `[low,high]`. Fixes `high<low`, nonpositive prices, and
  `open`/`close` outside `[low,high]` in one idempotent pass; rows with all-nonpositive OHLC are left
  untouched (never silently zeroed). Nonpositive values are excluded from the max/min so a spurious
  `0` never propagates into `low`. Same rule used earlier to fix the main 33-ticker series.
- `tests/test_clean_ohlc.py` — 7 tests (high<low, nonpositive-low exclusion, open/close-outside,
  valid-row no-op, idempotency, all-nonpositive left as-is, dir-level). All pass.

## Applied to VN100

`python -m src.data.clean_ohlc data/raw/prices/vn100` — 93/104 files changed, 1674 cells corrected.

Before → after (Parkinson-affecting defects):
- `high < low`: **54 → 0**
- nonpositive OHLC: 0 → 0
- `open`/`close` outside `[low,high]`: 3674 → 32 residual, all at machine-epsilon (max relative gap
  2.35e-16, i.e. `high == open/close` to float precision — not real violations; the vn100 quality
  test's rtol tolerance treats them as clean).

`tests/test_vn100_prices_quality.py`: **626 passed** (previously 533 passed / 93 xfailed — the
high<low xfail rows are now clean, so they pass normally).

## Verification / commands

- `pytest tests/test_clean_ohlc.py` → 7 passed.
- `python -m src.data.clean_ohlc data/raw/prices/vn100` → 1674 cells corrected.
- `pytest tests/test_vn100_prices_quality.py` → 626 passed.
- Re-scan: high<low=0, nonpositive=0, open/close-out=32 @ ~1e-16.

## Notes

- VN100 (`data/raw/prices/vn100/`) is a raw subfolder NOT read by active training (non-recursive
  `*_processed.csv` glob), so this does not affect current volatility results; it prepares VN100 for the
  cross-stock generalization experiment.
- The gross Yahoo glitch cases (e.g. ACB/TPB 2025-06-04..05 where adjusted high collapsed near zero)
  are corrected by the positive-aware max/min (the collapsed high is superseded by the max of the
  other positive OHLC values).
- Original VN100 files were committed by the crawl step, so git history is the backup.

## DoD checklist

- [x] Reusable script written (`src/data/clean_ohlc.py`).
- [x] Tests written + pass (7).
- [x] Applied + verified (high<low 54→0; vn100 test 626 passed).
- [x] Summary report (this file).
- [x] Push after task (below).
