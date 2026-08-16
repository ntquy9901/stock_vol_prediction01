# Summary of update — generic raw→Parkinson pipeline (VN30/VN100/HOSE/HNX/…) + process VN100

## What changed

Made the Parkinson processing pipeline universe-agnostic and processed the cleaned VN100 raw prices.

- `src/common/process_parkinson_pipeline.py` — `main()` now takes `--raw`/`--out` (argparse) so any
  ticker universe (VN30, VN100, HOSE, HNX, UPCoM, …) is processed by pointing at its folder of
  `<TICKER>_ohlcv.csv`. Defaults preserve the prior behaviour (VN30: `data/raw/prices` →
  `data/processed`). Reuses the tested `parkinson_utils.process_single_stock` (Parkinson variance +
  date normalization + NaN/inf drop + 0.1 clip). Returns an int exit code.
  - `python -m src.common.process_parkinson_pipeline`  → VN30 (default)
  - `... --raw data/raw/prices/vn100 --out data/processed/vn100`  → VN100
  - `... --raw data/raw/prices/hnx --out data/processed/hnx`  → HNX (same for HOSE/UPCoM)
- `tests/test_process_pipeline_cli.py` — 3 tests (arbitrary-universe dir processed; tz-aware dates
  normalized to plain `YYYY-MM-DD`; missing raw dir → exit 1). All pass.

## VN100 processed

Pipeline: crawl (`crawl_vietnam_stocks.py`) → clean (`clean_ohlc.py`) → **process
(`process_parkinson_pipeline.py --raw data/raw/prices/vn100 --out data/processed/vn100`)** → quality
tests.

- `data/processed/vn100/`: **104** `*_processed.csv` produced.
- Verified: all end 2026-08-14; no NaN/negative; dates strictly monotonic. Clean VN100 (high<low
  already fixed) → Parkinson well-defined.

## Verification / commands

- `pytest tests/test_process_pipeline_cli.py` → 3 passed.
- `pytest tests/test_data_processing.py` → 1 skipped (pre-existing retirement, unaffected).
- Default VN30 dry-run into a temp dir → rc=0, 33 files (VHM now 2058 rows = trimmed).
- `... --raw data/raw/prices/vn100 --out data/processed/vn100` → 104 files, all 2026-08-14, no issues.

## Notes

- `data/processed/vn100/` is a subfolder; the active trackA training globs `*_processed.csv`
  non-recursively at the top level, so VN100 is NOT pulled into current training — it is staged for
  the cross-stock generalization experiment.
- Full VN100 ingestion pipeline is now: crawl → clean → process → quality-test, each a reusable
  script/module.

## DoD checklist

- [x] Generic script (works for any universe via `--raw/--out`).
- [x] Tests written + pass (3), existing tests unaffected.
- [x] VN100 processed + verified (104 files, 2026-08-14, clean).
- [x] Summary report (this file).
- [x] Push after task (below).
