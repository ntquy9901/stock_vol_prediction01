# Phase 1 Tasks

## Task 1: Write tests for data adapter (Test-First)
- **File:** `tests/test_sp500/test_adapter.py`
- **Verify:** `adapt_to_vn30_format()` converts HF dataset columns to VN30 format
- **Test:** Dummy DataFrame with HF columns → assert VN30 columns present

## Task 2: Implement data adapter
- **File:** `src/common/data_adapters.py`
- **Verify:** `adapt_to_vn30_format(df, source="stocks_ohlcv")` works
- **Test pass:** Task 1 tests pass

## Task 3: Write tests for download script
- **File:** `tests/test_sp500/test_download.py`
- **Verify:** Download script creates CSV files with correct format
- **Test:** Mock HF dataset → assert CSV files created with correct columns

## Task 4: Implement download script
- **File:** `src/experiments/sp500/download_sp500.py`
- **Verify:** Downloads S&P 500 data, saves per-stock CSVs
- **Test pass:** Task 3 tests pass

## Task 5: Add --market arg to process_parkinson_pipeline.py
- **File:** `src/common/process_parkinson_pipeline.py`
- **Verify:** `python -m src.common.process_parkinson_pipeline --market sp500` works
- **Verify:** Default (no arg) still uses VN30 paths (backward compatible)

## Task 6: Smoke test — full pipeline on 3 stocks
- **Verify:** Download 3 stocks → process → HAR features → valid output
- **Test:** Run pipeline, assert output files exist with correct columns
