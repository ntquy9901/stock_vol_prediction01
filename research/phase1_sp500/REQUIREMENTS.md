# Phase 1 Spec: S&P 500 Data Download + Adapt

**Date:** 2026-08-01  
**Branch:** `global-benchmark`  
**Status:** Spec Draft - Needs Clarification

---

## 1. Goal

Download S&P 500 OHLCV data from Hugging Face (`siddharthmb/stocks-ohlcv`), convert to VN30-compatible format, and run existing processing pipeline — **zero impact on VN30 master branch**.

---

## 2. Input/Output

### Input
- Hugging Face dataset: `siddharthmb/stocks-ohlcv` (26.7M rows, 1.21GB)
- Fields: `date`, `act_symbol`, `open`, `high`, `low`, `close`, `volume`

### Output
- `data/raw/prices_sp500/{TICKER}.csv` — per-stock CSV (same format as VN30)
- `data/processed_sp500/{TICKER}.csv` — processed with Parkinson vol + HAR features
- `results/sp500_baseline_{timestamp}/` — training results

---

## 3. Acceptance Criteria

- [ ] S&P 500 data downloaded and saved to `data/raw/prices_sp500/`
- [ ] Each CSV has columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- [ ] `process_data.py --market sp500` runs successfully
- [ ] `data/processed_sp500/` contains processed files with Parkinson vol + HAR features
- [ ] Existing `python process_data.py` (default VN30) still works unchanged
- [ ] Smoke test: train model on 3 S&P 500 stocks for 2 epochs, verify output shape

---

## 4. [NEEDS CLARIFICATION]

1. **S&P 500 ticker list:** Dùng danh sách cố định (500 tickers) hay filter tự động từ dataset?
   - Đề xuất: Dùng danh sách cố định ~50 tickers representative (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, JNJ, V, PG, XOM, ...) để test nhanh trước. Sau đó mở rộng lên 500.
2. **Date range:** Lấy toàn bộ (2011-2026) hay filter (2006-2026 để match VN30)?
   - Đề xuất: 2011-2026 (dataset bắt đầu từ 2011).
3. **Download method:** `datasets` library (Hugging Face) hay direct CSV download?
   - Đề xuất: `datasets` library — dễ filter, caching tự động.

---

## 5. Scope (In/Out)

### In Scope
- Download script
- Data adapter (convert to VN30 format)
- `--market` argument cho `process_data.py`
- Smoke test (3 stocks, 2 epochs)

### Out of Scope (Later Phases)
- News/sentiment data (Phase 2)
- Market indicators (VIX, rates) (Phase 2)
- Cross-market experiments (Phase 4)
- Full 500-stock training (sau khi smoke test pass)

---

## 6. Files to Create/Modify

### New Files
| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/experiments/sp500/download_sp500.py` | Download + convert | ~60 |
| `src/common/data_adapters.py` | Universal adapter | ~50 |
| `tests/test_sp500/test_download.py` | Download + shape test | ~40 |
| `tests/test_sp500/test_adapter.py` | Adapter correctness | ~40 |

### Modified Files (Backward Compatible)
| File | Change | Lines |
|------|--------|-------|
| `src/common/process_data.py` | Add `--market` arg | ~10 |

---

## 7. Success Metrics

- Download completes without error
- CSV format matches VN30 exactly (column names, types)
- `process_data.py --market sp500` produces same output structure as VN30
- Smoke test: model trains 2 epochs on 3 stocks, outputs valid metrics
