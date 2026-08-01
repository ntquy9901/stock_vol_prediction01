# Phase 1 Design: S&P 500 Data Download + Adapt

**Date:** 2026-08-01  
**Branch:** `global-benchmark`

---

## 1. Architecture

```
src/experiments/sp500/download_sp500.py
    ↓ (downloads from HF, converts to VN30 format)
data/raw/prices_sp500/{TICKER}.csv
    ↓ (process_data.py --market sp500)
data/processed_sp500/{TICKER}.csv
    ↓ (train_parallel_enhanced.py --market sp500)
results/sp500_baseline_{timestamp}/
```

## 2. Data Flow

```
Hugging Face (siddharthmb/stocks-ohlcv)
    ↓ datasets.load_dataset()
26.7M rows DataFrame (date, act_symbol, open, high, low, close, volume)
    ↓ filter: act_symbol in SP500_TICKERS
~500 stocks × ~3800 days = ~1.9M rows
    ↓ group by act_symbol, save per-stock CSV
data/raw/prices_sp500/AAPL.csv (Date, Open, High, Low, Close, Volume)
    ↓ process_data.py --market sp500
data/processed_sp500/AAPL.csv (+ parkinson_vol, har_daily, har_weekly, har_monthly, target_5d)
```

## 3. File List

### New
| File | Purpose |
|------|---------|
| `src/experiments/sp500/__init__.py` | Package marker |
| `src/experiments/sp500/download_sp500.py` | Download + convert script |
| `src/common/data_adapters.py` | Universal adapter function |
| `tests/test_sp500/__init__.py` | Test package |
| `tests/test_sp500/test_download.py` | Download + shape tests |
| `tests/test_sp500/test_adapter.py` | Adapter correctness tests |

### Modified
| File | Change |
|------|--------|
| `src/common/process_data.py` | Add `--market` arg (default="vn30") |

## 4. SP500 Ticker List

Full S&P 500 ticker list (as of 2024) — ~500 tickers. Will be embedded in `download_sp500.py`.

## 5. Fallback Plan

If `datasets` library fails (network, auth, etc.):
- Fallback to direct CSV download from Hugging Face
- URL: `https://huggingface.co/datasets/siddharthmb/stocks-ohlcv/resolve/main/data.csv` (or similar)
- Use `pandas.read_csv()` instead of `datasets.load_dataset()`
