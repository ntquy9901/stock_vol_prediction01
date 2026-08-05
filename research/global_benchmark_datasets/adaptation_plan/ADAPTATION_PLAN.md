# Dataset Adaptation Plan: Global Benchmarks → VN30 Codebase

**Date:** 2026-08-01  
**Goal:** Adapt existing codebase to work with global benchmark datasets  
**Scope:** Minimal changes, reuse existing pipeline where possible

---

## 1. Current Codebase Architecture

```
data/raw/prices/          ← Per-stock CSV files (VN30)
  ├── ACB.csv
  ├── BCM.csv
  └── ...

process_data.py           ← Reads raw CSVs → computes Parkinson vol → HAR features
  ├── calculate_parkinson_volatility()
  ├── create_har_features()
  └── save processed data → data/processed/

src/lstm_gat_hybrid/      ← Model training
  ├── dataset_with_graph_method.py
  └── train_parallel_enhanced.py
```

---

## 2. Adaptation Strategy: "Adapter Pattern"

Instead of modifying existing code, create **adapter scripts** that:
1. Download global dataset
2. Transform to VN30-compatible format
3. Place in `data/raw/prices/` (or new subfolder)
4. Reuse existing pipeline unchanged

---

## 3. Phase 1: S&P 500 via siddharthmb/stocks-ohlcv (1-2 days)

### 3.1 Data Download Script

**File:** `research/global_benchmark_datasets/adaptation_plan/download_sp500.py`

```python
"""Download S&P 500 OHLCV from Hugging Face, convert to VN30 format."""
from datasets import load_dataset
import pandas as pd
import os

# Load dataset
ds = load_dataset("siddharthmb/stocks-ohlcv", split="train")
df = ds.to_pandas()

# S&P 500 tickers (as of 2024)
SP500_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", ...]  # Full list

# Filter to S&P 500
df_sp500 = df[df["act_symbol"].isin(SP500_TICKERS)]

# Convert to VN30 format
for ticker in SP500_TICKERS:
    ticker_df = df_sp500[df_sp500["act_symbol"] == ticker]
    ticker_df = ticker_df.rename(columns={"act_symbol": "Ticker"})
    ticker_df["Date"] = pd.to_datetime(ticker_df["date"]).dt.strftime("%Y-%m-%d")
    ticker_df = ticker_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    ticker_df = ticker_df.sort_values("Date")
    
    # Save in VN30 format
    os.makedirs("data/raw/prices_sp500", exist_ok=True)
    ticker_df.to_csv(f"data/raw/prices_sp500/{ticker}.csv", index=False)
```

### 3.2 Code Changes Required

| File | Change | Lines |
|------|--------|-------|
| `process_data.py` | Add `--data_dir` argument | ~5 |
| `process_data.py` | Default: `data/raw/prices/`, optional: `data/raw/prices_sp500/` | ~3 |
| **Total** | | **~8 lines** |

### 3.3 Verification

```bash
# Run on S&P 500 data
python process_data.py --data_dir data/raw/prices_sp500

# Train model
python src/lstm_gat_hybrid/train_parallel_enhanced.py --data_dir data/processed_sp500
```

---

## 4. Phase 2: FNS Dataset (3-5 days)

### 4.1 Data Download Script

**File:** `research/global_benchmark_datasets/adaptation_plan/download_fnspid.py`

```python
"""Download FNSPID dataset, extract price data, convert to VN30 format."""
import pandas as pd
import os

# Download from GitHub release or Kaggle
# URL: https://github.com/Zdong104/FNSPID_Financial_News_Dataset

# Price data structure:
# date, ticker, open, high, low, close, volume, adj_close

# Filter to specific index (e.g., S&P 500 or Nasdaq 100)
# Convert to VN30 format
```

### 4.2 Code Changes Required

| File | Change | Lines |
|------|--------|-------|
| `process_data.py` | Add `--dataset` argument (vn30, sp500, fnspid) | ~5 |
| `process_data.py` | Handle FNSPID column names if different | ~10 |
| **Total** | | **~15 lines** |

---

## 5. Phase 3: Oxford-Man Realized Volatility (3-5 days)

### 5.1 Key Difference: Target Variable

**Current VN30:**
```python
# Parkinson volatility (computed from OHLCV)
parkinson_vol = (np.log(high / low) ** 2) / (4 * np.log(2))
target = parkinson_vol.shift(-5)  # 5-day ahead
```

**Oxford-Man:**
```python
# Pre-computed realized volatility (5-minute returns)
target = realized_vol.shift(-5)  # 5-day ahead
```

### 5.2 Code Changes Required

| File | Change | Lines |
|------|--------|-------|
| `process_data.py` | Add `--vol_method` (parkinson, realized) | ~5 |
| `process_data.py` | Load realized vol from Oxford-Man CSV | ~15 |
| `src/common/evaluation.py` | No change (QLIKE, RMSE work with any vol) | 0 |
| **Total** | | **~20 lines** |

---

## 6. Phase 4: Cross-Market Experiments (1-2 weeks)

### 6.1 Experiment Design

| Experiment | Train | Test | Purpose |
|------------|-------|------|---------|
| **VN30 Only** | VN30 | VN30 | Baseline |
| **S&P 500 Only** | S&P 500 | S&P 500 | Global baseline |
| **VN30 → S&P 500** | VN30 | S&P 500 | Cross-market generalization |
| **S&P 500 → VN30** | S&P 500 | VN30 | Cross-market generalization |
| **Combined** | VN30 + S&P 500 | VN30 | Multi-market training |
| **Combined** | VN30 + S&P 500 | S&P 500 | Multi-market training |

### 6.2 Code Changes Required

| File | Change | Lines |
|------|--------|-------|
| `src/common/temporal_split.py` | Add `--market` argument | ~5 |
| `src/lstm_gat_hybrid/train_parallel_enhanced.py` | Load multi-market data | ~20 |
| **Total** | | **~25 lines** |

---

## 7. Summary: Total Code Changes

| Phase | Files Changed | Lines Added | Effort |
|-------|--------------|-------------|--------|
| Phase 1: S&P 500 | 1 | ~8 | 1-2 days |
| Phase 2: FNSPID | 1 | ~15 | 3-5 days |
| Phase 3: Oxford-Man RV | 1 | ~20 | 3-5 days |
| Phase 4: Cross-market | 2 | ~25 | 1-2 weeks |
| **Total** | **3-4 files** | **~68 lines** | **3-4 weeks** |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Data format mismatch | Low | Low | Adapter pattern, test with small sample first |
| Different trading calendars | Medium | Medium | Handle holidays, align dates |
| Model doesn't generalize | High | Medium | Expected - this is the research question |
| Download failures | Low | Low | Retry logic, cached downloads |

---

## 9. Success Criteria

- [ ] S&P 500 data downloaded and processed (Phase 1)
- [ ] Model trains on S&P 500 without code changes to model (Phase 1)
- [ ] Results comparable to VN30 (same metrics: RMSE, QLIKE, Dir Acc)
- [ ] Cross-market experiments run successfully (Phase 4)
- [ ] Results documented in `docs/reports/`

---

**Next Step:** Start with Phase 1 - download S&P 500 data and run existing pipeline.
