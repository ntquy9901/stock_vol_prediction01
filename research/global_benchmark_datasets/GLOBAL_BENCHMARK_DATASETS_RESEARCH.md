# Global Benchmark Datasets for Stock Volatility Forecasting

**Research Date:** 2026-08-01  
**Scope:** SOTA papers 2025-2026, benchmark datasets, data structures, adaptation plan  
**Target:** Adapt global datasets to current VN30 codebase for cross-market validation

---

## 1. SOTA Papers 2025-2026 (Volatility Forecasting)

### 1.1 Kronos: Foundation Model for Financial Markets (Aug 2025)
- **Paper:** arXiv 2508.02739
- **GitHub:** 35.2k stars (shiyu-coder/Kronos)
- **Key Innovation:** Specialized pre-training framework for financial K-line (OHLCV) data
- **Dataset:** Large-scale OHLCV pre-training corpus
- **Architecture:** Unique tokenizer + autoregressive pre-training
- **Tasks:** Forecasting + synthetic data generation
- **Relevance:** HIGH - directly uses OHLCV data like our project

### 1.2 Key Volatility Forecasting Papers (2025-2026)

From prior research (see `project-context.md` §2026-07-26):
- **Zhang/Pu/Cucuringu/Dong (IJF 2025):** Directed lead-lag graph for volatility spillover
- **Chi et al. (2026):** Multi-scale temporal-spatial attention
- **FNSPID (Zdong et al., 2024):** 15.7M time-aligned news + 29.7M prices (4,775 stocks, 1999-2023)

---

## 2. Benchmark Datasets Catalog

### 2.1 Hugging Face Datasets (Found)

| Dataset | Size | Format | Fields | Period | Stocks | Relevance |
|---------|------|--------|--------|--------|--------|-----------|
| **siddharthmb/stocks-ohlcv** | 26.7M rows, 1.21GB | CSV | date, act_symbol, open, high, low, close, volume | 2011-2026 | ~5,000+ US stocks | ⭐⭐⭐⭐⭐ |
| **KrossKinetic/SP500-Financial-News** | 4.59K rows, 17.6MB | JSON | id, links, symbol, company, Title, Text, Publishdate | 2006-2024 | 469 S&P 500 | ⭐⭐⭐⭐ |
| **Wenyan0110/Multimodal-Financial** | 754 rows | - | Image, Text, Table, TimeSeries | - | - | ⭐⭐ |
| **iioos/financial-time-series-lite** | - | - | - | - | - | ⭐⭐ |

### 2.2 Academic/Industry Benchmark Datasets (Known)

| Dataset | Source | Fields | Period | Stocks | Access |
|---------|--------|--------|--------|--------|--------|
| **FNSPID** | GitHub (Zdong104) | Date, Ticker, Headline, Body, Source, Price (OHLCV) | 1999-2023 | 4,775 US stocks | ✅ Open |
| **Oxford-Man Institute RV** | Oxford | Realized Volatility (5-min), Daily OHLCV | 2000-2025 | 21 indices | ✅ Open |
| **S&P 500 (Yahoo Finance)** | yfinance | OHLCV, Adjusted Close | 1990-2026 | 500 stocks | ✅ Free API |
| **CRSP/Compustat** | WRDS | OHLCV, Fundamentals, Returns | 1926-2026 | All US listed | 🔒 Paid |
| **TAQ (NYSE)** | WRDS | Tick-level OHLCV, Trades, Quotes | 1993-2026 | All NYSE | 🔒 Paid |
| **Nasdaq ITCH** | Nasdaq | Order-level, tick data | 2000-2026 | All Nasdaq | 🔒 Paid |
| **Kaggle: G-Research** | Kaggle | OHLCV, target (returns) | 2007-2022 | ~3,000 stocks | ✅ Free |
| **Kaggle: Optiver RV** | Kaggle | Order book, realized vol | 2021-2022 | 100+ stocks | ✅ Free |

---

## 3. Dataset Structures (Detailed)

### 3.1 siddharthmb/stocks-ohlcv (Hugging Face)

**Structure:**
```
Columns:
  - date: string (YYYY-MM-DD HH:MM:SS format)
  - act_symbol: string (ticker, 1-6 chars)
  - open: float64
  - high: float64
  - low: float64
  - close: float64
  - volume: int64
```

**Sample row:**
```
date: 2011-01-03 00:00:00
act_symbol: AAPL
open: 325.9
high: 330.26
low: 324.84
close: 329.57
volume: 15897201
```

**Comparison with our VN30 data:**
| Our VN30 Field | stocks-ohlcv Field | Match? |
|----------------|-------------------|--------|
| Date | date | ✅ Yes |
| Ticker | act_symbol | ✅ Yes |
| Open | open | ✅ Yes |
| High | high | ✅ Yes |
| Low | low | ✅ Yes |
| Close | close | ✅ Yes |
| Volume | volume | ✅ Yes |

**Adaptation effort:** MINIMAL - direct 1:1 mapping, only need to filter to specific index (S&P 500, Nasdaq 100, etc.)

### 3.2 FNSPID (GitHub: Zdong104/FNSPID_Financial_News_Dataset)

**Structure:**
```
Price data (29.7M records):
  - date: date
  - ticker: string
  - open, high, low, close: float
  - volume: int
  - adj_close: float

News data (15.7M records):
  - date: date
  - ticker: string
  - headline: string
  - body: text
  - source: string
  - url: string
```

**Comparison with our setup:**
| Our Component | FNSPID | Match? |
|---------------|--------|--------|
| OHLCV price data | ✅ Same fields | ✅ Direct |
| News articles | ✅ 15.7M articles | ✅ Direct |
| Per-ticker, per-date | ✅ Yes | ✅ Direct |
| Vietnamese sentiment | ❌ English only | ⚠️ Need PhoBERT |

**Adaptation effort:** LOW - same structure, just different market (US vs VN)

### 3.3 Oxford-Man Institute Realized Volatility

**Structure:**
```
Daily data per index:
  - Date: date
  - Open, High, Low, Close: float
  - Volume: int
  - Realized Vol (5-min): float (target)
  - Realized Kernel: float
  - Bipower Variation: float
```

**Indices covered:** S&P 500, FTSE 100, DAX, Nikkei 225, Hang Seng, etc. (21 total)

**Adaptation effort:** MEDIUM - has pre-computed realized vol (better than our Parkinson), but only indices not individual stocks

### 3.4 Kaggle: Optiver Realized Volatility Prediction

**Structure:**
```
train.csv:
  - stock_id: int
  - time_id: int
  - target: float (realized vol, 10-min window)

book_[train/test].parquet:
  - stock_id, time_id
  - bid/ask prices (levels 1-2)
  - bid/ask sizes (levels 1-2)

trade_[train/test].parquet:
  - stock_id, time_id
  - seconds_in_bucket
  - price, order_count, volume
```

**Adaptation effort:** HIGH - order book data, not daily OHLCV. Different granularity (intraday vs daily)

---

## 4. Benchmark Methodologies (How They Evaluate)

### 4.1 Common Evaluation Metrics

| Metric | Formula | Direction | Used By |
|--------|---------|-----------|---------|
| **MSE** | mean((y_true - y_pred)²) | ↓ lower | All |
| **RMSE** | sqrt(MSE) | ↓ lower | All |
| **MAE** | mean(|y_true - y_pred|) | ↓ lower | All |
| **R²** | 1 - SS_res/SS_tot | ↑ higher | All |
| **QLIKE** | log(pred) + true/pred | ↓ lower | Vol literature |
| **Dir Acc** | mean(sign(Δy_true) == sign(Δy_pred)) | ↑ higher | Our project |
| **MASE** | MAE / MAE_naive | ↓ lower | TimesFM |
| **SMAPE** | mean(2|y-p|/(|y|+|p|)) | ↓ lower | M4/M5 competitions |

### 4.2 Standard Benchmark Protocols

**Temporal Split (Standard):**
```
Train: 70-80% (earliest data)
Val:   10-15% (middle)
Test:  10-15% (latest)
```

**Rolling Window (Alternative):**
```
Window 1: Train[2000-2010] → Test[2011]
Window 2: Train[2001-2011] → Test[2012]
...
```

**Expanding Window (Alternative):**
```
Train: [2000-t] → Test: [t+1]
t expands each iteration
```

### 4.3 Baseline Models (Standard Comparison)

| Baseline | Description | Used By |
|----------|-------------|---------|
| **HAR-R** | Heterogeneous Autoregression (Corsi 2009) | Vol forecasting papers |
| **GARCH(1,1)** | Generalized ARCH | Econometrics standard |
| **EGARCH** | Exponential GARCH (asymmetric) | Vol forecasting |
| **Random Walk** | y_pred = y_{t-1} | Universal baseline |
| **Historical Mean** | y_pred = mean(train) | Universal baseline |
| **LSTM** | Standard LSTM | Deep learning papers |
| **TimesFM** | Foundation model (Google) | SOTA 2025-2026 |
| **Kronos** | Foundation model (K-line) | SOTA 2025 |

---

## 5. Adaptation Analysis: Which Datasets Work With Our Codebase

### 5.1 Compatibility Matrix

| Dataset | OHLCV Match | Needs New Loader | Needs New Features | Effort | Priority |
|---------|-------------|-----------------|-------------------|--------|----------|
| **siddharthmb/stocks-ohlcv** | ✅ 100% | ❌ Reuse existing | ❌ None | ⭐ Low | 🥇 #1 |
| **FNSPID (price only)** | ✅ 100% | ⚠️ Minor tweak | ❌ None | ⭐ Low | 🥈 #2 |
| **FNSPID (price + news)** | ✅ 100% | ⚠️ Minor tweak | ✅ Sentiment | ⭐⭐ Medium | 🥉 #3 |
| **Oxford-Man RV** | ✅ 100% | ⚠️ Minor tweak | ✅ RV target | ⭐⭐ Medium | #4 |
| **Yahoo Finance (S&P 500)** | ✅ 100% | ❌ Reuse existing | ❌ None | ⭐ Low | #5 |
| **Kaggle G-Research** | ✅ 90% | ⚠️ Minor tweak | ❌ None | ⭐ Low | #6 |
| **Kaggle Optiver RV** | ❌ Intraday | 🔴 New loader | 🔴 Order book | ⭐⭐⭐ High | Skip |

### 5.2 Recommended Adaptation Order

**Phase 1: Quick Wins (1-2 days)**
1. **siddharthmb/stocks-ohlcv** - Download, filter to S&P 500, run existing pipeline
2. **Yahoo Finance S&P 500** - Use yfinance, same format as VN30

**Phase 2: Moderate Effort (3-5 days)**
3. **FNSPID price data** - Download, adapt loader, compare results
4. **Oxford-Man RV** - Use pre-computed realized vol as target (better than Parkinson)

**Phase 3: Advanced (1-2 weeks)**
5. **FNSPID price + news** - Integrate sentiment pipeline (already deferred in our project)
6. **Kaggle G-Research** - Adapt to our framework, compare

---

## 6. Key Differences: VN30 vs Global Markets

| Aspect | VN30 (Our Data) | US Markets (Global) |
|--------|-----------------|---------------------|
| **Trading days/year** | ~250 | ~252 |
| **Market hours** | 9:00-15:00 ICT | 9:30-16:00 EST |
| **Price limits** | ±7% (ceiling/floor) | None (circuit breakers) |
| **Short selling** | Limited | Common |
| **Data quality** | Some gaps, missing data | Clean, complete |
| **Market efficiency** | Emerging market | Developed market |
| **Volatility level** | Higher (emerging) | Lower (developed) |
| **News impact** | Less efficient pricing | Faster incorporation |

**Implications for our model:**
- Model trained on VN30 may NOT generalize to US markets (different volatility regimes)
- Model trained on US markets may underperform on VN30 (higher volatility, less efficient)
- Cross-market validation is valuable for testing robustness

---

## 7. Actionable Recommendations

### 7.1 Immediate Actions (This Week)
1. Download `siddharthmb/stocks-ohlcv` from Hugging Face
2. Filter to S&P 500 tickers (~500 stocks)
3. Run existing `process_data.py` pipeline on S&P 500 data
4. Compare results: VN30 vs S&P 500 performance

### 7.2 Short-term (2-4 weeks)
5. Download FNS dataset (price + news)
6. Adapt sentiment pipeline for English (FinBERT instead of PhoBERT)
7. Run cross-market experiments: train on VN30 → test on S&P 500 (and vice versa)

### 7.3 Medium-term (1-3 months)
8. Implement Oxford-Man RV as alternative target
9. Compare Parkinson vol vs Realized Vol forecasting accuracy
10. Publish comparison: VN30 vs global markets

---

## 8. Data Sources & Links

| Dataset | URL | Access |
|---------|-----|--------|
| siddharthmb/stocks-ohlcv | https://huggingface.co/datasets/siddharthmb/stocks-ohlcv | Free |
| FNSPID | https://github.com/Zdong104/FNSPID_Financial_News_Dataset | Free |
| Oxford-Man RV | https://realized.oxford-man.ox.ac.uk/data/ | Free |
| Yahoo Finance | https://pypi.org/project/yfinance/ | Free API |
| Kaggle G-Research | https://www.kaggle.com/c/g-research-crypto-forecasting | Free |
| Kaggle Optiver | https://www.kaggle.com/c/optiver-realized-volatility-prediction | Free |
| Kronos | https://github.com/shiyu-coder/Kronos | Open Source |

---

**Next Step:** Create detailed adaptation plan with code changes needed for each dataset.
