# Market Data + News/Sentiment: Global Sources & Isolated Design

**Date:** 2026-08-01  
**Goal:** Deep research market data + news sources, design isolated architecture (zero impact on VN30)

---

## 1. Market Data Sources (Beyond OHLCV)

### 1.1 Macro Economic Indicators

| Source | Data | Frequency | Access | Cost |
|--------|------|-----------|--------|------|
| **FRED (Federal Reserve)** | Interest rates, GDP, CPI, unemployment, VIX | Daily/Monthly | API (fredapi) | Free |
| **World Bank** | GDP growth, inflation, trade balance | Annual | API | Free |
| **IMF** | Global economic outlook, exchange rates | Quarterly | API | Free |
| **Yahoo Finance (yfinance)** | ^VIX, ^DJI, ^GSPC, ^IXIC, ^TNX | Daily | Python | Free |
| **Alpha Vantage** | Forex, crypto, economic indicators | Daily | API | Free tier |
| **Quandl/Nasdaq Data Link** | Macro data, commodities, rates | Daily | API | Free tier |

**Key Indicators for Volatility:**
```
- VIX (CBOE Volatility Index) - "Fear index"
- Federal Funds Rate
- 10-Year Treasury Yield (^TNX)
- S&P 500 Index (^GSPC)
- USD Index (DXY)
- Oil Prices (WTI, Brent)
- Gold Prices
```

### 1.2 Market Index Data

| Index | Symbol (yfinance) | Description |
|-------|-------------------|-------------|
| S&P 500 | ^GSPC | 500 large-cap US stocks |
| Dow Jones | ^DJI | 30 large-cap US stocks |
| Nasdaq 100 | ^NDX | 100 non-financial Nasdaq stocks |
| VIX | ^VIX | Volatility index |
| Russell 2000 | ^RUT | Small-cap US stocks |
| FTSE 100 | ^FTSE | UK 100 largest stocks |
| DAX | ^GDAXI | German 40 largest stocks |
| Nikkei 225 | ^N225 | Japan 225 largest stocks |

### 1.3 Sector/Industry Data

| Source | Data | Access |
|--------|------|--------|
| **Yahoo Finance** | Sector ETFs (XLF, XLK, XLE, etc.) | yfinance |
| **FRED** | Industry production indices | API |
| **Compustat** | Company fundamentals, sector classification | Paid (WRDS) |

---

## 2. News/Sentiment Data Sources

### 2.1 Pre-labeled Sentiment Datasets (Hugging Face)

| Dataset | Size | Fields | Language | Source |
|---------|------|--------|----------|--------|
| **zeroshot/twitter-financial-news-sentiment** | 11.9K rows | text, label (-1,0,1) | English | Twitter + News |
| **Jean-Baptiste/financial_news_sentiment** | 1.78K rows | text, label | English | Financial news |
| **hw2942/financial-news-sentiment** | 2.33K rows | text, label | English | Financial news |
| **maguid28/combined_financial_phrasebank_twitter** | 16.8K rows | text, label | English | PhraseBank + Twitter |
| **lumen-models/finbert-financial-news-sentiment** | 285 rows | text, label | English | FinBERT training |
| **TheSunIsRising/finbert-sentiment-dataset** | 2.5K rows | text, label | English | FinBERT |

### 2.2 News + Price Combined Datasets

| Dataset | Size | Fields | Period | Stocks | Source |
|---------|------|--------|--------|--------|--------|
| **KrossKinetic/SP500-Financial-News** | 4.59K rows | id, links, symbol, company, Title, Text, Publishdate | 2006-2024 | 469 S&P 500 | CNBC |
| **FNSPID** | 15.7M news + 29.7M prices | date, ticker, headline, body, source, OHLCV | 1999-2023 | 4,775 US stocks | Multiple |
| **zhihangliu/alphavantage_financial_sentiment_news** | 20K rows | news, sentiment | - | - | Alpha Vantage |

### 2.3 News APIs (Live/Historical)

| API | Data | Cost | Rate Limit |
|-----|------|------|------------|
| **NewsAPI** | Global news, financial filter | Free tier: 100 req/day | 100/day |
| **Alpha Vantage** | News sentiment + stock data | Free tier: 25 req/day | 25/day |
| **Finnhub** | Real-time news, sentiment scores | Free tier: 60 req/min | 60/min |
| **Polygon.io** | News articles, sentiment | Free tier: 5 req/min | 5/min |
| **Tiingo** | News, sentiment, fundamentals | Free tier: 1000 req/month | 1000/month |
| **GDELT** | Global news events, sentiment | Free | Unlimited |

### 2.4 Sentiment Models (Pre-trained)

| Model | Language | Task | Source |
|-------|----------|------|--------|
| **FinBERT** (ProsusAI) | English | Sentiment (pos/neg/neutral) | Hugging Face |
| **FinBERT** (yiyanghkust) | English | Sentiment + emotion | Hugging Face |
| **PhoBERT** (vinai) | Vietnamese | Base model (need fine-tune) | Hugging Face |
| **BERTweet** | English | Twitter sentiment | Hugging Face |
| **RoBERTa** (twitter-financial) | English | Financial Twitter | Hugging Face |

---

## 3. Isolated Architecture Design

### 3.1 Core Principle: "Market-scoped folders"

Mọi thứ đều được scope theo **market** (vn30, sp500, global). Không có file nào chung giữa các market.

```
project_root/
├── data/
│   ├── raw/
│   │   ├── prices/                    # VN30 (EXISTING - DO NOT TOUCH)
│   │   │   ├── ACB.csv
│   │   │   └── VCB.csv
│   │   ├── prices_sp500/              # S&P 500 prices (NEW)
│   │   ├── prices_global/             # Other markets (NEW)
│   │   │
│   │   ├── news/                      # VN30 news (EXISTING - DO NOT TOUCH)
│   │   │   └── crawl_data/
│   │   ├── news_sp500/                # S&P 500 news (NEW)
│   │   │   ├── raw/
│   │   │   │   ├── CNBC/
│   │   │   │   ├── Reuters/
│   │   │   │   └── Bloomberg/
│   │   │   └── processed/
│   │   │       └── sp500_news.parquet
│   │   │
│   │   └── market_data/               # Macro/market indicators (NEW)
│   │       ├── vn30/
│   │       │   ├── vix_vn.csv         # VN volatility index
│   │       │   ├── interest_rates.csv
│   │       │   └── usd_vnd.csv
│   │       ├── sp500/
│   │       │   ├── vix.csv
│   │       │   ├── treasury_rates.csv
│   │       │   ├── sp500_index.csv
│   │       │   └── sector_etfs.csv
│   │       └── global/
│   │           ├── oil_prices.csv
│   │           ├── gold_prices.csv
│   │           └── dxy.csv
│   │
│   ├── processed/
│   │   ├── ACB.csv                    # VN30 processed (EXISTING)
│   │   └── VCB.csv
│   ├── processed_sp500/               # S&P 500 processed (NEW)
│   │   ├── AAPL.csv
│   │   └── MSFT.csv
│   │
│   └── sentiment/
│       ├── vn30/                      # VN30 sentiment (EXISTING)
│       │   └── ACB_sentiment.csv
│       ├── sp500/                     # S&P 500 sentiment (NEW)
│       │   ├── AAPL_sentiment.csv
│       │   └── MSFL_sentiment.csv
│       └── embedding/                 # Embedding caches (NEW)
│           ├── vn30/
│           └── sp500/
│
├── src/
│   ├── common/                        # Shared utilities (EXISTING)
│   │   ├── process_data.py            # ADD --market argument (backward compatible)
│   │   ├── data_adapters.py           # NEW - adapters for different sources
│   │   ├── market_data_loader.py      # NEW - macro/market data loader
│   │   └── sentiment_processor.py     # NEW - sentiment processing (FinBERT/PhoBERT)
│   │
│   ├── experiments/                   # NEW - isolated experiments
│   │   ├── sp500/
│   │   │   ├── download_sp500.py
│   │   │   ├── download_news_sp500.py
│   │   │   ├── download_market_data.py
│   │   │   └── run_sp500_baseline.py
│   │   └── global/
│   │       └── ...
│   │
│   └── lstm_gat_hybrid/               # EXISTING - DO NOT TOUCH
│       ├── train_parallel_enhanced.py # ADD --market argument (backward compatible)
│       └── dataset_with_graph_method.py
│
├── models/
│   ├── vn30_*/                        # VN30 models (EXISTING)
│   ├── sp500_*/                       # S&P 500 models (NEW)
│   └── global_*/                      # Global models (NEW)
│
├── results/
│   ├── vn30_*/                        # VN30 results (EXISTING)
│   ├── sp500_*/                       # S&P 500 results (NEW)
│   └── global_*/                      # Global results (NEW)
│
└── research/
    └── global_benchmark_datasets/     # Research docs (EXISTING)
```

### 3.2 Zero-Impact Guarantee

| Component | VN30 (Existing) | S&P 500 (New) | Conflict? |
|-----------|-----------------|---------------|-----------|
| Raw prices | `data/raw/prices/` | `data/raw/prices_sp500/` | ❌ No |
| Raw news | `data/raw/news/` | `data/raw/news_sp500/` | ❌ No |
| Market data | N/A | `data/raw/market_data/sp500/` | ❌ No |
| Processed | `data/processed/` | `data/processed_sp500/` | ❌ No |
| Sentiment | `data/sentiment/vn30/` | `data/sentiment/sp500/` | ❌ No |
| Models | `models/vn30_*/` | `models/sp500_*/` | ❌ No |
| Results | `results/vn30_*/` | `results/sp500_*/` | ❌ No |
| Code | `src/lstm_gat_hybrid/` | `src/experiments/sp500/` | ❌ No |

### 3.3 Backward Compatible Code Changes

**Only 2 files modified, both backward compatible:**

#### File 1: `src/common/process_data.py`

```python
# BEFORE (existing):
parser.add_argument("--remove_outliers", action="store_true")

# AFTER (backward compatible - default is still VN30):
parser.add_argument("--market", default="vn30", choices=["vn30", "sp500", "global"])
parser.add_argument("--data_dir", default=None)  # None = auto-detect from --market
parser.add_argument("--output_dir", default=None)  # None = auto-detect from --market

# Auto-detect logic:
if args.market == "vn30":
    args.data_dir = args.data_dir or "data/raw/prices"
    args.output_dir = args.output_dir or "data/processed"
elif args.market == "sp500":
    args.data_dir = args.data_dir or "data/raw/prices_sp500"
    args.output_dir = args.output_dir or "data/processed_sp500"
```

**Impact:** `python process_data.py` vẫn chạy như cũ (default `--market vn30`).

#### File 2: `src/lstm_gat_hybrid/train_parallel_enhanced.py`

```python
# BEFORE (existing):
parser.add_argument("--data_dir", default="data/processed")

# AFTER (backward compatible):
parser.add_argument("--market", default="vn30", choices=["vn30", "sp500", "global"])
parser.add_argument("--data_dir", default=None)

# Auto-detect logic (same as process_data.py)
```

**Impact:** `python src/lstm_gat_hybrid/train_parallel_enhanced.py` vẫn chạy như cũ.

### 3.4 New Files (No modification to existing)

| File | Purpose | Lines |
|------|---------|-------|
| `src/common/data_adapters.py` | Convert global datasets to VN30 format | ~80 |
| `src/common/market_data_loader.py` | Load macro/market indicators | ~60 |
| `src/common/sentiment_processor.py` | FinBERT/PhoBERT sentiment scoring | ~100 |
| `src/experiments/sp500/download_sp500.py` | Download S&P 500 OHLCV | ~40 |
| `src/experiments/sp500/download_news_sp500.py` | Download S&P 500 news | ~50 |
| `src/experiments/sp500/download_market_data.py` | Download VIX, rates, etc. | ~60 |
| `src/experiments/sp500/run_sp500_baseline.py` | Wrapper to run existing pipeline | ~30 |

**Total new code:** ~420 lines, **zero modifications** to existing baseline code.

---

## 4. Data Flow: S&P 500 Example

### 4.1 Step 1: Download Data

```bash
# Download S&P 500 OHLCV (26.7M rows from Hugging Face)
python src/experiments/sp500/download_sp500.py

# Download S&P 500 news (4.59K rows from Hugging Face)
python src/experiments/sp500/download_news_sp500.py

# Download market data (VIX, treasury rates, S&P 500 index)
python src/experiments/sp500/download_market_data.py
```

**Output:**
```
data/raw/prices_sp500/AAPL.csv
data/raw/prices_sp500/MSFT.csv
...
data/raw/news_sp500/processed/sp500_news.parquet
data/raw/market_data/sp500/vix.csv
data/raw/market_data/sp500/treasury_rates.csv
```

### 4.2 Step 2: Process Data

```bash
# Process S&P 500 prices (same pipeline as VN30)
python process_data.py --market sp500

# Process S&P 500 news sentiment (FinBERT)
python src/common/sentiment_processor.py --market sp500 --model finbert
```

**Output:**
```
data/processed_sp500/AAPL.csv
data/processed_sp500/MSFT.csv
...
data/sentiment/sp500/AAPL_sentiment.csv
data/sentiment/sp500/MSFT_sentiment.csv
```

### 4.3 Step 3: Train Model

```bash
# Train on S&P 500 (same model code as VN30)
python src/lstm_gat_hybrid/train_parallel_enhanced.py --market sp500
```

**Output:**
```
models/sp500_2026-08-01_*/
results/sp500_2026-08-01_*/
```

### 4.4 Step 4: Compare Results

```bash
# VN30 results (existing, unchanged)
results/vn30_2026-06-19_*/test_metrics.json

# S&P 500 results (new)
results/sp500_2026-08-01_*/test_metrics.json
```

---

## 5. Sentiment Processing: Dual Model Support

### 5.1 Architecture

```python
# src/common/sentiment_processor.py

SENTIMENT_MODELS = {
    "vn30": {
        "model": "vinai/phobert-base",
        "language": "vietnamese",
        "output_cols": ["sentiment_1d", "sentiment_confidence", "news_count_1d"]
    },
    "sp500": {
        "model": "ProsusAI/finbert",
        "language": "english",
        "output_cols": ["sentiment_1d", "sentiment_confidence", "news_count_1d"]
    }
}

def process_news_sentiment(market="vn30", news_dir=None, output_dir=None):
    """Process news to sentiment scores using market-specific model."""
    config = SENTIMENT_MODELS[market]
    model = AutoModelForSequenceClassification.from_pretrained(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    
    # Load news, score sentiment, save
    # ... (same logic for both markets, different model)
```

### 5.2 Usage

```bash
# VN30 (existing behavior, unchanged)
python src/common/sentiment_processor.py --market vn30

# S&P 500 (new, uses FinBERT)
python src/common/sentiment_processor.py --market sp500
```

---

## 6. Market Data Integration

### 6.1 What Market Data Adds

| Feature | Description | Source | Impact on Model |
|---------|-------------|--------|-----------------|
| **VIX** | Market fear index | Yahoo Finance | High (volatility proxy) |
| **Treasury Rates** | Risk-free rate, yield curve | FRED | Medium (macro context) |
| **S&P 500 Index** | Market-wide trend | Yahoo Finance | High (market beta) |
| **Sector ETFs** | Sector performance | Yahoo Finance | Medium (sector rotation) |
| **Oil/Gold Prices** | Commodity impact | Yahoo Finance | Low-Medium |
| **USD Index (DXY)** | Currency strength | Yahoo Finance | Medium (FX impact) |

### 6.2 How to Add to Model

```python
# Current model input: 3 HAR features
# With market data: 3 HAR + N market features = 3+N features

# Example: Add VIX + S&P 500 returns
market_features = pd.DataFrame({
    "vix": vix_data.set_index("date")["close"],
    "sp500_return": sp500_data.set_index("date")["close"].pct_change(),
    "treasury_10y": treasury_data.set_index("date")["close"],
})

# Merge with stock features
df = df.merge(market_features, left_on="Date", right_index=True, how="left")
```

### 6.3 Code Location

```python
# src/common/market_data_loader.py

def load_market_data(market="sp500", start_date="2006-01-01", end_date="2026-01-01"):
    """Download and cache market indicators."""
    
    if market == "sp500":
        tickers = {
            "vix": "^VIX",
            "sp500": "^GSPC",
            "treasury_10y": "^TNX",
            "dxy": "DX-Y.NYB",
        }
    elif market == "vn30":
        tickers = {
            "vix_vn": "VNINDEX.VN",  # VN-Index as proxy
            "usd_vnd": "VND=X",
        }
    
    # Download via yfinance, save to data/raw/market_data/{market}/
    # Return merged DataFrame
```

---

## 7. Summary: What Changes, What Doesn't

### 7.1 DOES NOT CHANGE (Guaranteed)

| File/Path | Status | Reason |
|-----------|--------|--------|
| `data/raw/prices/*.csv` | ✅ Unchanged | VN30 raw data |
| `data/processed/*.csv` | ✅ Unchanged | VN30 processed data |
| `data/raw/news/` | ✅ Unchanged | VN30 news |
| `data/sentiment/vn30/` | ✅ Unchanged | VN30 sentiment |
| `models/vn30_*/` | ✅ Unchanged | VN30 models |
| `results/vn30_*/` | ✅ Unchanged | VN30 results |
| `src/lstm_gat_hybrid/*.py` | ✅ Unchanged | Model code (only add optional arg) |
| `baselines/` | ✅ Unchanged | All existing baselines |

### 7.2 NEW (Isolated)

| File/Path | Purpose | Lines |
|-----------|---------|-------|
| `data/raw/prices_sp500/` | S&P 500 raw prices | - |
| `data/raw/news_sp500/` | S&P 500 raw news | - |
| `data/raw/market_data/` | Macro indicators | - |
| `data/processed_sp500/` | S&P 500 processed | - |
| `data/sentiment/sp500/` | S&P 500 sentiment | - |
| `models/sp500_*/` | S&P 500 models | - |
| `results/sp500_*/` | S&P 500 results | - |
| `src/common/data_adapters.py` | Dataset adapters | ~80 |
| `src/common/market_data_loader.py` | Market data loader | ~60 |
| `src/common/sentiment_processor.py` | Sentiment processor | ~100 |
| `src/experiments/sp500/` | S&P 500 experiment scripts | ~180 |

### 7.3 MINIMAL MODIFICATIONS (Backward Compatible)

| File | Change | Lines | Impact |
|------|--------|-------|--------|
| `process_data.py` | Add `--market` arg (default="vn30") | ~10 | None (default unchanged) |
| `train_parallel_enhanced.py` | Add `--market` arg (default="vn30") | ~10 | None (default unchanged) |

---

## 8. Recommended Implementation Order

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| **1** | Create folder structure | 30 min | 🥇 |
| **2** | `data_adapters.py` - download + convert S&P 500 | 2 hours | 🥇 |
| **3** | `process_data.py` - add `--market` arg | 30 min | 🥇 |
| **4** | Run pipeline on S&P 500 prices only | 1 hour | 🥇 |
| **5** | `market_data_loader.py` - download VIX, rates | 2 hours | 🥈 |
| **6** | `sentiment_processor.py` - FinBERT for S&P 500 | 3 hours | 🥈 |
| **7** | Train model with market data | 1 hour | 🥈 |
| **8** | Compare VN30 vs S&P 500 results | 2 hours | 🥉 |

**Total effort:** ~12 hours (1.5 days)

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Accidentally overwrite VN30 data | All new paths have `_sp500` suffix |
| Model code breaks | Only add optional args, default = existing behavior |
| Download fails | Cache downloads, retry logic |
| Sentiment model wrong language | Explicit model config per market |
| Market data misaligned dates | Forward-fill missing dates |

---

**Conclusion:** Zero impact on VN30 baselines. All new data/code is isolated in market-scoped folders. Only 2 files modified (backward compatible). ~420 lines of new code, all in new files.
