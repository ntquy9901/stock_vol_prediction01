# VN30 Sentiment Analysis Data Schema

**Project:** Stock Volatility Prediction - Sentiment Analysis Integration  
**Date:** 2026-06-27  
**Scope:** VN30 stocks sentiment data collection and processing

---

## Data Schema Overview

This document defines the data schemas for VN30 sentiment analysis, aligning with the existing volatility prediction pipeline structure and following ML/DS common rules.

---

## Raw Data Schemas

### 1. News Articles (`data/raw/vn30_sentiment/news/`)

**File Naming:** `{TICKER}_news_raw.csv` or `vn30_news_YYYY-MM.csv`

**Schema:**
```csv
article_id,ticker,pub_date,title,content,source,author,url,language,collected_at
ART_001,ACB,2026-06-27,"ACB Bank announces Q2 profit","Vietnam Asia Commercial Bank...",VN_Express,Minh Nguyen,https://vnexpress.net/...,vi,2026-06-27 10:30:00
ART_002,VCB,2026-06-27,"VCB leads market rally","Joint Stock Commercial Bank for...",CoffeeBank,Tuan Anh,https://cafef.vn/...,vi,2026-06-27 11:15:00
```

**Field Descriptions:**
- `article_id`: Unique identifier (string, UUID)
- `ticker`: VN30 stock symbol (string: ACB, BCM, BID, etc.)
- `pub_date`: Publication date (date: YYYY-MM-DD)
- `title`: Article headline (string, max 200 chars)
- `content`: Article body text (string, unlimited)
- `source`: News source name (string: VN_Express, CafeF, etc.)
- `author`: Article author (string, optional)
- `url`: Article URL (string)
- `language`: Content language (string: vi, en)
- `collected_at`: Data collection timestamp (datetime: YYYY-MM-DD HH:MM:SS)

---

### 2. Social Media Posts (`data/raw/vn30_sentiment/social_media/`)

**File Naming:** `social_media_raw_YYYY-MM.csv`

**Schema:**
```csv
post_id,ticker,created_at,platform,content,author,followers,likes,shares,url,sentiment_source
POST_001,ACB,2026-06-27 09:30:00,Twitter,"ACB showing strong momentum today...",@stock_trader_vn,15000,234,56,https://twitter.com/...,manual
POST_002,VNM,2026-06-27 10:15:00,StockTwits,"VNM milk products demand increasing...",@vietnam_invest,8500,189,23,https://stocktwits.com/...,manual
```

**Field Descriptions:**
- `post_id`: Unique identifier (string)
- `ticker`: Related stock symbol (string, nullable for general market)
- `created_at`: Post creation timestamp (datetime)
- `platform`: Platform name (string: Twitter, StockTwits, Facebook, Reddit)
- `content`: Post text content (string)
- `author`: Username/handle (string)
- `followers`: Author follower count (integer, optional)
- `likes`: Number of likes (integer, optional)
- `shares`: Number of shares/retweets (integer, optional)
- `url`: Post URL (string)
- `sentiment_source`: Source type (string: manual, api, scraped)

---

### 3. Press Releases (`data/raw/vn30_sentiment/press_releases/`)

**File Naming:** `{TICKER}_press_releases_raw.csv`

**Schema:**
```csv
release_id,ticker,pub_date,title,content,company,category,url,collected_at
PR_001,ACB,2026-06-27,"ACB Board approves dividend","The Board of Directors approved...",ACB Bank,Corporate,https://acb.com.vn/investor/...,2026-06-27 14:00:00
```

---

## Processed Data Schemas

### 4. Daily Sentiment Scores (`data/processed/vn30_sentiment/daily/`)

**File Naming:** `{TICKER}_sentiment_daily.csv`

**Schema:**
```csv
date,ticker,sentiment_score,sentiment_label,positive_score,negative_score,neutral_score,article_count,social_media_count,total_mentions,model_version,processed_at
2026-06-27,ACB,0.65,Positive,0.78,0.12,0.10,15,45,60,finbert_v1.0,2026-06-27 23:59:59
2026-06-27,VCB,0.45,Neutral,0.50,0.30,0.20,22,38,60,finbert_v1.0,2026-06-27 23:59:59
```

**Field Descriptions:**
- `date`: Date of sentiment analysis (date: YYYY-MM-DD)
- `ticker`: VN30 stock symbol (string)
- `sentiment_score`: Overall sentiment score (float: -1.0 to 1.0)
  - -1.0: Extremely negative
  - 0.0: Neutral
  - 1.0: Extremely positive
- `sentiment_label`: Discrete sentiment class (string: Positive, Negative, Neutral)
- `positive_score`: Probability of positive sentiment (float: 0.0 to 1.0)
- `negative_score`: Probability of negative sentiment (float: 0.0 to 1.0)
- `neutral_score`: Probability of neutral sentiment (float: 0.0 to 1.0)
- `article_count`: Number of news articles analyzed (integer)
- `social_media_count`: Number of social media posts (integer)
- `total_mentions`: Total sentiment data points (integer)
- `model_version`: Sentiment model version (string: finbert_v1.0)
- `processed_at`: Processing timestamp (datetime: YYYY-MM-DD HH:MM:SS)

---

### 5. Weekly Aggregated Sentiment (`data/processed/vn30_sentiment/weekly/`)

**File Naming:** `{TICKER}_sentiment_weekly.csv`

**Schema:**
```csv
week_start,week_end,ticker,avg_sentiment_score,max_sentiment,min_sentiment,sentiment_volatility,trend,total_articles,weekly_mentions
2026-06-21,2026-06-27,ACB,0.58,0.82,0.35,0.15,Increasing,125,315
```

---

### 6. HAR-like Sentiment Features (`data/processed/vn30_sentiment/features/`)

**File Naming:** `{TICKER}_har_sentiment_features.csv`

**Schema:**
```csv
date,ticker,sent_daily,sent_weekly,sent_monthly,sent_ma5,sent_ma22,sentiment_volatility,sentiment_momentum,positive_trend,negative_trend,volume_sentiment_corr
2026-06-27,ACB,0.65,0.58,0.52,0.61,0.55,0.12,0.08,True,False,0.45
```

**Field Descriptions:**
- `sent_daily`: Daily sentiment score (float)
- `sent_weekly`: Weekly average sentiment (float)
- `sent_monthly`: Monthly average sentiment (float)
- `sent_ma5`: 5-day moving average of sentiment (float)
- `sent_ma22`: 22-day moving average of sentiment (float)
- `sentiment_volatility`: Standard deviation of sentiment (float)
- `sentiment_momentum`: Rate of change in sentiment (float)
- `positive_trend`: Is positive sentiment increasing? (boolean)
- `negative_trend`: Is negative sentiment increasing? (boolean)
- `volume_sentiment_corr`: Correlation with trading volume (float)

---

### 7. Combined Price + Sentiment (`data/processed/vn30_sentiment/combined/`)

**File Naming:** `{TICKER}_price_sentiment_combined.csv`

**Schema:**
```csv
date,ticker,open,high,low,close,volume,parkinson_volatility,sentiment_score,sentiment_label,sent_daily,sent_weekly,sent_monthly,combined_signal
2026-06-27,ACB,28.50,29.20,28.30,28.90,1500000,0.018,0.65,Positive,0.65,0.58,0.52,Buy_Signal
```

**Combined Signal Logic:**
- `combined_signal`: Trading signal based on price + sentiment (string: Buy_Signal, Sell_Signal, Hold)
  - **Buy_Signal**: Price drop + High positive sentiment
  - **Sell_Signal**: Price rise + High negative sentiment
  - **Hold**: Neutral or conflicting signals

---

## Data Quality Requirements

### **Mandatory Constraints**
- No null values in critical fields (`date`, `ticker`, `sentiment_score`)
- Sentiment scores must be between -1.0 and 1.0
- Probability scores must sum to 1.0 (positive + negative + neutral)
- Dates must follow YYYY-MM-DD format
- Tickers must be valid VN30 symbols

### **Data Validation Rules**
```python
# Validation checklist
- [ ] Date format validation (YYYY-MM-DD)
- [ ] Ticker validation (VN30 symbols only)
- [ ] Sentiment score range (-1.0 to 1.0)
- [ ] Probability sum validation (positive + negative + neutral = 1.0)
- [ ] No duplicate date-ticker combinations
- [ ] Temporal ordering (chronological)
- [ ] Article count >= 0
- [ ] Model version tracking
```

---

## Data File Organization

### **By Stock (Ticker-based)**
```
vn30_sentiment/
├── daily/
│   ├── ACB_sentiment_daily.csv
│   ├── BCM_sentiment_daily.csv
│   └── ...
├── weekly/
│   ├── ACB_sentiment_weekly.csv
│   └── ...
```

### **By Date (Time-series)**
```
vn30_sentiment/
├── features/
│   ├── ACB_har_sentiment_features.csv
│   ├── VN30_combined_sentiment.csv
│   └── market_sentiment_index.csv
```

---

## Integration with Existing Volatility Pipeline

### **Merging Strategy**
```python
# Integration example
volatility_data = pd.read_csv('data/processed/vn30_only/ACB_processed.csv')
sentiment_data = pd.read_csv('data/processed/vn30_sentiment/daily/ACB_sentiment_daily.csv')

# Merge on date
combined_data = pd.merge(
    volatility_data, 
    sentiment_data, 
    on=['date', 'ticker'], 
    how='inner'
)
```

### **Feature Engineering**
```python
# HAR-like sentiment features for volatility prediction
sentiment_features = [
    'sentiment_score',      # Current sentiment
    'sent_daily',          # Daily HAR sentiment
    'sent_weekly',         # Weekly HAR sentiment  
    'sent_monthly',        # Monthly HAR sentiment
    'sentiment_volatility', # Sentiment variability
    'sentiment_momentum'   # Sentiment change rate
]
```

---

## Metadata and Versioning

### **Model Versioning**
- Format: `{model}_v{major}.{minor}` (e.g., `finbert_v1.0`)
- Track in all processed files
- Enable reproducibility and rollback

### **Data Lineage**
- Raw data → Processed data → Features
- Track transformation timestamps
- Document all processing steps

---

**Schema Version:** 1.0  
**Last Updated:** 2026-06-27  
**Compliance:** Follows ML/DS common rules and project data standards
