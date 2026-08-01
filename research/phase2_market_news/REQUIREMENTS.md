# Phase 2 Spec: Market Data + News Sentiment

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Status:** Spec Draft

---

## 1. Goal

Add market indicators (VIX, treasury rates, S&P 500 index) and news sentiment (FinBERT) for S&P 500 stocks — isolated from VN30.

---

## 2. Input/Output

### Market Data Input
- VIX (^VIX) via yfinance
- 10-Year Treasury (^TNX) via yfinance
- S&P 500 Index (^GSPC) via yfinance
- Period: 2011-2026 (match stock data)

### News Sentiment Input
- KrossKinetic/SP500-Financial-News (4.59K rows, Hugging Face)
- FinBERT (ProsusAI/finbert) for sentiment scoring

### Output
- `data/raw/market_data/sp500/vix.csv`
- `data/raw/market_data/sp500/treasury_10y.csv`
- `data/raw/market_data/sp500/sp500_index.csv`
- `data/sentiment/sp500/{TICKER}_sentiment.csv`

---

## 3. Acceptance Criteria

- [ ] Market data downloaded and saved to `data/raw/market_data/sp500/`
- [ ] Each market data CSV has: Date, Close columns
- [ ] News sentiment processed with FinBERT for 3 sample tickers
- [ ] Sentiment CSV has: date, sentiment_score, sentiment_confidence, news_count
- [ ] Zero impact on VN30 data/code
- [ ] Tests pass for market data loader + sentiment processor

---

## 4. [NEEDS CLARIFICATION]

1. **Market data merge strategy:** Merge market data into stock features at train time, or pre-merge into processed CSVs?
   - Đề xuất: Pre-merge vào processed CSVs (đơn giản hơn, giống HAR features)
2. **News sentiment scope:** Process all 4.59K news articles or just 3 sample tickers for Phase 2?
   - Đề xuất: 3 sample tickers (AAPL, MSFT, GOOGL) để verify pipeline, sau mở rộng

---

## 5. Scope (In/Out)

### In Scope
- Market data download (VIX, treasury, S&P 500 index)
- Market data loader + merge with stock features
- FinBERT sentiment for 3 sample tickers
- Tests for both components

### Out of Scope
- Full 500-ticker sentiment processing (Phase 3)
- Cross-market experiments (Phase 4)
- Model training with market data + sentiment (Phase 3)
