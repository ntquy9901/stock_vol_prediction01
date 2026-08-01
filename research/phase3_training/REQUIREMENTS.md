# Phase 3 Spec: Enhanced Features + Model Training

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Status:** Spec Draft

---

## 1. Goal

Merge market data (VIX, treasury, S&P 500 index) and news sentiment into stock features, train model with enhanced features, and compare performance against HAR-only baseline.

---

## 2. Input/Output

### Input
- `data/processed_sp500/{TICKER}_processed.csv` — HAR features (3 cols)
- `data/raw/market_data/sp500/*.csv` — Market indicators (3 cols)
- `data/sentiment/sp500/{TICKER}_sentiment.csv` — Sentiment scores (3 cols)

### Output
- `data/processed_sp500_enhanced/{TICKER}_enhanced.csv` — 9 features total
- `results/sp500_enhanced_{timestamp}/` — Training results
- Comparison report: HAR-only vs HAR+market vs HAR+market+sentiment

---

## 3. Feature Schema

| Feature | Source | Description |
|---------|--------|-------------|
| har_daily_vol | HAR | 1-day rolling mean |
| har_weekly_vol | HAR | 5-day rolling mean |
| har_monthly_vol | HAR | 22-day rolling mean |
| vix | Market | CBOE Volatility Index |
| treasury_10y | Market | 10-Year Treasury Rate |
| sp500_index | Market | S&P 500 Index level |
| sentiment_score | News | Daily sentiment (-1 to +1) |
| sentiment_confidence | News | Model confidence (0 to 1) |
| news_count | News | Number of articles per day |

---

## 4. Acceptance Criteria

- [ ] Feature merger creates CSVs with 9 features + target_5d
- [ ] Model trains with 9 features (input_size=9)
- [ ] Results saved with all 6 mandatory metrics
- [ ] Comparison report shows delta vs HAR-only baseline
- [ ] Zero impact on VN30 code/data
- [ ] Tests pass for feature merger

---

## 5. [NEEDS CLARIFICATION]

1. **Model architecture:** Reuse existing LSTM-GAT with input_size=9, or create new model?
   - Đề xuất: Reuse existing, just change input_size parameter
2. **Training config:** Same as VN30 (70 epochs, patience=15)?
   - Đề xuất: Yes, for fair comparison
3. **Baseline comparison:** Train HAR-only model on S&P 500 first, then compare?
   - Đề xuất: Yes — need S&P 500 HAR-only baseline before comparing

---

## 6. Scope (In/Out)

### In Scope
- Feature merger script
- Train model with 9 features (3 HAR + 3 market + 3 sentiment)
- Train HAR-only baseline on S&P 500 (for comparison)
- Comparison report with all 6 metrics

### Out of Scope
- Full 500-ticker training (use 3 sample tickers for Phase 3)
- Cross-market experiments (Phase 4)
- Hyperparameter tuning (use standard config)
