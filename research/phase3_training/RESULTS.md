# Phase 3 Results: Feature Comparison

**Date:** 2026-08-01
**Tickers:** AAPL, MSFT, GOOGL
**Epochs:** 10 (early comparison)

---

## Results Summary

| Feature Set | Features | Dir Acc | RMSE | QLIKE |
|-------------|----------|---------|------|-------|
| **HAR-only** | 3 (har_daily, har_weekly, har_monthly) | 50.89% | 0.000304 | 1.959 |
| **Full** | 9 (HAR + VIX + Treasury + S&P 500 + Sentiment) | 51.67% | 0.000292 | 1.952 |

## Delta (Full vs HAR-only)

| Metric | HAR-only | Full | Delta |
|--------|----------|------|-------|
| Dir Acc | 50.89% | 51.67% | +0.78pp |
| RMSE | 0.000304 | 0.000292 | -3.9% |
| QLIKE | 1.959 | 1.952 | -0.4% |

## Observations

1. **Full features slightly outperform HAR-only** on all 3 metrics after just 10 epochs
2. **RMSE improvement:** 3.9% lower with market + sentiment data
3. **Dir Acc improvement:** +0.78 percentage points
4. **Early results:** Only 10 epochs — full training (70 epochs) may show larger gaps

## Next Steps

1. Train both models for 70 epochs with early stopping
2. Test on full S&P 500 (500 tickers)
3. Cross-market experiment: train on S&P 500, test on VN30
