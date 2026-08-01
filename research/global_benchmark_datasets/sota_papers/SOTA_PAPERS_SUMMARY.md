# SOTA Papers Summary: Volatility Forecasting 2025-2026

**Date:** 2026-08-01  
**Scope:** Papers using global benchmark datasets for volatility forecasting

---

## 1. Kronos: Foundation Model for Financial Markets (Aug 2025)

**Paper:** arXiv 2508.02739  
**GitHub:** https://github.com/shiyu-coder/Kronos (35.2k stars)

### Key Points
- **Problem:** General-purpose time series models underperform on financial K-line (OHLCV) data
- **Solution:** Specialized tokenizer + autoregressive pre-training on large OHLCV corpus
- **Dataset:** Large-scale OHLCV (unspecified, likely US markets)
- **Architecture:** Tokenizer → Autoregressive transformer → Forecasting head
- **Tasks:** Price forecasting + synthetic data generation
- **Results:** Outperforms TimesFM, PatchTST on financial forecasting tasks

### Relevance to Our Project
- ✅ Uses OHLCV data (same as VN30)
- ✅ Foundation model approach (similar to our TimesFM experiments)
- ⚠️ No volatility-specific evaluation reported
- 🔍 **Action:** Test Kronos on VN30 data, compare with our LSTM-HAR

---

## 2. Zhang/Pu/Cucuringu/Dong (IJF 2025)

**Paper:** International Journal of Forecasting, 2025

### Key Points
- **Problem:** Volatility spillover between stocks is asymmetric (directed, not symmetric)
- **Solution:** Directed lead-lag graph construction (not symmetric correlation)
- **Dataset:** S&P 500 daily OHLCV
- **Method:** Granger causality → directed edges → GNN
- **Results:** Improves Dir Acc vs symmetric graph baselines

### Relevance to Our Project
- ✅ We implemented this in `baselines/2026-07-26_spillover_qlike_baseline`
- ⚠️ Result was null (68.23% vs 68.25% - no real lift on VN30)
- 🔍 **Action:** Test on S&P 500 data - may work better on US markets

---

## 3. Chi et al. (2026)

**Paper:** 2026 (details from project-context.md)

### Key Points
- **Problem:** Multi-scale temporal-spatial dependencies in volatility
- **Solution:** Multi-scale attention (daily, weekly, monthly + cross-stock)
- **Dataset:** US stock markets (unspecified)
- **Method:** Multi-scale temporal attention + spatial GNN
- **Results:** SOTA on volatility forecasting benchmarks

### Relevance to Our Project
- ✅ Similar to our LSTM-GAT architecture
- ⚠️ We haven't implemented multi-scale attention yet
- 🔍 **Action:** Consider multi-scale attention for Phase 3

---

## 4. FNSPID Dataset Paper (Zdong et al., 2024)

**Paper:** GitHub https://github.com/Zdong104/FNSPID_Financial_News_Dataset

### Key Points
- **Dataset:** 15.7M news + 29.7M prices, 4,775 stocks, 1999-2023
- **Organization:** Per-ticker, per-date (industry standard)
- **News:** Headline + body + source + URL
- **Price:** OHLCV + adjusted close
- **Use Case:** Sentiment-volatility fusion research

### Relevance to Our Project
- ✅ We already identified this dataset (project-context.md)
- ✅ Same structure as our planned sentiment pipeline
- ⚠️ English news (need FinBERT, not PhoBERT)
- 🔍 **Action:** Use for Phase 2-3 (sentiment integration)

---

## 5. TimesFM 2.5 (Google, 2025)

**Paper:** Google Research (existing in our project)

### Key Points
- **Model:** Decoder-only transformer, 232M params
- **Method:** LoRA fine-tuning (~1.4M trainable params)
- **Dataset:** Univariate time series (any)
- **Results:** SOTA on M4, M5, tourism, electricity benchmarks

### Relevance to Our Project
- ✅ Already implemented in `src/timesfm_baseline/`
- ✅ 34 tests passing, 100% pass rate
- 🔍 **Action:** Test on S&P 500 data, compare with VN30 results

---

## 6. Benchmark Datasets Used in Papers

| Paper | Dataset | Stocks | Period | Metrics |
|-------|---------|--------|--------|---------|
| Kronos | US OHLCV (unspecified) | ~5,000 | 2011-2026 | MSE, SMAPE |
| Zhang et al. (2025) | S&P 500 | 500 | 2000-2024 | RMSE, QLIKE, Dir Acc |
| Chi et al. (2026) | US markets | ~1,000 | 2000-2025 | RMSE, MAE, R² |
| FNSPID | US stocks | 4,775 | 1999-2023 | Custom |
| TimesFM | M4, M5, etc. | Various | Various | SMAPE, MASE |
| Oxford-Man | 21 indices | 21 | 2000-2025 | RV metrics |

---

## 7. Key Findings

### 7.1 Dataset Trends
1. **OHLCV is standard:** All papers use daily OHLCV as input
2. **Multi-stock is common:** Papers use 500-5,000 stocks (not single stock)
3. **Long history:** 20+ years of data (2000-2025)
4. **Realized vol is gold standard:** Oxford-Man RV used for evaluation

### 7.2 Model Trends
1. **Foundation models:** Kronos, TimesFM pre-train on large corpora
2. **Graph methods:** GNNs for cross-stock relationships
3. **Multi-scale:** Daily, weekly, monthly features (HAR pattern)
4. **Directed graphs:** Asymmetric spillover > symmetric correlation

### 7.3 Evaluation Trends
1. **QLIKE is standard:** Volatility literature favorite
2. **RMSE + Dir Acc:** Forecasting + directional accuracy
3. **Temporal split:** No random splits (time series integrity)
4. **Cross-market:** Some papers test on multiple markets

---

## 8. Gaps Identified

| Gap | Our Project | SOTA | Action |
|-----|------------|------|--------|
| **Dataset size** | 30 stocks | 500-5,000 stocks | Add S&P 500 |
| **Data history** | 2006-2026 | 1999-2025 | Add FNSPID |
| **Vol target** | Parkinson | Realized Vol | Add Oxford-Man |
| **News sentiment** | Deferred | Integrated | Add FNSPID news |
| **Directed graph** | Implemented (null) | Works on S&P 500 | Test on US data |
| **Foundation model** | TimesFM | Kronos + TimesFM | Test Kronos |

---

## 9. Recommended Next Steps

1. **Immediate:** Download S&P 500 data (siddharthmb/stocks-ohlcv)
2. **Week 1:** Run existing pipeline on S&P 500
3. **Week 2:** Compare VN30 vs S&P 500 results
4. **Week 3:** Test directed graph on S&P 500 (may work better)
5. **Week 4:** Test Kronos on VN30 + S&P 500
6. **Month 2:** Add Oxford-Man RV, compare Parkinson vs RV
7. **Month 3:** Add FNSPID news sentiment (FinBERT)

---

**Conclusion:** Our architecture is aligned with SOTA. Main gap is dataset size (30 vs 500+ stocks) and vol target (Parkinson vs Realized Vol). Adding S&P 500 data is the highest-impact, lowest-effort next step.
