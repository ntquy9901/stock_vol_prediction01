# Phase 4 Results: Cross-Market Experiments

**Date:** 2026-08-01
**Branch:** `global-benchmark`

---

## Experiment Results

| Experiment | Train | Test | Epochs | Dir Acc | RMSE | QLIKE |
|------------|-------|------|--------|---------|------|-------|
| **S&P 500 → VN30** | S&P 500 (257 tickers) | VN30 (32 tickers) | 38 (early stop) | 48.32% | 0.000229 | 0.0795 |
| **VN30 → S&P 500** | VN30 (32 tickers) | S&P 500 (3 tickers) | 13 (early stop) | 49.75% | 0.000638 | 0.5174 |

## Comparison with In-Market Baselines

| Experiment | Dir Acc | RMSE | QLIKE | vs In-Market |
|------------|---------|------|-------|-------------|
| S&P 500 → VN30 | 48.32% | 0.000229 | 0.0795 | Lower DirAcc than VN30-only (67.90%) |
| VN30 → S&P 500 | 49.75% | 0.000638 | 0.5174 | Lower DirAcc than S&P 500-only (51.67%) |

## Key Findings

1. **Cross-market generalization is poor:**
   - S&P 500 → VN30: 48.32% DirAcc (vs 67.90% VN30-only)
   - VN30 → S&P 500: 49.75% DirAcc (vs 51.67% S&P 500-only)

2. **Early stopping triggered quickly:**
   - S&P 500 → VN30: 38 epochs (model couldn't learn VN30 patterns)
   - VN30 → S&P 500: 13 epochs (very fast convergence, model memorized S&P 500)

3. **RMSE is lower for cross-market:**
   - S&P 500 → VN30: 0.000229 (lower than VN30-only ~0.0003)
   - But DirAcc is much worse → model predicts constant values

4. **Market-specific patterns dominate:**
   - VN30 (emerging market) has different volatility dynamics than S&P 500 (developed)
   - HAR features alone are not sufficient for cross-market transfer

## Implications

1. **Market-specific models are necessary:** Cross-market transfer doesn't work well with HAR-only features
2. **Need domain adaptation:** Techniques like fine-tuning, domain adversarial training, or market-specific normalization may help
3. **Multi-market training:** Combining data from both markets during training may improve generalization

## Next Steps

1. Test multi-market training (S&P 500 + VN30 combined)
2. Try domain adaptation techniques
3. Add market indicators as features (VIX, treasury rates)
4. Test on more tickers for S&P 500 test set
