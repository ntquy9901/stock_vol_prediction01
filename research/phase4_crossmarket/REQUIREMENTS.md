# Phase 4 Spec: Cross-Market Experiments

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Status:** Spec Draft

---

## 1. Goal

Run cross-market experiments: train on S&P 500 → test on VN30 (and vice versa), compare generalization across markets.

---

## 2. Experiments

| Experiment | Train | Test | Purpose |
|------------|-------|------|---------|
| **S&P 500 Only** | S&P 500 | S&P 500 | In-market baseline |
| **VN30 Only** | VN30 | VN30 | In-market baseline |
| **S&P 500 → VN30** | S&P 500 | VN30 | Cross-market generalization |
| **VN30 → S&P 500** | VN30 | S&P 500 | Cross-market generalization |
| **Combined** | S&P 500 + VN30 | VN30 | Multi-market training |
| **Combined** | S&P 500 + VN30 | S&P 500 | Multi-market training |

---

## 3. Acceptance Criteria

- [ ] Full S&P 500 (500 tickers) downloaded and processed
- [ ] All 6 experiments run with 70 epochs + early stopping
- [ ] Results saved with all 6 mandatory metrics
- [ ] Comparison report with cross-market generalization analysis
- [ ] Zero impact on VN30 code/data

---

## 4. [NEEDS CLARIFICATION]

1. **Full 500-ticker download:** May take 30+ minutes. Run now or use subset?
   - Đề xuất: Run full download (cached by HF, only first time slow)
2. **Cross-market model:** Need compatible feature sets. VN30 has 3 HAR features, S&P 500 has 9. How to align?
   - Đề xuất: Use HAR-only (3 features) for cross-market experiments (common denominator)

---

## 5. Scope (In/Out)

### In Scope
- Full S&P 500 download + process
- 70-epoch training with early stopping
- Cross-market experiments (HAR-only features)
- Comprehensive comparison report

### Out of Scope
- Hyperparameter tuning
- Full 9-feature cross-market (different feature sets)
- Production deployment
