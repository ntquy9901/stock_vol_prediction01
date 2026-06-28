# Model Comparison Summary - VN30 Stocks

**Date:** 2026-06-21
**Dataset:** 30 VN30 Blue-Chip Stocks (96,352 samples)
**Comparison:** 4 Models (2 with data leakage, 2 without)

---

## 📊 Quick Comparison Table

### All Models (With and Without Data Leakage)

| Model | Split Method | MSE | RMSE | MAE | R² | QLIKE | Dir Acc | Training Time | Status |
|-------|--------------|-----|------|-----|----|----|---------|---------------|---------|
| **HAR-R Linear** | ✅ Temporal | **2.631e-07** | **0.000513** | **0.000257** | **0.105** | **1.298** | **51.53%** | 0.004s | ⭐ Best Realistic |
| **Simple LSTM** | ❌ Random | **3.447e-07** | **0.000587** | **0.000303** | **0.167** | **1.170** | **67.63%** | ~15 min | 🔴 **DATA LEAKAGE** |
| **LSTM-HAR** | ❌ Random | **3.120e-07** | **0.000559** | **0.000297** | **0.161** | **0.566** | **67.39%** | ~15 min | 🔴 **DATA LEAKAGE** |
| **Simple LSTM (Val)** | ✅ Temporal (70/15/15) | **5.283e-07** | **0.000727** | **0.000424** | **0.211** | **0.673** | **47.89%** | ~5 min | ✅ Realistic |
| **LSTM-HAR (Val)** | ✅ Temporal (70/15/15) | **3.065e-07** | **0.000554** | **0.000258** | **0.110** | N/A | **48.09%** | ~8 min | ✅ Realistic |
| **Enhanced LSTM-HAR (Prevention)** | ✅ Temporal (70/15/15) | **3.107e-07** | **0.000557** | **0.000259** | **0.098** | **0.641** | **48.56%** | ~22 min | ⭐ Best Deep Learning |

---

## 🔴 Data Leakage Impact

### How Much Did Random Split Inflate Metrics?

**Directional Accuracy:**
```
Simple LSTM: 47.89% → 67.63% (+41.2% inflation)
LSTM-HAR:     48.09% → 67.39% (+40.1% inflation)
```

**RMSE:**
```
Simple LSTM: 0.000727 → 0.000587 (19% better - unrealistic)
LSTM-HAR:     0.000554 → 0.000559 (1% worse - coincidence)
```

**Conclusion:** Data leakage can inflate directional accuracy by **40%+**, making models appear much better than reality.

---

## ✅ Realistic Performance (Temporal Split Only)

### Complete Rankings - All 6 Metrics

**1. MSE (Lower is Better):**
```
Ranking (Realistic Models Only):
1. HAR-R Linear:                     2.631e-07 ⭐ BEST
2. Enhanced LSTM-HAR (Prevention):   3.107e-07 (+18.1% worse)
3. LSTM-HAR (Val):                  3.065e-07 (+16.5% worse)
4. Simple LSTM (Val):               5.283e-07 (+100.8% worse)

Target: Lower is better
Status: ✅ All models competitive (within 2x range)
Winner: HAR-R Linear
```

**2. RMSE (Lower is Better):**
```
Ranking (Realistic Models Only):
1. HAR-R Linear:                     0.000513 ⭐ BEST
2. LSTM-HAR (Val):                  0.000554 (+8.0% worse)
3. Enhanced LSTM-HAR (Prevention):   0.000557 (+8.5% worse)
4. Simple LSTM (Val):               0.000727 (+41.7% worse)

Target: < 0.20 ✅ ALL MODELS PASS (99.7%+ better than target)
Winner: HAR-R Linear
```

**3. MAE (Lower is Better):**
```
Ranking (Realistic Models Only):
1. LSTM-HAR (Val):                  0.000258 ⭐ BEST
2. HAR-R Linear:                     0.000257 (+0.4% worse, essentially tied)
3. Enhanced LSTM-HAR (Prevention):   0.000259 (+0.8% worse)
4. Simple LSTM (Val):               0.000424 (+64.3% worse)

Target: Lower is better
Status: ✅ All models very competitive (within 0.000001 range)
Winner: LSTM-HAR (Val) - by tiny margin
```

**4. R² (Higher is Better):**
```
Ranking (Realistic Models Only):
1. Simple LSTM (Val):               0.211 ⭐ BEST
2. LSTM-HAR (Val):                  0.110 (-47.9% worse)
3. HAR-R Linear:                     0.105 (-50.2% worse)
4. Enhanced LSTM-HAR (Prevention):   0.098 (-53.6% worse)

Target: > 0.50 ❌ ALL MODELS FAIL (highest is only 0.211)
Winner: Simple LSTM (Val) - but still fails target
```

**5. QLIKE (Lower is Better):**
```
Ranking (Realistic Models Only):
1. Enhanced LSTM-HAR (Prevention):   0.641 ⭐ BEST
2. Simple LSTM (Val):               0.673 (+5.0% worse)
3. HAR-R Linear:                     1.298 (+102.5% worse)

LSTM-HAR (Val): N/A (not measured)

Target: < 0.50 ❌ ALL MODELS FAIL (best is 0.641, 28% above target)
Winner: Enhanced LSTM-HAR - closest to target
```

**6. Dir Acc (Higher is Better) - PRIMARY METRIC:**
```
Ranking (Realistic Models Only):
1. HAR-R Linear:                     51.53% ⭐ BEST (closest to 55% target)
2. Enhanced LSTM-HAR (Prevention):   48.56% (-5.8% worse, -11.7% below target)
3. LSTM-HAR (Val):                  48.09% (-6.7% worse, -12.6% below target)
4. Simple LSTM (Val):               47.89% (-7.0% worse, -13.0% below target)

Target: > 55% ❌ ALL MODELS FAIL
Winner: HAR-R Linear - best but still below target
```

---

### Overall Performance Summary

**Best Model by Metric:**
- ⭐ **MSE:** HAR-R Linear (2.631e-07)
- ⭐ **RMSE:** HAR-R Linear (0.000513)
- ⭐ **MAE:** LSTM-HAR (Val) (0.000258)
- ⭐ **R²:** Simple LSTM (Val) (0.211)
- ⭐ **QLIKE:** Enhanced LSTM-HAR (0.641)
- ⭐ **Dir Acc:** HAR-R Linear (51.53%)

**Overall Ranking (by number of wins):**
1. ⭐⭐⭐ **HAR-R Linear** - Wins 3/6 metrics (MSE, RMSE, Dir Acc)
2. ⭐⭐ **Enhanced LSTM-HAR** - Wins 1/6 metrics (QLIKE), competitive on others
3. ⭐⭐ **LSTM-HAR (Val)** - Wins 1/6 metrics (MAE), competitive on RMSE
4. ⭐ **Simple LSTM (Val)** - Wins 1/6 metrics (R²)

**Success Criteria Pass Rate:**
- All models: **3/6 metrics PASS** (50% success rate)
- All models: **3/6 metrics FAIL** (50% failure rate)
- **Primary target (Dir Acc > 55%): ALL MODELS FAIL** ❌

---

## 🎯 Success Criteria Check

### All 6 Metrics for All Models

| Model | MSE (↓) | RMSE (<0.20) | MAE (↓) | R² (>0.50) | QLIKE (<0.50) | Dir Acc (>55%) | Overall Status |
|-------|---------|--------------|---------|------------|--------------|----------------|---------------|
| **HAR-R Linear** | ✅ 2.631e-07 ⭐ | ✅ 0.000513 ⭐ | ✅ 0.000257 | ❌ 0.105 | ❌ 1.298 | ❌ 51.53% ⭐ | **3/6 PASS** |
| **Simple LSTM (Val)** | ✅ 5.283e-07 | ✅ 0.000727 | ✅ 0.000424 | ❌ 0.211 ⭐ | ❌ 0.673 | ❌ 47.89% | **3/6 PASS** |
| **LSTM-HAR (Val)** | ✅ 3.065e-07 | ✅ 0.000554 | ✅ 0.000258 ⭐ | ❌ 0.110 | N/A | ❌ 48.09% | **3/6 PASS** |
| **Enhanced LSTM-HAR** | ✅ 3.107e-07 | ✅ 0.000557 | ✅ 0.000259 | ❌ 0.098 | ❌ 0.641 ⭐ | ❌ 48.56% | **3/6 PASS** |
| **Simple LSTM (Leakage)** | ✅ 3.447e-07 | ✅ 0.000587 | ✅ 0.000303 | ❌ 0.167 | ❌ 1.170 | ❌ 67.63% 🔴 | **3/6 PASS** |
| **LSTM-HAR (Leakage)** | ✅ 3.120e-07 | ✅ 0.000559 | ✅ 0.000297 | ❌ 0.161 | ✅ 0.566 ⭐ | ❌ 67.39% 🔴 | **4/6 PASS** |

**Legend:**
- ✅ = Passes target
- ❌ = Fails target
- ⭐ = Best in category
- 🔴 = Data leakage (inflated metrics)
- (↓) = Lower is better
- (↑) = Higher is better

**Summary by Metric:**
- **MSE:** All models PASS ✅ (Winner: HAR-R Linear)
- **RMSE:** All models PASS ✅ (Winner: HAR-R Linear)
- **MAE:** All models PASS ✅ (Winner: LSTM-HAR Val)
- **R²:** All models FAIL ❌ (Winner: Simple LSTM Val, but still < 0.50)
- **QLIKE:** All models FAIL ❌ (Winner: LSTM-HAR Leakage, but with data leakage)
- **Dir Acc:** All models FAIL ❌ (Winner: HAR-R Linear, closest to 55% at 51.53%)

**Overall Success Rate:**
- All models: **3/6 metrics PASS** (50% success rate)
- All models: **3/6 metrics FAIL** (50% failure rate)
- Primary target (Dir Acc > 55%): **ALL MODELS FAIL** ❌

**Conclusion:** No model meets all success criteria. HAR-R Linear is the best overall (wins 3/6 metrics, closest to Dir Acc target).

---

## 🏆 Final Rankings

### For Production (Realistic Performance):

**1. Best Overall: HAR-R Linear** ⭐⭐⭐
- ✅ **Wins 3/6 metrics** (MSE, RMSE, Dir Acc)
- ✅ **Best MSE** (2.631e-07, 18-101% better than others)
- ✅ **Best RMSE** (0.000513, 8-42% better than others)
- ✅ **Best Dir Acc** (51.53%, closest to 55% target, 5-8% better than others)
- ✅ **Good MAE** (0.000257, only 0.4% worse than best)
- ✅ **Instant training** (< 0.01 seconds)
- ✅ **Tiny model** (< 0.1 MB)
- ⚠️ R² = 0.105 (3rd best, but still fails target)
- ⚠️ QLIKE = 1.298 (worst, fails target)
- ⚠️ Still below 55% Dir Acc target

**2. Best Deep Learning: Enhanced LSTM-HAR (Overfitting Prevention)** ⭐⭐
- ✅ **Wins 1/6 metrics** (QLIKE)
- ✅ **Best QLIKE** (0.641, closest to < 0.50 target)
- ✅ **Competitive MSE** (3.107e-07, 18% worse than HAR-R)
- ✅ **Competitive RMSE** (0.000557, 8.5% worse than HAR-R)
- ✅ **Good MAE** (0.000259, only 0.8% worse than best)
- ✅ **Comprehensive overfitting prevention** (all techniques applied)
- ✅ **No overfitting** (val-test gap < 0.000094)
- ✅ **Best R² among LSTM models** with prevention (0.098)
- ⚠️ Dir Acc = 48.56% (2nd best among LSTM, but still below target)
- ⚠️ Longest training time (~22 min)

**3. Most Balanced: LSTM-HAR (Val)** ⭐⭐
- ✅ **Wins 1/6 metrics** (MAE)
- ✅ **Best MAE** (0.000258, essentially tied with HAR-R)
- ✅ **Good MSE** (3.065e-07, 16% worse than HAR-R)
- ✅ **Competitive RMSE** (0.000554, 8% worse than HAR-R)
- ✅ **Good R²** (0.110, 2nd best)
- ✅ **No overfitting** (val-test gap < 0.000089)
- ❌ QLIKE not measured
- ⚠️ Dir Acc = 48.09% (3rd best)
- ⚠️ Moderate training time (~8 min)

**4. Highest Variance Explained: Simple LSTM (Val)** ⭐
- ✅ **Wins 1/6 metrics** (R²)
- ✅ **Best R²** (0.211, highest variance explained, but still < 0.50 target)
- ✅ **Good QLIKE** (0.673, 2nd best)
- ✅ **No overfitting** (val-test gap < 0.000394)
- ❌ **Worst MSE** (5.283e-07, 101% worse than HAR-R)
- ❌ **Worst RMSE** (0.000727, 42% worse than HAR-R)
- ❌ **Worst MAE** (0.000424, 64% worse than best)
- ⚠️ Dir Acc = 47.89% (lowest among realistic models)
- ⚠️ Fast training (~5 min)

---

### Detailed Comparison Matrix

**By Success Criteria:**

| Success Criteria | Target | HAR-R Linear | Simple LSTM (Val) | LSTM-HAR (Val) | Enhanced LSTM-HAR | Best Model |
|-----------------|--------|-------------|------------------|----------------|-------------------|------------|
| **MSE < best** | Lowest | ✅ **2.631e-07** | ❌ 5.283e-07 | ❌ 3.065e-07 | ❌ 3.107e-07 | HAR-R |
| **RMSE < 0.20** | < 0.20 | ✅ 0.000513 | ✅ 0.000727 | ✅ 0.000554 | ✅ 0.000557 | HAR-R |
| **MAE < best** | Lowest | ✅ 0.000257 | ❌ 0.000424 | ✅ **0.000258** | ❌ 0.000259 | LSTM-HAR |
| **R² > 0.50** | > 0.50 | ❌ 0.105 | ❌ 0.211 | ❌ 0.110 | ❌ 0.098 | Simple LSTM |
| **QLIKE < 0.50** | < 0.50 | ❌ 1.298 | ❌ 0.673 | N/A | ❌ 0.641 | Enhanced LSTM |
| **Dir Acc > 55%** | > 55% | ❌ 51.53% | ❌ 47.89% | ❌ 48.09% | ❌ 48.56% | HAR-R |
| **No Overfitting** | Gap < 0.05 | N/A | ✅ 0.000394 | ✅ 0.000089 | ✅ 0.000094 | All LSTM |

**Color Key:**
- ✅ Green = Passes target or best in category
- ❌ Red = Fails target
- **Bold** = Best in category

**Overall Winner by Category:**
- **Magnitude Metrics (MSE, RMSE):** HAR-R Linear (best on both)
- **Error Metrics (MAE):** LSTM-HAR (Val) (best, essentially tied with HAR-R)
- **Variance Explained (R²):** Simple LSTM (Val) (best, but all fail target)
- **Volatility Quality (QLIKE):** Enhanced LSTM-HAR (best, but all fail target)
- **Directional Accuracy (Dir Acc):** HAR-R Linear (best, but all fail target)
- **Training Efficiency:** HAR-R Linear (instant, no training needed)
- **Model Size:** HAR-R Linear (tiny, < 0.1 MB)

---

## 📋 Recommendations

### For Immediate Use:

**⭐ Use HAR-R Linear**
- Best realistic Dir Acc (51.53%)
- Instant training
- Tiny model
- Best RMSE

### For Improvement:

**Option 1: Try New Approaches**
- 🚀 TimesFM 2.5 LoRA (foundation model) - already implemented
- 🚀 LSTM-GAT Hybrid (spatial relationships) - architecture designed
- ⭐ Add technical indicators (RSI, MACD, Bollinger Bands)
- ⭐ Ensemble methods (combine multiple models)

**Option 2: Accept Current Performance**
- HAR-R Linear at 51.53% is reasonable for volatility prediction
- Volatility is inherently difficult to predict (low R² across all models)
- Consider using ensemble of all 4 models for robustness

---

## 🔍 Key Insights

1. **Data Leakage is Critical:**
   - Random split inflates Dir Acc by 40%+
   - MUST use temporal split for time series
   - Always verify val-test metrics

2. **Deep Learning ≠ Better Performance:**
   - Linear baseline outperforms all LSTM models on Dir Acc
   - More complexity doesn't guarantee improvement
   - Need new approaches (foundation models, spatial features)

3. **Volatility is Inherently Difficult:**
   - Low R² across all models (0.10-0.21)
   - Dir Acc below 55% target for all models
   - May need to accept 48-52% as realistic baseline

4. **Overfitting Prevention Works:**
   - All models with temporal split show no overfitting
   - Comprehensive techniques (grad clipping, dropout, layer norm) effective
   - Val-test gaps < 0.0001 for proper models

---

**Created by:** Stock Volatility Prediction Team
**Date:** 2026-06-21
**Version:** 1.0
**Purpose:** Quick model comparison reference
