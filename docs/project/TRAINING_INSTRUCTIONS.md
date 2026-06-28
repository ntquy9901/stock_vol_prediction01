# Phase 3: User Training Instructions

**Date:** 2026-06-21  
**Status:** Ready for user training  
**Estimated Time:** 30 minutes (quick test) or 3-4 hours (full training)

---

## 🚀 Training Instructions

### Option 1: Quick Test (5 stocks, ~30 min)

**Purpose:** Verify architecture works before full training

```bash
# Activate your environment
# cd D:\bmad-projects\stock_vol_prediction01

# Train with correlation-based graph (from paper)
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method correlation
```

**Expected Output:**
- Training progress per epoch
- Validation metrics (Dir Acc, RMSE, etc.)
- Learning curves plots every 10 epochs
- Early stopping trigger (after ~15-40 epochs)

**Success Signs:**
- ✅ Training completes without errors
- ✅ Loss decreases over epochs
- ✅ Dir Acc > 0% (not constant predictions)
- ✅ RMSE reasonable (< 1.0)

**If Fails:**
- ❌ Constant predictions (Dir Acc = 0%)
- ❌ Loss doesn't decrease
- ❌ Training errors

---

### Option 2: Full Training (30 stocks, ~3-4 hours)

**Purpose:** Complete training on full dataset

```bash
# Train with correlation-based graph (recommended from paper)
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method correlation

# OR train with k-NN graph (current method)
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn
```

**Expected Output:**
- Results directory: `results/parallel_lstm_gnn_{graph_method}_{YYYY-MM-DD_HHMMSS}/`
- Files:
  - `best_parallel_model.pth` - Best model checkpoint
  - `training_results.json` - All metrics and configuration
  - `learning_curves_*.png` - Learning curve plots
  - Console output with per-epoch metrics

**Example Console Output:**
```
Epoch | Train Loss   | Val Loss    | Val Dir Acc | Val RMSE    | LR
------|--------------|-------------|-------------|-------------|------
    1 | 0.xxxxxx     | 0.xxxxxx    |     xx.xx%  |  0.xxxxxx   | 0.005000
    2 | 0.xxxxxx     | 0.xxxxxx    |     xx.xx%  |  0.xxxxxx   | 0.005000
   ...
```

---

## 📊 What to Provide for Evaluation

After training completes, provide:

### 1. **Results Directory Path**
```
Example: results/parallel_lstm_gnn_correlation_2026-06-21_143052/
```

### 2. **Which Graph Method You Used**
- `correlation` (from paper)
- `knn` (current method)
- Both (for comparison)

### 3. **Any Training Issues**
- Errors encountered
- Warnings observed
- Unexpected behavior

---

## 🎯 What I Will Evaluate

### 1. **Primary Metrics (6 Mandatory)**

From `training_results.json`:
- **MSE:** Mean Squared Error (lower is better)
- **RMSE:** Root Mean Squared Error (lower is better)
- **MAE:** Mean Absolute Error (lower is better)
- **R²:** Variance Explained (higher is better)
- **QLIKE:** Volatility metric (lower is better)
- **Dir Acc:** Directional Accuracy (higher is better)

### 2. **Success Criteria**

**Minimum Requirements:**
- ✅ Dir Acc > 50% (vs sequential 0.00%)
- ✅ RMSE < 0.25 (competitive with baselines)
- ✅ No constant predictions (variance > 1e-6)

**Stretch Goals:**
- ⭐ Dir Acc > 67.90% (beat LSTM-HAR Enhanced)
- ⭐ RMSE < 0.18 (beat all baselines)

### 3. **Comparison Analysis**

**Versus Baselines:**
- LSTM-HAR Enhanced: 67.90% Dir Acc
- Sequential LSTM-GAT: 0.00% Dir Acc (FAILED)
- LSTM-HAR Baseline: ~65-67% Dir Acc

**Expected Performance (based on paper):**
- Optimistic (70%): Dir Acc 55-65%, RMSE 0.18-0.22
- Realistic (20%): Dir Acc 40-55%, RMSE 0.22-0.28
- Pessimistic (10%): Dir Acc < 40%, RMSE > 0.28

### 4. **Overfitting Analysis**

**From Learning Curves:**
- Train vs Val loss gap
- Val vs Test metrics gap
- Recent trend (last 5 epochs)

**Healthy Signs:**
- Val-test gap < 0.05
- Val loss plateauing (not decreasing)
- No constant predictions

**Warning Signs:**
- Val-test gap > 0.10
- Val loss still decreasing at final epoch
- Predictions variance too low

---

## 📋 Evaluation Checklist

When you provide results, I will:

- ✅ Read `training_results.json`
- ✅ Analyze all 6 metrics
- ✅ Check prediction variance
- ✅ Evaluate overfitting (val-test gap)
- ✅ Compare with LSTM-HAR Enhanced (67.90%)
- ✅ Compare with Sequential LSTM-GAT (0.00%)
- ✅ Assess learning curves
- ✅ Provide comprehensive report

**Report will include:**
1. Executive summary
2. Metric-by-metric analysis
3. Overfitting assessment
4. Comparison with baselines
5. Success/failure determination
6. Recommendations for next steps

---

## 🔧 Troubleshooting

**If Training Errors Occur:**

**Issue 1: CUDA Out of Memory**
```bash
# Solution: Use CPU instead
# Edit src/lstm_gat_hybrid/config.py
# Change: device = 'cpu'
```

**Issue 2: Dataset Not Found**
```bash
# Ensure data is in correct location
ls data/processed/*.csv

# Should see 30 CSV files for VN30 stocks
```

**Issue 3: Module Import Errors**
```bash
# Ensure you're in project root
cd D:\bmad-projects\stock_vol_prediction01

# Try installing dependencies
pip install torch pandas numpy scikit-learn matplotlib
```

**Issue 4: Training Too Slow**
```bash
# Quick test with 5 stocks instead
# Edit src/lstm_gat_hybrid/config.py
# Change: num_stocks = 5 (temporary)
```

---

## ⏱️ Estimated Training Times

**Hardware: CPU (Intel i7 recommended)**

**Quick Test (5 stocks):**
- ~30 minutes total
- ~2-3 minutes per epoch
- ~10-15 epochs with early stopping

**Full Training (30 stocks):**
- ~3-4 hours total
- ~15-20 minutes per epoch
- ~20-40 epochs with early stopping

**Tips:**
- Start with quick test to verify setup
- Run full training overnight or during breaks
- Monitor console output for progress

---

## 📧 When Ready for Evaluation

**After training completes, provide:**

1. **Results Directory**
   ```
   Example: results/parallel_lstm_gnn_correlation_2026-06-21_143052/
   ```

2. **Graph Method Used**
   ```
   correlation or knn
   ```

3. **Any Issues**
   ```
   Training completed successfully
   OR
   Encountered error: [describe error]
   ```

**I will then:**
1. Analyze all results
2. Create comprehensive evaluation report
3. Compare with baselines
4. Provide recommendations

---

## 🎯 Expected Timeline

**Day 1 (Today):**
- You run training (30 min to 4 hours)
- I wait for results ✅

**Day 2 (Tomorrow):**
- You provide results
- I evaluate and analyze
- Create comprehensive report
- Provide recommendations

---

**Ready when you are!** 

Just provide the results directory when training completes, and I'll do the full evaluation.

**Good luck with training!** 🚀

---

**Status:** ⏸️ Waiting for user training to complete  
**Next:** Evaluation and analysis
