# Current Issues Summary - Parallel LSTM-GNN Training

**Date:** 2026-06-27
**Model:** Opus 4.6 (1M context)
**Status:** 🔄 Debugging

---

## 🔴 Current Issues

### Issue #1: Model Instability - NaN Predictions ⚠️ **HIGH PRIORITY**

**Symptoms:**
```
Batch 20/79: Loss=nan
Batch 40/79: Loss=nan
Average Train Loss: nan
predictions_norm mean: nan, std: nan
```

**Root Causes (Hypotheses):**
1. **Learning rate too high** (0.005 for complex model)
2. **Gradient explosion** in GNN layers
3. **Invalid graph operations** (sparse edges)
4. **Numerical instability** in attention mechanism

---

### Issue #2: Targets Not Normalized 🐛 **BUG - PARTIALLY FIXED**

**Symptoms:**
```
targets_norm mean: 9.424495  <- Should be ≈0.0
targets_norm std:  369.943481 <- Should be ≈1.0
targets_norm range: [-1.544472, 22237.529297]
```

**Fix Applied:**
```python
# Before (Line 860, 875, 890):
normalize=False  # ❌ Hardcoded

# After:
normalize=normalize  # ✅ Use parameter
```

**Status:** Fix applied, NEEDS VERIFICATION

---

## ✅ Fixes Already Applied

### Fix #1: PyTorch Compatibility
```python
# Removed deprecated 'verbose' parameter from ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
    # verbose=True  <- REMOVED (PyTorch 2.12+)
)
```

### Fix #2: Graph Threshold
```python
# Lower threshold to avoid empty graphs
graph_threshold = 0.1  # Was 0.3, caused 0 edges
```

### Fix #3: Normalization Parameter
```python
# Use parameter instead of hardcoded False
normalize=normalize  # Was normalize=False
```

---

## 🎯 Next Actions (Priority Order)

### 1. **Verify Normalization Fix** (5 min)
- Check if targets are now normalized (mean≈0, std≈1)
- If not, debug `__getitem__` in dataset_presplit.py

### 2. **Fix NaN Predictions** (15 min)
**Options:**
- **A. Reduce learning rate** (0.005 → 0.001 or 0.0005)
- **B. Add gradient clipping** (already exists, verify max_norm)
- **C. Add numerical stability checks** (clamp values, epsilon)
- **D. Simplify model** (reduce complexity temporarily)

### 3. **Quick Test After Fixes** (10 min)
- Run `--quick_test` (5 epochs)
- Verify: No NaN in predictions, normalized targets, finite loss

### 4. **Full Training** (2 hours)
- Run 70 epochs with stable configuration
- Compare with LSTM-HAR Enhanced (67.90% Dir Acc)

---

## 🔍 Debugging Commands

### Check Normalization Status:
```bash
cd D:/bmad-projects/stock_vol_prediction01
python -c "
from src.lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method_fixed
train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders_with_graph_method_fixed(
    data_dir='data/processed',
    graph_method='correlation',
    graph_threshold=0.1,
    normalize=True
)
x, adj, y, _ = datasets[0][0]
print(f'y mean: {y.mean():.6f}, std: {y.std():.6f}')
print(f'Normalized: {abs(y.mean()) < 0.1 and abs(y.std() - 1.0) < 0.1}')
"
```

### Test with Lower Learning Rate:
```bash
python -m src.lstm_gat_hybrid.train_parallel_enhanced \
    --graph_method correlation \
    --quick_test
# Then edit config.py: learning_rate = 0.001
```

---

## 📊 Expected Results After Fixes

**Normalization:**
```
targets_norm mean: ≈0.000 (±0.1)
targets_norm std:  ≈1.000 (±0.1)
```

**Training:**
```
Batch 20/79: Loss=0.001234  <- Finite
Average Train Loss: 0.001100
Val Loss: 0.001450         <- Comparable to test
Val DirAcc: 65-68%
```

**No NaN in predictions!**

---

**Last Updated:** 2026-06-27
**Priority:** Fix NaN → Verify Normalization → Quick Test → Full Training
