# Data Leakage Fix - Parallel LSTM-GNN

**Date:** 2026-06-21  
**Model:** Parallel LSTM-GNN (k-NN graph)  
**Issue:** Data leakage inflating Dir Acc from ~55% to 69.61%

---

## 🚨 Critical Issues Found

### **Issue 1: Adjacency Matrix Data Leakage** 🔴 HIGH

**Problem:**
- Graph was built from ENTIRE dataset before temporal split
- `all_volatility` contained data from all splits (train + val + test)
- Adjacency matrix was ONCE and reused for all sequences
- Training sequences saw graph patterns from test data

**Location:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`, Lines 197-242

**Original Code:**
```python
# Extract ALL volatility data
all_volatility = np.stack(all_volatility_padded, axis=1)  # [min_length, num_stocks]

# Build graph ONCE from entire dataset
if self.graph_method == 'knn':
    graph_data = {
        'volatility': all_volatility[:self.seq_length],  # ← Still from full dataset!
        'returns': all_volatility[:self.seq_length]
    }
    adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

# Use SAME adj_matrix for ALL sequences
for i in range(min_length - self.seq_length - self.forecast_horizon):
    sequences.append((x, adj_matrix, y))  # ← Same graph!
```

**Impact:**
- Estimated inflation: +10-15% Dir Acc
- Model learned from test data relationships

---

### **Issue 2: Normalization Data Leakage** 🔴 HIGH

**Problem:**
- Scalers were fit on ENTIRE dataset before temporal split
- `_initialize_normalizers()` called in `__init__` before split
- Validation/test statistics leaked into training via normalization

**Location:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`, Lines 176-186

**Original Code:**
```python
def __init__(self, ...):
    # Load data
    self.stock_data = self._load_multi_stock_data(data_dir)
    
    # Initialize AND FIT normalizers on ENTIRE dataset
    if normalize:
        self._initialize_normalizers()  # ← Fits on ALL data!

def _initialize_normalizers(self):
    for stock_name in self.stock_names:
        features = self.stock_data_with_har[stock_name][['har_daily_vol', ...]].values
        targets = self.stock_data_with_har[stock_name]['parkinson_volatility'].values
        
        # FIT on entire dataset (train + val + test)
        self.feature_normalizers[stock_name].fit(features)
        self.target_normalizers[stock_name].fit(targets.reshape(-1, 1))
```

**Impact:**
- Estimated inflation: +3-5% Dir Acc
- Test mean/std leaked into training normalization

---

## ✅ Fixes Applied

### **Fix 1: Per-Sequence Graph Construction**

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`, Lines 188-264

**New Code:**
```python
def _create_sequences(self) -> list:
    """Create sequences with PER-SEQUENCE graph construction (NO data leakage)"""
    sequences = []
    
    # Prepare volatility data
    all_volatility = np.stack(all_volatility_list, axis=1)  # [min_length, num_stocks]
    
    # Create sequences with PER-SEQUENCE graph construction
    for i in range(min_length - self.seq_length - self.forecast_horizon):
        # ====================================================================
        # BUILD GRAPH USING ONLY HISTORICAL DATA UP TO THIS POINT
        # ====================================================================
        historical_end_idx = i + self.seq_length
        historical_volatility = all_volatility[:historical_end_idx]
        
        if self.graph_method == 'knn':
            graph_data = {
                'volatility': historical_volatility[-self.seq_length:],
                'returns': historical_volatility[-self.seq_length:]
            }
            adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')
        
        # Create sequence with THIS sequence's graph
        x, y = ...
        sequences.append((x, adj_matrix, y))  # ← Different graph per sequence!
```

**Changes:**
- ✅ Graph built PER-SEQUENCE using only historical data
- ✅ For sequence at index i, only use data from [0, i+seq_length]
- ✅ No future information leaks into training
- ✅ More computationally expensive but correct

---

### **Fix 2: Training-Only Normalization**

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`, Lines 176-220, 430-492

**New Code:**

**Step 1: Only initialize scalers (don't fit)**
```python
def __init__(self, ...):
    # Load data
    self.stock_data = self._load_multi_stock_data(data_dir)
    
    # Initialize normalizers (DO NOT fit yet)
    if normalize:
        self._initialize_normalizers(fit=False)  # ← Only initialize!
```

**Step 2: Add method to fit on subset**
```python
def fit_normalizers_on_subset(self, subset_indices):
    """
    Fit normalizers on a subset of data (e.g., training data only)
    
    Args:
        subset_indices: List of indices to fit normalizers on
    """
    for stock_idx, stock_name in enumerate(self.stock_names):
        # Collect features and targets from subset
        subset_features = []
        subset_targets = []
        
        for idx in subset_indices:
            x, adj_matrix, y = self.sequences[idx]
            subset_features.append(x[:, stock_idx, :])
            subset_targets.append(y[stock_idx])
        
        # Fit normalizers on SUBSET only
        subset_features = np.concatenate(subset_features, axis=0)
        subset_targets = np.array(subset_targets)
        
        self.feature_normalizers[stock_name].fit(subset_features)
        self.target_normalizers[stock_name].fit(subset_targets.reshape(-1, 1))
```

**Step 3: Fit on training data only during dataloader creation**
```python
def create_multi_stock_dataloaders_with_graph_method(...):
    # Calculate split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Create datasets
    train_dataset = MultiStockDatasetWithGraphMethod(...)
    val_dataset = MultiStockDatasetWithGraphMethod(...)
    test_dataset = MultiStockDatasetWithGraphMethod(...)
    
    # ====================================================================
    # CRITICAL FIX: Fit normalizers on TRAINING data only
    # ====================================================================
    if normalize:
        # Fit on training data
        train_indices = list(range(0, train_end))
        train_dataset.fit_normalizers_on_subset(train_indices)
        
        # Copy fitted normalizers to val/test
        val_dataset.feature_normalizers = train_dataset.feature_normalizers
        val_dataset.target_normalizers = train_dataset.target_normalizers
        test_dataset.feature_normalizers = train_dataset.feature_normalizers
        test_dataset.target_normalizers = train_dataset.target_normalizers
    
    # Apply temporal split
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(val_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(test_dataset, range(val_end, n))
```

**Changes:**
- ✅ Scalers initialized but NOT fit during `__init__`
- ✅ Scalers fit ONLY on training data after temporal split
- ✅ Fitted scalers copied to validation and test datasets
- ✅ No test statistics leak into training

---

## 📊 Expected Impact

### **Performance Impact:**

| Metric | Before (with leakage) | After (estimated) | Change |
|--------|----------------------|-------------------|--------|
| **Dir Acc** | 69.61% | **54-56%** | -13-15% |
| **R²** | 0.711 | **0.55-0.60** | -15-25% |
| **RMSE** | 0.002650 | **0.001800** | Improvement |
| **MSE** | 7.024e-06 | **3.500e-06** | Improvement |

**Reasoning:**
- Remove adjacency leakage: -10-12% Dir Acc
- Remove normalization leakage: -3-5% Dir Acc
- Total: -13-17% Dir Acc
- **Estimated realistic Dir Acc: 54-56%**

---

### **Comparison with Baselines:**

| Model | Dir Acc | Status |
|-------|---------|--------|
| **Parallel LSTM-GNN (fixed)** | **54-56% (est)** | ✅ Still beats baseline! |
| Enhanced LSTM-HAR | 48.56% | ✅ Robust baseline |
| HAR-R Linear | 51.53% | ✅ Best simple model |

**Conclusion:** Even with fixes, Parallel LSTM-GNN should still beat Enhanced LSTM-HAR (54-56% vs 48.56%)!

---

## 🧪 Testing Plan

### **Step 1: Verify Data Leakage Fix**

```python
# Test 1: Verify per-sequence graphs
dataset = MultiStockDatasetWithGraphMethod(...)

# Check that graphs are different across sequences
graphs = []
for i in range(min(10, len(dataset))):
    x, adj_matrix, y = dataset[i]
    graphs.append(adj_matrix)

# Verify not all same
for i in range(1, len(graphs)):
    assert not torch.allclose(graphs[0], graphs[i]), "Graphs should be different!"

print("✅ Test 1 passed: Graphs are per-sequence")
```

### **Step 2: Verify Normalization Fix**

```python
# Test 2: Verify scalers fit on training only
train_loader, val_loader, test_loader, _ = create_multi_stock_dataloaders_with_graph_method(...)

# Check that val and test use train scalers
train_dataset = train_loader.dataset.dataset
val_dataset = val_loader.dataset.dataset
test_dataset = test_loader.dataset.dataset

# Verify scalers are shared
for stock in train_dataset.stock_names:
    assert train_dataset.feature_normalizers[stock].mean_ == \
           val_dataset.feature_normalizers[stock].mean_, \
           "Val should use train scalers!"
    assert train_dataset.feature_normalizers[stock].mean_ == \
           test_dataset.feature_normalizers[stock].mean_, \
           "Test should use train scalers!"

print("✅ Test 2 passed: Val/test use train scalers")
```

### **Step 3: Re-train Model**

```bash
# Re-train with fixed pipeline
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn
```

**Expected Results:**
- Dir Acc: 54-56% (down from 69.61%)
- R²: 0.55-0.60 (down from 0.711)
- RMSE: Improved (lower than before)
- Training time: Longer (per-sequence graph construction)

---

## 📋 Checklist

### **Code Changes:**
- [x] Fix adjacency matrix construction (per-sequence)
- [x] Fix normalization (training-only fit)
- [x] Update `_create_sequences()` method
- [x] Add `fit_normalizers_on_subset()` method
- [x] Update dataloader creation function

### **Testing:**
- [ ] Verify per-sequence graphs are different
- [ ] Verify scalers fit on training only
- [ ] Run quick test (5 epochs)
- [ ] Re-train full model (50 epochs)

### **Validation:**
- [ ] Check val-test gap is reasonable (<0.05)
- [ ] Verify Dir Acc is realistic (54-56%)
- [ ] Compare with Enhanced LSTM-HAR baseline
- [ ] Document new results

---

## 🚀 Next Steps

### **Immediate:**
1. ✅ **Apply fixes** - COMPLETED
2. 🔲 **Run tests** - Verify fixes work correctly
3. 🔲 **Quick test** - 5 epochs to verify pipeline
4. 🔲 **Full re-train** - 50 epochs with fixed pipeline

### **Short-term:**
1. 🔲 **Compare results** - Fixed vs inflated
2. 🔲 **Update documentation** - Add data leakage warning
3. 🔲 **Update report** - Correct performance numbers
4. 🔲 **Best practices doc** - How to avoid data leakage

---

## 📚 Lessons Learned

### **Data Leakage Patterns:**

1. **Graph Construction Leakage:**
   - ❌ Build graph from entire dataset
   - ✅ Build graph per-sequence from historical data

2. **Normalization Leakage:**
   - ❌ Fit scalers on entire dataset
   - ✅ Fit scalers on training data only

3. **Feature Engineering Leakage:**
   - ❌ Compute features using all data
   - ✅ Compute features per-sequence

### **Detection Signs:**

- ⚠️ Dir Acc > 10% better than robust baseline
- ⚠️ R² > 0.5 for volatility prediction (very hard)
- ⚠️ No val-test gap reported
- ⚠️ Graph/normalization happens before split

### **Prevention:**

1. ✅ Always split data FIRST (temporal for time series)
2. ✅ Fit preprocessing on TRAINING data only
3. ✅ Apply preprocessing to val/test using training parameters
4. ✅ Build dynamic features per-sequence (not once)
5. ✅ Report val-test gap to detect leakage

---

**Created:** 2026-06-21  
**Status:** ✅ Fixes applied, ready for testing  
**Next:** Re-train model with corrected pipeline
