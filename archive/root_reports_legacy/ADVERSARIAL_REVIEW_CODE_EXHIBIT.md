# ADVERSARIAL REVIEW - CODE EXHIBIT

## 11 Suspicious Code Points from Data Leakage Implementation

---

### **Point 1: Graph Construction Claims "Different" But Only 0.000002 Difference**

**Location:** `src/lstm_gat_hybrid/dataset_presplit.py` lines 100-118

```python
for i in range(min_length - self.seq_length - self.forecast_horizon):
    # Build graph using ONLY this sequence's data window
    sequence_volatility = all_volatility[i:i+self.seq_length]  # Line 102

    if self.graph_method == 'correlation':
        from .graph_correlation import construct_correlation_graph
        adj_matrix = construct_correlation_graph(
            sequence_volatility,
            corr_threshold=self.graph_threshold
        )
    elif self.graph_method == 'knn':
        graph_data = {
            'volatility': sequence_volatility,
            'returns': sequence_volatility  # ← SAME DATA USED TWICE
        }
        adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')
```

**Issue:** Using identical volatility for both 'volatility' and 'returns' fields (line 112-113). Graphs may be nearly identical because inputs are the same. The 0.000002 difference is likely floating-point rounding error.

---

### **Point 2: Hiding Warnings Instead of Fixing Root Cause**

**Location:** `run_quick_test_no_warnings.py` lines 7-11

```python
# Suppress ALL numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Also suppress scipy warnings if any
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
```

**Issue:** Just silences divide-by-zero warnings in correlation calculation. Should handle zero variance explicitly:

```python
# CORRECT approach would be:
if np.std(series_i) == 0 or np.std(series_j) == 0:
    correlation = 0.0
else:
    correlation = np.corrcoef(series_i, series_j)[0, 1]
```

---

### **Point 3: Common Stocks Filter Removes Critical Information**

**Location:** `dataset_with_graph_method.py` lines 817-831

```python
# Get common stock names across all splits
common_stocks = set(train_har.keys()) & set(val_har.keys()) & set(test_har.keys())
common_stocks = sorted(common_stocks)

# Filter HAR data to only common stocks
train_har_common = {k: v for k, v in train_har.items() if k in common_stocks}
val_har_common = {k: v for k, v in val_har.items() if k in common_stocks}
test_har_common = {k: v for k, v in test_har.items() if k in common_stocks}
```

**Issue:** If stock "ABC" has data in train but insufficient data in val/test, it gets dropped ENTIRELY from all three splits. This artificially constrains the dataset. You're training on 32 stocks instead of 30+ that might be available.

**Better approach:**
```python
# Keep stocks that exist in train + at least one of val/test
train_stocks = set(train_har.keys())
val_stocks = set(val_har.keys())
test_stocks = set(test_har.keys())

valid_stocks = train_stocks & (val_stocks | test_stocks)  # Union of val/test
```

---

### **Point 4: HAR Features Mean Difference Doesn't Prove Correct Computation**

**Location:** `_generate_har_for_split` function, lines 679-708

```python
def _generate_har_for_split(
    raw_split: Dict[str, pd.DataFrame],
    split_name: str
) -> Dict[str, pd.DataFrame]:
    """Generate HAR features for a single split"""
    split_with_har = {}

    for stock_name, df in raw_split.items():
        df_copy = df.copy()
        df_har = generate_har_features(df_copy)  # ← Still uses rolling windows
        df_har['parkinson_volatility'] = df_copy['parkinson_volatility'].values
        split_with_har[stock_name] = df_har

    return split_with_har
```

**Issue:** `generate_har_features(df_copy)` still uses `df.rolling(window=22).mean()` on the split data. But rolling means look BACKWARDS, not forwards. The difference in means (0.015 vs 0.125) could be because:

1. Train period: 2012-2020 (low volatility regime)
2. Test period: 2020-2026 (high volatility regime - COVID, inflation)

This is NOT evidence of correct HAR computation - just evidence of different time periods having different volatility characteristics.

---

### **Point 5: No Temporal Verification**

**Missing verification code:**

```python
# NO CODE EXISTS to verify:
assert train_dataset.sequences[0][0].shape[0] == seq_length, "Sequence length wrong"
# OR verify date ranges:
train_first_date = train_dataset.stock_data_with_har['ACB']['date'].iloc[0]
train_last_date = train_dataset.stock_data_with_har['ACB']['date'].iloc[-1]
val_first_date = val_dataset.stock_data_with_har['ACB']['date'].iloc[0]
assert train_last_date < val_first_date, "Temporal split violated!"
```

**What actually exists:** Just checking sequence counts, not actual dates.

---

### **Point 6: Two Functions Exist - Naming Confusion**

**Location:** `dataset_with_graph_method.py` lines 365 and 729

```python
# Line 365: OLD function (with bugs)
def create_multi_stock_dataloaders_with_graph_method(...):
    # Creates ONE dataset then splits (WRONG)
    full_dataset = MultiStockDatasetWithGraphMethod(data_dir=data_dir, ...)
    train_dataset = Subset(full_dataset, range(0, train_end))

# Line 729: NEW function (with fixes)
def create_multi_stock_dataloaders_with_graph_method_fixed(...):
    # Splits raw data FIRST (CORRECT)
    train_raw, val_raw, test_raw = _split_raw_data_by_date(...)
```

**Issue:** Both functions exist in same file. Training script calls the "fixed" version, but other code might call the old version. No deprecation warning on the old function.

---

### **Point 7: Paper Comparison Is Apples-to-Oranges**

**Claim made:**
```
Paper MSE: 0.00144
Current MSE: 0.000007
"200x better!"
```

**Reality:**
- Paper: 10 crypto stocks, different volatility measure
- Current: 30 VN30 stocks, Parkinson volatility
- Different data scales → MSE not comparable

**What's missing:** Paper doesn't report Dir Acc, so no way to know if 68% is realistic.

---

### **Point 8: Val-Test Gap Analysis**

**Claim:** "Val-test gap of 1.18% is acceptable"

**Missing analysis:**
```python
# NO CODE to check if val and test have similar volatility regimes:
val_volatility_std = np.std([seq[2].std() for seq in val_dataset.sequences])
test_volatility_std = np.std([seq[2].std() for seq in test_dataset.sequences])

# If val_std ≈ test_std, they're similar regimes
# If gap is small but regimes are different, something's wrong
```

**The gap being small** could mean:
1. Model generalizes well (GOOD)
2. Val and test both benefit from same leakage (BAD)
3. Val and test have similar volatility patterns by chance (MISLEADING)

---

### **Point 9: Normalization Re-Initialization Is Redundant**

**Location:** `dataset_with_graph_method.py` lines 895-902

```python
# Step 1: Initialize normalizers (lines 896-902)
for stock_name in common_stocks:
    train_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
    train_dataset.target_normalizers[stock_name] = VolatilityNormalizer()
    # ... (12 normalizer initializations total)

# Step 2: Re-initialize again in loop (lines 904-927)
for stock_idx, stock_name in enumerate(train_dataset.stock_names):
    # ... then fit them

# Why initialize twice? The second loop could just fit without re-initializing.
```

**Better approach:**
```python
# Initialize once in MultiStockDatasetWithPreSplitData.__init__
# Then just fit in the loop
```

---

### **Point 10: No Verification of Time-Based Isolation**

**What's missing:**

```python
# NO CODE to verify that sequence at index i uses time range [i:i+seq_length]
# Should check:
for i in range(3):  # Sample a few sequences
    x, adj, y, _ = train_dataset[i]

    # Verify x comes from correct time range
    expected_start_idx = i
    expected_end_idx = i + seq_length

    # Get actual dates from HAR data
    actual_dates = train_dataset.stock_data_with_har['ACB']['date'].iloc[expected_start_idx:expected_end_idx]

    # Verify no data leakage to future
    assert actual_dates.max() < val_dataset.stock_data_with_har['ACB']['date'].min()
```

---

### **Point 11: Performance Anomaly Not Investigated**

**Expected:** Dir Acc 54-56% (10-12% drop from 67.88%)
**Actual:** Dir Acc 68.02% (no drop)

**What should happen but doesn't:**

```python
# NO CODE exists to investigate:
print(f"Expected Dir Acc drop: {67.88 - 54:.2f}%")
print(f"Actual Dir Acc drop: {67.88 - 68.02:.2f}%")
print(f"Anomaly: Drop is {-0.20}% instead of expected -13.88%")

if abs(actual_drop - expected_drop) > 5.0:
    print("WARNING: Performance didn't drop as expected!")
    print("Possible causes:")
    print("1. Fixes not actually applied")
    print("2. Additional leakage source exists")
    print("3. Expected performance was wrong")
```

---

### **Point 12: Pandas DataFrame Index Alignment**

**Location:** `dataset_presplit.py` line 125

```python
x_seq = stock_feats[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values
```

**Issue:** Uses `.iloc[i:i+self.seq_length]` which relies on DataFrame having correct positional index. But after splitting:

```python
val_raw[stock_name] = df.iloc[train_end_idx:val_end_idx].copy()  # Line 668
```

The copied DataFrame starts at index 0, NOT at train_end_idx. So when you do `.iloc[i]`, you're accessing relative to split start, not absolute time position.

**Example:**
- Train ends at index 891
- Val starts at index 891 in original data
- But `val_raw['ACB']` has index 0, 1, 2, ... (reset after `.copy()`)
- When you do `val_dataset.sequences[0]`, it uses `.iloc[0:22]`
- This is correct relative to val split, but NO VERIFICATION that this matches expected absolute time range

---

## SUMMARY

**Core pattern:** Code looks correct on paper but has subtle issues:
1. Duplicate inputs (volatility == returns)
2. No temporal verification
3. Index confusion (relative vs absolute)
4. Performance anomaly not investigated
5. Functions duplicated instead of replaced

**Recommendation:** Add comprehensive verification tests before accepting 68% as "realistic."
