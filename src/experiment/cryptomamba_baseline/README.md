# CryptoMamba Baseline - Phase 1 Proof-of-Concept

**Status:** ✅ Ready for Training
**Phase:** 1 - Proof-of-Concept
**Objective:** Test if CryptoMamba beats LSTM baseline (48% Dir Acc)

---

## 📚 Background

**CryptoMamba** is a novel State Space Model (SSM) architecture published at **IEEE ICBC 2025** for cryptocurrency price prediction. This implementation adapts it for VN30 stock volatility forecasting.

**Key Advantages over LSTM:**
- ✅ Better long-range dependency modeling (SSM specialty)
- ✅ Input-adaptive parameters (adjusts to volatility regimes)
- ✅ More efficient (136K params vs LSTM 204K)
- ✅ Proven in volatile markets (cryptocurrency)

**Research Paper:** [arXiv:2501.01010v2](https://arxiv.org/html/2501.01010v2)
**Original Code:** [GitHub](https://github.com/MShahabSepehri/CryptoMamba)

---

## 🎯 Phase 1 Objectives

### **Primary Goal:**
Test if CryptoMamba architecture beats your LSTM baseline

### **Success Criteria:**
- ✅ **Beat LSTM:** Dir Acc > 48.32% (all LSTM variants)
- 🎯 **Approach HAR-R:** Dir Acc > 51% (stretch goal)
- ✅ **Improve RMSE:** < 0.00055 (current LSTM performance)

### **Current Baselines:**
```
LSTM Models:      48.01-48.32% Dir Acc ❌
HAR-R Baseline:   51.53% Dir Acc ✅
Target:          >48% (minimum), >51% (optimal)
```

---

## 📁 File Structure

```
src/cryptomamba_baseline/
├── __init__.py           # Package initialization
├── model.py              # CryptoMamba architecture (simplified)
├── dataset.py            # Volatility dataset (HAR features)
├── config.py             # Hyperparameters and settings
├── train.py              # Training script with 3-way temporal split
└── README.md             # This file
```

---

## 🔧 Implementation Details

### **Architecture Simplifications:**

**Original CryptoMamba:**
- Complex modular structure (C-Blocks, CMBlocks, Merge)
- Multiple sequence lengths (14, 16, 32)
- Volume-inclusive variants

**This Implementation (Phase 1):**
- ✅ Simplified single-model architecture
- ✅ Fixed sequence length: 22 days (HAR monthly)
- ✅ HAR features + optional volume
- ✅ Direct volatility prediction (not price)
- ✅ Compatible with project's evaluation framework

### **Model Configuration:**

```python
CryptoMambaVolatility(
    num_features = 5,          # HAR (3) + volume (1) + 1
    hidden_dim = 64,           # Moderate size (training efficiency)
    num_layers = 3,            # Balance capacity vs overfitting
    d_state = 16,              # Default from paper
    d_conv = 4,                # Convolution kernel
    expand = 2,                # Expansion factor
    dropout = 0.1,             # Same as LSTM (reduced regularization)
    seq_length = 22            # HAR monthly window
)
```

### **Training Configuration:**

```python
# Same as LSTM (fair comparison)
learning_rate = 0.001         # 10× faster (from learning rate fix)
weight_decay = 1e-6           # Reduced regularization
batch_size = 32
num_epochs = 100
patience = 10
min_epochs = 15               # Prevent premature stopping

# Split: 70/15/15 temporal (same as LSTM)
```

---

## 🚀 Usage

### **Quick Start:**

```bash
# 1. Ensure processed data exists
python -m src.common.process_data

# 2. Train CryptoMamba with validation
python -m src.cryptomamba_baseline.train

# 3. Check results in:
#    results/cryptomamba_val_YYYY-MM-DD_HHMMSS/cryptomamba_val_results.json
```

### **Expected Output:**

```
Phase 1 Proof-of-Concept: Test if CryptoMamba beats LSTM
LSTM Baseline: 48.01-48.32% Dir Acc
HAR-R Baseline: 51.53% Dir Acc
CryptoMamba Target: >48% (beat LSTM), approach 51% (HAR-R)

Training... (will take ~15-30 minutes on CPU, ~5-10 min on GPU)

Test Results:
  Test Dir Acc: XX.XX%
  LSTM Baseline: 48.01-48.32%
  Status: [✅ SUCCESS / ⚠️ PARTIAL / ❌ NEEDS INVESTIGATION]
```

---

## 📊 Expected Outcomes

### **Conservative Estimate:**
Based on CryptoMamba vs LSTM performance in cryptocurrency:
```
Dir Acc: 48% → 50-52% (+2-4%)
RMSE: 0.00055 → 0.00050-0.00052 (+5-9%)
QLIKE: 0.64 → 0.60-0.62 (+3-6%)
```

### **Optimistic Estimate:**
If SSM architecture suits volatility better than LSTM:
```
Dir Acc: 48% → 52-54% (+4-6%)
RMSE: 0.00055 → 0.00048-0.00050 (+9-12%)
QLIKE: 0.64 → 0.56-0.60 (+6-12%)
# Could approach or beat HAR-R!
```

---

## 🔄 Phase Progression

### **Phase 1 (Current) ✅**
**Timeline:** 1 day implementation + training time
**Goal:** Beat LSTM baseline (>48% Dir Acc)
**Action:** Train and evaluate

**Decision Criteria:**
- ✅ **Dir Acc > 48%:** Success → Proceed to Phase 2
- ⚠️ **Dir Acc 47-48%:** Partial → Tune hyperparameters
- ❌ **Dir Acc < 47%:** Failed → Investigate issues

### **Phase 2 (If Phase 1 Successful)**
**Timeline:** 3-5 days
**Goal:** Beat HAR-R baseline (>55% Dir Acc)
**Actions:**
1. HAR-Mamba integration (better features)
2. Add technical indicators
3. Fine-tune architecture
4. Comprehensive evaluation

### **Phase 3 (Optional)**
**Timeline:** 5-7 days
**Goal:** Best performance, multi-asset
**Actions:**
1. Multi-stock extension (all 30 VN30 stocks)
2. Compare with LSTM-GAT hybrid
3. Production deployment

---

## ⚙️ Configuration

### **Modify hyperparameters in `config.py`:**

```python
CRYPTOMAMBA_CONFIG = {
    # Architecture
    'hidden_dim': 64,        # Increase for more capacity
    'num_layers': 3,         # More layers = deeper model
    'dropout': 0.1,         # Reduce if overfitting

    # Training
    'learning_rate': 0.001,  # Adjust if not converging
    'weight_decay': 1e-6,    # Regularization strength
    'num_epochs': 100,
    'patience': 10,
    'min_epochs': 15,
}
```

---

## 🔍 Troubleshooting

### **Common Issues:**

**1. Import Errors:**
```bash
# Ensure mamba_ssm is installed (optional for simplified version)
# If errors occur, simplified implementation doesn't require it
```

**2. Data Loading Errors:**
```bash
# Ensure processed data exists
python -m src.common.process_data

# Check CSV files have required columns:
# - har_daily_vol, har_weekly_vol, har_monthly_vol
# - target_5d
# - volume (optional)
```

**3. CUDA Errors:**
```python
# Modify train.py to force CPU training
device = torch.device('cpu')  # Line 106
```

**4. Poor Performance:**
```python
# Try different hyperparameters:
# - Increase hidden_dim: 64 → 128
# - Increase num_layers: 3 → 4
# - Reduce dropout: 0.1 → 0.05
# - Adjust learning rate: 0.001 → 0.0005 or 0.002
```

---

## 📚 Key References

### **Papers:**
1. **CryptoMamba:** [IEEE ICBC 2025](https://ieeexplore.ieee.org/document/11114565/)
2. **Mamba Original:** [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3. **S-Mamba:** Neurocomputing 619 (2025)

### **Code:**
1. **CryptoMamba:** [GitHub](https://github.com/MShahabSepehri/CryptoMamba)
2. **MambaTS:** [GitHub](https://github.com/XiudingCai/MambaTS-pytorch)
3. **Mamba Core:** [GitHub](https://github.com/state-spaces/mamba)

### **Related Research:**
- Mamba-LSTM-Attention (MLA): Hybrid for long-term dependencies
- SAMForecast: Self-attention + Mamba
- DTMamba: Dual Twin Mamba for time series

---

## 🎓 Technical Notes

### **Why Simplified Implementation?**

**Original CryptoMamba:** ~5000 lines, complex modular structure
**This Implementation:** ~500 lines, clean and maintainable

**Rationale:**
- Phase 1 needs proof-of-concept, not production code
- Simpler = easier to debug and adapt
- Faster training and iteration
- Easier to understand and modify

### **What's Missing (Phase 2+):**

1. **Full selective scan implementation** (uses simplified SSM)
2. **Hierarchical C-Block structure** (uses flat layers)
3. **Fused CUDA kernels** (uses PyTorch operations)
4. **Complex trading algorithms** (only prediction metrics)

**These can be added in Phase 2 if PoC successful.**

---

## ✅ Checklist

Before training, ensure:
- [ ] Processed data exists in `data/processed/`
- [ ] CSV files have required columns (HAR features, target_5d)
- [ ] Project dependencies installed (`torch`, `pandas`, `sklearn`)
- [ ] Enough disk space for results (~50MB)
- [ ] GPU available (optional, but recommended)

---

## 📝 Results Interpretation

### **After Training:**

1. **Check `cryptomamba_val_results.json`**
   ```json
   {
     "test_metrics": {
       "directional_accuracy": XX.XX,
       "rmse": 0.000XXX,
       "qlike": 0.XXX
     },
     "baseline_comparison": {
       "beats_lstm": true/false,
       "beats_harr": true/false
     }
   }
   ```

2. **Compare with LSTM:**
   - ✅ Dir Acc > 48.32: Success!
   - ⚠️ Dir Acc 47-48%: Partial success
   - ❌ Dir Acc < 47%: Needs investigation

3. **Decision:**
   - **Success (✅):** Proceed to Phase 2
   - **Partial (⚠️):** Tune hyperparameters
   - **Failed (❌):** Debug implementation

---

**Implementation Date:** 2026-06-19
**Phase:** 1 - Proof-of-Concept
**Status:** ✅ Ready for Training
**Expected Duration:** 15-30 min training (CPU), 5-10 min (GPU)

---

## 🚀 Next Steps

After Phase 1 training completes:

1. **Review results** in `results/cryptomamba_val_*/`
2. **Compare with LSTM** performance
3. **Decide on Phase 2** based on success criteria
4. **If successful:** Implement HAR-Mamba integration
5. **If unsuccessful:** Tune hyperparameters or investigate issues

---

**Author:** Stock Volatility Prediction Team
**Last Updated:** 2026-06-19
**Version:** 1.0 (Phase 1 Proof-of-Concept)
