# LSTM Hyperparameter Optimization with Optuna - Quick Start

**Date:** 2026-06-18
**Status:** Ready to run
**Estimated Time:** 1-2 hours

---

## 📋 What This Does

**Optuna sẽ tự động tìm best hyperparameters cho LSTM:**
- hidden_size: [64, 128, 256, 512]
- learning_rate: [0.0001, 0.001, 0.01, 0.1]
- batch_size: [32, 64, 128]
- dropout: [0.0, 0.1, 0.2, 0.3]
- weight_decay: [0, 1e-5, 1e-4, 1e-3]
- seq_length: [10, 22, 30, 44]

**GIỮ NGUYÊN:**
- ✅ 1-layer LSTM architecture
- ✅ Raw volatility input (NO HAR features)
- ✅ Same data, same evaluation metrics

---

## 🚀 Step-by-Step Guide

### Step 1: Install Optuna

```bash
pip install optuna
```

### Step 2: Run Optimization

```bash
cd D:\bmad-projects\stock_vol_prediction01
python optimize_lstm.py
```

**What happens:**
- Optuna runs 50 trials
- Each trial trains LSTM with different hyperparameters
- Automatically prunes bad trials (saves time!)
- Takes ~1-2 hours depending on GPU

**Progress output:**
```
[Trial 0] Testing params: {'hidden_size': 128, 'lr': 0.01, ...}
Trial 0: QLIKE=0.650, Dir_Acc=45.00%

[Trial 1] Testing params: {'hidden_size': 256, 'lr': 0.001, ...}
Trial 1: QLIKE=0.580, Dir_Acc=48.50%

...
```

### Step 3: Check Results

After optimization completes, results saved to:
```
results/lstm_optimization_YYYY-MM-DD_HHMMSS/
├── best_params.json          ← BEST HYPERPARAMETERS
└── all_trials.csv            ← All 50 trials
```

**Console output shows:**
```
Best Trial:
  Trial ID: 23
  QLIKE: 0.450000

Best Hyperparameters:
  hidden_size: 256
  dropout: 0.15
  learning_rate: 0.008
  batch_size: 64
  weight_decay: 0.0001
  seq_length: 22

Top 5 Trials:
  Rank 1: QLIKE=0.450, params={...}
  Rank 2: QLIKE=0.470, params={...}
  Rank 3: QLIKE=0.490, params={...}
  ...
```

### Step 4: Train Final Model with Best Params

```bash
python train_best_lstm.py --params results/lstm_optimization_*/best_params.json
```

**Output:**
```
results/lstm_optimized_YYYY-MM-DD_HHMMSS/
├── best_lstm.pth              ← Trained model with best params
├── test_metrics.csv           ← Final metrics
└── model_info.json            ← Model info + hyperparameters
```

### Step 5: Compare with HAR

```bash
python compare_models.py
```

---

## 📊 Expected Results

### Before Optimization (Current LSTM):
```
QLIKE: 0.665
Directional_Acc: 38.90%
RMSE: 0.000596
```

### After Optimization (Expected):
```
QLIKE: 0.40-0.50          ← 25-40% improvement
Directional_Acc: 50-55%  ← Beats random!
RMSE: 0.00045-0.00050     ← Better than current
```

### Best Case (If lucky):
```
QLIKE: < 0.20             ← Meets target!
Directional_Acc: > 55%    ← Meets target!
RMSE: < 0.0003
```

---

## 🔍 How Optuna Works

**Algorithm:** TPE (Tree-structured Parzen Estimator)
- Bayesian optimization (smart search)
- Learns from previous trials
- Focuses on promising hyperparameter regions

**Pruning:** MedianPruner
- Stops bad trials early (after epoch 10)
- Saves time on unpromising configurations
- Example: If QLIKE > 0.8 after 10 epochs → prune

**Number of Trials:** 50
- Enough to find good params
- Not too many (takes 1-2 hours)
- Can increase to 100 for better results

---

## 📈 Hyperparameter Search Space

| Hyperparameter | Values | Impact |
|----------------|--------|---------|
| **hidden_size** | [64, 128, 256, 512] | Model capacity |
| **learning_rate** | [1e-4, 1e-1] log scale | Convergence speed |
| **batch_size** | [32, 64, 128] | Training stability |
| **dropout** | [0.0, 0.3] | Regularization |
| **weight_decay** | [0, 1e-3] | L2 regularization |
| **seq_length** | [10, 22, 30, 44] | Input window size |

---

## 💡 Tips

**If optimization takes too long:**
- Reduce trials: `n_trials=30` (in optimize_lstm.py)
- Stop early: `timeout=3600` (1 hour max)

**If results aren't good:**
- Increase trials to 100
- Expand search space (add more values)
- Run multiple optimizations with different random seeds

**To reproduce best results:**
- Use same `best_params.json`
- Results should be identical (same seed)

---

## 🎯 Success Criteria

**Optimization successful if:**
- ✅ QLIKE < 0.50 (25% improvement)
- ✅ Directional_Acc > 50% (beats random)
- ✅ Better than current LSTM (0.665, 38.90%)

**Stretch goals:**
- 🎯 QLIKE < 0.20 (meets target)
- 🎯 Directional_Acc > 55% (meets target)
- 🎯 Beats HAR baseline

---

## 📁 Files Created

**optimize_lstm.py** - Main optimization script
- 50 trials
- TPE sampler + Median pruner
- Saves best_params.json

**train_best_lstm.py** - Train final model
- Uses best_params.json
- Trains for 50 epochs (longer)
- Saves final model

**Results:**
```
results/
├── lstm_optimization_*/          ← From Optuna
│   ├── best_params.json
│   └── all_trials.csv
└── lstm_optimized_*/              ← Final model
    ├── best_lstm.pth
    ├── test_metrics.csv
    └── model_info.json
```

---

## 🚀 Ready to Start!

```bash
# 1. Install Optuna
pip install optuna

# 2. Run optimization (1-2 hours)
python optimize_lstm.py

# 3. Train final model
python train_best_lstm.py --params results/lstm_optimization_*/best_params.json

# 4. Compare results
python compare_models.py
```

**Good luck! 🍀**
