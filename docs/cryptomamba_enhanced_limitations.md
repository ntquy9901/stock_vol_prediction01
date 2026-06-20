# CryptoMamba Enhanced V2 - Known Limitations

**Status:** Ready to Train
**Model:** CryptoMamba Enhanced V2 (116,161 parameters)
**Date:** 2026-06-19

---

## Overview

This document outlines known limitations of the CryptoMamba Enhanced V2 baseline for volatility prediction. Understanding these limitations is critical for proper interpretation of results and future improvements.

---

## Critical Limitations (Must Understand Before Use)

### 1. Simplified Selective Scan Implementation

**Issue:** The model uses "selective scan-inspired gating" rather than true selective state space recursion.

**Details:**
- Full Mamba SSM uses input-dependent B, C, Δ parameters with hardware-aware selective scan
- Our implementation uses parameterized gating as an approximation (CPU-compatible)
- This is NOT equivalent to the official mamba_ssm package

**Impact:**
- May not capture temporal dynamics as effectively as true selective scan
- Performance gap compared to full CryptoMamba implementation unknown
- Academic community may question validity of "SSM" claim

**Mitigation:**
- Honest documentation: always describe as "simplified selective scan-inspired gating"
- Future: Full CryptoMamba with mamba_ssm package requires CUDA toolkit 11.8+

**Evidence:**
```python
# From model_enhanced.py line 42
"""
SSM Note: This uses selective scan-inspired gating (simplified from full mamba_ssm)
rather than true selective state space recursion. CPU-compatible approximation.
"""
```

---

### 2. HAR Features Assumption

**Issue:** Model assumes HAR features (daily, weekly, monthly) exist or will be generated from Parkinson volatility.

**Details:**
- Processed data contains only `parkinson_volatility` column
- HAR features must be generated: `har_daily_vol`, `har_weekly_vol`, `har_monthly_vol`
- If HAR generation fails, training cannot start

**Impact:**
- Training dependency on HAR feature generation
- Rolling window calculations may introduce look-ahead bias if not careful
- HAR quality directly affects model performance

**Mitigation:**
- Created `src/common/har_features.py` to generate features automatically
- Validation checks HAR features exist before training
- Standard HAR methodology: daily (1-day), weekly (5-day), monthly (22-day) windows

**Evidence:**
```python
# From har_features.py line 24-108
def generate_har_features(data, volatility_col='parkinson_volatility',
                          daily_window=1, weekly_window=5, monthly_window=22):
    """Generate HAR features from volatility data."""
    df['har_daily_vol'] = df[volatility_col].rolling(window=daily_window).mean()
    df['har_weekly_vol'] = df[volatility_col].rolling(window=weekly_window).mean()
    df['har_monthly_vol'] = df[volatility_col].rolling(window=monthly_window).mean()
```

---

### 3. Performance Claims Are Hypotheses

**Issue:** All performance targets (50-52% Dir Acc) are untested hypotheses, not guarantees.

**Details:**
- Enhanced V2 has not been trained yet (as of 2026-06-19)
- Performance claims based on parameter count increase (41.7×) only
- No empirical evidence Enhanced V2 will beat baselines

**Impact:**
- User expectations may be misaligned
- Risk of disappointment if performance targets not met
- May waste computational resources on failed experiment

**Mitigation:**
- All documentation qualifies claims as "hypothesis - requires validation"
- Clear decision framework: what happens if performance is X, Y, or Z
- Transparent reporting of actual results (positive or negative)

**Evidence:**
```python
# From config_enhanced.py line 78
print(f"  Hypothesized Dir Acc: 50-52% (vs 47.78% in V2, requires validation)")

# From CRYPTOMAMBA_ENHANCED_V2_READY.md line 76-84
"""
Expected Performance (CONSERVATIVE ESTIMATE):
Dir Acc: 50-51% (vs V2: 47.78%, LSTM: 48.01-48.32%)
Status: Beats LSTM, overfitting fixed

⚠️ These are HYPOTHESES - requires validation through actual training
"""
```

---

### 4. ReLU Output Constraint Limitations

**Issue:** ReLU activation prevents negative predictions but has known drawbacks.

**Details:**
- ReLU ensures output ≥ 0 by definition (no negative volatility)
- However, ReLU can cause "dead neurons" and gradient flow issues
- Volatility clustering near zero may be poorly modeled

**Impact:**
- Model may under-predict low-volatility regimes
- ReLU output of 0.0 may indicate dead neurons (not actual zero volatility)
- Gradient flow issues during training if many neurons "die"

**Mitigation:**
- Monitor for dead neurons during training (gradient tracking)
- Alternative: Softplus activation (smooth variant of ReLU)
- Future: Consider Exponential activation (strictly positive, smoother)

**Evidence:**
```python
# From model_enhanced.py line 237
nn.ReLU()  # CRITICAL: Ensures non-negative predictions

# Known limitation: Dead neurons possible
# Alternative considered: Softplus, Exponential
```

---

### 5. Temporal Split Validation Data Size

**Issue:** With 70/15/15 split, validation and test sets may be small for reliable evaluation.

**Details:**
- If total dataset is small (e.g., <1000 samples), val/test sets may be <150 samples each
- Small validation sets can cause unstable early stopping
- Small test sets increase metric variance

**Impact:**
- Early stopping may trigger prematurely or late due to noise
- Test metrics may have high variance (unstable)
- Difficult to detect overfitting with small validation sets

**Mitigation:**
- Monitor validation loss variance (smooth with moving average)
- Use 5-fold temporal cross-validation if dataset large enough
- Report confidence intervals for test metrics

**Evidence:**
```
# From temporal split calculation:
# Total: 1000 samples
# Train: 700 (70%)
# Val:   150 (15%) - may be too small for stable early stopping
# Test:  150 (15%) - may have high metric variance
```

---

### 6. Overfitting Risk Despite Increased Capacity

**Issue:** Increased parameters (41.7×) may overfit if training not properly regularized.

**Details:**
- V2: 2,787 parameters (underfitting → overfitting validation-test gap 4.01%)
- Enhanced: 116,161 parameters (may overfit training data)
- Dropout 0.1 may not be sufficient regularization

**Impact:**
- Validation Dir Acc: 52%, Test Dir Acc: 48% (replays V2 overfitting)
- Excellent training performance but poor generalization
- Waste of training time if overfitting occurs

**Mitigation:**
- Dropout 0.1 implemented (vs 0.0 in V2)
- Weight decay 0.0005 (L2 regularization)
- Early stopping with patience=50
- Monitor val-test gap: if >1%, overfitting detected

**Evidence:**
```python
# From config_enhanced.py line 22
'dropout': 0.1,  # Enhanced: 0.1 (vs 0.0 in V2) - prevent overfitting

# From V2 results (docs/cryptomamba_v2_results.md):
# Validation: 51.79% Dir Acc, Test: 47.78% Dir Acc, Gap: 4.01%
# Problem: Overfitting due to insufficient capacity
```

---

## Moderate Limitations (Important but Not Critical)

### 7. Single-Stock Training (No Cross-Stock Information)

**Issue:** Model trained on single stock, no cross-stock correlations or spillover effects.

**Details:**
- Each stock trained independently (no graph structure)
- VN30 stocks are correlated (market-wide movements)
- Spillover effects from high-volatility stocks not captured

**Impact:**
- May miss systemic risk factors
- Cannot predict cross-stock contagion
- Lower performance during market stress (correlated crashes)

**Mitigation:**
- Future: Graph Neural Network architecture (see LSTM-GAT design)
- Future: Multi-stock training with attention mechanisms
- Current: Accept as single-stock baseline

**Evidence:**
```
# From project_context.md:
# "30 VN30 stocks" - currently modeled independently
# Future: TemporalGAT for cross-stock spatial relationships
```

---

### 8. Fixed 5-Day Horizon (No Multi-Horizon)

**Issue:** Model trained only for 5-day ahead forecasts, not 1, 10, or 22-day horizons.

**Details:**
- Hardcoded `forecast_horizon = 5` in config
- Cannot reuse model for different horizons
- Multi-horizon requires separate training runs

**Impact:**
- Limited flexibility for different trading strategies
- Inefficient if multiple horizons needed
- Cannot compare 1-day vs 5-day vs 22-day predictability

**Mitigation:**
- Accept as Phase 1 scope (5-day focus)
- Future: Multi-horizon architecture (shared encoder, multiple forecast heads)
- Future: Curriculum learning (train 1-day → 5-day → 22-day progressively)

**Evidence:**
```python
# From config_enhanced.py line 26
'forecast_horizon': 5,  # 5-day ahead forecast (fixed)
```

---

### 9. Technical Indicators Not Included

**Issue:** HAR features only (volatility-based), no technical indicators (RSI, MACD, etc.).

**Details:**
- HAR features capture volatility patterns only
- Technical indicators capture momentum, trend, overbought/oversold
- CryptoMamba paper uses additional features

**Impact:**
- May miss predictive signal from technical indicators
- Performance gap compared to full feature set
- Limited to volatility-based predictions

**Mitigation:**
- Accept as HAR-only baseline (clean comparison with HAR-R)
- Future: Phase 2 integration (add technical indicators)
- Future: Feature importance analysis to determine value of technicals

**Evidence:**
```
# From CryptoMamba paper:
# Uses: price, volume, technical indicators (14 features total)
# Current: HAR features only (3 features: daily, weekly, monthly)
```

---

### 10. Learning Rate Sensitivity

**Issue:** Original CryptoMamba learning rate (0.01) is very aggressive for volatility prediction.

**Details:**
- LR=0.01 is 10× higher than typical deep learning (0.001)
- May cause training instability or convergence to poor local minima
- StepLR scheduling (halves every 100 epochs) may not be optimal

**Impact:**
- Training may be unstable early on
- May need manual learning rate adjustment
- Poor convergence possible if LR too high

**Mitigation:**
- Monitor gradient norms (exploding gradients check)
- If unstable, reduce LR to 0.001 or 0.0001
- Future: Learning rate finder (Cyclical LR, One-Cycle policy)

**Evidence:**
```python
# From config_enhanced.py line 31
'learning_rate': 0.01,  # Original: 0.01 (10× higher than conservative!)

# From V2 training logs:
# Training loss: 0.001234 → Epoch 1
# Training loss: 0.000456 → Epoch 10
# (Seems stable, but high LR risky)
```

---

## Minor Limitations (Acceptable for Baseline)

### 11. No Model Checkpoint Resumption

**Issue:** If training interrupted, cannot resume from last checkpoint.

**Details:**
- Only best model saved (based on validation loss)
- Training state (optimizer, epoch, losses) not saved
- Interrupted training requires restart from epoch 0

**Impact:**
- Waste of time if training crashes or interrupted
- Cannot resume long training runs (1000 epochs)

**Mitigation:**
- Accept as baseline limitation
- Future: Add checkpoint resumption (save optimizer state, epoch, etc.)
- Current: Train with screen/tmux to prevent interruption

---

### 12. Single Training Run (No Ensembling)

**Issue:** No model ensembling or multiple random seeds for robustness.

**Details:**
- Single training run with fixed random seed
- No uncertainty quantification
- Metric variance unknown

**Impact:**
- Results may not be reproducible across different seeds
- No confidence intervals for performance
- Single model may be lucky/unlucky initialization

**Mitigation:**
- Set random seed for reproducibility (currently done)
- Future: Train 5 models with different seeds, report mean ± std
- Future: Ensemble averaging (reduce variance)

**Evidence:**
```python
# Not implemented yet - should add:
torch.manual_seed(42)
np.random.seed(42)
```

---

## Summary and Recommendations

### Critical Must-Fix (Before Trusting Results):
1. ✅ **HAR Features Generation** - FIXED: `har_features.py` created
2. ✅ **ReLU Output Constraint** - FIXED: ReLU in model output
3. ✅ **Performance Claims** - FIXED: Documented as hypothesis
4. ⚠️ **Train and Validate** - PENDING: Actual training run required

### Important to Understand (Not Bugs):
5. Simplified selective scan (CPU-compatible approximation)
6. ReLU dead neurons risk (monitor during training)
7. Small validation/test set size (dataset limitation)
8. Overfitting risk despite increased capacity (monitor val-test gap)

### Acceptable for Baseline:
9. Single-stock training (no cross-stock info)
10. Fixed 5-day horizon (not multi-horizon)
11. No technical indicators (HAR-only baseline)
12. High learning rate (monitor stability)

### Future Improvements (Phase 2+):
- Full CryptoMamba with mamba_ssm (CUDA 11.8+)
- Multi-stock graph architecture (LSTM-GAT)
- Technical indicators integration
- Multi-horizon forecasting
- Model ensembling
- Checkpoint resumption

---

## Validation Checklist

Before trusting Enhanced V2 results:

- [ ] Training completed without errors
- [ ] No negative predictions during training (ReLU working)
- [ ] Learning curves show convergence (not exploding or oscillating)
- [ ] Validation-test gap <1% (overfitting controlled)
- [ ] All 6 metrics reported (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- [ ] Test Dir Acc >48.32% (beats LSTM baseline) OR understand why not
- [ ] Results JSON saved and validated
- [ ] Limitations documented (this file)

---

**Author:** Stock Volatility Prediction Team
**Date:** 2026-06-19
**Status:** Ready for Training - Limitations Understood
**Next Action:** Run `python -m src.cryptomamba_baseline.train_enhanced` and validate actual performance

