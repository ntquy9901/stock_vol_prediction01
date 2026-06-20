# TimesFM 2.5 Architecture for Parkinson Volatility Forecasting

**Author:** Stock Volatility Prediction Team
**Date:** 2026-06-20
**Model:** TimesFM 2.5 (Time Series Foundation Model)
**Method:** LoRA (Low-Rank Adaptation) Fine-Tuning

---

## Table of Contents

1. [Overview](#overview)
2. [TimesFM 2.5 Architecture](#timesfm-25-architecture)
3. [LoRA Fine-Tuning](#lora-fine-tuning)
4. [Volatility Forecasting Adaptation](#volatility-forecasting-adaptation)
5. [Hyperparameter Justification](#hyperparameter-justification)
6. [Performance Characteristics](#performance-characteristics)
7. [Comparison with Alternatives](#comparison-with-alternatives)
8. [References](#references)

---

## Overview

**TimesFM 2.5** is a decoder-only transformer architecture pre-trained on 100 billion time points across diverse datasets. It serves as a **foundation model** for time series forecasting, similar to how GPT serves as a foundation model for text.

**Our Application:** Fine-tune TimesFM 2.5 with LoRA for forecasting Parkinson volatility of VN30 stocks (5-day ahead forecasts).

**Key Innovation:** Parameter-efficient fine-tuning adapts the general-purpose time series model to financial volatility patterns without retraining all 232M parameters.

---

## TimesFM 2.5 Architecture

### Decoder-Only Transformer

TimesFM 2.5 uses a **decoder-only transformer** architecture (similar to GPT), adapted for time series:

```
Input: Time Series (univariate)
    ↓
Patch Creation (32-point patches)
    ↓
Patch Embedding + Positional Encoding
    ↓
20 × Transformer Decoder Layers
    ├─ Multi-Head Self-Attention (16 heads)
    ├─ Rotary Position Embeddings (RoPE)
    ├─ QK Normalization
    ├─ Per-Dimension Attention Scaling
    └─ Swish Activation
    ↓
Continuous Quantile Head
    ↓
Output: Probabilistic Forecast (9 quantiles)
```

### Key Architectural Components

**1. Input Processing**
- **Patching**: Input time series divided into 32-point patches (similar to tokens in LLMs)
- **Patch Embedding**: Each patch embedded into 1280-dimensional vector
- **Positional Encoding**: Rotary position embeddings (RoPE) for temporal awareness

**2. Transformer Decoder**
- **Layers**: 20 decoder layers
- **Hidden Size**: 1280 dimensions
- **Attention Heads**: 16 heads (head dimension = 80)
- **Normalization**: QK normalization for stable attention
- **Activation**: Swish (not ReLU/GELU)

**3. Output Head**
- **Continuous Quantile Function**: Predicts 9 quantiles (0.1, 0.2, ..., 0.9)
- **Point Forecast**: Median (0.5 quantile) used as deterministic forecast
- **Probabilistic Forecast**: Full quantile distribution for uncertainty quantification

### Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Parameters** | 232M (reduced from 500M in v2.0) | Total model parameters |
| **Context Length** | 16,384 (increased from 2,048) | Maximum input sequence length |
| **Patch Length** | 32 | Size of each input patch |
| **Horizon Length** | 128 | Maximum forecast horizon |
| **Hidden Size** | 1,280 | Dimension of hidden representations |
| **Layers** | 20 | Number of transformer decoder layers |
| **Attention Heads** | 16 | Number of attention heads per layer |

### Pre-Training

**Training Data:** 100 billion time points from:
- Real-world datasets (finance, weather, traffic, health, etc.)
- Synthetic datasets (augmented real data)
- Diverse domains and temporal granularities

**Training Objective:** Next token prediction (autoregressive) on patched time series

**Result:** Model learns universal time series patterns applicable across domains

---

## LoRA Fine-Tuning

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that adds trainable rank decomposition matrices to existing weights:

```python
# Standard linear layer: y = Wx
# LoRA-modified: y = Wx + ΔWx = Wx + BAx

# Where:
# W: Frozen pre-trained weights (232M params)
# B, A: trainable low-rank matrices (r=4, ~1.4M params)
```

### LoRA Architecture for TimesFM 2.5

```
Base TimesFM 2.5 Model (232M params - FROZEN)
    ↓
Inject LoRA Adapters into All Linear Layers
    ├─ Query (Q) projection
    ├─ Key (K) projection
    ├─ Value (V) projection
    ├─ Output projection
    ├─ Feed-forward networks
    └─ Output layer
    ↓
Fine-Tuned Model with LoRA (1.4M trainable params)
```

### LoRA Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **r (rank)** | 4 | Low rank keeps parameters low (~0.6%), sufficient for domain adaptation |
| **alpha** | 8 | Scaling factor = 2×r, standard practice for stability |
| **target_modules** | "all-linear" | Apply to all linear layers for maximum adaptation |
| **dropout** | 0.05 | Light regularization to prevent overfitting |
| **bias** | "none" | Don't train bias terms (reduces params further) |

### Trainable Parameters

**Calculation:**
- Base model: 232M parameters (all frozen)
- LoRA adapters: 1.4M parameters (trainable)
- **Percentage: 0.6%**

**Per-Layer Breakdown:**
```
For each linear layer with shape (d_in, d_out):
- Standard: d_in × d_out parameters
- LoRA: d_in × r + r × d_out = r × (d_in + d_out) parameters

With r=4, d_in=1280, d_out=1280:
- Standard: 1,638,400 parameters
- LoRA: 4 × (1280 + 1280) = 10,240 parameters
- Reduction: 99.4%
```

### Why LoRA Works

**1. Low-Intrinsic Dimension:**
- Pre-trained models operate in low-dimensional subspace
- Adaptation requires updating only a few directions
- LoRA captures these directions efficiently

**2. Domain Adaptation:**
- Pre-training learns universal patterns (trends, seasonality)
- Fine-tuning learns domain-specific patterns (volatility clustering)
- LoRA separates domain knowledge from general knowledge

**3. Training Efficiency:**
- 99.4% fewer parameters = 10× faster training
- Lower GPU memory = larger batch sizes
- Faster convergence (10 epochs vs 100)

**4. Modularity:**
- LoRA adapters can be swapped for different domains
- Ensemble multiple adapters for diverse stocks
- Smaller artifact size (~6MB vs ~900MB)

---

## Volatility Forecasting Adaptation

### Data Pipeline

**Input:** OHLCV data for VN30 stocks (2006-2024)

```
OHLCV Data (5 columns)
    ↓
Parkinson Volatility Estimator
    σ² = (log(H/L)²) / (4 × log(2))
    ↓
Univariate Time Series (single column)
    ↓
Temporal Split (70/15/15 chronological)
    ↓
Random Window Sampling (training)
    - 5,000 random (context, horizon) windows per epoch
    - Context: 64 days
    - Horizon: 5 days
    ↓
Last Window Sampling (validation/test)
    - Most recent (context, horizon) window
```

### Why Univariate Input?

**TimesFM 2.5 Constraint:**
- `past_values` expects **1D time series** (univariate)
- NO multivariate input support in standard architecture

**Our Approach:**
- **Phase 1:** Parkinson volatility only (univariate)
- **Rationale:** Volatility captures most relevant OHLCV information
- **If Underperforms:** Add HAR features as static covariates

### Internal Normalization (RevIN)

**TimesFM 2.5 applies Reversible Instance Normalization (RevIN) internally:**

```python
# NO external normalization needed!

# Input: Raw Parkinson values
context = [0.0012, 0.0015, 0.0011, ...]

# TimesFM applies RevIN:
# 1. Compute mean, std from context
# 2. Normalize: (context - mean) / std
# 3. Forecast in normalized space
# 4. Denormalize: forecast × std + mean
# 5. Output in original scale

# Output: Forecast in original scale
forecast = [0.0013, 0.0014, ...]
```

**Benefits:**
- No manual normalization required
- Consistent across all datasets
- Reversible (no information loss)

### Forecasting Objective

**Training Loss:** MSE (computed internally by TimesFM)
```python
loss = MSE(forecast, target)
```

**Evaluation Metrics:** All 6 mandatory metrics
1. **MSE:** Mean Squared Error (loss function)
2. **RMSE:** Root Mean Squared Error (primary accuracy metric)
3. **MAE:** Mean Absolute Error (robustness check)
4. **R²:** Variance Explained (goodness of fit)
5. **QLIKE:** Academic standard for volatility
6. **Dir Acc:** Directional Accuracy (economic significance)

---

## Hyperparameter Justification

### Model Hyperparameters

| Hyperparameter | Value | Justification | Source |
|----------------|-------|---------------|--------|
| **context_len** | 64 | TimesFM default (multiple of 32), sufficient for volatility patterns | timesfm-google |
| **horizon_len** | 5 | Project target (5-day forecasts) | Project spec |
| **model_id** | google/timesfm-2.5-200m-transformers | Latest Transformers integration, LoRA support | Hugging Face |

### LoRA Hyperparameters

| Hyperparameter | Value | Justification | Source |
|----------------|-------|---------------|--------|
| **lora_r** | 4 | Proven sufficient for domain adaptation (~0.6% params) | timesfm-google |
| **lora_alpha** | 8 | 2×r scaling factor (standard practice) | PEFT documentation |
| **lora_dropout** | 0.05 | Light regularization | timesfm-google |
| **target_modules** | "all-linear" | Maximum adaptation flexibility | timesfm-google |

### Training Hyperparameters

| Hyperparameter | Value | Justification | Source |
|----------------|-------|---------------|--------|
| **optimizer** | AdamW | Proven better than SGD for Transformers | timesfm-google |
| **lr** | 1e-4 | Conservative LR prevents catastrophic forgetting | timesfm-google |
| **weight_decay** | 0.01 | Standard regularization for AdamW | Transformers best practice |
| **scheduler** | CosineAnnealingLR | Smooth LR decay, proven for Transformers | timesfm-google |
| **epochs** | 10 | LoRA converges faster than full fine-tuning | timesfm-google |
| **batch_size** | 32 | Balance GPU memory and gradient stability | timesfm-google |
| **max_grad_norm** | 1.0 | Prevent gradient explosions | timesfm-google |
| **num_train_samples** | 5000 | Random window samples per epoch | Chronos-2 practice |

### Why AdamW (Not SGD)?

**Paper-based approach (SGD):**
- Older research used SGD with momentum
- Requires careful LR tuning
- Slower convergence

**Proven approach (AdamW):**
- Working implementation from timesfm-google
- Adaptive learning rates
- Faster convergence
- Better stability

**Decision:** Use **AdamW** (proven working code) over SGD (paper-based)

### Why Cosine Annealing (Not Warmup)?

**Paper-based approach (warmup + cosine):**
- Linear warmup for 5 epochs
- Then cosine decay
- Total: 100 epochs

**Proven approach (cosine only):**
- Direct cosine annealing
- No warmup needed
- Total: 10 epochs

**Decision:** Use **cosine annealing only** (LoRA converges faster)

---

## Performance Characteristics

### Training Performance

**Expected Performance (based on timesfm-google):**

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Time** | ~30-60 minutes (Colab GPU) | 10 epochs @ 32 batch size |
| **Convergence** | 5-10 epochs | LoRA converges quickly |
| **GPU Memory** | ~8-16 GB | Depends on batch size |
| **Adapter Size** | ~6 MB | vs ~900 MB for full model |

### Inference Performance

**Expected Performance (5-day forecasts):**

| Metric | Zero-Shot | LoRA Fine-Tuned | Improvement |
|--------|-----------|-----------------|-------------|
| **RMSE** | ~0.25 | < 0.20 | ≥ 20% ↓ |
| **Dir Acc** | ~50% | > 55% | ≥ 10% ↑ |
| **QLIKE** | ~0.15 | < 0.12 | ≥ 20% ↓ |

**Success Criteria:**
- RMSE < 0.20
- Directional Accuracy > 55%

### Comparison with Baselines

**Expected Ranking:**

1. **TimesFM 2.5 LoRA** (best) - Foundation model + domain adaptation
2. **LSTM-HAR Enhanced** - Current best baseline (67.90% Dir Acc)
3. **TimesFM 2.5 Zero-Shot** - Foundation model without adaptation
4. **HAR-R** - Linear baseline

---

## Comparison with Alternatives

### vs Full Fine-Tuning

| Aspect | LoRA | Full Fine-Tuning |
|--------|------|------------------|
| **Trainable Params** | 1.4M (0.6%) | 232M (100%) |
| **Training Time** | ~1 hour | ~10 hours |
| **GPU Memory** | 8-16 GB | 32-64 GB |
| **Storage** | ~6 MB | ~900 MB |
| **Convergence** | 10 epochs | 100 epochs |
| **Flexibility** | Swap adapters | Retrain for each domain |
| **Proven** | ✅ Working code | ❌ Paper-only |

**Winner:** LoRA (10× faster, 99% fewer params)

### vs Traditional Models

| Aspect | TimesFM 2.5 LoRA | HAR-R | LSTM |
|--------|------------------|-------|------|
| **Parameters** | 1.4M (fine-tuned) | ~4 | ~50K |
| **Training Data** | 100B (pre) + domain | Domain only | Domain only |
| **Architecture** | Transformer (decoder-only) | Linear regression | Recurrent |
| **Zero-Shot** | ✅ Yes | ❌ No | ❌ No |
| **Transfer Learning** | ✅ Yes | ❌ No | ⚠️ Limited |
| **Probabilistic** | ✅ 9 quantiles | ❌ Point only | ❌ Point only |
| **Domain Adaptation** | ✅ LoRA | ❌ N/A | ⚠️ Retrain |

**Winner:** TimesFM 2.5 (zero-shot + transfer learning + probabilistic)

### vs Other Foundation Models

| Model | Parameters | Context | Status |
|-------|-----------|---------|--------|
| **TimesFM 2.5** | 200M | 16K | ✅ Production-ready |
| **Chronos** | 70M | 512 | ✅ Available |
| **Lag-Llama** | 1B | 2048 | ⚠️ Beta |

**Winner:** TimesFM 2.5 (longest context, production-ready)

---

## References

### Papers

1. **TimesFM Paper:** "A Decoder-Only Foundation Model for Time-Series Forecasting" (2024)
   - Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou
   - arXiv: [link]

2. **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
   - Hu et al.
   - arXiv: 2106.09685

3. **Chronos-2:** "Chronos-2: A Higher-Capacity, General-Purpose Time Series Forecasting Model" (2024)
   - Ansari et al.
   - Random window sampling methodology

### Code & Documentation

4. **TimesFM Google Repository:**
   - https://github.com/UberGuidoZ/timesfm-google
   - Working LoRA implementation

5. **Hugging Face Transformers:**
   - https://huggingface.co/docs/transformers/model_doc/timesfm2_5
   - Model API and usage

6. **PEFT Library:**
   - https://github.com/huggingface/peft
   - LoRA implementation and best practices

### External Resources

7. **TimesFM-2.5 + External Covariates:**
   - https://aihorizonforecast.substack.com/p/timesfm-25-external-covariates-a
   - Static covariates and xreg layers

8. **statmike's TimesFM Notebook:**
   - https://colab.research.google.com/github/statmike/vertex-ai-mlops/blob/main/Applied%20Forecasting/TimesFM%20-%20Time%20Series%20Foundation%20Model.ipynb
   - Colab notebook patterns

### Project Documentation

9. **Project Context:**
   - `project-context.md`
   - `CLAUDE.md`
   - `docs/requirements.md`

10. **Implementation Specification:**
    - `_bmad-output/implementation-artifacts/spec-3-4-timesfm-parkinson-volatility-finetuning.md`

---

## Appendix: Architecture Diagrams

### Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Data                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Parkinson Volatility                           │
│         σ² = (log(H/L)²) / (4 × log(2))                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Split (70/15/15)                     │
│              Train | Val | Test (chronological)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Random Window Sampling (Train)                   │
│         5,000 random (64 context + 5 horizon) windows            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              Last Window Sampling (Val/Test)                      │
│              Most recent (64 context + 5 horizon)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TimesFM 2.5 Base Model                        │
│                  (232M params, pre-trained)                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Patch Creation (32-point patches)                       │  │
│  │  ↓                                                        │  │
│  │  Patch Embedding + Positional Encoding                  │  │
│  │  ↓                                                        │  │
│  │  20 × Transformer Decoder Layers                         │  │
│  │  ↓                                                        │  │
│  │  Continuous Quantile Head (9 quantiles)                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA Adapter Injection                         │
│                   (1.4M params, trainable)                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Low-Rank Matrices (r=4)                                │  │
│  │  ΔW = BA, where B, A ∈ R^(d×r)                         │  │
│  │  Applied to all linear layers                           │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                Training Process (AdamW, LR=1e-4)                  │
│  • Cosine Annealing Scheduler                                  │
│  • Gradient Clipping (max_norm=1.0)                            │
│  • MSE Loss (internal RevIN normalization)                    │
│  • 10 epochs, batch size 32                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Fine-Tuned Model (LoRA)                         │
│               1.4M trainable params saved                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Evaluation (6 Metrics)                           │
│  • MSE, RMSE, MAE (accuracy metrics)                           │
│  • R² (variance explained)                                     │
│  • QLIKE (volatility academic standard)                        │
│  • Dir Acc (economic significance)                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              Success Criteria: RMSE < 0.20, Dir Acc > 55%        │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA Architecture Detail

```
Standard Linear Layer (Frozen):
┌─────────────────────────────────────────┐
│  Input: x (d_in,)                       │
│  Output: y = Wx (d_out,)                │
│  Parameters: d_in × d_out               │
└─────────────────────────────────────────┘

LoRA-Modified Layer:
┌─────────────────────────────────────────┐
│  Input: x (d_in,)                       │
│  ↓                                      │
│  Branch 1: Wx (frozen)                 │
│  Branch 2: BAx (trainable)              │
│  ↓                                      │
│  Output: y = Wx + BAx                   │
│  Parameters:                            │
│    - W: d_in × d_out (frozen)          │
│    - B: d_out × r (trainable)          │
│    - A: r × d_in (trainable)           │
│    - Total trainable: r × (d_in + d_out)│
└─────────────────────────────────────────┘

Example: d_in=1280, d_out=1280, r=4
  Standard: 1,638,400 parameters
  LoRA: 4 × (1280 + 1280) = 10,240 parameters
  Reduction: 99.4%
```

---

**End of Document**
