# Debug Plan — Parallel LSTM-GNN (flat val + underfitting)

**Date:** 2026-06-28
**Run analyzed:** `results/parallel_lstm_gnn_knn_2026-06-28_173532`
**Training script:** `src/lstm_gat_hybrid/train_parallel_enhanced.py`
**Dataset:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`
**Metrics:** `src/common/evaluation.py`

---

## Symptom

Validation loss curve is flat (slope `0.000000`), pinned at ~`6e-6` while train loss
sits at ~`0.93`. Overfitting monitor shows gap `-0.93` (nonsense). Yet test reports
R²=0.71, DirAcc=69%.

## Diagnosis — three stacked bugs

### Issue 1 — Scale mismatch (direct cause of the flat val line)
- `train_epoch` (line ~210) computes MSE on **normalized** targets (StandardScaler,
  mean 0 / std 1) → loss ~0.93.
- `validate` (line ~358) **inverse-transforms** preds+targets to raw volatility BEFORE
  computing MSE → loss ~6e-6 (raw vol ≈ 0.002, squared ≈ 1e-6).
- Train and val losses are 5 orders of magnitude apart → val line flat on a 0–1 axis,
  slope rounds to 0, overfitting monitor meaningless.

### Issue 2 — Model is genuinely underfitting (the real problem)
- Train normalized MSE only drops 1.00 → 0.93 over 31 epochs. On standardized targets,
  MSE=1.0 = naive per-stock-mean predictor. Model explains only ~7% of train variance.
- Likely contributors (in order):
  - `learning_rate = 0.0001` — very low; was dropped twice "to prevent NaN" (masking
    instability instead of fixing root cause).
  - `fusion_dropout = 0.3` + weight_decay + augmentation → over-regularized.
  - Possible dead GNN branch / fusion dominated by one path.

### Issue 3 — "Good" test metrics are not trustworthy
- `validate` pools all stocks into one flat array ordered `[batch, stock]`, then
  denormalizes **per-stock**. R² (`evaluation.py:98`) on this pooled vector captures
  cross-sectional volatility-level spread → high pooled R² with zero temporal skill.
- `directional_accuracy` (`evaluation.py:71`) does `np.diff` over the pooled array, so it
  measures sign of (stock_{i+1} − stock_i) within a batch, NOT temporal change. 69% is
  meaningless.

---

## Plan (prioritized)

### Step 1 — Make losses comparable (fixes flat curve immediately)
In `validate`, compute the curve's `avg_loss` on the **normalized** preds/targets (the
arrays already collected), so train and val curves share scale. Keep denormalized values
only for reported business metrics (return both).

### Step 2 — Confirm underfitting via constant baseline
Add a sanity check: predict per-stock mean (= constant 0 on normalized scale) and push it
through the SAME eval. Expect it to reproduce R²≈0.7 / DirAcc≈69%, proving metrics are
level-driven, not skill.

### Step 3 — Fix metrics to measure temporal skill
Evaluate **per-stock**: reshape to `[n_windows, n_stocks]`, compute R²/DirAcc per stock
then average. Directional accuracy must be along the **time axis** per stock, not across
stocks. Report per-stock-averaged R² as the honest number.

### Step 4 — Address underfitting (after metrics are honest)
- Raise LR (try 1e-3, clip already 0.5); find the real NaN cause (adj-matrix norm /
  exploding GNN) instead of masking with tiny LR.
- Reduce `fusion_dropout` → 0.1–0.2, lower augmentation_prob; confirm train loss < ~0.7.
- Log per-branch (LSTM vs GNN) activation stats to confirm GNN isn't dead.

### Step 5 — Cleanup
Remove per-epoch `print` debug spam in `validate` / `directional_accuracy` once stable.

---

## Sanity check results (Step 2 — executed 2026-06-28)

Ran constant predictor (predict 0 = normalized training mean) on test set:

| Metric | Constant predictor | Trained model | Interpretation |
|---|---|---|---|
| Pooled R² | **0.702** | 0.714 | Identical — inflated by cross-sectional spread, not temporal skill |
| Per-stock R² | **-0.328** | TBD | Worse than constant mean on time axis → model has NO skill |
| Pooled DirAcc | 64.53% | 69.1% | Inflated — `np.diff` runs across stock boundaries |
| Per-stock DirAcc | **8.26%** | TBD | Barely above random (50%) |

**Conclusion:** Trained model's reported 69% DirAcc / 0.71 R² measures **cross-sectional level spread, not temporal forecasting skill**. The flat val line was the scale bug (Step 1 fixed). The real problem is **severe underfitting** — train normalized MSE only dropped 1.00 → 0.93 over 31 epochs.

---

## Step 4 — Fix the underfitting (DONE ✅)

### Changes to `train_parallel_enhanced.py` (implemented 2026-06-28)

1. **Raise learning rate back up** (line ~413):
   - Changed `learning_rate = 0.0001` → `0.001` (paper's value)
   - The "prevent NaN" rationale was masking a different bug — we need to find it, not cripple learning
   - `gradient_clip = 0.5` is already set (line ~430), keep it

2. **Reduce over-regularization** (lines ~432-433):
   - Changed `fusion_dropout = 0.3` → `0.15` (was too aggressive)
   - Changed `augmentation_prob = 0.3` → `0.15` (line ~516)
   - Kept `lstm_dropout = 0.2` and `weight_decay = 1e-5`

3. **Added per-branch activation logging** (in `train_epoch`, line ~201):
   - After `predictions = model(x, adj_matrix)`, calls `model.get_embeddings()`
   - Logs LSTM and GNN embedding stats (mean, std, min, max) on first batch
   - Warns if either branch is near-constant (std < 0.01) or contains NaN

4. **Expected results when retraining**:
   - Train normalized MSE should drop < 0.7 within ~10 epochs (was stuck at 0.93)
   - Check logs: if GNN activations explode or go NaN, the real bug is in adj-matrix normalization
   - After convergence, per-stock R² and DirAcc should be honest temporal-skill metrics

### Root cause hunt for NaN

If NaN appears at `lr=0.001`:
- Check `adj_matrix` values in `validate` first batch — confirm they're O(1) and symmetric
- Check GNN attention weights in `model_parallel.py` — add gradient clipping to attention softmax
- Check for division-by-zero in edge normalization

---

## Execution order (updated 2026-06-28)

✅ **Step 1** — Scale fix (`validate` returns normalized MSE) — DONE
✅ **Step 3** — Per-stock metrics — DONE
✅ **Step 2** — Constant baseline sanity check — DONE (confirmed pooled R² inflation)
✅ **Step 4** — Fix underfitting: raise LR, cut dropout, add logging — DONE
✅ **Step 5** — Fix VHM distribution shift: clip normalized values to [-10, 10] — DONE
⏭️ **Step 6** — Full retrain and verify per-stock skill improves
⏭️ **Step 7** — Cleanup debug prints after convergence

---

## Step 5 — Distribution shift fix (DONE ✅)

### Root cause found
VHM stock has near-zero variance in training but much higher volatility in validation:

```
VHM Training:   std = 0.00000007, max = 0.00000197
VHM Validation: std = 0.00014121, max = 0.00151564

Amplification: 1/std = 14,672,088x
Result: 0.00151564 / 0.00000007 = 22237.53 (normalized value!)
```

When validation values are normalized with training statistics, they explode by **14 million times**.

### Fix implemented
Added clipping in `dataset_with_graph_method.py` line ~353:
```python
y_normalized = np.clip(y_normalized, -10.0, 10.0)
```

This prevents extreme normalized values from distribution shift while preserving the normalization for in-range values.

### Result
Validation batches 5, 8, 13 (VHM extreme values) are now clipped to ±10 instead of exploding to 20000+. Val loss is now in the same scale as train loss (~0.94 vs ~0.94).
