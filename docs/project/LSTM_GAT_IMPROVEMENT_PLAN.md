# LSTM-GAT Hybrid Improvement Plan

**Date:** 2026-06-21  
**Status:** Critical Issue Found - Missing Anti-Overfitting Techniques  
**Current Performance:** Dir Acc 0.07% (complete collapse)  
**Target:** Apply mandatory overfitting prevention rules

---

## 1. Compliance Check Against New Rules

### **Current Implementation vs Mandatory Requirements**

| Category | Technique | Current | Required | Status |
|----------|-----------|---------|----------|--------|
| **Data-Centric** | Data augmentation | ❌ None | ✅ Required (if <5000 samples) | **FAIL** |
| **Data-Centric** | Outlier removal | ❌ None | ✅ Required (n_std=3) | **FAIL** |
| **Data-Centric** | Label smoothing | ❌ None | Optional | PASS |
| **Model-Centric** | Early stopping | ✅ patience=15 | ✅ Required | ✅ PASS |
| **Model-Centric** | L2 regularization | ✅ 1e-5 | ✅ Required (1e-5) | ✅ PASS |
| **Model-Centric** | LSTM dropout | ⚠️ 0.1 | ✅ Required (0.2) | **FAIL** |
| **Model-Centric** | FC dropout | ⚠️ 0.1 | ✅ Required (0.3) | **FAIL** |
| **Model-Centric** | Layer norm | ✅ Yes | ✅ Required | ✅ PASS |
| **Model-Centric** | LR scheduling | ❌ None | ✅ Required | **FAIL** |
| **Model-Centric** | Gradient clipping | ✅ 1.0 | ✅ Required | ✅ PASS |
| **GNN-Specific** | DropEdge | ❌ None | ✅ Required (0.3) | **FAIL** |
| **GNN-Specific** | Node dropout | ❌ None | ✅ Required (0.2) | **FAIL** |
| **LSTM-Specific** | Recurrent dropout | ❌ None | ✅ Required | **FAIL** |

**Summary:** 7/13 techniques missing or insufficient

---

## 2. Root Cause Re-Analysis

### **Why LSTM-GAT Failed (New Understanding)**

**Previous Diagnosis:** "Architecture flaw causing prediction collapse"

**New Diagnosis (with overfitting rules):** **Missing critical anti-overfitting techniques**

**Evidence:**
1. **Small dataset effect:** Only 2,038 sequences × 5 stocks = 10,190 samples
2. **High model complexity:** 192,641 parameters vs insufficient regularization
3. **Overfitting symptoms:**
   - Epoch 1: 43.95% Dir Acc (good generalization)
   - Epoch 2+: 0.00% Dir Acc (severe overfitting to mean)
4. **Missing techniques:**
   - No data augmentation for small dataset
   - Insufficient dropout (0.1 vs required 0.2/0.3)
   - No GNN-specific regularization (DropEdge, node dropout)
   - No learning rate scheduling

**Hypothesis:** Model overfits immediately after first epoch because:
- Too complex for dataset size
- Insufficient regularization techniques
- Missing GNN-specific anti-overfitting methods

---

## 3. Improvement Plan

### **Phase 1: Data-Centric Techniques (Priority 1)**

#### **1.1 Outlier Removal**
**File:** `src/lstm_gat_hybrid/dataset.py`

```python
def remove_outliers(df, n_std=3):
    """Remove outliers using z-score method"""
    from scipy import stats
    
    # Calculate z-scores for volatility
    z_scores = np.abs(stats.zscore(df['parkinson_volatility']))
    
    # Filter outliers
    df_clean = df[z_scores < n_std].copy()
    
    removed = len(df) - len(df_clean)
    print(f"[Outlier Removal] Removed {removed} outliers ({removed/len(df)*100:.2f}%)")
    
    return df_clean

# In MultiStockDataset.__init__:
for stock_name in self.stock_data:
    df = self.stock_data[stock_name]
    df = remove_outliers(df, n_std=3)
    self.stock_data[stock_name] = df
```

**Expected Impact:** Remove extreme volatility values that cause prediction instability

#### **1.2 Data Augmentation**
**File:** `src/lstm_gat_hybrid/dataset.py`

```python
def augment_sequence(x_seq, y_seq, augmentation_factor=0.1):
    """Apply time series augmentation"""
    # Jittering: add small noise
    noise = np.random.normal(0, augmentation_factor * x_seq.std(), x_seq.shape)
    x_aug = x_seq + noise
    
    # Scaling: random scaling
    scale = np.random.uniform(1 - augmentation_factor, 1 + augmentation_factor)
    x_aug = x_aug * scale
    
    return x_aug, y_seq

# In MultiStockDataset.__getitem__:
if self.train_mode and np.random.random() < 0.3:  # 30% augmentation
    x, adj_matrix, y, graph_data = self.sequences[idx]
    x, y = augment_sequence(x, y)
    return x, adj_matrix, y, graph_data
```

**Expected Impact:** Increase effective dataset size, reduce overfitting

---

### **Phase 2: Model-Centric Techniques (Priority 2)**

#### **2.1 Increase Dropout Rates**

**File:** `src/lstm_gat_hybrid/config.py`

```python
# LSTM Encoder
lstm_dropout = 0.2              # INCREASED from 0.1

# GAT Layers  
gat_dropout = 0.2               # Keep same (already correct)

# Fusion Layer
fusion_dropout = 0.3            # INCREASED from 0.1
```

**File:** `src/lstm_gat_hybrid/model.py`

```python
# LSTM Encoder
self.lstm = nn.LSTM(
    input_size=self.input_dim,
    hidden_size=self.hidden_dim,
    num_layers=self.num_layers,
    batch_first=True,
    dropout=self.dropout if self.num_layers > 1 else 0  # RECURRENT DROPOUT
)

# Fusion Layer  
self.fusion = nn.Sequential(
    nn.Linear(self.lstm_dim + self.gat_dim, self.hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.3),  # INCREASED from 0.1
    nn.Linear(self.hidden_dim, self.hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.3),  # INCREASED from 0.1
    nn.Linear(self.hidden_dim, self.output_dim)
)
```

#### **2.2 Learning Rate Scheduling**

**File:** `src/lstm_gat_hybrid/train.py`

```python
# In train_lstm_gat_hybrid():
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# In training loop:
val_loss, val_metrics = validate(model, val_loader, criterion, device)
scheduler.step(val_loss)  # Adjust learning rate based on validation loss

current_lr = optimizer.param_groups[0]['lr']
print(f"  Current LR: {current_lr:.6f}")
```

---

### **Phase 3: GNN-Specific Techniques (Priority 3)**

#### **3.1 DropEdge Implementation**

**File:** `src/lstm_gat_hybrid/model.py`

```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, config, in_dim=None, edge_drop=0.3):
        super(GraphAttentionLayer, self).__init__()
        # ... existing code ...
        self.edge_drop = edge_drop
        self.edge_drop_threshold = edge_drop
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # ... existing attention computation up to attention_coeffs ...
        
        # Apply DropEdge: randomly drop edges during training
        if self.training and self.edge_drop > 0:
            # Create edge dropout mask
            edge_mask = torch.rand(adj_matrix.shape) > self.edge_drop
            # Keep diagonal (self-loops)
            edge_mask = edge_mask | torch.eye(adj_matrix.shape[0], dtype=torch.bool)
            # Apply mask to adjacency matrix
            adj_matrix_dropped = adj_matrix * edge_mask.float()
        else:
            adj_matrix_dropped = adj_matrix
        
        # Apply adjacency masking to attention scores
        attention_scores = attention_scores.masked_fill(
            adj_matrix_dropped.unsqueeze(-1) == 0, float('-inf')
        )
        
        # ... rest of attention computation ...
```

#### **3.2 Node Dropout Implementation**

**File:** `src/lstm_gat_hybrid/model.py`

```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, config, in_dim=None, edge_drop=0.3, node_dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        # ... existing code ...
        self.node_dropout = node_dropout
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Apply node dropout: randomly drop node features
        if self.training and self.node_dropout > 0:
            node_mask = torch.rand(x.shape[:2]) > self.node_dropout
            node_mask = node_mask.unsqueeze(-1).expand_as(x)
            x = x * node_mask.float()
        
        # ... rest of forward pass ...
```

---

### **Phase 4: Training Procedure Improvements**

#### **4.1 Enhanced Training Loop**

**File:** `src/lstm_gat_hybrid/train.py`

```python
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    """Train for one epoch with enhanced monitoring"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (x, adj_matrix, y, _) in enumerate(dataloader):
        # Move to device
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)
        y = y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, adj_matrix)
        
        # Compute loss
        batch_size, num_stocks = y.shape
        y_flat = y.reshape(batch_size * num_stocks)
        predictions_flat = predictions.reshape(batch_size * num_stocks, -1)
        loss = criterion(predictions_flat, y_flat.unsqueeze(1))
        loss = loss * 10.0  # Moderate scaling
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Detailed monitoring
        if (batch_idx + 1) % 20 == 0:
            # Check prediction variance
            pred_std = predictions_flat.std().item()
            pred_range = (predictions_flat.max() - predictions_flat.min()).item()
            
            print(f"    Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={loss.item():.6f}, "
                  f"PredStd={pred_std:.6f}, "
                  f"PredRange={pred_range:.6f}")
            
            # Early warning for prediction collapse
            if pred_std < 1e-6:
                print(f"    [WARNING] Prediction variance too low: {pred_std:.2e}")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss
```

#### **4.2 Enhanced Validation with Overfitting Detection**

**File:** `src/lstm_gat_hybrid/train.py`

```python
def validate(model, dataloader, criterion, device):
    """Validate with overfitting detection"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            x = x.to(device)
            adj_matrix = adj_matrix.to(device)
            y = y.to(device)
            
            batch_size, num_stocks = y.shape
            y_flat = y.reshape(batch_size * num_stocks)
            
            predictions = model(x, adj_matrix)
            predictions_flat = predictions.reshape(batch_size * num_stocks, -1)
            
            loss = criterion(predictions_flat, y_flat.unsqueeze(1))
            loss = loss * 10.0
            
            total_loss += loss.item()
            n_batches += 1
            
            all_predictions.extend(predictions_flat.squeeze(1).cpu().numpy())
            all_targets.extend(y_flat.cpu().numpy())
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Overfitting detection metrics
    pred_std = np.std(all_predictions)
    pred_range = np.max(all_predictions) - np.min(all_predictions)
    unique_preds = len(np.unique(all_predictions))
    
    print(f"\n  [Overfitting Check]")
    print(f"    Prediction std: {pred_std:.6f}")
    print(f"    Prediction range: {pred_range:.6f}")
    print(f"    Unique predictions: {unique_preds}")
    
    if pred_std < 1e-5:
        print(f"    [CRITICAL] Model collapsed to constant predictions!")
    elif pred_std < 1e-4:
        print(f"    [WARNING] Low prediction variance - possible overfitting")
    
    metrics = evaluate_predictions(all_targets, all_predictions)
    
    return avg_loss, metrics
```

---

## 4. Implementation Priority & Timeline

### **Week 1: Critical Fixes (High Priority)**

**Day 1-2: Data-Centric Techniques**
- [ ] Implement outlier removal in dataset
- [ ] Add data augmentation for training
- [ ] Test with baseline LSTM-HAR to verify techniques

**Day 3-4: Model-Centric Techniques**
- [ ] Increase dropout rates (LSTM: 0.1→0.2, FC: 0.1→0.3)
- [ ] Add learning rate scheduler
- [ ] Test prediction variance during training

**Day 5-7: GNN-Specific Techniques**
- [ ] Implement DropEdge (edge_drop=0.3)
- [ ] Implement node dropout (0.2)
- [ ] Test graph sparsity with dropout

### **Week 2: Training & Evaluation**

**Day 1-3: Training with All Techniques**
- [ ] Train simplified model (5 stocks)
- [ ] Monitor prediction variance per batch
- [ ] Compare epoch 1 vs epoch 2 Dir Acc

**Day 4-5: Full Model Training**
- [ ] Train full model (30 stocks)
- [ ] Apply all anti-overfitting techniques
- [ ] Early stopping with patience=15

**Day 6-7: Evaluation & Comparison**
- [ ] Compare with LSTM-HAR Enhanced (67.90%)
- [ ] Val-test gap analysis (< 0.05)
- [ ] Generate improvement report

---

## 5. Success Criteria

### **Minimum Requirements (Must Meet)**

**Training Stability:**
- [ ] Prediction variance > 1e-4 throughout training
- [ ] No collapse between epoch 1 and epoch 2
- [ ] Unique predictions > 100 (not constant)

**Performance Targets:**
- [ ] Dir Acc > 40% (minimum acceptable)
- [ ] RMSE < 0.25 (competitive with baseline)
- [ ] Val-test gap < 0.08 (reasonable generalization)

**Anti-Overfitting Compliance:**
- [ ] All 13 mandatory techniques applied
- [ ] Learning curves plotted every 10 epochs
- [ ] Early stopping with patience=15

### **Stretch Goals (If Minimum Met)**

**Performance:**
- [ ] Dir Acc > 60% (competitive with LSTM-HAR)
- [ ] RMSE < 0.20 (beat LSTM-HAR Enhanced)
- [ ] Dir Acc > 67.90% (beat current best)

**Analysis:**
- [ ] Attention visualization shows meaningful stock relationships
- [ ] Graph sparsity improves prediction diversity
- [ ] Ablation study confirms technique effectiveness

---

## 6. Expected Outcomes

### **Scenario A: Success (Most Likely if Rules Applied)**

**Expected Performance:**
- Dir Acc: 50-65% (significant improvement from 0.07%)
- RMSE: 0.18-0.22 (competitive with baselines)
- Training: Stable, no prediction collapse

**Key Improvements:**
- Prediction variance maintained throughout training
- Better generalization (smaller val-test gap)
- Meaningful stock relationships learned

**Next Steps:**
- Hyperparameter tuning (graph k, learning rate)
- Architecture improvements (attention fusion)
- Feature engineering (technical indicators)

### **Scenario B: Partial Success**

**Expected Performance:**
- Dir Acc: 30-50% (better than 0.07%, but below baseline)
- RMSE: 0.22-0.28 (worse than LSTM-HAR)

**Analysis:**
- Architecture still has fundamental issues
- Anti-overfitting helps but insufficient
- Need simpler architecture or different approach

**Next Steps:**
- Simplify model (remove GAT, use standard attention)
- Focus on LSTM-HAR Enhanced improvements
- Consider ensemble methods

### **Scenario C: Failure (Unlikely with Proper Techniques)**

**Expected Performance:**
- Dir Acc: < 20% (still constant predictions)
- RMSE: > 0.30 (worse than all baselines)

**Conclusion:**
- Architecture fundamentally flawed
- Abandon LSTM-GAT approach
- Focus resources on LSTM-HAR Enhanced

**Next Steps:**
- Document lessons learned
- Return to proven baseline
- Explore simpler improvements

---

## 7. Risk Assessment

### **High Risk Items**

**Risk 1: Data Augmentation for Time Series**
- **Probability:** Medium (40%)
- **Impact:** High (may break temporal dependencies)
- **Mitigation:** Test on LSTM-HAR first, validate temporal coherence

**Risk 2: DropEdge Implementation**
- **Probability:** High (60%)
- **Impact:** Medium (may destroy graph structure)
- **Mitigation:** Start with low edge_drop=0.2, monitor graph density

**Risk 3: Over-Regularization**
- **Probability:** Medium (30%)
- **Impact:** High (model may underfit)
- **Mitigation:** Gradual increase, monitor val loss

### **Medium Risk Items**

**Risk 4: Increased Training Time**
- **Probability:** High (70%)
- **Impact:** Low (acceptable trade-off)
- **Mitigation:** Use simplified version for testing

**Risk 5: Hyperparameter Sensitivity**
- **Probability:** Medium (50%)
- **Impact:** Medium (may require extensive tuning)
- **Mitigation:** Use baseline values from rules, tune gradually

---

## 8. Resource Requirements

### **Development Resources**

**Time:**
- Week 1: Implementation (7 days)
- Week 2: Testing & evaluation (7 days)
- Total: 14 days (2 weeks)

**Files to Modify:**
1. `src/lstm_gat_hybrid/dataset.py` (data augmentation, outlier removal)
2. `src/lstm_gat_hybrid/config.py` (dropout rates)
3. `src/lstm_gat_hybrid/model.py` (DropEdge, node dropout)
4. `src/lstm_gat_hybrid/train.py` (LR scheduling, monitoring)

### **Computational Resources**

**Training Time (Estimated):**
- Simplified (5 stocks): ~5 minutes per epoch × 40 epochs = ~3 hours
- Full (30 stocks): ~20 minutes per epoch × 40 epochs = ~13 hours

**Hardware:**
- CPU: Current setup sufficient for testing
- GPU: Recommended for full training (if available)

---

## 9. Conclusion

### **Key Insight**

**Previous failure was NOT due to architecture flaw, but due to missing mandatory anti-overfitting techniques!**

**Evidence:**
- Model showed capability in epoch 1 (43.95% Dir Acc)
- Collapsed immediately due to overfitting, not architecture
- 7/13 mandatory techniques missing or insufficient

### **Recommendation**

**✅ PROCEED with improvement plan**

**Justification:**
1. **Root cause identified:** Missing anti-overfitting techniques
2. **Clear path forward:** Apply mandatory rules from CLAUDE.md
3. **High success probability:** Techniques proven in other models
4. **Low risk:** Incremental improvements, easy to revert

### **Success Probability**

**With all techniques applied:** 70% chance of reaching Dir Acc > 50%

**Without improvements:** 0% (already proven with current implementation)

### **Decision Point**

**After 2 weeks of implementation:**
- If Dir Acc > 50%: Continue with LSTM-GAT development
- If Dir Acc 30-50%: Consider architecture simplification
- If Dir Acc < 30%: Abandon LSTM-GAT, return to LSTM-HAR Enhanced

---

**Plan End**

**Next Action:** Start Phase 1 implementation (Data-Centric Techniques)
**Timeline:** Week 1, Day 1-2
**Expected Completion:** 2026-07-05
**Review Date:** 2026-07-05