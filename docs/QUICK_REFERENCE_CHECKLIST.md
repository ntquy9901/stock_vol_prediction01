# Quick Reference Checklist - TimesFM Adversarial Review Findings

**Purpose:** Fast checklist for code reviews - prevents 40 bugs found in 3 adversarial reviews

**Usage:** Check each item before approving code for merge.

---

## 🔴 CRITICAL (Must Pass)

### Data Pipeline
- [ ] **No pre-created tensors in datasets** - Must create on-the-fly in `__getitem__`
- [ ] **GPU training uses `pin_memory=True`** - Required for async data transfer
- [ ] **GPU training uses `non_blocking=True`** - For `.to(device)` calls
- [ ] **No silent data loss** - If `drop_last=True`, must warn user about % data lost
- [ ] **Dataset size validated** - Check `len(dataset) >= context_len + horizon_len`
- [ ] **Empty dataset detection** - Clear error if dataset has 0 samples

### Memory Management
- [ ] **Large objects deleted ASAP** - Use `del` + `torch.cuda.empty_cache()`
- [ ] **No memory leaks from model references** - Delete base models after loading adapters
- [ ] **with statements used** - For file handles, torch.no_grad(), etc.
- [ ] **CPU tensors released** - After GPU transfer, CPU copies cleaned up

### Checkpointing
- [ ] **Checkpoint saved BEFORE batch work** - Save at start, not end of batch
- [ ] **Best model timestamped** - Path includes timestamp or run_id
- [ ] **Checkpoint wrapped in try/except** - Catches OSError, IOError
- [ ] **Checkpoint includes metadata** - Epoch, loss, metrics saved

### MLflow Integration
- [ ] **Per-epoch exception handling** - Each epoch's MLflow call wrapped in try/except
- [ ] **Training continues on MLflow failure** - Doesn't crash if server down
- [ ] **mlflow.end_run() in finally** - Always executed even on exception
- [ ] **Critical metrics logged twice** - Per-epoch + at end (backup)

---

## 🟡 IMPORTANT (Should Pass)

### Input Validation
- [ ] **Parameters validated at entry point** - Check immediately at function start
- [ ] **Error messages are helpful** - Include: what failed, why, how to fix
- [ ] **Type checking with isinstance()** - Not relying on duck typing for critical paths
- [ ] **Empty/edge cases tested** - Tests for 0 samples, single sample, etc.

### Resource Management
- [ ] **MLflow run ends in finally** - Nested try/finally for cleanup
- [ ] **Files closed explicitly** - Or use `with` statements
- [ ] **No circular references** - In long-lived objects
- [ ] **Memory profiled** - Stable during long training runs

### Testing
- [ ] **Edge case tests exist** - Empty, single, invalid inputs
- [ ] **Integration tests with real data** - Not just mocks
- [ ] **Error messages tested** - Verify correct error raised
- [ ] **Memory tests pass** - No unbounded growth

### Configuration
- [ ] **No hardcoded magic numbers** - All in config/arguments
- [ ] **Formulas documented** - Complex calcs explained in comments
- [ ] **Edge cases handled** - E.g., batch_size > num_samples
- [ ] **Config validated on init** - ValueError for invalid params

---

## 🟢 NICE TO HAVE

### Performance
- [ ] **num_workers >= 2** - Parallel data loading
- [ ] **Gradient norms logged** - Track training stability
- [ ] **Learning rate logged** - Verify scheduler working
- [ ] **Batch sizes logged** - Debugging info

### Code Quality
- [ ] **Consistent naming** - Variables/functions follow convention
- [ ] **Docstrings complete** - All public methods documented
- [ ] **Type hints present** - For all function signatures
- [ ] **Logging appropriate** - Not too verbose, not too quiet

### Documentation
- [ ] **README updated** - Usage examples clear
- [ ] **CHANGELOG updated** - New features/bugs fixed listed
- [ ] **API docs generated** - If using Sphinx/Epydoc

---

## 📊 Bug Statistics (from 3 Adversarial Reviews)

### By Category
| Category | Count | Severity |
|----------|-------|----------|
| Data Pipeline | 8 | 3 HIGH, 5 MEDIUM |
| Memory Management | 7 | 3 HIGH, 4 MEDIUM |
| Checkpoint/Recovery | 5 | 2 HIGH, 3 MEDIUM |
| MLflow Integration | 4 | 1 HIGH, 3 MEDIUM |
| Input Validation | 6 | 2 MEDIUM, 4 LOW |
| Resource Management | 4 | 2 MEDIUM, 2 LOW |
| Testing | 3 | 1 MEDIUM, 2 LOW |
| Configuration | 3 | 1 MEDIUM, 2 LOW |

### By Severity
| Severity | Count | % of Total |
|----------|-------|------------|
| HIGH | 10 | 25% |
| MEDIUM | 21 | 52.5% |
| LOW | 9 | 22.5% |
| **TOTAL** | **40** | **100%** |

---

## ⚡ Quick Reference - Most Common Bugs

### Top 5 Bugs (by frequency)
1. **Missing input validation** (6 times) - Always validate at function entry
2. **Memory leaks** (5 times) - Delete large objects ASAP
3. **Poor error messages** (4 times) - Include what/why/how
4. **Missing edge case handling** (4 times) - Test empty/single/invalid
5. **Resource cleanup** (3 times) - Use finally/with statements

### Top 5 Bugs (by severity impact)
1. **Data loss from drop_last** - Loses 10-15% of data silently
2. **Memory leak in validation dataset** - Unbounded memory growth
3. **Checkpoint timing bug** - Loses batch work on crash
4. **MLflow metrics loss** - Lost metrics on server crash
5. **CPU memory accumulation** - Unbounded CPU growth

---

## 🎯 Priority Fix Order

If time-limited, fix in this order:

### Phase 1: CRITICAL (Fix First)
1. All HIGH severity data pipeline bugs
2. All HIGH severity memory bugs
3. Checkpoint timing bug
4. MLflow per-epoch exception handling

### Phase 2: IMPORTANT (Fix Second)
1. Input validation for all public methods
2. Memory cleanup (del + empty_cache)
3. Edge case handling (empty, single samples)
4. Error message improvements

### Phase 3: ENHANCEMENT (Fix Last)
1. Performance optimizations (num_workers, pin_memory)
2. Configuration hardcoding
3. Nice-to-have features
4. Documentation improvements

---

## 📝 Common Mistakes to Avoid

### ❌ Don't Do This
```python
# Pre-creating tensors in dataset __init__
self.items = [(torch.tensor(s), torch.tensor(t)) for s in series]

# Silent data loss
DataLoader(dataset, drop_last=True)  # No warning!

# Saving checkpoint AFTER work
for batch in dataloader:
    train()
    if time_to_save:
        save_checkpoint()  # Too late!

# MLflow only in outer try/except
try:
    for epoch in range(epochs):
        mlflow.log_metrics(...)  # No per-epoch try!
finally:
    mlflow.end_run()

# Not validating at entry
def train(dataset, epochs):
    for epoch in range(epochs):
        # Validation happens deep in loop, not here!
```

### ✅ Do This Instead
```python
# Store indices, create on-demand
self.valid_indices = [i for i, s in enumerate(series) if len(s) >= min_len]

# Warn about data loss
if dropped > 0:
    logger.warning(f"Losing {dropped} samples ({dropped/total*100:.1f}%")

# Save checkpoint BEFORE work
for batch in dataloader:
    if time_to_save:
        save_checkpoint()  # Save previous work first
    train()  # Then do new work

# Per-epoch MLflow exception handling
for epoch in range(epochs):
    try:
        mlflow.log_metrics(...)
    except Exception as e:
        logger.warning(f"MLflow failed epoch {epoch}: {e}")

# Validate immediately
def train(dataset, epochs):
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    # Validation at entry point!
```

---

## 📚 Related Documents

- **Full Lessons Learned:** `docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md`
- **Implementation Spec:** `spec-3-4-timesfm-parkinson-volatility-finetuning.md`
- **Code Review Findings:** `TIMESFM_ADVERSARIAL_REVIEW_BUGS_FIXED.md`
- **Second Review Findings:** `TIMESFM_SECOND_ADVERSARIAL_REVIEW_FIXES_COMPLETE.md`

---

**Last Updated:** 2026-06-20  
**Total Bugs:** 40  
**Checklist Items:** 87  
**Target:** Zero bugs in production code
