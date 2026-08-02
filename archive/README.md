# Archive Directory

> **⛔ OUT OF SCOPE FOR ALL AUDITS AND REVIEWS.** Everything under `archive/` (any depth) is
> intentionally retired — not in active use, not part of the paper, not a target for fixes.
> Code review, adversarial audits, lint, coverage, and any AI-driven "find issues"/"audit the
> repo" pass MUST skip this entire tree. Do not report findings, propose fixes, or flag bugs for
> anything under `archive/` — if you're an AI agent auditing this repo and you weren't explicitly
> asked to review `archive/` itself, treat it as if it doesn't exist. See CLAUDE.md's Definition
> of Done → "Audit/Review scope" for the project-wide rule this enforces.

This directory contains deprecated and archived code from the stock volatility prediction project.

---

## 📁 Archive Structure

```
archive/
├── data_leakage_scripts/     ← Scripts with CRITICAL data leakage bugs
│   ├── README.md             ← Detailed documentation
│   ├── train_all_baselines.py
│   ├── train_enhanced_lstm_har_vn30.py
│   └── train_enhanced_lstm_har_vn30_progress.py
│
├── baselines/                ← Full timestamped baselines no longer in scope for the paper
│   └── 2026-07-18_resttext_baseline/  ← Best raw QLIKE (0.5431) of all 23 baselines, but not
│                                         seed-verified and dropped from paper scope 2026-08-02
│                                         (user decision) in favor of the per-ticker-gate
│                                         news-fusion lineage. Not a bug archive — moved here
│                                         only to shrink `baselines/`'s active scope.
│
├── data_raw/                 ← Raw price folders confirmed unused by any live script
│   │                            (verified via repo-wide grep before moving, 2026-08-02)
│   ├── hnx/, hnx_enhanced/   ← HNX exchange data, unrelated to the VN30 project scope.
│   │                            Was gitignored in its old location (data/raw/hnx/) — .gitignore
│   │                            updated to the new path, so these stay untracked here too.
│   ├── vn30_enhanced/        ← Duplicate ticker list of data/raw/vn30/ (the one actually used),
│   │                            referenced by nothing except a one-off comparison script
│   │                            (src/experiment/compare_crawl_results.py).
│   ├── vn100/, vn100_enhanced/ ← VN100 track dropped from project scope entirely, 2026-08-02
│   │                            (user decision — "không xử lý vn100"). Was gitignored in its old
│   │                            location; .gitignore updated to the new path.
│   └── test/, test_combined/ ← Only referenced by an already-archived script
│                                 (archive/data_scripts/quick_test_crawl.py).
│
├── data_processed/
│   └── vn100_only/           ← Processed output of the now-archived VN100 raw data, 2026-08-02.
│
├── vn100_scripts/            ← 5 VN100-exclusive root-level scripts, 2026-08-02 (user decision):
│                                 train_and_test_vn100.py, process_vn100_data.py, evaluate_vn100.py,
│                                 evaluate_vn100_simple.py, evaluate_vn100_train_only.py.
│                                 Confirmed not imported by any other live script before moving.
│                                 NOT archived: src/experiment/train_with_config.py,
│                                 src/data/crawl_vietnam_stocks.py, src/data/combine_datasets.py,
│                                 src/sentiment/data_collection/tickers.py — these are generic,
│                                 multi-universe utilities also used for VN30/other scopes, not
│                                 VN100-exclusive, so archiving them would remove VN30 functionality.
│
│   NOT archived: data/raw/all_available/ + process_all_available.py + data/processed_all/ — a
│   distinct, broader "generalization test" universe (210 stocks), not literally "VN100". Flagged
│   but left alone pending explicit instruction, per the same scope-matching principle above.
│
└── [future archives]
```

---

## 🚨 Data Leakage Scripts Archive

**Location:** `archive/data_leakage_scripts/`

**Status:** 🔴 **CRITICAL - DO NOT USE**

**What Happened:**
These scripts used `torch.utils.data.random_split` for time series data, causing:
- Future data to leak into training set
- Inflated performance metrics
- 17.8x worse test performance in reality

**Impact:**
- Test RMSE: 0.009943 (with leakage) vs 0.000557 (correct) → **17.8x worse**
- Val-Test Gap: 0.00937 (with leakage) vs 0.000094 (correct) → **99x larger**

**For Detailed Information:**
See: `archive/data_leakage_scripts/README.md`

---

## ✅ Safe to Use (Not Archived)

These files are CORRECT and use proper temporal split:

### In Project Root:
- ✅ `train_all_models_vn30.py` - Calls correct scripts
- ✅ `train_all_with_validation.py` - Calls correct scripts

### In Source Directories:
- ✅ `src/lstm_har_enhanced/train_with_overfitting_prevention.py` - Best choice
- ✅ `src/lstm_har_enhanced/archive/train_with_validation_DEPRECATED_2026-06-20.py` - Old version
- ✅ `src/lstm_baseline/train_with_validation.py` - Temporal split
- ✅ `src/lstm_har_baseline/train_with_validation.py` - Temporal split

---

## 📚 Key Rules

### Time Series Data Splitting:

**❌ NEVER:**
```python
# Random split shuffles time → data leakage
train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
```

**✅ ALWAYS:**
```python
# Temporal split maintains chronology
from src.common.temporal_split import TemporalSplitter
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train, val, test = splitter.create_dataloaders(dataset)
```

---

## 🎯 Purpose of Archive

1. **Educational:** Examples of what NOT to do
2. **Historical:** Track project evolution
3. **Comparison:** Show before/after fixes
4. **Prevention:** Prevent future mistakes

---

**Last Updated:** 2026-06-21
**Maintained by:** Stock Volatility Prediction Team
