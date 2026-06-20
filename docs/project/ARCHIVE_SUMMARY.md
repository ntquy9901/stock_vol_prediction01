# Archive Cleanup Summary

**Date:** 2026-06-19
**Status:** ✅ Cleanup Complete

---

## What Was Archived

### 1. Data Scripts (`archive/data_scripts/`)

**Files archived:**
- ✅ `crawl_vietnam_enhanced.py` - Enhanced crawler (performed worse than original)
- ✅ `vietnam_web_scraper.py` - Web scraper framework (not implemented)
- ✅ `quick_test_crawl.py` - Quick test script (testing complete)

**Reason:**
- Enhanced crawler performed WORSE (-10 stocks vs original)
- Web scraper not needed (208 stocks is sufficient)
- Test script already used

**Active files kept:**
- ✅ `crawl_vietnam_stocks.py` - Original crawler (208 stocks, 90% success)
- ✅ `combine_datasets.py` - Dataset combiner (currently used)

---

### 2. Historical Documentation (`archive/historical_docs/`)

**Files archived:**
- ✅ `CLAUDE_OLD_BACKUP.md` - Old backup (1,798 lines vs 350 lines current)
- ✅ `LEARNING_RATE_FIX_SUMMARY_2026-06-19.md` - Historical summary
- ✅ `MODEL_COMPARISON_AFTER_LR_FIX_2026-06-19.md` - Historical comparison
- ✅ `MODEL_COMPARISON_TEMPORAL_SPLIT_2026-06-19.md` - Historical evaluation
- ✅ `PERFORMANCE_FIXES_IMPLEMENTED_2026-06-19.md` - Historical fixes
- ✅ `CRYPTOMAMBA_PHASE1_IMPLEMENTATION_SUMMARY.md` - Phase 1 summary
- ✅ `CRYPTOMAMBA_V2_IMPLEMENTATION_SUMMARY.md` - V2 summary
- ✅ `TRAINING_HYPERPARAMETERS_UPDATED_2026-06-19.md` - Hyperparameter changes

**Reason:**
- Historical records of completed work
- Superseded by current documentation
- Completed phases/fixes

**Active files kept:**
- ✅ `CLAUDE.md` - Current project rules (350 lines, streamlined)
- ✅ `CRAWL_ISSUES_FIXED.md` - Recent analysis (2026-06-19)
- ✅ `DATA_EXPANSION_COMPLETE.md` - Recent status (2026-06-19)
- ✅ `README.md` - Project overview
- ✅ `project-context.md` - Project background
- ✅ `CRYPTOMAMBA_FULL_IMPLEMENTATION_GUIDE.md` - Implementation guide

---

## Archive Structure

```
archive/
├── data_scripts/
│   ├── README.md
│   ├── crawl_vietnam_enhanced.py
│   ├── vietnam_web_scraper.py
│   └── quick_test_crawl.py
├── historical_docs/
│   ├── README.md
│   ├── CLAUDE_OLD_BACKUP.md
│   ├── LEARNING_RATE_FIX_SUMMARY_2026-06-19.md
│   ├── MODEL_COMPARISON_AFTER_LR_FIX_2026-06-19.md
│   ├── MODEL_COMPARISON_TEMPORAL_SPLIT_2026-06-19.md
│   ├── PERFORMANCE_FIXES_IMPLEMENTED_2026-06-19.md
│   ├── CRYPTOMAMBA_PHASE1_IMPLEMENTATION_SUMMARY.md
│   ├── CRYPTOMAMBA_V2_IMPLEMENTATION_SUMMARY.md
│   └── TRAINING_HYPERPARAMETERS_UPDATED_2026-06-19.md
├── debug_output.txt
└── old_train_scripts/
```

---

## Clean Project Root

### Before Cleanup
```
project_root/
├── CLAUDE.md (1,798 lines)
├── CLAUDE_OLD_BACKUP.md ❌
├── LEARNING_RATE_FIX_SUMMARY_2026-06-19.md ❌
├── MODEL_COMPARISON_AFTER_LR_FIX_2026-06-19.md ❌
├── MODEL_COMPARISON_TEMPORAL_SPLIT_2026-06-19.md ❌
├── PERFORMANCE_FIXES_IMPLEMENTED_2026-06-19.md ❌
├── CRYPTOMAMBA_*_SUMMARY.md ❌ (2 files)
├── TRAINING_HYPERPARAMETERS_UPDATED_2026-06-19.md ❌
└── [other active docs]
```

### After Cleanup
```
project_root/
├── CLAUDE.md ✅ (350 lines, streamlined)
├── CRAWL_ISSUES_FIXED.md ✅ (Recent analysis)
├── DATA_EXPANSION_COMPLETE.md ✅ (Recent status)
├── README.md ✅
├── project-context.md ✅
├── CRYPTOMAMBA_FULL_IMPLEMENTATION_GUIDE.md ✅
└── archive/ ✅ (New organized archive)
```

---

## Benefits

### 1. Cleaner Project Root
- **Before:** 15+ markdown files
- **After:** 6 core files
- **Improvement:** 60% reduction in clutter

### 2. Easier Navigation
- **Active docs:** Easy to find current information
- **Historical docs:** Archived but accessible
- **Clear distinction:** Current vs historical

### 3. Better Organization
- **Data scripts:** Unused scripts archived
- **Historical docs:** Completed work archived
- **Active files:** Only current/essential files

---

## Restoration Guide

### If you need archived files:

**1. Data Scripts:**
```bash
# Restore a specific script
cp archive/data_scripts/[script].py src/data/

# Or move back
mv archive/data_scripts/[script].py src/data/
```

**2. Historical Documentation:**
```bash
# Restore a specific document
cp archive/historical_docs/[doc].md ./
```

**3. Check archive contents:**
```bash
# List all archived items
find archive/ -type f -name "*.md" -o -name "*.py"

# Read archive READMEs
cat archive/data_scripts/README.md
cat archive/historical_docs/README.md
```

---

## Summary

**Total items archived:**
- **3 data scripts** (enhanced crawler, web scraper, test script)
- **8 historical documents** (summaries, comparisons, backups)

**Total items remaining in root:**
- **6 core markdown files** (project documentation)
- **2 active data scripts** (original crawler, dataset combiner)

**Improvement:**
- **60% reduction** in root directory files
- **Clear separation** of current vs historical
- **Organized archive** with READMEs

---

## Next Steps

✅ **Cleanup complete!**

**Current state:**
- ✅ Clean project root
- ✅ Organized archive structure
- ✅ Only essential files active
- ✅ Historical context preserved

**Ready to:**
- Focus on model training with clean codebase
- Easy navigation of current documentation
- Access historical context when needed

---

**Last Updated:** 2026-06-19
**Status:** ✅ Archive cleanup complete
