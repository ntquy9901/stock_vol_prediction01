# Archived Data Scripts

**Date Archived:** 2026-06-19
**Reason:** Not currently needed, but kept for reference

---

## Archived Files

### 1. crawl_vietnam_enhanced.py
**Status:** ❌ Performed worse than original
**Issue:** Lost 10 stocks due to rate limiting in parallel mode
**Original:** 208 stocks (90% success)
**Enhanced:** 198 stocks (86% success)
**Verdict:** Use original `crawl_vietnam_stocks.py` instead

**Why keep:** Reference for retry logic and parallel processing patterns
**When to use:** Only if Yahoo Finance changes rate limiting behavior

---

### 2. vietnam_web_scraper.py
**Status:** ⚠️ Framework not implemented
**Issue:** Web scraping needs implementation for Cafef/Vietstock/Fintext
**Verdict:** Not needed currently (208 stocks is sufficient)

**Why keep:** Framework ready for future implementation if needed
**When to use:** If need +10-15% more stocks and willing to spend 4 hours

---

### 3. quick_test_crawl.py
**Status:** ✅ Testing complete
**Issue:** Test script no longer needed
**Verdict:** Successfully tested 10/13 stocks (77% success rate)

**Why keep:** Quick test template for future data sources
**When to use:** Need to quickly test new data source or API

---

## Active Files (NOT Archived)

### ✅ crawl_vietnam_stocks.py (Original)
**Status:** Active - Best performing crawler
**Results:** 208 stocks (90% success rate)
**Location:** `src/data/crawl_vietnam_stocks.py`
**Why keep:** This is the best performing solution

### ✅ combine_datasets.py (Active)
**Status:** Active - Currently used
**Purpose:** Combine multiple datasets
**Location:** `src/data/combine_datasets.py`
**Why keep:** Needed for combining vn30, vn100, hnx datasets

---

## Restoration Guide

If you need to restore any archived file:

```bash
# Copy back to src/data/
cp archive/data_scripts/[filename].py src/data/

# Or move back
mv archive/data_scripts/[filename].py src/data/
```

---

## Summary

**Archived:** 3 files
**Active:** 2 files
**Reason:** Current data is sufficient (208 stocks), enhanced versions performed worse

**Next Steps:**
- Use original crawler for any future data needs
- Implement web scrapers only if additional data needed
- Focus on model training with current excellent dataset
