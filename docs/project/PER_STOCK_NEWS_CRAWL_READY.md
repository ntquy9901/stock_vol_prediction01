# Per-Stock News Crawl — Quick Start Guide

**Date:** 2026-06-28
**Status:** ✅ Infrastructure Ready — Ready to crawl

---

## What Was Created

### New Files (3):
1. `src/sentiment/data_collection/crawlers/base_crawler.py` — Base crawler class
2. `src/sentiment/data_collection/crawlers/cafef_crawler.py` — Cafef.vn crawler
3. `src/sentiment/data_collection/crawlers/__init__.py` — Package init
4. `src/sentiment/data_collection/per_stock_crawl.py` — Main orchestrator

### Folder Structure:
```
data/sentiment/raw/
├── VCB/
│   ├── 2020-01.csv  (month 2020-01)
│   ├── 2020-02.csv
│   └── ...
├── VNM/
│   ├── 2020-01.csv
│   └── ...
├── VIC/
│   └── ...
└── ... (one folder per ticker)
```

---

## How to Use

### Option 1: Test Single Ticker (Recommended First)

```bash
# Test VCB for 1 month (June 2026)
python src/sentiment/data_collection/per_stock_crawl.py \
    --ticker VCB \
    --start 2026-06-01 \
    --end 2026-06-30
```

**Expected Output:**
```
[2026-06-28 10:00:00] CRAWLING VCB news from 2026-06-01 to 2026-06-30
Fetching VCB news...
  Fetched: VCB completes 12.5 trillion VND profit...
  Fetched: Ngân hàng nhà nước duyệt nhất thị trường...
✅ Successfully crawled 45 articles for VCB
   Location: data/sentiment/raw/VCB/
```

### Option 2: Crawl Multiple Tickers

```bash
# Crawl 3 banks (VCB, VNM, VIC) for 2020-2023
python src/sentiment/data_collection/per_stock_crawl.py \
    --tickers VCB VNM VIC \
    --start 2020-01-01 \
    --end 2023-12-31
```

### Option 3: Crawl All VN30 (Full Production)

```bash
# Crawl all 30 tickers for 2020-2026
python src/sentiment/data_collection/per_stock_crawl.py \
    --all \
    --start 2020-01-01 \
    --end 2026-06-30
```

**Expected Timeline:**
- **1 ticker (1 month):** ~2-5 minutes
- **3 tickers (3 years):** ~10-20 minutes
- **30 tickers (6 years):** ~2-4 hours

---

## Folder Organization (Per-Stock)

Each ticker gets its own folder:

```
data/sentiment/raw/{TICKER}/
├── 2020-01.csv  (articles from Jan 2020)
├── 2020-02.csv  (articles from Feb 2020)
├── 2020-03.csv
├── ...
└── 2026-06.csv  (most recent)
```

**CSV Structure (per month file):**
```csv
title,url,date,content,ticker,source,author,tags
"VCB completes profit...", "https://...", "2020-01-15", "VCB completes profit...", "VCB", "cafef", "Author", "tag1,tag2"
```

---

## Date Alignment with Price Data

**Price Data Location:** `data/processed/{TICKER}_processed.csv`
- Columns: `date`, `parkinson_volatility`
- Date range: 2006-2026 (~3,326 days per stock)

**News Data Location:** `data/sentiment/raw/{TICKER}/{YYYY-MM}.csv`
- Filter: Automatically filters articles by date range
- Only keeps articles within specified date range

**Matching:**
- Both datasets use same date format: `YYYY-MM-DD`
- Sentiment features will be aligned during processing phase

---

## Testing Checklist

Before full production crawl, validate:

- [ ] **Single ticker test** — Run Option 1 for VCB June 2026
- [ ] **Check output file** — Verify `data/sentiment/raw/VCB/2026-06.csv` exists
- [ ] **Validate content** — Open CSV, check if articles make sense
- [ ] **Check date alignment** — Verify dates match price data range
- [ ] **Rate limiting test** — Confirm no block/ban from Cafef

---

## Rate Limiting & Best Practices

**Built-in Protections:**
- ⏱️ Rate limit: 2 seconds between requests (configurable)
- 🔄 Retry logic: Up to 3 attempts with exponential backoff
- 📦 Per-month saving: Reduces file size, easier to manage

**Tips to Avoid Blocking:**
1. **Start small** — Test 1 ticker, 1 month first
2. **Crawl during off-peak hours** — Vietnam: 10PM-6AM (less traffic)
3. **Monitor logs** — If many "Failed to fetch" errors, increase rate_limit
4. **Don't crawl all at once** — Break into batches (e.g., 10 tickers at a time)

---

## Troubleshooting

**Error: "Failed to fetch"**
```bash
# Solution: Increase rate limit
python per_stock_crawl.py --ticker VCB --start 2026-06-01 --end 2026-06-30
# Then manually edit cafef_crawler.py: self.rate_limit = 5.0
```

**Error: "No articles found"**
```bash
# Possible causes:
# 1. Wrong ticker symbol (check VN30 list)
# 2. Date range with no news (check manually on cafef.vn)
# 3. Website structure changed (update selectors)
```

**Error: "Import issues"**
```bash
# Ensure crawlers package is initialized
touch src/sentiment/data_collection/crawlers/__init__.py
```

---

## Next Steps After Crawl

**After successful crawl, process articles:**

```bash
# Step 1: Generate sentiment features
python src/sentiment/processing/sentiment_integration.py

# Step 2: Validate alignment with HAR dataset
python tools/validate_sentiment_coverage.py

# Step 3: Train sentiment-enhanced model (after baseline complete)
python src/lstm_gat_hybrid/train_parallel_enhanced.py \
    --graph_method knn \
    --load_sentiment \
    --sentiment_dir data/processed/sentiment_features
```

---

**Ready to test?** Run Option 1 first (single ticker, 1 month) to validate!
