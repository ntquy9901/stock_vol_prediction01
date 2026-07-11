# Comprehensive VN30 Historical News Crawl Plan

**Date:** 2026-06-28
**Purpose:** Crawl toàn bộ historical news cho 30 VN30 stocks (2006-2026)

---

## Current Data Status

**Existing Data:**
- **Date Range:** 2026-06-15 to 2026-06-30 (~15 days)
- **Stocks Covered:** ~17/30 VN30 stocks
- **Total Articles:** ~780 articles (estimated)
- **Location:** `data/processed/vn30_sentiment/daily/`

**Gap Analysis:**
| Metric | Current | Target | Gap |
|---------|---------|--------|-----|
| Date Range | ~15 days | ~20 years (2006-2026) | **~5,000 trading days** |
| Stocks Covered | ~17/30 | 30/30 | **13 stocks missing** |
| Total Articles | ~780 | ~50,000+ | **~49,000+ missing** |

---

## Crawl Strategy

### Option 1: Full Historical Crawl (2006-2026) — RECOMMENDED

**Scope:** Crawl toàn bộ news từ 2006 đến 2026 cho 30 stocks

**Approach:**
```
Phase 1: Recent News (2020-2026) — HIGH PRIORITY
  ├─ Date range: 2020-01-01 to 2026-06-30
  ├─ Sources: Cafef, Vietstock, Tinnhanhchungkhoan
  ├─ Estimated: ~30,000 articles
  └─ Timeline: 2-3 weeks

Phase 2: Historical News (2006-2019) — MEDIUM PRIORITY
  ├─ Date range: 2006-01-01 to 2019-12-31
  ├─ Sources: Archives, major news only
  ├─ Estimated: ~20,000 articles
  └─ Timeline: 3-4 weeks
```

**Pros:**
- ✅ Comprehensive coverage
- ✅ Enables long-term analysis
- ✅ Better model training (more data)

**Cons:**
- ❌ Time-consuming (5-7 weeks total)
- ❌ High computational cost
- ❌ May hit rate limits

---

### Option 2: Incremental Crawl (2020-2026) — FASTEST

**Scope:** Chỉ crawl recent news (2020-2026) — giai đoạn thị trường biến động mạnh

**Approach:**
```
Focus: 2020-01-01 to 2026-06-30
  ├─ Sources: Cafef, Vietstock, Tinnhanhchungkhoan
  ├─ Estimated: ~30,000 articles
  └─ Timeline: 2-3 weeks
```

**Pros:**
- ✅ Faster (2-3 weeks)
- ✅ Covers COVID-19 recovery period (high volatility)
- ✅ Sufficient for initial sentiment model training

**Cons:**
- ❌ Misses 2006-2019 data (14 years)
- ❌ Cannot analyze long-term historical patterns

---

### Option 3: Synthetic + Real — HYBRID

**Scope:** Kết hợp real crawl (2020-2026) + synthetic data (2006-2019)

**Approach:**
```
Phase 1: Real Crawl (2020-2026) — 2-3 weeks
  ├─ Cafef, Vietstock, Tinnhanhchungkhoan
  └─ ~30,000 articles

Phase 2: Synthetic Generation (2006-2019) — 1 week
  ├─ Sample articles → LLM rewrite → Synthetic news
  ├─ Preserve sentiment distribution
  └─ ~20,000 synthetic articles
```

**Pros:**
- ✅ Fastest (3-4 weeks total)
- ✅ Full coverage (2006-2026)
- ✅ Good for initial model testing

**Cons:**
- ❌ Synthetic data may not reflect real historical patterns
- ❌ Need validation on synthetic quality

---

## Technical Implementation

### News Sources (Vietnamese Financial)

**Priority Sources (High Article Volume):**
1. **Cafef.vn** — 5,000+ articles/year
   - URL: https://cafef.vn/tai-chinh-ngan-hang
   - Tickers: VN30 (explicitly mentioned)
   - Crawl: Daily RSS feed + category pages

2. **Vietstock.vn** — 3,000+ articles/year
   - URL: https://vietstock.vn/tai-chinh
   - Tickers: VN30 tagged
   - Crawl: Daily updates + tag search

3. **Tinnhanhchungkhoan.vn** — 2,000+ articles/year
   - URL: https://tinnhanhchungkhoan.vn/tin-tuc
   - Tickers: All stocks
   - Crawl: Daily feed

4. **Kenhtaichinh.vn** — 1,500+ articles/year
   - URL: https://kenhtaichinh.vn
   - Tickers: Major stocks
   - Crawl: Weekly summary

**Secondary Sources (Supplemental):**
5. **VnExpress.net** (Kinh doanh section) — 1,000+ articles/year
6. **Vietstock.vn** (Blog section) — Analysis articles

### Crawl Architecture

```python
# Proposed structure: src/sentiment/data_collection/comprehensive_crawler.py

class ComprehensiveVN30Crawler:
    """
    Comprehensive news crawler for VN30 stocks (2006-2026)
    
    Features:
    - Parallel crawling (multiple sources simultaneously)
    - Rate limiting (respect robots.txt)
    - Retry logic (handle network failures)
    - Incremental updates (resume from last crawl)
    - Deduplication (remove duplicate articles)
    """
    
    def __init__(self):
        self.sources = {
            'cafef': CafeFCrawler(),
            'vietstock': VietstockCrawler(),
            'tinnhanhchungkhoan': TinNhanChungKhoanCrawler(),
            'kenhtaichinh': KenhTaiChinhCrawler()
        }
        self.tickers = get_all_vn30_tickers()  # 30 stocks
        
    def crawl_date_range(self, start_date: str, end_date: str):
        """
        Crawl all news for date range
        
        Args:
            start_date: '2006-01-01'
            end_date: '2026-06-30'
        """
        all_articles = []
        
        for date in pd.date_range(start_date, end_date, freq='D'):
            daily_articles = []
            
            # Parallel crawl for all sources
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for source, crawler in self.sources.items():
                    future = executor.submit(
                        crawler.fetch_daily, 
                        date, 
                        self.tickers
                    )
                    futures.append(future)
                
                for future in futures:
                    daily_articles.extend(future.result())
            
            # Save daily
            self.save_daily(date, daily_articles)
            all_articles.extend(daily_articles)
            
            # Rate limiting: sleep 2s between days
            time.sleep(2)
        
        # Save combined
        self.save_combined(all_articles)
```

---

## Implementation Steps (Option 2: Fastest — 2020-2026)

### Week 1: Infrastructure + Recent Crawl

**Days 1-2: Setup crawler infrastructure**
```bash
# Install dependencies
pip install requests beautifulsoup4 lxml
pip install aiohttp aiofiles
pip install pandas numpy

# Create crawler modules
mkdir -p src/sentiment/data_collection/crawlers
touch src/sentiment/data_collection/crawlers/__init__.py
touch src/sentiment/data_collection/crawlers/cafef_crawler.py
touch src/sentiment/data_collection/crawlers/vietstock_crawler.py
touch src/sentiment/data_collection/crawlers/tinnhanh_crawler.py
touch src/sentiment/data_collection/comprehensive_crawler.py
```

**Days 3-5: Implement individual crawlers**
```bash
# Priority order:
1. cafef_crawler.py — Most articles
2. vietstock_crawler.py — Second most
3. tinnhanh_crawler.py — Good coverage
```

**Days 6-7: Test crawlers**
```bash
# Test on recent dates (2026-06-01 to 2026-06-30)
python src/sentiment/data_collection/comprehensive_crawler.py \
    --start 2026-06-01 --end 2026-06-30 --test
```

### Week 2: Production Crawl (2020-2026)

**Days 1-7: Full crawl**
```bash
# Crawl 2020-2023 (Year by Year)
python comprehensive_crawler.py --start 2020-01-01 --end 2023-12-31

# Crawl 2024-2026
python comprehensive_crawler.py --start 2024-01-01 --end 2026-06-30
```

**Expected Output:**
- Total articles: ~30,000
- Storage: ~500MB (raw text) + ~50MB (processed)
- Processing time: ~5-7 days (depending on rate limits)

### Week 3: Sentiment Processing

**Days 1-3: Run FinBERT/LLM Agent**
```bash
# Process all crawled articles
python src/sentiment/agents/llm_sentiment_agent.py \
    --input data/sentiment/raw/ \
    --output data/sentiment/processed/
```

**Days 4-5: Create sentiment features**
```bash
# Align with HAR dataset
python src/sentiment/processing/sentiment_integration.py
```

**Days 6-7: Validate & Quality Check**
```bash
# Validate sentiment coverage
python tools/validate_sentiment_coverage.py

# Check quality metrics
python tools/sentiment_quality_report.py
```

---

## Estimated Timeline & Resources

| Phase | Duration | Output | Resources |
|-------|----------|--------|------------|
| **Infrastructure** | Week 1 | Crawlers ready | Developer: 40h |
| **Production Crawl** | Week 2 | 30,000 articles | Developer: 20h, Server: 168h (continuous) |
| **Sentiment Processing** | Week 3 | Sentiment features | GPU: 40h, Developer: 20h |
| **Total** | **3 weeks** | **Ready for training** | **~80h + GPU time** |

---

## Risk Mitigation

### Risk 1: Rate Limiting (HIGH)
**Mitigation:**
- Implement exponential backoff (1s → 2s → 4s → 8s)
- Use rotating proxies (if available)
- Crawl during off-peak hours (Vietnam: 10PM-6AM)

### Risk 2: Website Structure Changes (MEDIUM)
**Mitigation:**
- Build flexible parsers (CSS selectors, not hard-coded paths)
- Implement daily validation tests
- Fallback to archive sources

### Risk 3: Missing Historical Data (HIGH)
**Mitigation:**
- Focus on 2020-2026 (most recent, most relevant)
- Use Google Cache/API for older data
- Accept gaps in 2006-2019 (use synthetic or skip)

### Risk 4: Storage/Cost (LOW)
**Mitigation:**
- Raw text: ~500MB (negligible)
- Processed features: ~50MB (negligible)
- No database needed (CSV files sufficient)

---

## Success Criteria

**Coverage Targets:**
- [ ] All 30 VN30 stocks covered
- [ ] At least 1 article/day/stock (average)
- [ ] Date range: 2020-2026 covered
- [ ] Total articles: ≥ 20,000

**Quality Targets:**
- [ ] Sentiment accuracy: ≥ 75% (sample validation)
- [ ] Duplicate rate: < 5%
- [ ] Missing data: < 10% per stock

**Integration Targets:**
- [ ] Sentiment features align with HAR dataset
- [ ] Ready for LSTM-GNN training
- [ ] Documented in research report

---

## Recommendation: Start with Option 2 (2020-2026)

**Reason:**
1. Fastest time to value (3 weeks vs 7 weeks)
2. Covers most volatile period (COVID-19 + recovery)
3. Sufficient data for initial model training (~30,000 articles)
4. Can expand to 2006-2019 later if needed

**After Option 2 completes:**
- Train LSTM-GNN + Sentiment baseline (Phase 1)
- Validate results
- If good, expand to 2006-2019 (Option 1 full historical)

---

## Next Steps

**Immediate Actions (This Week):**
1. ✅ Create crawler infrastructure files
2. ✅ Implement Cafef crawler (highest priority)
3. ✅ Test crawler on recent dates (2026-06-01 to 2026-06-30)
4. ✅ Validate crawler output format

**Week 2 Actions:**
1. Implement remaining crawlers (Vietstock, Tinnhanhchungkhoan)
2. Run production crawl (2020-2026)
3. Monitor and fix issues

**Week 3 Actions:**
1. Process crawled articles with LLM Agent
2. Create sentiment features
3. Validate and document

---

**Ready to implement?** Choose Option 2 for fastest results or Option 1 for comprehensive coverage.
