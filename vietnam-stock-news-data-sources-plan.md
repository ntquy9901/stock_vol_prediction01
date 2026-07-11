# Vietnam Stock News Data Sources - Research & Implementation Plan

**Date:** 2026-06-28
**Status:** RSS feeds tested - NOT VIABLE
**Goal:** Crawl historical news for VN30 stocks (2006-2026) for sentiment analysis

---

## Summary of Findings

### Current Status: RSS Feeds Not Viable ❌

**Tested Sources:**
1. **Cafef.vn RSS Feeds** (`home.rss`, `category.rss`)
   - Total entries: 60
   - Stock-related: 0/60 (0%)
   - Content: General news (zodiac, health, beauty)

2. **CafeBiz.vn RSS** (`index.rss`)
   - Total entries: 0
   - Error: Feed not accessible

3. **Google News Sitemap** (`google-news-sitemap.xml`)
   - Contains article URLs with metadata
   - Some financial content (LPB stock, gold prices, banking)
   - Mostly general news
   - **Limitation:** URLs only, need full content crawling

**Conclusion:** RSS feeds are not a viable source for stock-specific news.

---

## Data Source Options Analysis

### Option 1: GitHub Dataset (REJECTED ❌)

**Source:** [Vietnamese-stock-article-classification](https://github.com/209sontung/Vietnamese-stock-article-classification)

**Dataset Characteristics:**
- Shape: (1005, 2)
- Columns: ['title', 'label']
- Distribution: 187 negative, 249 neutral, 569 positive
- Source: CafeF.vn (correct source)
- **CRITICAL LIMITATION:** No dates, no tickers, titles only

**Why Unsuitable:**
- ❌ No temporal dimension (can't create time series)
- ❌ No ticker information (can't align with stock data)
- ❌ Fixed set of 1,005 articles (not comprehensive for 2006-2026)
- ✅ Good for classification, NOT for time series features

**Verdict:** REJECTED - Cannot create daily sentiment features per ticker

---

### Option 2: Cafef.vn Direct Crawling (BLOCKED ⛔)

**Approaches Attempted:**

1. **Search URL Pattern** - FAILED
   ```
   https://cafef.vn/tim-kiem/{ticker}  → 404 Error
   ```

2. **Google Site Search** - FAILED
   ```
   https://www.google.com/search?q={ticker}+site%3Acafef.vn&tbm=nws
   Results: 0 articles
   ```

3. **RSS Feeds** - FAILED
   - 0% stock-related content
   - Mostly general news

**Current Status:** NO viable crawling method found for Cafef.vn

**Blockers:**
- No public API
- Search URLs return 404
- RSS feeds don't filter by stock
- Google search returns 0 results

---

### Option 3: Vietstock.vn (RECOMMENDED ✅)

**Potential Approach:**

1. **Investigate API Availability:**
   - Check if Vietstock.vn offers public API
   - Look for documentation on rate limits, authentication

2. **Alternative: Direct Crawling:**
   - Analyze HTML structure for article listings
   - Implement per-ticker crawler (similar to Cafef approach)
   - Parse article content, extract dates

3. **Timeline:** 1-2 weeks

**Advantages:**
- May have better API/crawling infrastructure
- Similar content to Cafef (Vietnamese financial news)

**Risks:**
- May also block automated crawling
- Unknown API availability

---

### Option 4: vnstock Package (PENDING ⏸️)

**Source:** [vnstock on PyPI](https://pypi.org/project/vnstock/0.1.1/)

**Approach:**
- Investigate if package provides news data
- Check documentation for news/sentiment features
- May already handle Cafef/Vietstock crawling

**Timeline:** 2-3 days (investigation + testing)

---

### Option 5: Third-Party News APIs (PENDING ⏸️)

**Potential Services:**
- NewsAPI.org (may have Vietnam business news)
- Bing News API (web search + news)
- Serper API (Google search programmatic access)

**Advantages:**
- Structured JSON responses
- Rate-limited but reliable
- No HTML parsing needed

**Disadvantages:**
- May require subscription/paid plan
- Limited historical data (usually 30 days)
- May not have Vietnamese financial news coverage

---

## Implementation Roadmap

### Phase 1: Quick Investigation (2-3 days)

**Priority 1: Check vnstock package**
```bash
pip install vnstock
py -c "import vnstock; print(dir(vnstock))"
```

**Priority 2: Research third-party APIs**
- NewsAPI.org documentation
- Bing News API coverage for Vietnam
- Serper API capabilities

**Priority 3: Manual inspection of Vietstock.vn**
- Visit site, check for API documentation
- Inspect HTML structure for article listings
- Check if search by ticker is possible

### Phase 2: Implementation (1-2 weeks)

**Based on Phase 1 findings:**

**If vnstock has news:**
- ✅ Use vnstock (fastest path)
- Implement sentiment pipeline
- Create daily features

**If Vietstock.vn is accessible:**
- Implement crawler (follow `src/sentiment/data_collection/crawlers/base_crawler.py` pattern)
- Test with 1 ticker first (e.g., VCB)
- Scale to all 30 VN30 stocks

**If third-party API works:**
- Integrate API client
- Handle rate limits, pagination
- Extract ticker mentions from content

### Phase 3: Sentiment Pipeline (1 week)

**Input:** Crawled news articles (title, content, date, ticker)

**Steps:**

1. **Extract Ticker Mentions:**
   - Use regex pattern matching for VN30 tickers
   - Handle variations (VCB, Vietcombank, NH TMCP CP Ngoai Thuoc Viet Nam)

2. **Generate Sentiment Scores:**
   - Use PhoBERT (Vietnamese BERT)
   - Or FinBERT (multilingual financial BERT)
   - Output: sentiment_score, confidence, label (neg/neu/pos)

3. **Create Daily Features:**
   - Aggregate sentiment by date and ticker
   - Features: sentiment_score, news_count, avg_confidence
   - Align with HAR dataset dates (2006-2026)

4. **Validate:**
   - Check coverage (% dates with news)
   - Check distribution (pos/neg/neu ratio)
   - Save to `data/processed/sentiment_features/{TICKER}_sentiment.csv`

---

## Technical Requirements

### Sentiment Models

**Option 1: PhoBERT (VinAIResearch)**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
```

**Option 2: FinBERT (Multilingual)**
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
```

### Feature Engineering

**Output Format (Per Ticker):**
```csv
date,sentiment_score_3d,sentiment_confidence,news_count_norm
2006-01-02,0.123,0.89,0.5
2006-01-03,0.456,0.92,0.8
...
```

**Integration with LSTM-GNN:**
- Current: 3 HAR features (daily, weekly, monthly volatility)
- Enhanced: 3 HAR + 3 sentiment features = 6 features per stock
- Architecture: Early fusion (sentiment as node features)

---

## Comparison Matrix

| Source | Viability | Timeline | Coverage | Historical Data | Cost |
|--------|-----------|----------|----------|-----------------|------|
| GitHub Dataset | ❌ Rejected | N/A | Fixed 1005 articles | ❌ No dates | Free |
| Cafef RSS | ❌ Not viable | N/A | 0% stock content | ❌ General news only | Free |
| Cafef Crawl | ⛔ Blocked | N/A | Unknown | ❌ No access method | Free |
| Vietstock.vn | ✅ Recommended | 1-2 weeks | Unknown | ✅ Likely 2006-2026 | Free |
| vnstock Package | ⏸️ Pending | 2-3 days | Unknown | Unknown | Free |
| NewsAPI.org | ⏸️ Pending | 3-5 days | Vietnam business | ❌ 30 days only | Free tier limited |
| Bing News API | ⏸️ Pending | 3-5 days | Vietnam business | ❌ 30 days only | Paid |

---

## Recommended Next Steps

### Immediate Actions (This Week)

1. **Investigate vnstock package** (2-3 days)
   ```bash
   pip install vnstock
   py -c "import vnstock; help(vnstock)"
   ```

2. **Manual inspection of Vietstock.vn** (1 day)
   - Check for API documentation
   - Test search functionality for ticker "VCB"
   - Inspect HTML structure

3. **Research third-party APIs** (1 day)
   - Read NewsAPI.org docs (Vietnam coverage)
   - Check Bing News API historical data limits
   - Evaluate pricing for full 2006-2026 data

### Decision Points

**If vnstock has news:**
- ✅ Use vnstock → Implement sentiment pipeline
- Timeline: 1 week total

**If Vietstock.vn is accessible:**
- ✅ Implement Vietstock crawler → Test with VCB → Scale to VN30
- Timeline: 2 weeks total

**If third-party API works:**
- ⚠️ Evaluate cost vs. benefit (may not have 2006-2026 historical)
- Timeline: 3-5 days investigation + 1 week implementation

**If ALL options fail:**
- ⚠️ Defer sentiment integration
- Complete LSTM-GNN baseline first (68.02% Dir Acc)
- Revisit sentiment after baseline complete

---

## User Decision Required

Based on the research, here are the current options:

**A. Continue with vnstock investigation** (2-3 days)
   - Quick check if package handles news data
   - Fastest path if successful

**B. Switch to Vietstock.vn crawler** (1-2 weeks)
   - Manual site inspection first
   - Implement crawler if accessible
   - More certain path but longer timeline

**C. Defer sentiment integration** (RECOMMENDED given baseline incomplete)
   - Focus on completing LSTM-GNN baseline (68.02% Dir Acc target)
   - Use HAR features only (no sentiment)
   - Revisit sentiment data after baseline complete

**D. Research alternative approaches** (1 week)
   - Check other Vietnamese financial news sites
   - Investigate Kaggle for Vietnamese datasets
   - Explore academic datasets (Mendeley, etc.)

---

**Current Recommendation:** Option C (Defer sentiment)
**Rationale:** Baseline LSTM-GNN not complete (68.02% Dir Acc, need 70%+). Sentiment integration is enhancement, not core requirement. Better to complete baseline first, then revisit data crawling with more time.

**Sources:**
- [Vietnamese Stock Article Classification Dataset](https://github.com/209sontung/Vietnamese-stock-article-classification)
- [Crawl News Stock From Cafef](https://github.com/bazzi24/Crawl_NewsStock_From_Cafef)
- [Vietnamese Banks Public Sentiment & Stock Performance](https://data.mendeley.com/datasets/3w8955hk5t/1)
- [vnstock Python Package](https://pypi.org/project/vnstock/0.1.1/)
- [NewsAPI.org Documentation](https://newsapi.org/docs)
- [Bing News API](https://www.microsoft.com/en-us/bing/apis/bing-news-search-api)
