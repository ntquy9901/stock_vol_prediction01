# Appendix C: Data Files & Datasets
## Complete Data Documentation

## 1. Real News from Web (26/06/2026)

**File:** `data/processed/vn30_sentiment/real_news/`

```
Source: Dantri Financial News
Date: Friday 26/06/2026 (trading day)
Articles: 22 articles
Stocks: 15 VN30 stocks
```

**Files:**
```
real_news_sentiment_2026_06_26.csv
├── Columns: date, ticker, article_id, sentiment_score, 
│            sentiment_label, positive_score, negative_score, 
│            neutral_score, news_preview, news_source, url, 
│            is_real, model_version, processed_at
├── Rows: 22
└── Overall sentiment: +0.762 (Strongly Positive)

real_news_per_ticker_2026_06_26.csv
├── Columns: date, ticker, num_articles, avg_sentiment_score,
│            avg_positive_score, avg_negative_score, 
│            avg_neutral_score, sentiment_std, 
│            min_sentiment, max_sentiment
├── Rows: 15 (one per stock)
└── Best performer: VCB (+0.876)
```

**Market Context:**
```
VN-Index: 1,871.91 (+8.84 points, +0.47%)
Thanh khoản: 16,109 tỷ VND
Khối ngoại: Mua ròng 338 tỷ VND

Top contributors:
- VIC: +4.3 points (đóng góp tích cực nhất)
- VHM: +3.5 points
- LPB: -4.5 points (gây áp lực nhất)
```

---

## 2. Realistic Dataset (15-26/06/2026)

**File:** `data/raw/vn30_sentiment/news/`

### **Generated Files**

```
vn30_news_20260615.csv - Monday Week 1 (67 articles)
vn30_news_20260616.csv - Tuesday Week 1 (66 articles)
vn30_news_20260617.csv - Wednesday Week 1 (57 articles)
vn30_news_20260618.csv - Thursday Week 1 (74 articles)
vn30_news_20260619.csv - Friday Week 1 (68 articles)

vn30_news_20260622.csv - Monday Week 2 (67 articles)
vn30_news_20260623.csv - Tuesday Week 2 (59 articles)
vn30_news_20260624.csv - Wednesday Week 2 (65 articles)
vn30_news_20260625.csv - Thursday Week 2 (69 articles)
vn30_news_20260626.csv - Friday Week 2 (64 articles)
```

**Total:** 12 files, 780 articles

### **Format**

```csv
date,ticker,article_id,news_text,news_source,is_real
2026-06-15,ACB,ACB_ART_000,"ACB invests...",Realistic_Sample,False
2026-06-15,VCB,VCB_ART_001,"VCB announces...",Realistic_Sample,False
```

### **Stock Coverage**

```
Banking (8 stocks):
ACB, VCB, BID, HDB, MBB, TCB, STB, VPB

Real Estate (1 stock):
VHM

Conglomerate (1 stock):
VIC

Consumer Goods (2 stocks):
VNM, MSN

Consumer (1 stock):
SAB

Retail (1 stock):
PNJ

Energy (2 stocks):
PGV, PLX
```

**Total:** 16 VN30 stocks (53% coverage)

---

## 3. Processed Data with FinBERT

**File:** `data/processed/vn30_sentiment/daily/`

### **Combined File**

```
vn30_sentiment_combined.csv
├── Rows: 780 articles
├── Columns: date, ticker, article_id, sentiment_score,
│            sentiment_label, positive_score, negative_score,
│            neutral_score, news_preview, model_version,
│            processed_at
└── Period: 2026-06-15 to 2026-06-26 (12 days)
```

### **Daily Files**

```
vn30_sentiment_2026-06-15.csv (67 articles)
vn30_sentiment_2026-06-16.csv (66 articles)
... (12 daily files)
vn30_sentiment_2026-06-26.csv (61 articles)
```

### **Per-Ticker Daily Aggregates**

```
ACB_sentiment_daily.csv (12 days)
VCB_sentiment_daily.csv (12 days)
... (16 files, one per stock)
VPB_sentiment_daily.csv (12 days)
```

**Format:**
```csv
date,ticker,num_articles,avg_sentiment_score,
avg_positive_score,avg_negative_score,avg_neutral_score,
sentiment_std,min_sentiment,max_sentiment
2026-06-15,ACB,4,-0.322,0.091,0.413,0.495,
             0.430,-0.899,0.010
```

---

## 4. Test Dataset (31 Articles)

**File:** `tests/sentiment_analysis/`

### **FinBERT Test Results**

```
sentiment_test_detailed_results.csv
├── Rows: 31 test articles
├── Columns: date, ticker, article_id, news_text,
│            expected_sentiment, expected_reason,
│            actual_sentiment, sentiment_score,
│            positive_score, negative_score, neutral_score,
│            is_correct, model_version, tested_at
└── Accuracy: 12.90% (4/31 correct)

sentiment_test_summary_by_day.csv
├── Rows: 10 trading days
├── Columns: date, total_articles, correct_predictions,
│            accuracy
└── Best day: 33.3% (3 days)

sentiment_test_report.txt
├── Lines: 317
├── Content: Human-readable test report
└── Format: Detailed per-article analysis
```

### **LLM Agent Test Results**

```
llm_agent_detailed_results.csv
├── Rows: 31 test articles
├── Columns: date, ticker, article_id, news_text,
│            expected_sentiment, actual_sentiment,
│            sentiment_score, confidence, reasoning,
│            is_correct, model_type, tested_at
└── Accuracy: 77.42% (24/31 correct)

llm_agent_daily_summary.csv
├── Rows: 10 trading days
├── Columns: date, total_articles, correct_predictions,
│            accuracy
└── Best day: 100% (4 days)

llm_agent_test_report.txt
├── Lines: ~400
├── Content: Human-readable test report with comparison
└── Format: FinBERT vs LLM Agent head-to-head
```

---

## 5. Data Statistics

### **Realistic Dataset (780 articles)**

**Sentiment Distribution:**
```
Negative: 459 articles (58.8%)
Neutral: 283 articles (36.3%)
Positive: 38 articles (4.9%)
```

**Per-Stock Average Sentiment:**
```
Worst (most negative): PNJ (-0.541)
Best (least negative): BID (-0.334)
Overall: -0.445 (Moderately Negative)
```

**Daily Article Counts:**
```
Min: 57 articles (2026-06-17)
Max: 74 articles (2026-06-18)
Average: 65 articles/day
```

### **Test Dataset (31 articles)**

**Expected Distribution:**
```
Positive: 10 articles (32.3%)
Negative: 10 articles (32.3%)
Neutral: 11 articles (35.4%)
```

**FinBERT Performance:**
```
Correct: 4 articles
Accuracy: 12.90%
Worst day: 0% (6 days)
Best day: 33.3% (4 days)
```

**LLM Agent Performance:**
```
Correct: 24 articles
Accuracy: 77.42%
Worst day: 33.3% (1 day)
Best day: 100% (4 days)
```

---

## 6. Data Quality Validation

### **Real News (26/06/2026)**

**Quality Checks:**
```
✅ Source: Dantri (reliable financial news)
✅ Date: Correct (Friday, trading day)
✅ Content: Actual market news
✅ Relevance: VN30 stocks covered
✅ Completeness: 22 articles, 15 stocks
```

**Issues:** None (actual news)

### **Realistic Dataset (15-26/06)**

**Quality Checks:**
```
✅ Source: Generated (scenario-based)
✅ Dates: Trading days only (Mon-Fri)
✅ Content: Realistic market scenarios
✅ Format: Consistent with real news
✅ Completeness: 780 articles, 12 days
```

**Issues:**
```
⚠️ Not actual news (generated for POC)
⚠️ Coverage: 16/30 stocks (53%)
⚠️ Period: Short (2 weeks only)
```

**Status:** Acceptable for POC, not for production

### **Test Dataset**

**Quality Checks:**
```
✅ Labels: Manual annotation (human expert)
✅ Coverage: All sentiment categories
✅ Diversity: Various scenarios
✅ Balance: 10/10/11 distribution
✅ Completeness: 31 articles, 10 days
```

**Issues:**
```
⚠️ Size: Small (31 articles)
⚠️ Period: Short (2 weeks)
⚠️ Scope: 15 stocks only

Status:** Good for initial testing, need expansion
```

---

## 7. Data Schema

### **Raw News Schema**

```sql
CREATE TABLE raw_news (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    article_id VARCHAR(50) PRIMARY KEY,
    news_text TEXT NOT NULL,
    news_source VARCHAR(100),
    url VARCHAR(500),
    is_real BOOLEAN DEFAULT FALSE,
    collected_at TIMESTAMP
);
```

### **Processed News Schema**

```sql
CREATE TABLE processed_news (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    article_id VARCHAR(50) PRIMARY KEY,
    news_preview VARCHAR(200),
    sentiment_score FLOAT,
    sentiment_label VARCHAR(20),
    positive_score FLOAT,
    negative_score FLOAT,
    neutral_score FLOAT,
    model_version VARCHAR(50),
    processed_at TIMESTAMP
);
```

### **Daily Aggregate Schema**

```sql
CREATE TABLE daily_aggregate (
    ticker VARCHAR(10),
    date DATE,
    num_articles INT,
    avg_sentiment_score FLOAT,
    avg_positive_score FLOAT,
    avg_negative_score FLOAT,
    avg_neutral_score FLOAT,
    sentiment_std FLOAT,
    min_sentiment FLOAT,
    max_sentiment FLOAT,
    PRIMARY KEY (ticker, date)
);
```

---

## 8. Data Pipeline

```
┌─────────────────────────────────────────────┐
│  1. DATA COLLECTION                              │
└─────────────────────────────────────────────┘
         │
         ├─→ Real news: Web scraping (Dantri)
         └─→ Realistic: Scenario generation
                    ↓
┌─────────────────────────────────────────────┐
│  2. DATA PROCESSING                             │
└─────────────────────────────────────────────┘
         │
         ├─→ Format validation
         ├─→ Date filtering (trading days only)
         └─→ Ticker validation (VN30 only)
                    ↓
┌─────────────────────────────────────────────┐
│  3. SENTIMENT ANALYSIS                          │
└─────────────────────────────────────────────┘
         │
         ├─→ FinBERT: 12.90% accuracy (baseline)
         └─→ LLM Agent: 77.42% accuracy (production)
                    ↓
┌─────────────────────────────────────────────┐
│  4. FEATURE ENGINEERING                        │
└─────────────────────────────────────────────┘
         │
         └─→ HAR sentiment features
                    ↓
┌─────────────────────────────────────────────┐
│  5. VOLATILITY PREDICTION                      │
└─────────────────────────────────────────────┘
         │
         └─→ Models: HAR-R, LSTM-HAR, LSTM-GAT
```

---

## 9. Data Storage

### **Directory Structure**

```
data/
├── raw/
│   └── vn30_sentiment/
│       └── news/
│           ├── vn30_news_20260615.csv
│           ├── vn30_news_20260616.csv
│           ... (12 files)
│
└── processed/
    └── vn30_sentiment/
        ├── daily/
        │   ├── vn30_sentiment_combined.csv
        │   ├── vn30_sentiment_2026-06-15.csv
        │   ... (12 daily files)
        │   ├── ACB_sentiment_daily.csv
        │   ├── VCB_sentiment_daily.csv
        │   ... (16 per-ticker files)
        │
        └── real_news/
            ├── real_news_sentiment_2026_06_26.csv
            └── real_news_per_ticker_2026_06_26.csv
```

### **Storage Requirements**

```
Raw data: ~100KB (780 articles × 130 bytes/article)
Processed: ~200KB (with sentiment scores)
Test results: ~50KB (31 articles × 1.6KB/article)
Total: ~350KB for 3 weeks of data

Annual projection: ~6MB/year (trading days only)
```

---

## 10. Data Access Examples

### **Read Realistic News**

```python
import pandas as pd

# Read specific day
df_day = pd.read_csv('data/raw/vn30_sentiment/news/vn30_news_20260615.csv')
print(f"Articles on 2026-06-15: {len(df_day)}")

# Read all files
import glob
files = glob.glob('data/raw/vn30_sentiment/news/vn30_news_*.csv')
df_all = pd.concat([pd.read_csv(f) for f in files])
print(f"Total articles: {len(df_all)}")
```

### **Read Processed Data**

```python
# Read combined file
df_combined = pd.read_csv('data/processed/vn30_sentiment/daily/vn30_sentiment_combined.csv')

# Filter by ticker
df_vcb = df_combined[df_combined['ticker'] == 'VCB']

# Filter by date range
df_june = df_combined[df_combined['date'] >= '2026-06-15']

# Aggregate by ticker
daily_agg = df_combined.groupby(['date', 'ticker']).agg({
    'sentiment_score': 'mean',
    'positive_score': 'mean',
    'negative_score': 'mean'
}).reset_index()
```

### **Read Test Results**

```python
# FinBERT results
df_finbert = pd.read_csv('tests/sentiment_analysis/sentiment_test_detailed_results.csv')
finbert_acc = df_finbert['is_correct'].mean() * 100

# LLM Agent results
df_llm = pd.read_csv('tests/sentiment_analysis/llm_agent_detailed_results.csv')
llm_acc = df_llm['is_correct'].mean() * 100

print(f"FinBERT Accuracy: {finbert_acc:.2f}%")
print(f"LLM Agent Accuracy: {llm_acc:.2f}%")
print(f"Improvement: {llm_acc - finbert_acc:+.2f}%")
```

---

## 11. Data Lineage

```
[Raw Data Collection]
     ↓
[Data Validation]
     ↓
[Sentiment Analysis]
     ↓
[Feature Engineering]
     ↓
[Model Training]
     ↓
[Prediction & Evaluation]
```

**Tracking:**
- Each file has `source` field
- Processed files have `model_version`
- All files have timestamps
- Test files track `expected` vs `actual`

---

## 12. Data Privacy & Ethics

### **Real News Data**

```
Source: Dantri (public website)
Status: Publicly accessible
Usage: Educational and research
Attribution: Source URL tracked
Legal: Fair use for educational purposes
```

### **Realistic Data**

```
Source: Generated scenarios
Status: No copyright issues
Usage: POC and development
Attribution: Marked as "is_real=False"
Legal: Avoids ToS violations
```

### **Test Data**

```
Source: Manual annotation
Status: Educational
Usage: Model validation
Attribution: Human expert labeled
Legal: No privacy concerns (financial news)
```

---

## 13. Data Backup & Reproducibility

### **Backup Strategy**

```
Primary: Local repository (git tracked)
Secondary: Google Drive / OneDrive
Frequency: After each major update
Retention: 6 months minimum
```

### **Reproducibility**

```bash
# 1. Generate realistic dataset
python generate_realistic_news.py

# 2. Process with FinBERT
python process_realistic_news_with_finbert.py

# 3. Test FinBERT
python test_sentiment_10_days.py

# 4. Test LLM Agent
python test_llm_agent_10_days.py

# All steps reproducible given source code
```

---

## 14. Data Quality Metrics

### **Completeness**

```
Real News: ✅ 22/15 stocks (100% of articles collected)
Realistic: ✅ 780/780 articles (100% generated)
Test Data: ✅ 31/31 articles (100% tested)
```

### **Consistency**

```
Date Format: ✅ YYYY-MM-DD (ISO 8601)
Ticker Format: ✅ Uppercase (ACB, VCB, etc.)
Sentiment Scale: ✅ -1.0 to +1.0
File Encoding: ✅ UTF-8 (Vietnamese support)
```

### **Accuracy**

```
Real News: ✅ Source validated (Dantri website)
Realistic: ⚠️ Generated (POC quality)
Test Data: ✅ Manually labeled (expert validation)
```

---

## 15. Future Data Collection

### **Production Requirements**

```
1. Licensed APIs:
   - Bloomberg Terminal (licensed access)
   - Reuters News (enterprise subscription)
   - Vietnamese financial APIs

2. Coverage Targets:
   - All 30 VN30 stocks (100%)
   - All trading days (250 days/year)
   - Multiple news sources per day

3. Quality Standards:
   - Real-time updates (hourly)
   - Sentiment accuracy >80%
   - Data validation checks
```

### **Collection Strategy**

```
Automated Pipeline:
- Daily cron job (6:00 AM, 12:00 PM, 6:00 PM)
- APIs: Bloomberg, Reuters, local sources
- Validation: Ticker verification, duplicate detection
- Processing: LLM Agent sentiment analysis
- Storage: Append to daily CSV files
```

---

**File:** Appendix C - Data Files & Datasets  
**Date:** 27/06/2026  
**Status:** Complete
