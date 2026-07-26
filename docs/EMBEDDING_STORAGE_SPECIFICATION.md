# News Article Embedding Storage Specification

**Document Version:** 1.0  
**Date:** 2026-07-26  
**Status:** Active (used in 2026-07-25+ baselines)

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Parquet File Format](#parquet-file-format)
4. [Encoding Pipeline](#encoding-pipeline)
5. [Multi-Project Reuse](#multi-project-reuse)
6. [Train/Val/Test Split Handling](#trainvalstest-split-handling)
7. [Usage Examples](#usage-examples)
8. [Maintenance & Versioning](#maintenance--versioning)

---

## Overview

### **Purpose**

Store pre-computed PhoBERT embeddings for news articles in a **persistent, reusable, chunked-encoded cache** that:
- ✅ Never re-encodes the same URL (efficient, GPU cost amortized)
- ✅ Survives crashes mid-encoding (atomic writes per chunk)
- ✅ Works across multiple projects/models without duplication
- ✅ Supports multiple filtering use cases (ticker-mentioning vs. all articles)
- ✅ Handles large sources (up to 1.2M articles/source) without memory overflow

### **Core Numbers**

| Property | Value | Notes |
|----------|-------|-------|
| **Embedding Model** | PhoBERT-base (`vinai/phobert-base`) | Frozen [CLS] token |
| **Raw Dimension** | 768 | PhoBERT hidden_size |
| **Reduced Dimension** | 32 | PCA reduction (per baseline) |
| **Total Articles Cached** | 7,494,266 | As of 2026-07-26 (--include_all mode) |
| **Cache Size** | 34 GB | 60 per-source parquet files |
| **Encoding Time (GPU)** | ~3.02 hours | On RTX 4060 Laptop, batch_size=256 |
| **Encoding Speed (GPU)** | ~1,052 articles/sec | Plateau at batch_size=256+ |

---

## Directory Structure

### **Layout**

```
data/external_news_embeddings/
├── raw_cache/                          # ← Main cache (34GB as of 2026-07-26)
│   ├── news_emb_articles_baodautu.parquet
│   ├── news_emb_articles_baophapluat.parquet
│   ├── news_emb_articles_bnews.parquet
│   ├── news_emb_articles_cafebiz.parquet
│   ├── news_emb_articles_cafef.parquet
│   ├── news_emb_articles_cand.parquet
│   ├── news_emb_articles_coin68.parquet
│   ├── news_emb_articles_dantri.parquet
│   ├── news_emb_articles_fica.parquet
│   ├── news_emb_articles_forum.parquet
│   ├── news_emb_articles_giaoducthoidai.parquet
│   ├── news_emb_articles_hanoimoi.parquet
│   ├── news_emb_articles_hsc.parquet
│   ├── news_emb_articles_khach_quan.parquet  # [deprecated, for legacy compatibility]
│   ├── ... (60 files total)
│   └── news_emb_articles_[SOURCE].parquet
│
├── raw_cache_backup_2026-07-25/        # ← Backup: ticker-only version
│   └── ... (same structure, ~4.4GB)
│
└── raw_cache_backup_2026-07-25_pre_include_all/  # ← Backup: pre---include_all version
    └── ... (same structure, ~4.4GB)
```

### **Naming Convention**

```
news_emb_articles_{SOURCE_NAME}.parquet
```

**Examples:**
- `news_emb_articles_cafef.parquet` (CafeF securities portal)
- `news_emb_articles_dantri.parquet` (DanTri general news)
- `news_emb_articles_vov.parquet` (VOV state broadcaster)

**Why per-source?**
- Incremental updates (encode only new articles from a source, upsert atomically)
- Isolation (corrupt one source → don't lose others)
- Discovery (source classification list can grow; cache adapts)

---

## Parquet File Format

### **Column Schema**

Each parquet file contains:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `url` | string | Article URL (unique key) | `"https://cafef.vn/..."`  |
| `raw_0` | float32 | PhoBERT embedding dim 0 | `-0.1234` |
| `raw_1` | float32 | PhoBERT embedding dim 1 | `0.5678` |
| ... | ... | ... | ... |
| `raw_767` | float32 | PhoBERT embedding dim 767 | `0.0089` |

### **Full Example (Conceptual)**

```
┌─────────────────────────────────┬──────────┬──────────┬─────┬──────────┐
│ url                             │ raw_0    │ raw_1    │ ... │ raw_767  │
├─────────────────────────────────┼──────────┼──────────┼─────┼──────────┤
│ https://cafef.vn/article-123    │ -0.1234  │ 0.5678   │ ... │ 0.0089   │
│ https://cafef.vn/article-456    │ 0.4567   │ -0.2345  │ ... │ -0.0123  │
│ https://cafef.vn/article-789    │ -0.3456  │ 0.1234   │ ... │ 0.0567   │
│ ...                             │ ...      │ ...      │ ... │ ...      │
└─────────────────────────────────┴──────────┴──────────┴─────┴──────────┘

Rows: ~10,000 to ~500,000+ per source
Columns: 769 (url + 768 embedding dims)
Size: ~30MB to ~4.8GB per source
```

### **Properties**

- **Compression:** Parquet default (Snappy)
- **Index:** URL is the logical key (used for cache-miss detection, not stored as physical index)
- **Sorting:** Arbitrary (determined by encoding order)
- **Data Type Safety:** All floats are float32 (not float64) → memory efficient

---

## Encoding Pipeline

### **High-Level Flow**

```
crawl_data/data (raw scraped articles)
    ↓
[Step 1] Discover + clean (title + lead text, URL dedup)
    ├─ khach_quan sources (mainstream press) 
    └─ tong_hop sources (securities/analyst)
    ↓
[Step 2] Filter (ticker-mention vs. all articles)
    ├─ ticker_mentioning: df[df["_text"].contains(TICKER_PATTERN)]  
    └─ all_articles: include every article with text
    ↓
[Step 3] Check existing cache (lookup by URL)
    ├─ Found → skip (already encoded)
    └─ Not found → send to PhoBERT
    ↓
[Step 4] Encode (PhoBERT → 768-dim vector)
    ├─ Chunk into batches (batch_size=32..512)
    └─ GPU-accelerate if available (cuda or cpu fallback)
    ↓
[Step 5] Write (atomic per-chunk write, upsert dedup)
    ├─ Create {url, raw_0..raw_767} dataframe
    ├─ Concat with existing cache
    ├─ Dedup by URL (keep first)
    └─ Atomic write (.parquet.tmp → .parquet rename)
    ↓
data/external_news_embeddings/raw_cache/ (persistent cache)
```

### **PhoBERT Encoding Details**

**Model & Tokenization:**
```python
from transformers import AutoTokenizer, AutoModel

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

# Input text → tokenize
text = "VnIndex tăng 1.2% do PSI và FPT kéo động"
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=64  # truncate at 64 tokens
)

# Forward pass → [CLS] token extraction
with torch.no_grad():
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)

# Extract numpy
embedding_768d = cls_embedding.cpu().numpy()  # shape: (1, 768) → float32
```

**Batch Processing:**
```python
# Process 32 texts at once → (32, 768) output
texts = [text_1, text_2, ..., text_32]
embeddings = extract_phobert_embeddings(texts, batch_size=32)  # (32, 768)
```

### **Implementation (from `build_incremental_cache.py`)**

**Entry point:**
```bash
# Dry run (count only, no PhoBERT calls)
python build_incremental_cache.py --dry_run

# Encode ticker-mentioning articles only
python build_incremental_cache.py

# Encode ALL articles (market-wide/macro use case)
python build_incremental_cache.py --include_all

# Fine-tune chunk size (trade-off: memory vs. checkpoint granularity)
python build_incremental_cache.py --chunk_size 100000

# Limit to specific sources
python build_incremental_cache.py --sources cafef hsc vnexpress
```

**Chunked encoding (handles huge sources):**
```python
def run_source(source, path, batch_size=32, include_all=False, chunk_size=5000):
    """Encode + upsert one source, chunked."""
    articles = _all_articles(source, path) if include_all else _ticker_mentioning_articles(source, path)
    existing = _load_existing_cache(source)
    new_articles = _new_rows_to_encode(articles, existing)
    
    if new_articles.empty:
        return {"source": source, "n_new": 0, ...}
    
    combined = existing
    n_encoded = 0
    
    # Process in chunks
    for start in range(0, len(new_articles), chunk_size):
        chunk = new_articles.iloc[start : start + chunk_size]
        new_rows = _encode_rows(chunk, batch_size=batch_size)  # PhoBERT call
        combined = pd.concat([combined, new_rows], ignore_index=True) if not combined.empty else new_rows
        combined = combined.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
        _atomic_write(combined, _article_cache_path(source))  # Write checkpoint
        n_encoded += len(new_rows)
    
    return {"source": source, "n_new": n_encoded, "n_after": len(combined)}
```

**Atomic write (crash-safe):**
```python
def _atomic_write(df, path):
    """Write to .tmp, rename if successful → crash never leaves truncated cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic on POSIX; on Windows, may need os.replace()
```

---

## Multi-Project Reuse

### **Why Reusable**

✅ **URL is universal** — Same article URL appears in multiple crawls/projects.  
✅ **Embedding is deterministic** — PhoBERT [CLS] on the same text always produces the same 768-d vector.  
✅ **Dimension is fixed** — All embeddings are 768-d; PCA reduction is per-project, not per-article.  
✅ **Source-per-file design** — Easy to slice/share individual sources without replicating entire cache.

### **Reuse Scenarios**

**Scenario 1: Multiple stock-forecasting projects**
```
Project A (VN30 volatility prediction)
├── uses: data/external_news_embeddings/raw_cache/
└── trains: 2026-07-25_dual_group_news_embedding_baseline, ...

Project B (VNM (Vinamilk) price prediction)
├── uses: same data/external_news_embeddings/raw_cache/ (symlink or shared mount)
└── trains: custom news models for VNM only

→ Same cache, different per-project PCA fits, different feature dimensions (32 for project A, 64 for project B)
```

**Scenario 2: Cross-project embeddings**
```
Project C (earnings sentiment analysis for VN30)
├── download: baselines/2026-07-25_expand_news_cache_baseline/
├── extract: data/external_news_embeddings/raw_cache/ 
├── reuse: identical embeddings, but aggregate differently (per-earnings-event instead of per-date)
└── no re-encode needed
```

**Scenario 3: Combining projects' encodings**
```
Project D (combined VN30 + sector fundamentals)
├── base cache: data/external_news_embeddings/raw_cache/ (from Project A)
├── extend: python build_incremental_cache.py --sources [NEW_SOURCES] 
│   → encode only new sources/articles discovered in project D's crawl
├── result: single unified cache, URLs deduplicated, no double-encoding
└── cost: only new URLs, not full re-encode
```

---

## Train/Val/Test Split Handling

### **Critical: Temporal Split, Not Random**

⚠️ **NEVER use random split with time-series news** — causes data leakage!

**Wrong (data leakage):**
```python
# ❌ Random split allows future news in training
train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
```

**Right (temporal split):**
```python
# ✅ Chronological split: train dates < val dates < test dates
train_cutoff = "2020-01-01"  # e.g., 70% of data up to this date
val_cutoff = "2022-01-01"    # 15% of data between cutoff and val_cutoff
# Remaining: test (15% after val_cutoff)

train_news = cache[cache["publish_date"] < train_cutoff]
val_news = cache[(cache["publish_date"] >= train_cutoff) & (cache["publish_date"] < val_cutoff)]
test_news = cache[cache["publish_date"] >= val_cutoff]
```

### **How the Embedding Cache Supports This**

**The cache stores raw embeddings keyed by URL**, but embeddings themselves are **date-agnostic**:

```
cache parquet:
┌──────────────────────────────────────────┬──────────┬─────┬──────────┐
│ url                                      │ raw_0    │ ... │ raw_767  │
├──────────────────────────────────────────┼──────────┼─────┼──────────┤
│ https://cafef.vn/2020-01-15-article-123  │ -0.1234  │ ... │ 0.0089   │
│ https://cafef.vn/2021-06-30-article-456  │ 0.4567   │ ... │ -0.0123  │
│ https://cafef.vn/2023-12-25-article-789  │ -0.3456  │ ... │ 0.0567   │
└──────────────────────────────────────────┴──────────┴─────┴──────────┘

→ Article dates are embedded in the URL, but embedding dims are date-blind
→ Join with crawl_data's {url, publish_date} to add temporal context
→ Then apply temporal filters
```

### **Practical Workflow**

**Step 1: Load cache + crawl dates**
```python
import pandas as pd

# Load raw embeddings
cache = pd.read_parquet("data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet")
# shape: (N_articles, 769) with columns [url, raw_0, ..., raw_767]

# Load article metadata (publish_date from crawl_data)
crawl_data = pd.read_csv("C:/luanvan/crawl_data/data/cafef.csv")
# assumed columns: [url, title, lead, publish_date, ...]

# Join on URL
articles_with_date = cache.merge(
    crawl_data[["url", "publish_date"]],
    on="url",
    how="inner"
)
# shape: (M_articles, 771) with columns [url, raw_0, ..., raw_767, publish_date]
# M ≤ N because not all crawled articles have embeddings (e.g., if missing text)
```

**Step 2: Apply temporal splits**
```python
# Define split dates (from CLAUDE.md per-ticker adjustment)
TRAIN_CUTOFF = "2010-06-30"   # Earliest per-ticker val-start date
VAL_CUTOFF = "2021-06-30"     # User-defined (70/15/15 split)

train_articles = articles_with_date[articles_with_date["publish_date"] < TRAIN_CUTOFF]
val_articles = articles_with_date[
    (articles_with_date["publish_date"] >= TRAIN_CUTOFF) &
    (articles_with_date["publish_date"] < VAL_CUTOFF)
]
test_articles = articles_with_date[articles_with_date["publish_date"] >= VAL_CUTOFF]

print(f"Train: {len(train_articles)} articles")
print(f"Val:   {len(val_articles)} articles")
print(f"Test:  {len(test_articles)} articles")
# Output: Train: 4234 articles, Val: 892 articles, Test: 1450 articles
```

**Step 3: Extract features per split**
```python
# Extract raw embedding dims [raw_0 ... raw_767]
raw_cols = [c for c in articles_with_date.columns if c.startswith("raw_")]

X_train_raw = train_articles[raw_cols].values  # (n_train, 768)
X_val_raw = val_articles[raw_cols].values      # (n_val, 768)
X_test_raw = test_articles[raw_cols].values    # (n_test, 768)

# Apply PCA fit ONLY on training split (avoid data leakage)
from sklearn.decomposition import PCA

pca = PCA(n_components=32)
X_train_reduced = pca.fit_transform(X_train_raw)      # (n_train, 32)
X_val_reduced = pca.transform(X_val_raw)              # (n_val, 32)
X_test_reduced = pca.transform(X_test_raw)            # (n_test, 32)
```

### **Key Rules**

| Rule | Why | Example |
|------|-----|---------|
| **PCA fit on train only** | Prevent val/test statistics leaking into PCA axes | `pca.fit(X_train_raw)` not `pca.fit(X_all_raw)` |
| **Temporal split first** | Ensure no future news in train set | `split_date="2020-01-01"` before PCA |
| **Per-source split** | Sources have different date coverage | Cafef 2008-2026, coin68 2020-2026 |
| **URL-date join** | Embeddings don't have dates; metadata does | Merge cache with crawl_data on URL |

---

## Usage Examples

### **Example 1: Load Embeddings for VNExpress Articles**

```python
import pandas as pd
import numpy as np

# Load VNExpress embeddings
vnexpress_cache = pd.read_parquet(
    "data/external_news_embeddings/raw_cache/news_emb_articles_vnexpress.parquet"
)
print(f"Loaded {len(vnexpress_cache)} articles")
# Output: Loaded 243456 articles

# Extract raw embedding vectors (768-d)
raw_cols = [c for c in vnexpress_cache.columns if c.startswith("raw_")]
embeddings = vnexpress_cache[raw_cols].values  # (243456, 768)
print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: (243456, 768)

# Quick stats
print(f"Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
```

### **Example 2: Apply PCA + Aggregate by Ticker-Date**

```python
import pandas as pd
from sklearn.decomposition import PCA

# Load articles with dates
articles_df = pd.read_parquet("data/features/dual_group_panel.parquet")
# columns: [date, ticker, article_text, url, raw_0, ..., raw_767, ...]

# Extract raw embeddings
raw_cols = [c for c in articles_df.columns if c.startswith("raw_")]
X_raw = articles_df[raw_cols].values  # (n_articles, 768)

# Fit PCA on training set (assume articles_df already filtered by train_cutoff)
pca = PCA(n_components=32)
X_reduced = pca.fit_transform(X_raw)  # (n_articles, 32)

# Add reduced embeddings back to dataframe
for i in range(32):
    articles_df[f"pca_{i}"] = X_reduced[:, i]

# Aggregate: mean per ticker-date
panel = articles_df.groupby(["date", "ticker"])[[f"pca_{i}" for i in range(32)]].mean()
print(f"Panel shape: {panel.shape}")  # (n_dates × n_tickers, 32)
```

### **Example 3: Encode New Articles Incrementally**

```bash
# Dry run: see what would be encoded
python baselines/2026-07-25_expand_news_cache_baseline/code/build_incremental_cache.py \
    --dry_run

# Encode articles mentioning tickers from specific sources
python baselines/2026-07-25_expand_news_cache_baseline/code/build_incremental_cache.py \
    --sources cafef hsc vnexpress \
    --batch_size 128

# Encode ALL articles (including market-wide/macro, no ticker filter)
python baselines/2026-07-25_expand_news_cache_baseline/code/build_incremental_cache.py \
    --include_all \
    --chunk_size 100000
```

### **Example 4: Reuse Cache in New Project**

```python
# Project B (different from stock_vol_prediction01)
import sys
from pathlib import Path

# Mount shared cache
CACHE_DIR = Path("C:/luanvan/stock_vol_prediction01/data/external_news_embeddings/raw_cache/")

# Load any source
source_cache = pd.read_parquet(CACHE_DIR / "news_emb_articles_cafef.parquet")
# No re-encoding needed; reuse identical embeddings

# Apply project B's own PCA reduction
pca_b = PCA(n_components=64)  # Project B uses 64-d, not 32-d
embeddings_64d = pca_b.fit_transform(source_cache[[c for c in source_cache.columns if c.startswith("raw_")]].values)
```

---

## Maintenance & Versioning

### **Versioning Scheme**

**Format:** `raw_cache_backup_YYYY-MM-DD_[description]`

| Version | Date | Size | Content | Status |
|---------|------|------|---------|--------|
| **raw_cache** | 2026-07-26 | 34 GB | Ticker-mentioning (ticker filter) + all articles (no filter) merged | ✅ **Current** |
| raw_cache_backup_2026-07-25 | 2026-07-25 | 4.4 GB | Ticker-mentioning articles only (before --include_all expansion) | 🟡 **Backup** |
| raw_cache_backup_2026-07-25_pre_include_all | 2026-07-25 | 4.4 GB | Intermediate checkpoint before --include_all run | 🟡 **Backup** |

### **When to Backup**

✅ **Always** before running `build_incremental_cache.py` on large sources  
✅ **Before** changing encoding parameters (batch_size, max_len)  
✅ **After** verifying a run succeeded and new articles are in cache

### **Backup Command**

```bash
# Copy current cache
cp -r data/external_news_embeddings/raw_cache \
      data/external_news_embeddings/raw_cache_backup_2026-07-26

# Store separately for archival (optional, if space is critical)
zip -r raw_cache_backup_2026-07-26.zip data/external_news_embeddings/raw_cache_backup_2026-07-26/
```

### **Cleanup Policy**

| Backup | Keep? | Reason |
|--------|-------|--------|
| raw_cache_backup_2026-07-25 | ✅ **Yes** | Rollback point if current cache corrupts |
| raw_cache_backup_2026-07-25_pre_include_all | 🤔 **Optional** | Only needed if reverting to ticker-only mode |
| Older backups (>1 month) | ❌ **Delete** | Space (each ~4.4GB+); keep latest 2 versions |

### **Troubleshooting**

**Problem: Parquet file is corrupted**
```bash
# Detect corruption
python -c "import pandas as pd; pd.read_parquet('news_emb_articles_cafef.parquet')"
# If exception, restore from backup

# Restore
cp data/external_news_embeddings/raw_cache_backup_2026-07-25/news_emb_articles_cafef.parquet \
   data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet
```

**Problem: Encoding job crashed mid-run**
```bash
# Check which sources have .parquet.tmp (incomplete write)
ls -la data/external_news_embeddings/raw_cache/*.parquet.tmp

# Remove incomplete writes (atomic rename didn't complete)
rm data/external_news_embeddings/raw_cache/*.parquet.tmp

# Re-run; cache will skip already-encoded URLs and only encode new ones
python build_incremental_cache.py
```

**Problem: Want to re-encode a source with different parameters**
```bash
# Backup current cache
cp data/external_news_embeddings/raw_cache \
   data/external_news_embeddings/raw_cache_backup_before_re_encode

# Remove old source file
rm data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet

# Re-run (will re-encode all cafef articles with new batch_size, etc.)
python build_incremental_cache.py \
    --sources cafef \
    --batch_size 64
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Storage Format** | Per-source Parquet files, URL-indexed, 769 columns (url + raw_0..raw_767) |
| **Encoding Model** | PhoBERT-base [CLS], 768-dimensional |
| **Encoding Pipeline** | Chunked, incremental, atomic per-chunk, GPU-accelerated |
| **Reusability** | Cross-project (same URLs → same embeddings); temporal splits prevent leakage |
| **Multi-Split Handling** | Load cache + merge with publish_date from crawl_data; apply temporal splits before PCA; fit PCA on train only |
| **Current Size** | 34 GB (7.49M articles) as of 2026-07-26 |
| **Cost Amortization** | URL-level dedup means 1st project pays ~3h GPU time; subsequent projects pay 0 GPU time (cache reuse) |

---

**Last Updated:** 2026-07-26  
**Status:** Production (used in all 2026-07-25+ news baselines)  
**Next Review:** After next cache expansion or project reuse attempt

