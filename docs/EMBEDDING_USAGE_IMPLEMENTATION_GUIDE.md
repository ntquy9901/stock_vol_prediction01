# Embedding Usage & Implementation Guide

**Document Version:** 1.0  
**Date:** 2026-07-26  
**Audience:** ML engineers implementing news-fusion baselines  
**Companion:** [EMBEDDING_STORAGE_SPECIFICATION.md](EMBEDDING_STORAGE_SPECIFICATION.md)

---

## Quick Start

**You want to use pre-computed news embeddings in your baseline. Here's the shortest path:**

```python
# Step 1: Load cache for one source
import pandas as pd

cache = pd.read_parquet("data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet")
# Returns: (N, 769) DataFrame with columns [url, raw_0, ..., raw_767]

# Step 2: Get raw embeddings (768-dimensional)
raw_cols = [c for c in cache.columns if c.startswith("raw_")]
X_raw = cache[raw_cols].values  # (N, 768) numpy array

# Step 3: Fit PCA on training data only (avoid leakage)
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
X_train_raw = ...  # Subset: articles in your training date range
X_train_reduced = pca.fit_transform(X_train_raw)  # (n_train, 32)

# Step 4: Transform validation and test data
X_val_reduced = pca.transform(X_val_raw)    # (n_val, 32)
X_test_reduced = pca.transform(X_test_raw)  # (n_test, 32)

print(f"Reduced embeddings: train {X_train_reduced.shape}, val {X_val_reduced.shape}, test {X_test_reduced.shape}")
# Output: Reduced embeddings: train (4234, 32), val (892, 32), test (1450, 32)
```

---

## Table of Contents

1. [Overview: Why Use Cached Embeddings](#overview-why-use-cached-embeddings)
2. [Loading the Cache](#loading-the-cache)
3. [PCA Reduction (Date-Safe)](#pca-reduction-date-safe)
4. [Aggregating to Ticker-Date Level](#aggregating-to-ticker-date-level)
5. [Integration with Dataset Classes](#integration-with-dataset-classes)
6. [Multi-Source Aggregation](#multi-source-aggregation)
7. [Train/Val/Test Split Handling](#trainvalstest-split-handling)
8. [Debugging & Validation](#debugging--validation)
9. [Real Example: Macro News Baseline](#real-example-macro-news-baseline)

---

## Overview: Why Use Cached Embeddings

### **Cost Benefit**

| Task | Time (CPU) | Time (GPU) | Cost | Recurring? |
|------|-----------|----------|------|-----------|
| **PhoBERT encode** 7.49M articles | N/A | 3+ hours | $$$ (GPU) | ❌ No |
| **Load + PCA** from cache | 10 sec | N/A | Free | ✅ Yes per train |
| **Reuse cache** across projects | <1 sec | N/A | Free | ✅ Yes |

**Bottom line:** First baseline pays 3 hours of GPU time. Every subsequent baseline on the same cache pays 10 seconds and $0.

### **Data Leakage Prevention**

✅ Cache is encoding-only (no temporal info)  
✅ Temporal splits applied when loading/aggregating  
✅ PCA fit only on training dates (not val/test)

---

## Loading the Cache

### **Single Source**

```python
import pandas as pd

source = "cafef"
cache = pd.read_parquet(f"data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet")

print(f"Loaded {len(cache)} articles from {source}")
print(f"Columns: {cache.columns.tolist()[:5]}...")  # [url, raw_0, raw_1, ...]
print(f"Shape: {cache.shape}")  # (N, 769)
```

### **Multiple Sources**

```python
import pandas as pd
from pathlib import Path

sources = ["cafef", "hsc", "vnexpress", "dantri", "cand"]
cache_dir = Path("data/external_news_embeddings/raw_cache/")

all_caches = []
for source in sources:
    try:
        df = pd.read_parquet(cache_dir / f"news_emb_articles_{source}.parquet")
        print(f"Loaded {len(df):>7} articles from {source}")
        all_caches.append(df)
    except FileNotFoundError:
        print(f"⚠️  {source} not cached yet")

merged_cache = pd.concat(all_caches, ignore_index=True)
print(f"\nTotal: {len(merged_cache)} articles from {len(all_caches)} sources")
# Output:
# Loaded    243456 articles from cafef
# Loaded     10234 articles from hsc
# Loaded    567890 articles from vnexpress
# Loaded    432156 articles from dantri
# Loaded    234567 articles from cand
#
# Total: 1488303 articles from 5 sources
```

### **Extract Raw Embeddings**

```python
import numpy as np

raw_cols = sorted([c for c in cache.columns if c.startswith("raw_")])
print(f"Found {len(raw_cols)} embedding dimensions")  # 768

X = cache[raw_cols].values
print(f"Embedding matrix: {X.shape}, dtype: {X.dtype}")  # (N, 768), float32

# Quick sanity check
print(f"Mean: {X.mean(axis=0)[:5]}, Std: {X.std(axis=0)[:5]}")
print(f"Range: [{X.min():.4f}, {X.max():.4f}]")
```

---

## PCA Reduction (Date-Safe)

### **The Leakage Trap** 🚨

```python
# ❌ WRONG: Fit PCA on ALL data (includes val/test dates!)
pca_bad = PCA(n_components=32)
pca_bad.fit(X_all)  # X_all has val+test dates
X_train_reduced = pca_bad.transform(X_train)
X_val_reduced = pca_bad.transform(X_val)
# → VAL/TEST statistics leaked into PCA axes!
```

### **The Correct Way** ✅

```python
# ✅ CORRECT: Fit PCA only on training data
pca = PCA(n_components=32)
pca.fit(X_train)  # Fit on training dates ONLY
X_train_reduced = pca.transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)
```

### **Full Workflow**

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime

# 1. Load cache + metadata
cache = pd.read_parquet("data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet")
# Must have: columns [url, raw_0, ..., raw_767]

# 2. Get publish dates from crawl_data
crawl_df = pd.read_csv("C:/luanvan/crawl_data/data/cafef.csv")
# Must have: columns [url, publish_date]

# 3. Join cache with dates
articles = cache.merge(
    crawl_df[["url", "publish_date"]],
    on="url",
    how="inner"
)
articles["publish_date"] = pd.to_datetime(articles["publish_date"])

print(f"Matched {len(articles)} / {len(cache)} articles with publish dates")
# Output: Matched 243400 / 243456 articles with publish dates (56 articles have no date)

# 4. Define temporal split points
TRAIN_CUTOFF = pd.Timestamp("2020-06-30")
VAL_CUTOFF = pd.Timestamp("2022-06-30")

train_mask = articles["publish_date"] < TRAIN_CUTOFF
val_mask = (articles["publish_date"] >= TRAIN_CUTOFF) & (articles["publish_date"] < VAL_CUTOFF)
test_mask = articles["publish_date"] >= VAL_CUTOFF

# 5. Extract raw embeddings per split
raw_cols = sorted([c for c in articles.columns if c.startswith("raw_")])
X_train = articles[train_mask][raw_cols].values
X_val = articles[val_mask][raw_cols].values
X_test = articles[test_mask][raw_cols].values

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
# Output: Train: (4234, 768), Val: (892, 768), Test: (1450, 768)

# 6. Fit PCA on training dates ONLY
pca = PCA(n_components=32)
pca.fit(X_train)

# 7. Transform all splits
X_train_reduced = pca.transform(X_train)  # (4234, 32)
X_val_reduced = pca.transform(X_val)      # (892, 32)
X_test_reduced = pca.transform(X_test)    # (1450, 32)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
# Output: PCA explained variance: 84.53%

print(f"Reduced: train {X_train_reduced.shape}, val {X_val_reduced.shape}, test {X_test_reduced.shape}")
# Output: Reduced: train (4234, 32), val (892, 32), test (1450, 32)
```

### **Saving PCA Fit for Reuse**

```python
import joblib

# Save the fitted PCA
joblib.dump(pca, "models/pca_cafef_32d_2020_2022_split.pkl")

# Later, load and reuse (e.g., for inference or another experiment)
pca_loaded = joblib.load("models/pca_cafef_32d_2020_2022_split.pkl")
X_test_reduced = pca_loaded.transform(X_test)
```

---

## Aggregating to Ticker-Date Level

### **Why Aggregate?**

Each article has 1 URL and 1 embedding. But your model likely works at ticker-date resolution:
- Input: `[date, ticker] → news features for that ticker-date`
- Not: `[date, ticker, article_id] → features for individual articles`

**Aggregation reduces sparsity:** Multiple articles per ticker-date → 1 dense feature vector.

### **Aggregation Methods**

**Option 1: Simple Mean (most common)**
```python
# Group by ticker-date, take mean of embeddings
panel = articles.groupby(["date", "ticker"])[[f"pca_{i}" for i in range(32)]].mean()
# Result: (n_ticker_dates, 32) with shape dependent on date coverage

print(f"Panel shape: {panel.shape}")
# Output: Panel shape: (450, 32)  # 450 unique ticker-date pairs
```

**Option 2: Max-pooling (capture extreme sentiment)**
```python
panel = articles.groupby(["date", "ticker"])[[f"pca_{i}" for i in range(32)]].max()
```

**Option 3: Weighted aggregation (by recency)**
```python
# Weight recent articles higher
articles["weight"] = 1.0 / (1.0 + (article_datetime - articles["publish_date"]).dt.days)
panel = articles.groupby(["date", "ticker"]).apply(
    lambda grp: (grp[[f"pca_{i}" for i in range(32)]] * grp["weight"].values[:, None]).mean(axis=0)
)
```

### **Full Example: Dual-Group Aggregation**

This is what the `2026-07-25_macro_news_baseline` does:

```python
import pandas as pd
from sklearn.decomposition import PCA

# Define source groups
KHACH_QUAN_SOURCES = {
    "cafef", "hsc", "vnexpress", "thanhnien", "tuoitre", ..., "dantri", "cand", "hanoimoi", ...
}
TONG_HOP_SOURCES = {
    "ssi", "vndirect", "vnstock", "vietstock", "vsdc", "forum", ...
}

# Load + PCA for each group
def load_and_reduce_group(sources, pca_dim=32):
    all_dfs = []
    for source in sources:
        cache = pd.read_parquet(f"data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet")
        crawl = pd.read_csv(f"C:/luanvan/crawl_data/data/{source}.csv")
        articles = cache.merge(crawl[["url", "publish_date", "ticker_mentioned"]], on="url", how="inner")
        articles["source_group"] = "khach_quan" if source in KHACH_QUAN_SOURCES else "tong_hop"
        all_dfs.append(articles)
    
    group_df = pd.concat(all_dfs, ignore_index=True)
    
    # Separate training and all data for PCA
    train_mask = group_df["publish_date"] < "2020-06-30"
    raw_cols = sorted([c for c in group_df.columns if c.startswith("raw_")])
    
    pca = PCA(n_components=pca_dim)
    pca.fit(group_df[train_mask][raw_cols].values)
    
    for i in range(pca_dim):
        group_df[f"pca_group_{i}"] = pca.transform(group_df[raw_cols].values)[:, i]
    
    return group_df[["date", "ticker_mentioned", "pca_group_0", ..., "pca_group_31"]], pca

# Load both groups
khach_quan_df, pca_kq = load_and_reduce_group(KHACH_QUAN_SOURCES)
tong_hop_df, pca_th = load_and_reduce_group(TONG_HOP_SOURCES)

# Aggregate to ticker-date (per-group)
khach_quan_panel = khach_quan_df.groupby(["date", "ticker_mentioned"]).mean()
tong_hop_panel = tong_hop_df.groupby(["date", "ticker_mentioned"]).mean()

# Concat horizontally (per-group dual features)
dual_group_panel = pd.concat(
    [
        khach_quan_panel.rename(columns={c: c.replace("pca_group", "pca_kq") for c in khach_quan_panel.columns}),
        tong_hop_panel.rename(columns={c: c.replace("pca_group", "pca_th") for c in tong_hop_panel.columns})
    ],
    axis=1
)

print(f"Dual-group panel shape: {dual_group_panel.shape}")
# Output: Dual-group panel shape: (28800, 64)  # 64 = 32 khach_quan + 32 tong_hop

dual_group_panel.to_parquet("data/features/dual_group_panel.parquet")
```

---

## Integration with Dataset Classes

### **Minimal PyTorch Dataset**

```python
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NewsEmbeddingDataset(Dataset):
    """Loads pre-computed embeddings for specific ticker-date pairs."""
    
    def __init__(self, panel_path, volatility_path, ticker_dates):
        """
        Args:
            panel_path: path to aggregated news panel (ticker-date × embedding_dims)
            volatility_path: path to volatility target
            ticker_dates: list of (date, ticker) tuples to use
        """
        self.panel = pd.read_parquet(panel_path)
        self.volatility = pd.read_parquet(volatility_path)
        self.ticker_dates = ticker_dates
    
    def __len__(self):
        return len(self.ticker_dates)
    
    def __getitem__(self, idx):
        date, ticker = self.ticker_dates[idx]
        
        # Fetch news embedding (32-d)
        news_emb = self.panel.loc[(date, ticker)].values.astype(np.float32)  # (32,)
        
        # Fetch target volatility
        vol = self.volatility.loc[(date, ticker), "volatility"]
        
        return {
            "news_embedding": torch.from_numpy(news_emb),
            "volatility": torch.tensor(vol, dtype=torch.float32),
            "date": date,
            "ticker": ticker
        }

# Usage
dataset = NewsEmbeddingDataset(
    panel_path="data/features/dual_group_panel.parquet",
    volatility_path="data/features/volatility_target.parquet",
    ticker_dates=train_ticker_dates  # list of (date, ticker) tuples
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    X_news = batch["news_embedding"]  # (32, 32) for batch_size=32
    y_vol = batch["volatility"]       # (32,)
    # → use in your model
```

### **With Time Series Context (LSTM)**

```python
class TimeSeriesNewsDataset(Dataset):
    """Loads news embeddings in time windows for LSTM input."""
    
    def __init__(self, panel_path, volatility_path, ticker, lookback=20, step=1):
        """
        Args:
            panel_path: (date, ticker, embedding_32)
            volatility_path: (date, ticker, volatility)
            ticker: specific ticker to load
            lookback: # of past days to include in LSTM input
            step: stride between windows (1 = every day, 5 = every 5 days)
        """
        self.panel = pd.read_parquet(panel_path)
        self.volatility = pd.read_parquet(volatility_path)
        self.ticker = ticker
        self.lookback = lookback
        
        # Get all dates for this ticker, sorted
        ticker_dates = sorted(self.panel.xs(ticker, level="ticker").index)
        
        # Generate windows
        self.windows = []
        for i in range(lookback, len(ticker_dates), step):
            self.windows.append(ticker_dates[i - lookback : i])
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window_dates = self.windows[idx]
        last_date = window_dates[-1]
        
        # Stack news embeddings (lookback, 32)
        X_news = np.stack([
            self.panel.loc[(date, self.ticker)].values
            for date in window_dates
        ], axis=0).astype(np.float32)  # (20, 32)
        
        # Get future volatility (e.g., next day's vol)
        y_vol = self.volatility.loc[(last_date, self.ticker), "volatility"]
        
        return {
            "X_news": torch.from_numpy(X_news),
            "y_vol": torch.tensor(y_vol, dtype=torch.float32)
        }

# Usage
dataset = TimeSeriesNewsDataset(
    panel_path="data/features/dual_group_panel.parquet",
    volatility_path="data/features/volatility_target.parquet",
    ticker="VNM",
    lookback=20
)
loader = DataLoader(dataset, batch_size=16)

for batch in loader:
    X_news = batch["X_news"]  # (16, 20, 32)
    y_vol = batch["y_vol"]    # (16,)
    # → LSTM forward(X_news) → (16, hidden)
    # → FC layer → (16, 1) volatility prediction
```

---

## Multi-Source Aggregation

### **Scenario: Combine Khach Quan + Tong Hop + Macro News**

The `2026-07-25_macro_news_baseline` does this:

```python
import pandas as pd
import numpy as np

# Step 1: Dual-group panel (khach_quan + tong_hop, 64-d)
dual_group = pd.read_parquet("data/features/dual_group_panel.parquet")
# shape: (ticker_dates, 64)

# Step 2: Macro news panel (date-only aggregation, broadcast to all tickers, 32-d)
macro = pd.read_parquet("data/features/macro_news_panel.parquet")
# shape: (dates, 32) — no ticker dimension; will broadcast

# Step 3: Concat horizontally
# First, expand macro to ticker-date index
dates = macro.index
tickers = ["AAA", "ABB", ..., "VPB"]  # All 30 VN30 tickers
macro_expanded = pd.concat(
    [macro] * len(tickers),
    keys=tickers,
    names=["ticker"]
)
macro_expanded.reset_index(inplace=True)
macro_expanded.set_index(["date", "ticker"], inplace=True)

# Now concat with dual_group
combined = pd.concat([dual_group, macro_expanded], axis=1)
# shape: (ticker_dates, 96)  # 64 + 32

print(f"Combined panel: {combined.shape}")
# Output: Combined panel: (28800, 96)

# Step 4: Fill missing dates (optional)
combined = combined.fillna(0)  # Or forward/backward fill

# Step 5: Use in model
# X_news = combined.values  # (28800, 96)
```

### **Weighted Combination**

```python
# Weight dual-group more than macro (can tune)
combined_weighted = pd.concat([
    dual_group * 0.7,  # 70% weight on ticker-specific news
    macro_expanded * 0.3  # 30% weight on market-wide news
], axis=1)
```

---

## Train/Val/Test Split Handling

### **Critical Pattern** ⚠️

**DO NOT:**
```python
# ❌ WRONG: Split embeddings randomly
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(range(len(panel)), test_size=0.2)
X_train = panel.iloc[train_idx]
X_test = panel.iloc[test_idx]
```

**DO:**
```python
# ✅ CORRECT: Split by publish date
TRAIN_CUTOFF = "2020-06-30"
VAL_CUTOFF = "2022-06-30"

articles["date"] = pd.to_datetime(articles["date"])
train_mask = articles["date"] < TRAIN_CUTOFF
val_mask = (articles["date"] >= TRAIN_CUTOFF) & (articles["date"] < VAL_CUTOFF)
test_mask = articles["date"] >= VAL_CUTOFF

X_train = articles[train_mask]
X_val = articles[val_mask]
X_test = articles[test_mask]
```

### **Handling Missing Dates**

Not all ticker-date pairs have news:

```python
# Raw panel may have shape (4,234, 32) for cafef + training period
# But your ticker-date grid may be (23,000) — many dates have 0 articles

# Strategy 1: Forward fill (propagate last known embedding)
combined = combined.fillna(method="ffill")

# Strategy 2: Zero fill (articles → 0-vector for that ticker-date)
combined = combined.fillna(0)

# Strategy 3: PCA centroid fill (use mean of training embeddings)
centroid = combined.iloc[:4234].mean()  # mean over training dates
combined = combined.fillna(centroid)

# Strategy 4: Separate "no-news" indicator
combined["has_news"] = ~combined.isna().any(axis=1)
combined = combined.fillna(0)
# Model can then learn: "on dates with news, use these features; on dates without, do X"
```

---

## Debugging & Validation

### **Sanity Checks**

```python
import pandas as pd
import numpy as np

cache = pd.read_parquet("data/external_news_embeddings/raw_cache/news_emb_articles_cafef.parquet")

# Check 1: No NaN in embeddings
raw_cols = sorted([c for c in cache.columns if c.startswith("raw_")])
assert not cache[raw_cols].isna().any().any(), "Found NaN in embeddings!"
print("✓ No NaN values")

# Check 2: Embedding dimension
assert len(raw_cols) == 768, f"Expected 768 dims, got {len(raw_cols)}"
print("✓ Embedding dimension is 768")

# Check 3: URL uniqueness
assert cache["url"].nunique() == len(cache), "URLs are not unique!"
print("✓ URLs are unique")

# Check 4: Reasonable value ranges (PhoBERT should be roughly [-1, +1] after normalization)
X = cache[raw_cols].values
print(f"✓ Embedding range: [{X.min():.4f}, {X.max():.4f}]")
print(f"✓ Embedding mean: {X.mean():.6f}, std: {X.std():.6f}")

# Check 5: Embedding norms (should be roughly normalized)
norms = np.linalg.norm(X, axis=1)
print(f"✓ Norm range: [{norms.min():.2f}, {norms.max():.2f}], mean: {norms.mean():.2f}")
```

### **Debugging PCA Fit**

```python
from sklearn.decomposition import PCA

X_train = ...  # (4234, 768)
pca = PCA(n_components=32)
pca.fit(X_train)

# Check 1: Variance explained
explained_var = pca.explained_variance_ratio_.cumsum()
print(f"PCA cumulative variance: {explained_var}")
# Should see: [0.15, 0.26, 0.35, ..., 0.8453] — reaches 80%+ quickly

# Check 2: Principal components (should look like noise)
print(f"First component shape: {pca.components_[0].shape}")  # (768,)
print(f"First component norm: {np.linalg.norm(pca.components_[0]):.4f}")  # ~1.0

# Check 3: Reduced embeddings (should be ~0-centered, unit variance)
X_reduced = pca.transform(X_train)
print(f"Reduced mean: {X_reduced.mean(axis=0)[:5]}")  # ~[0, 0, 0, 0, 0]
print(f"Reduced std: {X_reduced.std(axis=0)[:5]}")   # ~[1, 0.9, 0.8, ...] (decreasing)
```

### **Debugging Aggregation**

```python
# Before aggregation
articles_by_ticker_date = articles.groupby(["date", "ticker"]).size()
print(f"Articles per ticker-date:")
print(articles_by_ticker_date.describe())
# Should see: mean ~5-10 articles/date/ticker (varies by source)

# After aggregation
panel_filled = panel.fillna(0)
print(f"Panel sparsity: {(panel_filled == 0).sum().sum() / panel_filled.size:.2%} zeros")
# Should see: 20-50% sparse (many ticker-dates have no articles)

# Check that mean aggregation works
sample_date_ticker = ("2020-07-01", "VNM")
raw_articles = articles[(articles["date"] == sample_date_ticker[0]) & 
                        (articles["ticker"] == sample_date_ticker[1])][f"pca_0"]
manual_mean = raw_articles.mean()
aggregated_mean = panel.loc[sample_date_ticker, "pca_0"]
assert np.isclose(manual_mean, aggregated_mean), f"Aggregation mismatch: {manual_mean} vs {aggregated_mean}"
print("✓ Aggregation verified")
```

---

## Real Example: Macro News Baseline

This example shows how everything comes together:

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

# ============================================================================
# PART 1: LOAD AND REDUCE EMBEDDINGS
# ============================================================================

# Load all sources (khach_quan + tong_hop)
KHACH_QUAN = ["cafef", "hsc", "vnexpress", "thanhnien", "tuoitre", "dantri", "cand"]
TONG_HOP = ["ssi", "vndirect", "vnstock", "vietstock", "forum"]

def load_and_reduce(sources, group_name, pca_dim=32):
    """Load, reduce, and aggregate to ticker-date."""
    all_dfs = []
    
    for source in sources:
        # Load raw cache
        cache = pd.read_parquet(f"data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet")
        
        # Load crawl metadata (need dates)
        crawl = pd.read_csv(f"C:/luanvan/crawl_data/data/{source}.csv")
        articles = cache.merge(crawl[["url", "publish_date"]], on="url", how="inner")
        
        articles["source"] = source
        all_dfs.append(articles)
    
    # Combine all sources for this group
    group_df = pd.concat(all_dfs, ignore_index=True)
    group_df["publish_date"] = pd.to_datetime(group_df["publish_date"])
    
    # Extract raw embeddings
    raw_cols = sorted([c for c in group_df.columns if c.startswith("raw_")])
    X_raw = group_df[raw_cols].values  # (N, 768)
    
    # Fit PCA on training dates only
    TRAIN_CUTOFF = pd.Timestamp("2020-06-30")
    train_mask = group_df["publish_date"] < TRAIN_CUTOFF
    pca = PCA(n_components=pca_dim)
    pca.fit(X_raw[train_mask])
    
    # Transform all dates
    X_reduced = pca.transform(X_raw)  # (N, 32)
    
    # Add reduced embeddings to dataframe
    for i in range(pca_dim):
        group_df[f"pca_{group_name}_{i}"] = X_reduced[:, i]
    
    return group_df[["publish_date", "pca_" + group_name + "_0"] + 
                    [f"pca_{group_name}_{i}" for i in range(1, pca_dim)]], pca

# Load both groups
kq_df, pca_kq = load_and_reduce(KHACH_QUAN, "kq", pca_dim=32)
th_df, pca_th = load_and_reduce(TONG_HOP, "th", pca_dim=32)

# ============================================================================
# PART 2: AGGREGATE TO TICKER-DATE
# ============================================================================

# Aggregate per ticker-date (add ticker_mentioned from raw articles)
def aggregate_to_panel(articles_df, group_name):
    """Aggregate articles to ticker-date level."""
    # Assume articles_df has: publish_date, ticker_mentioned, pca_* columns
    panel = articles_df.groupby(["publish_date", "ticker_mentioned"]).mean()
    panel.index.names = ["date", "ticker"]
    return panel

kq_panel = aggregate_to_panel(kq_df, "kq")
th_panel = aggregate_to_panel(th_df, "th")

# Combine dual-group
dual_panel = pd.concat([kq_panel, th_panel], axis=1)
dual_panel.to_parquet("data/features/dual_group_panel.parquet")

# ============================================================================
# PART 3: LOAD MACRO NEWS PANEL (separate process, already done)
# ============================================================================

macro_panel = pd.read_parquet("data/features/macro_news_panel.parquet")  # (dates, 32)

# Broadcast macro to all tickers
dates = macro_panel.index
tickers = ["AAA", "ABB", ..., "VPB"]  # 30 VN30 tickers
macro_expanded = pd.concat([macro_panel] * len(tickers), keys=tickers, names=["ticker"])
macro_expanded.reset_index(inplace=True)
macro_expanded.set_index(["date", "ticker"], inplace=True)

# ============================================================================
# PART 4: CONCATENATE ALL FEATURES
# ============================================================================

combined_panel = pd.concat([dual_panel, macro_expanded], axis=1)
# Shape: (28800, 96)  # 64 from dual-group + 32 from macro
combined_panel.to_parquet("data/features/combined_panel.parquet")

# ============================================================================
# PART 5: USE IN DATASET + TRAIN
# ============================================================================

class CombinedNewsDataset(torch.utils.data.Dataset):
    def __init__(self, panel_path, vol_path, ticker_dates):
        self.panel = pd.read_parquet(panel_path)
        self.vols = pd.read_parquet(vol_path)
        self.ticker_dates = ticker_dates
    
    def __len__(self):
        return len(self.ticker_dates)
    
    def __getitem__(self, idx):
        date, ticker = self.ticker_dates[idx]
        X = self.panel.loc[(date, ticker)].fillna(0).values.astype(np.float32)  # (96,)
        y = self.vols.loc[(date, ticker), "volatility"]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)

# Create dataset + loader
train_ticker_dates = ...  # your split logic
dataset = CombinedNewsDataset(
    "data/features/combined_panel.parquet",
    "data/features/volatility_target.parquet",
    train_ticker_dates
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = YourNewsModel(input_dim=96)  # 96-d input = dual-group + macro
for epoch in range(50):
    for X_batch, y_batch in train_loader:
        # X_batch: (32, 96)
        # y_batch: (32,)
        y_pred = model(X_batch)
        loss = mse(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Summary

| Task | Function | File Location | Notes |
|------|----------|----------------|-------|
| **Load cache** | `pd.read_parquet(f"news_emb_articles_{source}.parquet")` | `data/external_news_embeddings/raw_cache/` | 769 columns: url + 768-d embedding |
| **Extract raw** | `df[[c for c in df.columns if c.startswith("raw_")]].values` | N/A | Returns (N, 768) numpy array |
| **Fit PCA** | `pca.fit(X_train_only)` | sklearn | **Critical:** fit on train dates ONLY |
| **Transform** | `X_reduced = pca.transform(X_raw)` | sklearn | Apply to train/val/test separately |
| **Aggregate** | `df.groupby(["date", "ticker"]).mean()` | pandas | Reduce to ticker-date resolution |
| **Combine groups** | `pd.concat([kq_panel, th_panel], axis=1)` | pandas | Dual-group: 64-d (32+32) |
| **Add macro** | `pd.concat([dual_panel, macro_expanded], axis=1)` | pandas | Full: 96-d (64+32) |
| **Dataset loader** | `DataLoader(NewsDataset(...))` | PyTorch | Batch loading for training |

---

**Last Updated:** 2026-07-26  
**Next:** See [EMBEDDING_STORAGE_SPECIFICATION.md](EMBEDDING_STORAGE_SPECIFICATION.md) for cache maintenance.

