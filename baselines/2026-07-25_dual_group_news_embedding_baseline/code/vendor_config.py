"""Path/constant config for the vendored data_eda aggregation code (Story 1.3).

Mirrors the role of data_eda's own `config/__init__.py` (PROJECT_ROOT, CRAWL_DATA_ROOT,
FEATURES_DIR, EDA_TICKERS/PRICE_DATA_DIR) but every path resolves against THIS project
(stock_vol_prediction01), never a hardcoded absolute literal (CLAUDE.md code-hygiene rule) —
each path is computed relative to this file's location or its established sibling directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# code/vendor_config.py -> parents[0]=code, [1]=<baseline folder>, [2]=baselines, [3]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Same sibling crawl_data directory already used by this project's own
# src/data_aggregation/aggregate_news_sources.py (RAW_DIR = _ROOT.parent / "crawl_data" / "data").
CRAWL_DATA_ROOT = PROJECT_ROOT.parent / "crawl_data" / "data"

# Copied PhoBERT article-embedding cache (Story 1.1) — NOT data_eda's own data/features/.
FEATURES_DIR = PROJECT_ROOT / "data" / "external_news_embeddings" / "raw_cache"

# This project's existing per-ticker processed price files (date + parkinson_volatility),
# used as the trading calendar (data_eda used its own OHLCV dir for the same purpose).
PRICE_DATA_DIR = PROJECT_ROOT / "data" / "processed"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOT this project's own src/sentiment/data_collection/tickers.py::VN30_TICKERS (45 entries,
# only 28 of which have data/processed/ files — an inflated/legacy ticker list). The copied
# PhoBERT cache (Story 1.1) was built against data_eda's ticker-mention filter, which uses the
# real 30-constituent VN30 index (data_eda/config/__init__.py::VN30_TICKERS). Using the broader
# 45-ticker list here made the vendored TICKER_PATTERN match ~5,499 MORE articles than the cache
# covers (discovered via build_dual_group_panel.py's cache-miss assertion, Story 2.2) — those
# extra tickers were never a candidate for data_eda's own encoding pass, so they're genuinely
# uncached, not a bug in the copy. Using the same 30-ticker list data_eda used keeps this
# baseline's ticker-mention filter IDENTICAL to what produced the cache -> 0 cache misses.
# (Verified: all 30 have a data/processed/{TICKER}_processed.csv in this project.)
#
# [FIX 2026-07-26] This 30-item list itself was missing VPB and VRE even though both ARE in this
# project's own 32-ticker data/processed/ universe — dual_group_news_panel.parquet had ZERO rows
# for either (verified), so every news-fusion baseline's x_news for VPB/VRE was an all-zero
# vector (any "gate value" for them was a network-bias-term artifact, not a signal — see memory
# project_vn30_ticker_universe_mismatch.md). Adding them here is safe: the news cache now covers
# ALL articles regardless of ticker mention (2026-07-25 --include_all GPU expansion), so this
# still produces 0 cache misses (verified by the 2026-07-26 dual-group panel rebuild).
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SHB", "SSB",
    "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM",
    "VPB", "VRE",
]

EDA_TICKERS = list(VN30_TICKERS)
