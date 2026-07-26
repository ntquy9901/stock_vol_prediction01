"""One-off script (Story 2): rebuild the dual-group news-embedding aggregation (PCA + single
30d EWMA) against the copied PhoBERT article-embedding caches, WITHOUT ever running PhoBERT.

Articles whose `url` isn't in the copied cache (data/external_news_embeddings/raw_cache/,
snapshotted from data_eda 2026-07-24 21:49) — i.e. crawl_data grew past that snapshot — are
SKIPPED, not encoded (user decision 2026-07-25: 316 such articles were found across all 30
sources at build time; excluding them keeps 0 PhoBERT calls, at the cost of missing that small
slice of the most-recent news). This is enforced by
`vendor_data_eda.news_embeddings._get_article_embeddings` (cache-only lookup); this script logs
how many were skipped instead of asserting zero (CLAUDE.md: don't assume — verify, and report
what actually happened).

Usage: python build_dual_group_panel.py
Output: ../../../data/features/dual_group_news_panel.parquet  (ticker, date, 146 cols)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import pandas as pd  # noqa: E402

import vendor_config  # noqa: E402
from vendor_data_eda import news_embeddings as ne  # noqa: E402
from vendor_data_eda.dual_news_features import ADV_FEATURES_DUAL, EWMA_FEATURES, build_advanced_features  # noqa: E402

# requirements.md §3 Simplicity Gate: this baseline's scope is basic-dual (80) + single-EWMA
# (66) = 146 cols, NOT the 39 extra legacy tong_hop-only columns that build_advanced_features
# also emits for backward compat with data_eda's own "price+news_adv" panel (irrelevant here —
# this project has no such consumer, and shipping them would silently double the feature count
# vs. what requirements.md documents and the go/no-go criteria were written against).
KEEP_COLS = ADV_FEATURES_DUAL + EWMA_FEATURES

OUT_PATH = vendor_config.PROJECT_ROOT / "data" / "features" / "dual_group_news_panel.parquet"


def _count_cache_misses() -> int:
    """Sum of articles (per source, across both groups) whose `url` is NOT already in the
    copied per-source cache — i.e. how many articles `build_advanced_features` will silently
    skip (per `_get_article_embeddings`'s cache-only lookup). Read-only: does not write
    anything; purely informational logging before the real build."""
    total_new = 0
    for group in ("khach_quan", "tong_hop"):
        news = ne._load_group(group)
        if news.empty:
            continue
        for source, sub_news in news.groupby("source"):
            cache_path = ne._article_cache_path(source)
            cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame({"url": []})
            known = set(cached["url"]) if not cached.empty else set()
            new_rows = sub_news[~sub_news["url"].isin(known)]
            total_new += len(new_rows)
    return total_new


def main():
    t0 = time.time()
    print("[1/2] Checking for cache misses (these will be SKIPPED, never encoded)...", flush=True)
    n_new = _count_cache_misses()
    print(f"      cache misses = {n_new} article(s) will be skipped (0 PhoBERT calls, by design)",
          flush=True)

    print("[2/2] Building dual-group panel (mode='ewma')...", flush=True)
    panel = build_advanced_features(mode="ewma")
    if panel.empty:
        raise RuntimeError("build_advanced_features returned an empty panel — check raw_cache/crawl_data.")

    missing = [c for c in KEEP_COLS if c not in panel.columns]
    if missing:
        raise RuntimeError(f"expected columns missing from build_advanced_features output: {missing}")
    panel = panel[["ticker", "date"] + KEEP_COLS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)

    n_tickers = panel["ticker"].nunique()
    n_dates = panel["date"].nunique()
    n_cols = panel.shape[1]
    has_any_news = panel.drop(columns=["ticker", "date"]).notna().any(axis=1).mean()
    print(f"[done] {OUT_PATH} — {panel.shape[0]} rows, {n_cols} cols, "
          f"{n_tickers} tickers x {n_dates} dates, "
          f"{has_any_news * 100:.2f}% rows have >=1 non-NaN news feature "
          f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
