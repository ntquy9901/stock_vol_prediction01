# Design (Plan) — Expand News-Embedding Cache

## 1. Data flow

```
crawl_data/data/*.csv
   -> discover_source_files()          [read-only import, vendor_data_eda/discover_news.py]
   -> load_source(source, path)        [normalizes url/pub_date/lead/source cols]
   -> filter: has url, non-empty text, mentions a VN30 ticker (TICKER_PATTERN, same regex
              as news_embeddings.py — imported read-only, not redefined, so filter behavior
              is IDENTICAL to what the panel-builder baseline expects)
   -> diff against existing cache (data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet)
      by `url` -> only the NOT-yet-cached rows go to PhoBERT
   -> extract_phobert_embeddings(texts)  [read-only import, vendor_data_eda/phobert_embeddings.py]
   -> new_rows_df = {url, raw_0..raw_767, <original metadata cols kept for schema parity>}
   -> upsert: pd.concat([existing_cache, new_rows_df]).drop_duplicates(subset="url", keep="first")
   -> atomic write: write to <path>.tmp, then os.replace(tmp, final)  [never leaves a half-written
      parquet if the process dies mid-write]
```

## 2. Source classification (new sources only)

12 newly-discovered sources classified as `khach_quan` (mainstream press — matches the existing
group's composition: cafef, vnexpress, thanhnien, tuoitre, nld, vietnamplus, baodautu, cafebiz,
coin68, fica, nhadautu, nhipsongkinhdoanh, theinvestor, thoibaotaichinhvietnam,
thuonghieucongluan, tinnhanhchungkhoan, vietbao, vietnambiz, vietnamfinance, vietnamnet,
vneconomy — all mainstream news portals, none are securities-firm/analyst content):

```python
NEW_KHACH_QUAN_SOURCES = {
    "baophapluat", "bnews", "cand", "dantri", "giaoducthoidai", "hanoimoi",
    "plo", "sggp", "tapchicongthuong", "tienphong", "viettimes", "vov",
}
```

This list is used ONLY to decide which of the 52 discovered sources this script bothers
encoding (it does not need `tong_hop` classification — this script doesn't do group-based
aggregation, that's the other baseline's job). Concretely: this script encodes **every**
discovered source's ticker-mentioning articles (whether already known to `KHACH_QUAN_SOURCES`/
`TONG_HOP_SOURCES` from the other baseline, or newly added here) — building the cache is a
superset operation, cheaper to do for all sources than to special-case. The classification
constant above exists for documentation/traceability of "what's new," not as a runtime filter.

## 3. Why cache-diff instead of re-encoding everything

`extract_phobert_embeddings` for ~1.3M total discovered articles across 52 sources (most
without a ticker mention) would be wasteful and slow. Filtering to ticker-mentioning articles
first (13,818 total across all sources, per scoping) keeps this to a ~10-15 min CPU job. Only
NEW urls (not already in a source's existing cache) are actually sent to PhoBERT — existing
cached rows are never re-encoded or modified.

## 4. Safety

- **Backup first:** copy `data/external_news_embeddings/raw_cache/` to a timestamped sibling dir
  before any write (raw_cache holds real PhoBERT compute — expensive to regenerate; a bug in the
  upsert logic must not be able to destroy existing work).
- **Atomic write per source:** write `.tmp` then `os.replace()` — a crash mid-run leaves the old
  cache file intact for that source (never truncated/partial).
- **Dedup rule:** `drop_duplicates(subset="url", keep="first")` — existing cached rows always win
  ties, so even if this script somehow re-encoded an already-cached url, the OLD (already-used-
  by-committed-results) embedding is kept, not silently replaced by a new one that could shift
  downstream numbers.

## 5. Gates

- **Simplicity Gate:** one script (`build_incremental_cache.py`) + one small helper module,
  no new framework/config layer. Reuses the other baseline's discovery/encode code as-is
  (read-only import), does not reimplement ticker filtering or PhoBERT wrapping.
- **Anti-Abstraction Gate:** no wrapper class needed — a few top-level functions
  (`_diff_new_rows`, `_upsert_write`, `run_source`) are enough; matches the existing
  `build_dual_group_panel.py` sibling's style (plain functions, no classes).

## 6. Out of scope / deferred

- Re-running `build_dual_group_panel.py` (the panel rebuild) — separate, cheap, user can trigger
  after this if they want the new coverage reflected in a retrained baseline.
- Re-training any model — not requested.
