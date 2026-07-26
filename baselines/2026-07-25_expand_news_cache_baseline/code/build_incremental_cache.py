"""Incrementally extend data/external_news_embeddings/raw_cache/ with newly crawled articles.

crawl_data/data (C:\\luanvan\\crawl_data\\data) grew a lot since the cache was last built
(2026-07-24 snapshot, see sibling `2026-07-25_dual_group_news_embedding_baseline`): 12 brand-new
sources (no cache at all) + ~10 existing sources with new articles. This script encodes ONLY the
not-yet-cached, ticker-mentioning articles via real PhoBERT calls (unlike the sibling baseline's
`news_embeddings._get_article_embeddings`, which is deliberately cache-only and must never call
PhoBERT) and upserts them into the same per-source parquet cache files, so the sibling baseline's
panel-builder picks up the new coverage next time it runs — no code change needed there.

Isolated (CLAUDE.md §3.F): does not modify anything in
`2026-07-25_dual_group_news_embedding_baseline/` — only imports its discovery/encode utilities
read-only (established precedent: `2026-07-18_gated_crossattn_baseline` already imports read-only
from `2026-07-07_embedding_baseline` the same way).

Run (dry-run, no PhoBERT calls, just counts): python build_incremental_cache.py --dry_run
Run (real):                                   python build_incremental_cache.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from vendor_data_eda.discover_news import discover_source_files, load_source  # noqa: E402
from vendor_data_eda.news_embeddings import TICKER_PATTERN, _article_cache_path, RAW_DIM  # noqa: E402
from vendor_data_eda.phobert_embeddings import extract_phobert_embeddings  # noqa: E402

# Documentation only (see design.md §2) — this script encodes every discovered source's
# ticker-mentioning articles regardless of khach_quan/tong_hop classification, so this set is
# not read at runtime; it records which 12 sources are newly covered by this session's crawl.
NEW_KHACH_QUAN_SOURCES = {
    "baophapluat", "bnews", "cand", "dantri", "giaoducthoidai", "hanoimoi",
    "plo", "sggp", "tapchicongthuong", "tienphong", "viettimes", "vov",
}


def _clean_articles(source: str, path: Path) -> pd.DataFrame:
    """Load one source, return rows with a non-empty url + non-empty title+lead text
    (de-duped by url). No ticker filter — shared by both `_ticker_mentioning_articles` and
    `_all_articles` below."""
    df = load_source(source, path)
    if "url" not in df.columns:
        return pd.DataFrame({"url": pd.Series([], dtype=object),
                              "_text": pd.Series([], dtype=object)})
    title = df.get("title", pd.Series(index=df.index)).fillna("")
    lead = df.get("lead", pd.Series(index=df.index)).fillna("")
    text = (title.astype(str) + " " + lead.astype(str)).str.strip()
    mask = df["url"].notna() & (text.str.len() > 0)
    df = df[mask].copy()
    df["_text"] = text[mask]
    return df.dropna(subset=["url"]).drop_duplicates(subset="url").reset_index(drop=True)


def _ticker_mentioning_articles(source: str, path: Path) -> pd.DataFrame:
    """Rows from `_clean_articles` whose text mentions a VN30 ticker.

    Mirrors news_embeddings._load_group's per-article filter (title+lead text, url present,
    de-duped, ticker-mention regex) but applied to ANY discovered source, not just ones already
    classified into KHACH_QUAN_SOURCES/TONG_HOP_SOURCES (this script's whole point is to cover
    sources that aren't classified yet).
    """
    df = _clean_articles(source, path)
    return df[df["_text"].str.contains(TICKER_PATTERN, regex=True, na=False)].reset_index(drop=True)


def _all_articles(source: str, path: Path) -> pd.DataFrame:
    """All of `_clean_articles`'s rows — no ticker-mention filter. Used for the
    market-wide/macro embedding use case (--include_all), where articles that never mention a
    specific VN30 ticker are still wanted."""
    return _clean_articles(source, path)


def _load_existing_cache(source: str) -> pd.DataFrame:
    """Existing per-source cache, or an empty (url-only) frame if missing/corrupted/wrong-dim."""
    path = _article_cache_path(source)
    if not path.exists():
        return pd.DataFrame({"url": []})
    try:
        cached = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame({"url": []})
    raw_cols = [c for c in cached.columns if c.startswith("raw_")]
    if raw_cols and len(raw_cols) != RAW_DIM:
        return pd.DataFrame({"url": []})
    return cached


def _new_rows_to_encode(articles: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Rows in `articles` whose url is NOT already in `existing`."""
    known = set(existing["url"]) if not existing.empty else set()
    return articles[~articles["url"].isin(known)].reset_index(drop=True)


def _encode_rows(new_articles: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Run PhoBERT on new_articles['_text'], return {url, raw_0..raw_{RAW_DIM-1}}."""
    if new_articles.empty:
        return pd.DataFrame({"url": []})
    embs = extract_phobert_embeddings(new_articles["_text"].tolist(), batch_size=batch_size)
    if embs.shape[1] != RAW_DIM:
        raise RuntimeError(f"PhoBERT returned dim {embs.shape[1]}, expected {RAW_DIM}")
    emb_cols = {f"raw_{i}": embs[:, i] for i in range(embs.shape[1])}
    return pd.concat(
        [new_articles[["url"]].reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write to a .tmp file then rename — a crash mid-write never leaves a truncated cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


CHUNK_SIZE = 5000  # bounds peak memory + limits crash-loss to <=1 chunk for huge (--include_all) sources


def run_source(source: str, path: Path, batch_size: int = 32, include_all: bool = False,
               chunk_size: int = CHUNK_SIZE) -> dict:
    """Encode + upsert-cache one source. Returns a stats dict for logging (no side effects if
    there is nothing new to encode). `include_all=True` encodes every article with text (not
    just ticker-mentioning ones) — the market-wide/macro embedding use case, which can be up to
    ~1M+ articles for a single large source. Processed in `chunk_size`-row chunks, writing
    (atomically) after each chunk, so a crash partway through a huge source loses at most one
    chunk's work, not the whole source, and peak memory stays bounded regardless of source size.
    """
    select = _all_articles if include_all else _ticker_mentioning_articles
    articles = select(source, path)
    existing = _load_existing_cache(source)
    new_articles = _new_rows_to_encode(articles, existing)
    if new_articles.empty:
        return {"source": source, "n_candidates": len(articles), "n_before": len(existing),
                "n_new": 0, "n_after": len(existing)}

    combined = existing
    n_encoded = 0
    for start in range(0, len(new_articles), chunk_size):
        chunk = new_articles.iloc[start:start + chunk_size]
        new_rows = _encode_rows(chunk, batch_size=batch_size)
        combined = (pd.concat([combined, new_rows], ignore_index=True) if not combined.empty
                    else new_rows)
        combined = combined.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
        _atomic_write(combined, _article_cache_path(source))
        n_encoded += len(new_rows)

    return {"source": source, "n_candidates": len(articles), "n_before": len(existing),
            "n_new": n_encoded, "n_after": len(combined)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None,
                    help="limit to these source names (default: all discovered)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                    help="rows per encode+write chunk within a source (larger = fewer PhoBERT "
                         "model-reload calls, more memory/less checkpointing per chunk)")
    ap.add_argument("--dry_run", action="store_true",
                    help="only report counts, never call PhoBERT or write")
    ap.add_argument("--include_all", action="store_true",
                    help="encode ALL articles with text, not just ticker-mentioning ones "
                         "(market-wide/macro embedding use case)")
    args = ap.parse_args()
    select = _all_articles if args.include_all else _ticker_mentioning_articles

    files = discover_source_files()
    if args.sources:
        unknown = sorted(set(args.sources) - set(files))
        if unknown:
            raise ValueError(f"--sources contains unknown source name(s): {unknown} "
                              f"(discovered: {sorted(files)})")
    targets = {s: p for s, p in files.items() if args.sources is None or s in args.sources}
    print(f"[expand_news_cache] {len(targets)} source(s) to process "
          f"(dry_run={args.dry_run}, include_all={args.include_all})")

    t0 = time.time()
    total_new = 0
    for source, path in sorted(targets.items()):
        if args.dry_run:
            articles = select(source, path)
            existing = _load_existing_cache(source)
            n_new = len(_new_rows_to_encode(articles, existing))
            print(f"  {source:<24} n_candidates={len(articles):>7} "
                  f"n_before={len(existing):>7} n_new={n_new:>7}")
            total_new += n_new
            continue
        stats = run_source(source, path, batch_size=args.batch_size, include_all=args.include_all,
                           chunk_size=args.chunk_size)
        total_new += stats["n_new"]
        print(f"  {stats['source']:<24} n_candidates={stats['n_candidates']:>7} "
              f"n_before={stats['n_before']:>7} n_new={stats['n_new']:>7} "
              f"n_after={stats['n_after']:>7}")

    print(f"[done] total new rows encoded: {total_new} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
