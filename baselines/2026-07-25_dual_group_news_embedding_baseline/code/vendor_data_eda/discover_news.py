"""Dynamic discovery + schema-normalizing loader for news CSVs under ``CRAWL_DATA_ROOT``.

Vendored (trimmed) from C:\\luanvan\\data_eda\\src\\data\\discover_news.py (2026-07-25) — copy
only, data_eda itself is not modified. Dropped: `load_all_sources`/`log_processing`
(processing-log traceability, unused by the aggregation path this baseline needs — only
`discover_source_files` + `load_source` are called by `news_embeddings.py`).

Three schemas are recognized:
- OLD (cafef/ssi/vndirect/hsc-style): ``title``, ``pub_date``, ``lead`` (DD/MM or ISO
  mixed dates, per the existing project convention).
- NEW "tier" schema (``objective/`` subdirectory): ``title``, ``publish_time`` (ISO 8601 UTC),
  ``source_tier``, ``raw_text`` (full body, richer than the old ``lead``).
- VNSTOCK schema (``vnstock_articles.csv``): ``title``, ``date`` — a PDF-crawl metadata file
  that tracks brokerage PDF downloads per ticker, using ``date`` instead of ``pub_date``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vendor_config import CRAWL_DATA_ROOT

# Backup/archive/duplicate snapshots of the SAME underlying vnstock PDF crawl (verified: their
# 'source' column holds brokerage names like "Vietstock"/"MBS"/"KBSV" identical to
# vnstock_articles.csv, with overlapping/subset row counts) — not distinct sources.
_DENYLIST = {
    "data.csv",
    "data_2021_2025.csv",
    "data_archive.csv",
    "vnstock_pdf_raw.csv",
    "vnstock_pdfs_extracted.csv",
    # news_articles.csv is the literal union of cafef+ssi+vndirect+hsc (row counts verified to
    # match exactly) — reading it AND its constituents would double-count every article.
    "news_articles.csv",
}
# Rolling consolidated snapshot (explicitly a partial union), not a distinct source; the
# underlying tier files (vietstock_records.csv, vsdc_records.csv, ...) are read directly instead.
_SNAPSHOT_PREFIX = "objective_v"

OLD_SCHEMA_COLS = {"title", "pub_date"}
NEW_SCHEMA_COLS = {"title", "publish_time", "source_tier"}
VNSTOCK_SCHEMA_COLS = {"title", "date"}


def _infer_source_name(path: Path) -> str:
    """Derive a short source name from a filename, e.g. 'news_unenriched_vnexpress_records.csv'
    -> 'vnexpress', 'cafef_articles.csv' -> 'cafef'."""
    name = path.stem
    if name.startswith("news_unenriched_"):
        name = name[len("news_unenriched_"):]
    for suffix in ("_articles", "_records"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower()


def discover_source_files() -> dict[str, Path]:
    """{source_name: path} for every schema-valid news CSV under ``CRAWL_DATA_ROOT``
    (recursive), excluding known backups/duplicates/snapshots.

    Two files under different subdirectories can infer the SAME source name (e.g. a top-level
    ``thanhnien_articles.csv`` historical backfill alongside
    ``objective/news_unenriched_thanhnien_records.csv``, a distinct tier-classified crawl).
    Silently keeping only the alphabetically-last path would drop the other source's articles
    entirely. Instead, when a name collision is detected, EVERY colliding path is disambiguated
    by its parent directory (e.g. ``thanhnien`` + ``thanhnien_objective``) so no source is ever
    silently dropped — the single-file (non-colliding) case is unaffected (bare name, as before).
    """
    found: dict[str, Path] = {}
    if not CRAWL_DATA_ROOT.exists():
        return found
    candidates: list[tuple[str, Path]] = []
    for p in sorted(CRAWL_DATA_ROOT.rglob("*.csv")):
        if p.name in _DENYLIST or p.name.startswith(_SNAPSHOT_PREFIX):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=0, encoding="utf-8").columns)
        except Exception:
            continue
        if not (OLD_SCHEMA_COLS <= cols or NEW_SCHEMA_COLS <= cols or VNSTOCK_SCHEMA_COLS <= cols):
            continue
        candidates.append((_infer_source_name(p), p))

    by_name: dict[str, list[Path]] = {}
    for name, p in candidates:
        by_name.setdefault(name, []).append(p)

    for name, paths in by_name.items():
        if len(paths) == 1:
            found[name] = paths[0]
            continue
        for p in paths:
            suffix = p.parent.name if p.parent != CRAWL_DATA_ROOT else "root"
            disambiguated = f"{name}_{suffix}"
            found[disambiguated] = p
    return found


def load_source(source: str, path: Path) -> pd.DataFrame:
    """Load one discovered file, normalized to include ``source``/``pub_date``/``lead``/``url``
    columns regardless of which schema the file uses."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = set(df.columns)
    if "url" not in cols and "article_url" in cols:
        # cafef_articles.csv uses `article_url` instead of `url` — without this rename, every
        # downstream consumer that requires a `url` column (article-embedding cache, ticker
        # explode) silently drops 100% of this source's rows via dropna(subset=["url"]).
        df = df.rename(columns={"article_url": "url"})
        cols = set(df.columns)
    if NEW_SCHEMA_COLS <= cols:
        df["pub_date"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True).dt.tz_localize(None)
        df["lead"] = df.get("raw_text", pd.Series(index=df.index)).fillna("")
    elif VNSTOCK_SCHEMA_COLS <= cols and "pub_date" not in cols:
        df["pub_date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce", utc=True
        ).dt.tz_localize(None)
        df["lead"] = df.get("lead", pd.Series(index=df.index)).fillna("")
        if "url" not in cols:
            df["url"] = df.get("pdf_url", pd.Series(index=df.index)).fillna("")
    else:
        df["pub_date"] = pd.to_datetime(
            df.get("pub_date"), format="mixed", dayfirst=True, errors="coerce", utc=True
        ).dt.tz_localize(None)
        df["lead"] = df.get("lead", pd.Series(index=df.index)).fillna("")
    df["source"] = source
    return df
