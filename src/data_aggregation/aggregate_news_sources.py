"""Aggregate multi-source crawled news into ONE unified, deduplicated dataset.

ISOLATED (hard requirement):
  - Reads ONLY from:  D:/bmad-projects/crawl_data/data
  - Writes ONLY to:   D:/bmad-projects/crawl_data/aggregated
  - Does NOT touch:   project data/raw/news, data/processed, src/sentiment_baseline, any old file.

Two source schema families are unified into a superset schema:
  - Family A (title-only): id,title,source,date,pdf_url,pdf_filename,downloaded_at
      → data.csv, data_2021_2025.csv, data_archive.csv, vnstock_articles.csv
  - Family B (richer — has lead/category/author): id,source,title,category,pub_date,url,author,lead,...
      → cafef_articles.csv, ssi_articles.csv, vndirect_articles.csv, hsc_articles.csv, news_articles.csv

Unified output schema:
  unified_id, source, title, lead, category, author,
  date (YYYY-MM-DD), pub_datetime, url, pdf_url, pdf_filename, collected_at, origin_file

Dedup: by normalized URL; fallback to title-hash when URL empty. Prefers rows WITH lead.

Usage:
  python -m src.data_aggregation.aggregate_news_sources
"""
from __future__ import annotations
import re
import hashlib
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]  # project root (src/data_aggregation/)
RAW_DIR = _ROOT.parent / "crawl_data" / "data"  # [HIGH-2] sibling crawl_data dir (D:/bmad-projects/crawl_data)
OUT_DIR = _ROOT.parent / "crawl_data" / "aggregated"

FILES_A = ["data.csv", "data_2021_2025.csv", "data_archive.csv", "vnstock_articles.csv",
           "vnstock_pdf_raw.csv"]  # vnstock_pdf_raw: new (2026-07-11), has body + full schema
FILES_B = ["cafef_articles.csv", "ssi_articles.csv", "vndirect_articles.csv",
           "hsc_articles.csv", "news_articles.csv"]

UNIFIED_COLS = ["unified_id", "source", "title", "lead", "body", "category", "author",
                "date", "pub_datetime", "url", "pdf_url", "pdf_filename",
                "collected_at", "origin_file"]


def _read(name: str) -> pd.DataFrame:
    # [MED-5 reverted] engine="python" + callable BROKE body text parsing (embedded
    # newlines in body → treated as bad lines → 90%+ rows silently dropped).
    # C engine (default) correctly handles quoted fields with embedded newlines/commas.
    # on_bad_lines="skip" drops only genuinely malformed rows (rare).
    df = pd.read_csv(RAW_DIR / name, dtype=str, encoding="utf-8-sig",
                     keep_default_na=False, on_bad_lines="skip")
    return df


def _pick(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return the first existing column from df as a Series (empty if none)."""
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series("", index=df.index)


def normalize_a(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    # Family A: title-only. url absent → use pdf_url as url (often the report link).
    return pd.DataFrame({
        "unified_id": df.get("id", ""),
        "source": df.get("source", ""),
        "title": df.get("title", ""),
        "lead": _pick(df, "lead"),   # vnstock_pdf_raw has lead; Family A originals don't (="")
        "body": _pick(df, "body"),   # vnstock_pdf_raw has body; Family A originals don't
        "category": "",
        "author": "",
        "date": df.get("date", ""),
        "pub_datetime": df.get("date", ""),
        "url": df.get("pdf_url", ""),
        "pdf_url": df.get("pdf_url", ""),
        "pdf_filename": df.get("pdf_filename", ""),
        "collected_at": df.get("downloaded_at", ""),
        "origin_file": origin,
    })[UNIFIED_COLS]


def normalize_b(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    # Family B: richer schema but column names vary across sources.
    # cafef uses article_url (not url), section (not category), and has no source.
    # news_articles.csv has no 'id' column → _pick returns "".
    return pd.DataFrame({
        "unified_id": _pick(df, "id"),
        "source": _pick(df, "source"),
        "title": _pick(df, "title"),
        "lead": _pick(df, "lead"),
        "body": _pick(df, "body"),   # cafef/ssi/vndirect now have body text (2026-07-11 crawl)
        "category": _pick(df, "category", "section"),
        "author": _pick(df, "author"),
        "date": _pick(df, "pub_date"),
        "pub_datetime": _pick(df, "pub_date"),
        "url": _pick(df, "url", "article_url"),
        "pdf_url": _pick(df, "pdf_url"),
        "pdf_filename": _pick(df, "pdf_filename"),
        "collected_at": _pick(df, "collected_at"),
        "origin_file": origin,
    })[UNIFIED_COLS]


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DMY_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})")


def _parse_one(s):
    s = str(s).strip()
    if not s:
        return pd.NaT
    if _ISO_RE.match(s):
        return pd.to_datetime(s, errors="coerce")
    m = _DMY_RE.match(s)
    if m:
        d, mo, y = m.groups()
        y = int(y)
        y = 2000 + y if y < 100 else y
        try:
            return pd.Timestamp(year=y, month=int(mo), day=int(d))
        except ValueError:
            return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize mixed tz: cafef ISO dates carry a tz offset (+0700) while broker
    # reports are tz-naive (DD/MM/YYYY). pd.to_datetime coerces tz-aware rows to
    # NaT when mixed with tz-naive in one datetime64 column -> cafef dates were
    # silently LOST (the "cafef 0% date coverage" bug). utc=True + tz_localize(None)
    # collapses all to tz-naive UTC so no row is dropped on the tz boundary.
    parsed = pd.to_datetime(df["date"].map(_parse_one), errors="coerce", utc=True).dt.tz_localize(None)
    df["pub_datetime"] = parsed
    df["date"] = parsed.dt.strftime("%Y-%m-%d")
    return df


def _norm_url(u) -> str:
    u = str(u).strip().lower()
    if not u:
        return ""
    u = re.sub(r"\?.*$", "", u)            # strip query string
    u = u.removesuffix("/")                 # normalize trailing slash
    return u


def _title_hash(t) -> str:
    return hashlib.md5(str(t).strip().lower().encode("utf-8")).hexdigest()[:16]


def dedup(df: pd.DataFrame):
    df = df.copy()
    df["_url"] = df["url"].map(_norm_url)
    df["_has_lead"] = (df["lead"].str.strip().str.len() > 0).astype(int)
    df["_title_hash"] = df["title"].map(_title_hash)
    df["_key"] = df["_url"].where(df["_url"] != "", "T:" + df["_title_hash"])
    # Prefer rows WITH lead, then newer dates, when collapsing duplicates.
    df = df.sort_values(["_has_lead", "date"], ascending=[False, False], kind="mergesort")
    before = len(df)
    df = df.drop_duplicates("_key", keep="first")
    after = len(df)
    # Fill missing unified_id with the dedup key so every row has a stable id.
    need_id = df["unified_id"].str.strip() == ""
    df.loc[need_id, "unified_id"] = df.loc[need_id, "_key"]
    df = df.drop(columns=["_url", "_has_lead", "_title_hash", "_key"])
    return df, before, after


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, raw_counts = [], {}

    for name in FILES_A + FILES_B:
        path = RAW_DIR / name
        if not path.exists():
            print(f"[skip] {name} not found")
            continue
        df = _read(name)
        raw_counts[name] = len(df)
        norm = normalize_a(df, name) if name in FILES_A else normalize_b(df, name)
        frames.append(norm)
        print(f"[load] {name:28s} rows={len(df):>6d}")

    big = pd.concat(frames, ignore_index=True)
    print(f"\n[concat] total raw rows = {len(big)}")

    # Fill empty source from origin_file (e.g., cafef_articles.csv → cafef).
    origin_label = (big["origin_file"].str.replace(r"_articles\.csv$", "", regex=True)
                    .str.replace(r"\.csv$", "", regex=True))
    empty_source = big["source"].str.strip() == ""
    big.loc[empty_source, "source"] = origin_label[empty_source]

    big = parse_dates(big)
    parsed_ok = big["pub_datetime"].notna().sum()
    print(f"[dates] parsed OK = {parsed_ok}/{len(big)} ({parsed_ok/len(big)*100:.1f}%)")

    big, before, after = dedup(big)
    print(f"[dedup] {before} -> {after} (removed {before-after} duplicates)")

    # Final tidy: empty strings → for clean CSV, keep as "".
    big = big[UNIFIED_COLS].sort_values("date", ascending=False, kind="mergesort").reset_index(drop=True)

    out_csv = OUT_DIR / "unified_articles.csv"
    big.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[write] {out_csv}  ({len(big)} rows)")

    # ---- Stats report ----
    has_lead = (big["lead"].str.strip().str.len() > 0).sum()
    big["_year"] = big["date"].str[:4].replace("", pd.NA)
    by_year = big["_year"].value_counts(dropna=True).sort_index()
    by_origin = big["origin_file"].value_counts().sort_values(ascending=False)
    by_source = big["source"].str.strip().replace("", "(empty)").value_counts().head(15)

    lines = []
    lines.append("=== NEWS AGGREGATION STATS ===\n")
    lines.append(f"Total raw rows (concat): {len(frames) and sum(raw_counts.values())}")
    lines.append("Raw rows per file:")
    for k, v in raw_counts.items():
        lines.append(f"  {k:28s} {v:>6d}")
    lines.append(f"\nAfter dedup: {len(big)}")
    lines.append(f"Rows WITH lead (rich text for embeddings): {has_lead} ({has_lead/len(big)*100:.1f}%)")
    lines.append(f"Date parsed OK: {parsed_ok}")
    lines.append("\nUnique contribution per origin file (post-dedup):")
    for k, v in by_origin.items():
        lines.append(f"  {k:28s} {v:>6d}")
    lines.append("\nTop sources:")
    for k, v in by_source.items():
        lines.append(f"  {k:20s} {v:>6d}")
    lines.append("\nPer-year coverage (post-dedup):")
    for y, v in by_year.items():
        lines.append(f"  {y}  {v:>6d}")
    no_date = big["_year"].isna().sum()
    lines.append(f"\nRows with no parseable date: {no_date}")

    stats = "\n".join(lines)
    stats_path = OUT_DIR / "aggregation_stats.txt"
    stats_path.write_text(stats, encoding="utf-8")
    print(f"[write] {stats_path}\n")
    print(stats)


if __name__ == "__main__":
    main()
