"""Analyze per-year news sparsity of the unified dataset vs the VN30 trading calendar.

Question being answered: is the news data still sparse when compared across years?

Metrics per year:
  - articles: total news articles with a parseable date in that year
  - matched_title: articles whose TITLE matches >=1 VN30 ticker (whole-word)
  - matched_full: articles whose title OR lead matches (lead available for ~30% of rows)
  - stockdays_news: unique (ticker, date) pairs with news (the modeling-relevant unit)
  - trading_stockdays: total possible (ticker, trading-date) pairs from the price calendar
  - coverage %: stockdays_news / trading_stockdays  <- the true density metric

ISOLATED: read-only on unified_articles.csv, processing_summary.csv, {TICKER}_processed.csv.
Writes: D:/bmad-projects/crawl_data/aggregated/sparsity_report.txt

Usage: python -m src.data_aggregation.analyze_news_sparsity
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CRAWL_AGGREGATED = ROOT.parent / "crawl_data" / "aggregated"  # sibling repo, per aggregate_news_sources.py
UNIFIED = CRAWL_AGGREGATED / "unified_articles.csv"
STOCK_DIR = ROOT / "data" / "processed"
SUMMARY = STOCK_DIR / "processing_summary.csv"
OUT = CRAWL_AGGREGATED / "sparsity_report.txt"


def load_tickers():
    with open(SUMMARY, encoding="utf-8-sig") as f:
        return [(r.get("ticker") or "").strip() for r in csv.DictReader(f) if (r.get("ticker") or "").strip()]


def main():
    tickers = load_tickers()
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b")
    print(f"[tickers] VN30 = {len(tickers)}")

    # ---- Trading calendar: total (ticker, trading-date) pairs per year ----
    trading = Counter()
    for t in tickers:
        f = STOCK_DIR / f"{t}_processed.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, usecols=["date"])
        except Exception:
            continue
        for d in df["date"].astype(str):
            y = d[:4]
            if y.isdigit() and len(y) == 4:
                trading[y] += 1
    print(f"[calendar] total trading stock-days = {sum(trading.values())}")

    # ---- News: scan unified articles ----
    df = pd.read_csv(UNIFIED, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    articles = Counter()
    match_title = Counter()
    match_full = Counter()
    sd_title = defaultdict(set)   # year -> {(ticker, date)}
    sd_full = defaultdict(set)
    no_date = 0

    cols = df["date"].tolist()
    titles = df["title"].tolist()
    leads = df["lead"].tolist()
    bodies = df["body"].tolist() if "body" in df.columns else [""] * len(df)
    for i in range(len(df)):
        y = (cols[i] or "")[:4]
        if not (y.isdigit() and len(y) == 4):
            no_date += 1
            continue
        articles[y] += 1
        title = titles[i] or ""
        lead = leads[i] or ""
        body = bodies[i] or ""
        mt = set(pat.findall(title))
        # search title + body + lead (body now available from 2026-07-11 crawl)
        search_text = (title + " " + body + " " + lead).strip()
        mf = set(pat.findall(search_text)) if search_text else mt
        if mt:
            match_title[y] += 1
            for t in mt:
                sd_title[y].add((t, cols[i]))
        if mf:
            match_full[y] += 1
            for t in mf:
                sd_full[y].add((t, cols[i]))

    # ---- Build report ----
    years = sorted(articles.keys())
    lines = []
    lines.append("=== NEWS SPARSITY ANALYSIS (unified vs VN30 trading calendar) ===\n")
    lines.append(f"VN30 tickers: {len(tickers)}")
    lines.append(f"Total trading stock-days (calendar): {sum(trading.values())}")
    lines.append(f"Unified articles with parseable year: {sum(articles.values())}  (no date: {no_date})\n")
    lines.append("Per-year breakdown:")
    hdr = (
        f"{'year':>6} {'articles':>9} {'mtch_title':>10} {'mtch_full':>9} "
        f"{'sd_news':>8} {'sd_total':>9} {'cover%':>7}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for y in years:
        a = articles[y]
        mt = match_title[y]
        mf = match_full[y]
        sdn = len(sd_full[y])            # use full (title+lead) as the best-case density
        sdt = trading.get(y, 0)
        cov = (100.0 * sdn / sdt) if sdt else 0.0
        lines.append(f"{y:>6} {a:>9} {mt:>10} {mf:>9} {sdn:>8} {sdt:>9} {cov:>6.1f}%")

    # Splits per design (train 2006-2020 / val 2020-2021 / test 2021-2026)
    def agg(years_set, fn):
        tot_a = sum(articles[y] for y in years_set)
        tot_sdn = len(set().union(*(sd_full[y] for y in years_set if sd_full[y])))
        tot_sdt = sum(trading[y] for y in years_set)
        return tot_a, tot_sdn, tot_sdt

    lines.append("\nBy design split (train/val/test):")
    train = [y for y in years if y < "2020"]
    val = [y for y in years if "2020" <= y < "2022"]
    test = [y for y in years if y >= "2021"]
    for name, ys in [("train (<2020)", train), ("val (2020-2021)", val), ("test (>=2021)", test)]:
        a, sdn, sdt = agg(set(ys), None)
        cov = (100.0 * sdn / sdt) if sdt else 0.0
        lines.append(
            f"  {name:16s} articles={a:>6}  stockdays_news={sdn:>6}  "
            f"stockdays_total={sdt:>7}  cover={cov:5.1f}%"
        )

    # Match-rate
    tot_a = sum(articles.values())
    lines.append(f"\nTitle match rate (all years): {sum(match_title.values())}/{tot_a} "
                 f"({100*sum(match_title.values())/max(1,tot_a):.1f}%)")
    lines.append(f"Title+lead match rate:         {sum(match_full.values())}/{tot_a} "
                 f"({100*sum(match_full.values())/max(1,tot_a):.1f}%)")

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(f"\n[write] {OUT}\n")
    # Console-safe print (ASCII only, avoid cp1252 crash on Vietnamese)
    print(text)


if __name__ == "__main__":
    main()
