"""Fetch GICS sector labels for the S&P 500 and write a provenance CSV.

Source: the maintained datahub constituents table
``https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv``
-- the SAME source ``src/data/download_sp500.py`` already uses for the ticker universe, so the
sector labels are aligned to the downloaded raw files by construction. It exposes a ``GICS Sector``
column (the 11 official GICS sectors) mirroring the Wikipedia "List of S&P 500 companies" table.

Ticker sanitization matches the raw files: the index list uses ``.`` (BRK.B, BF.B) while the Yahoo
raw CSVs use ``-`` (BRK-B, BF-B); we store the dash form so ``load_sector_map`` keys line up with
``<TICKER>_ohlcv.csv`` and the processed panel.

No wall-clock is read here: the ``fetched_date`` is passed in (fixed string) so the CSV is
reproducible and diff-stable.

Run: python fetch_sectors.py --fetched-date 2026-08-29 --out sp500_gics_sectors.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def sanitize(symbol: str) -> str:
    """Index dot-form symbol -> Yahoo dash-form used by the raw ``*_ohlcv.csv`` files (BRK.B -> BRK-B)."""
    return str(symbol).strip().replace(".", "-")


def build_sector_map(df: pd.DataFrame) -> dict[str, str]:
    """``{ticker(dash-form): GICS Sector}`` from a constituents frame.

    Requires ``Symbol`` and ``GICS Sector`` columns. Rows with a blank sector are skipped (the ticker
    stays unmapped -> singleton own-sector downstream)."""
    if "Symbol" not in df.columns or "GICS Sector" not in df.columns:
        raise ValueError(f"constituents frame missing Symbol/GICS Sector; has {list(df.columns)}")
    out: dict[str, str] = {}
    for sym, sec in zip(df["Symbol"], df["GICS Sector"]):
        sec = "" if pd.isna(sec) else str(sec).strip()
        tk = sanitize(sym)
        if tk and sec:
            out[tk] = sec
    return out


def write_sector_csv(sector_map: dict[str, str], out_path: str | Path, source_url: str,
                     fetched_date: str) -> Path:
    """Write ``ticker,sector,source_url,fetched_date`` (sorted by ticker) with provenance."""
    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "sector", "source_url", "fetched_date"])
        for tk in sorted(sector_map):
            w.writerow([tk, sector_map[tk], source_url, fetched_date])
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched-date", required=True,
                    help="fixed provenance date string (e.g. 2026-08-29); NOT read from the clock")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "sp500_gics_sectors.csv"))
    ap.add_argument("--url", default=CONSTITUENTS_URL)
    a = ap.parse_args()
    df = pd.read_csv(a.url)
    sm = build_sector_map(df)
    out = write_sector_csv(sm, a.out, a.url, a.fetched_date)
    n_sec = len(set(sm.values()))
    print(f"wrote {len(sm)} tickers / {n_sec} GICS sectors -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
