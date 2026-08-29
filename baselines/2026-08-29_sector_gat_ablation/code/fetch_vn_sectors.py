"""Fetch ICB industry (sector) labels for Vietnamese tickers and write a canonical provenance CSV.

GICS/Wikipedia does not cover Vietnamese equities, so the classification comes from the ``vnstock``
library's ICB taxonomy via ONE bulk call ``Listing().symbols_by_industries()`` (columns
``symbol, industry_code, industry_name`` for ~700 listed symbols). One request for the whole market
respects vnstock's ~20 req/min per-IP limit (never one request per ticker).

Output schema matches the panel-agnostic ``sector_adjacency.load_sector_map`` reader:
``ticker,sector,source_url,fetched_date`` (sector = ICB ``industry_name``). ``fetched_date`` is a
fixed string passed in -- never read from the clock -- so the CSV is reproducible.

vnstock is imported LAZILY inside ``fetch_industries_df`` so this module (and its pure builder) import
cleanly under the GPU venv used by the quality gate, where vnstock is not installed.

Build the canonical CSV from an already-fetched raw dump (no network):
  python fetch_vn_sectors.py --from-raw vn_icb_sectors.csv --fetched-date 2026-08-29 --out vn_sectors.csv
Live fetch (run under the vnstock venv, one network call):
  python fetch_vn_sectors.py --live --fetched-date 2026-08-29 --out vn_sectors.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

SOURCE = "vnstock Listing().symbols_by_industries() (ICB industry_name)"


def fetch_industries_df() -> pd.DataFrame:  # pragma: no cover (needs vnstock; run under the vnstock venv)
    """ONE bulk vnstock call -> DataFrame[symbol, industry_code, industry_name]. Lazy import."""
    from vnstock import Listing  # noqa: PLC0415  (lazy: vnstock absent in the gate venv)
    return Listing().symbols_by_industries()


def build_vn_sector_map(df: pd.DataFrame, level: str = "industry_name") -> dict[str, str]:
    """``{ticker: sector}`` from a symbols-by-industries frame. Blank sectors are skipped (unmapped)."""
    if "symbol" not in df.columns or level not in df.columns:
        raise ValueError(f"frame missing symbol/{level}; has {list(df.columns)}")
    out: dict[str, str] = {}
    for sym, sec in zip(df["symbol"], df[level]):
        sec = "" if pd.isna(sec) else str(sec).strip()
        tk = str(sym).strip().upper()
        if tk and sec:
            out[tk] = sec
    return out


def write_sector_csv(sector_map: dict[str, str], out_path: str | Path, source: str,
                     fetched_date: str) -> Path:
    """Write canonical ``ticker,sector,source_url,fetched_date`` (sorted) with provenance."""
    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "sector", "source_url", "fetched_date"])
        for tk in sorted(sector_map):
            w.writerow([tk, sector_map[tk], source, fetched_date])
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched-date", required=True, help="fixed provenance date string; NOT from the clock")
    ap.add_argument("--from-raw", default=None, help="path to an already-fetched symbols_by_industries CSV")
    ap.add_argument("--live", action="store_true", help="hit vnstock (one bulk call) instead of --from-raw")
    ap.add_argument("--level", default="industry_name", choices=["industry_name", "industry_code"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "vn_sectors.csv"))
    a = ap.parse_args()
    if a.live:
        df = fetch_industries_df()  # pragma: no cover (network path)
    elif a.from_raw:
        df = pd.read_csv(a.from_raw)
    else:
        raise SystemExit("provide --from-raw PATH or --live")
    sm = build_vn_sector_map(df, level=a.level)
    out = write_sector_csv(sm, a.out, SOURCE, a.fetched_date)
    print(f"wrote {len(sm)} tickers / {len(set(sm.values()))} ICB sectors -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
