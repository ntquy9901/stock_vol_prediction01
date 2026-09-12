"""Build the S&P 500 index (^GSPC) daily series as the US market-index target for the complex-network
volatility experiment, with a FRED cross-check.

Source
------
Primary: Yahoo Finance ``^GSPC`` (via yfinance), daily OHLC + Adj Close + Volume, 2000-01-03 onward.
A single source covers the whole range, so no stitching is needed (unlike VNINDEX).
Cross-check: FRED series ``SP500`` (free CSV, Close only, about the last 10 years) compared on the overlap.

Note on index volume: the volume of a price index is not comparable to a stock or ETF volume; the
experiment uses OHLC (Close for the return-volatility target). Volume is stored for completeness only.

Output
------
``data/raw/prices/_market_index/gspc.csv``   date,open,high,low,close,adj_close,volume  (ascending)
``data/raw/prices/_market_index/gspc_provenance.json``  source, range, FRED cross-check stats

Run (GPU venv has yfinance): .venv_gpu_encode/Scripts/python scripts/etl_vn_index/build_gspc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_vnindex as BV  # noqa: E402  (reuse validate + cross_check)

IDX_DIR = REPO / "data" / "raw" / "prices" / "_market_index"
YAHOO_CSV = IDX_DIR / "_sources" / "yahoo_gspc_2000_2026.csv"
FRED_CSV = IDX_DIR / "_sources" / "fred_sp500.csv"
OUT_CSV = IDX_DIR / "gspc.csv"
OUT_PROV = IDX_DIR / "gspc_provenance.json"
OHLC = ["open", "high", "low", "close"]


def parse_yahoo(path: Path) -> pd.DataFrame:
    """Parse the yfinance ^GSPC export into date,open,high,low,close,adj_close,volume (ascending)."""
    g = pd.read_csv(path)
    g = g.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low",
                          "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    g["date"] = pd.to_datetime(g["date"])
    cols = ["date", *OHLC, "adj_close", "volume"]
    return g[cols].sort_values("date").reset_index(drop=True)


def parse_fred(path: Path) -> pd.DataFrame:
    """Parse the FRED SP500 CSV (observation_date, SP500) into date,close (ascending, non-null)."""
    f = pd.read_csv(path)
    f.columns = ["date", "close"]
    f["date"] = pd.to_datetime(f["date"])
    f["close"] = pd.to_numeric(f["close"], errors="coerce")
    return f.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def main() -> None:
    g = parse_yahoo(YAHOO_CSV)
    BV.validate(g)
    fred = parse_fred(FRED_CSV)
    xc = BV.cross_check(g, fred)   # both have 'close'; compares on the overlap
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT_CSV, index=False)
    prov = {
        "series": "SP500 (^GSPC)", "built_from": "single source (Yahoo Finance ^GSPC) + FRED cross-check",
        "output_rows": int(len(g)), "date_start": str(g["date"].min().date()),
        "date_end": str(g["date"].max().date()),
        "sources": [
            {"name": "Yahoo Finance ^GSPC (yfinance)", "covers": f"{g['date'].min().date()}..{g['date'].max().date()}",
             "role": "primary OHLC + Adj Close + volume", "license": "Yahoo terms (research use)"},
            {"name": "FRED SP500", "covers": f"{fred['date'].min().date()}..{fred['date'].max().date()}",
             "role": "Close cross-check (~10 years)", "license": "public domain (US gov)"},
        ],
        "cross_check_close_yahoo_vs_fred": xc,
        "volume_note": "Index volume is not comparable to a stock/ETF volume; the experiment uses OHLC "
                       "(Close for the return-volatility target). Volume stored for completeness only.",
    }
    OUT_PROV.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_CSV} rows={len(g)} {prov['date_start']}..{prov['date_end']}")
    print(f"cross-check close vs FRED: mean {xc['close_pct_diff_mean']:.5f}% max {xc['close_pct_diff_max']:.4f}% "
          f"(overlap {xc['overlap_n']} days, {xc['n_days_gt_0.5pct']} days > 0.5%)")


if __name__ == "__main__":  # pragma: no cover
    main()
