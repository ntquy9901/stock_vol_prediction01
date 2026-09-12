"""Build a full-history daily VNINDEX series by stitching two independent sources, with a documented
cross-check, for the complex-network market-index volatility experiment (Complexity-2026 replication).

Sources
-------
1. Zenodo DOI 10.5281/zenodo.21873555 "VN-Index Daily Dataset for Volatility Forecasting (2000-2024)"
   (CC BY 4.0), file ``Dữ liệu Lịch sử VN Index.csv``: daily OHLC + volume 2000-07-31 to 2024-12-16
   (Investing.com-style export; columns in Vietnamese, DD/MM/YYYY, volume in "K" display units).
2. vnstock VCI feed: daily OHLCV 2024-12-17 onward (the community feed caps 1-day history at ~8 years,
   so it supplies only the recent tail; here it fills 2024-12-17 to the last available session).

The two sources overlap on 2018-09 to 2024-12; ``cross_check`` compares Close on the overlap and records
the discrepancy (mean/max abs percent) as provenance. OHLC index points are on the same scale across
sources (verified); the volume DISPLAY UNIT differs (Zenodo "K" vs vnstock raw share count), so volume
is kept per-source with a ``vol_source`` column and the avg-log-volume target must not stitch across the
boundary (the headline volatility target is Close-based and unaffected).

Output
------
``data/raw/prices/_market_index/vnindex.csv``      date,open,high,low,close,volume,vol_source  (ascending)
``data/raw/prices/_market_index/vnindex_provenance.json``  sources, ranges, cross-check stats, license

Run (vnstock venv, UTF-8): PYTHONUTF8=1 .venv_vnstock/Scripts/python scripts/etl_vn_index/build_vnindex.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IDX_DIR = REPO / "data" / "raw" / "prices" / "_market_index"
ZENODO_CSV = IDX_DIR / "_sources" / "zenodo_vnindex_2000_2024.csv"
OUT_CSV = IDX_DIR / "vnindex.csv"
OUT_PROV = IDX_DIR / "vnindex_provenance.json"
STITCH_BOUNDARY = pd.Timestamp("2024-12-17")  # Zenodo <= 2024-12-16; vnstock supplies from here on
OHLC = ["open", "high", "low", "close"]


def parse_zenodo(path: Path) -> pd.DataFrame:
    """Parse the Zenodo Investing-style CSV into date,open,high,low,close,volume (ascending)."""
    z = pd.read_csv(path)
    z = z.rename(columns={"Ngày": "date", "Lần cuối": "close", "Mở": "open",
                          "Cao": "high", "Thấp": "low", "KL": "volume"})
    z["date"] = pd.to_datetime(z["date"], dayfirst=True)
    for c in OHLC:
        z[c] = pd.to_numeric(z[c].astype(str).str.replace(",", ""), errors="coerce")
    z["volume"] = z["volume"].apply(_parse_k)
    z = z[["date", *OHLC, "volume"]].sort_values("date").reset_index(drop=True)
    return z


def _parse_k(x) -> float:
    """Parse an Investing 'KL' cell like '538.93K' / '1.2M' / '' into a float share count."""
    s = str(x).strip().replace(",", "")
    if s in ("", "nan", "None"):
        return np.nan
    mult = 1.0
    if s[-1] in "KMB":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def fetch_vnstock(start: str, end: str) -> pd.DataFrame:
    """Fetch daily VNINDEX OHLCV from the vnstock VCI feed (imported lazily; needs the vnstock venv)."""
    from vnstock.api.quote import Quote

    v = Quote(symbol="VNINDEX", source="VCI").history(start=start, end=end, interval="1D")
    v = v.rename(columns={"time": "date"})
    v["date"] = pd.to_datetime(v["date"])
    return v[["date", *OHLC, "volume"]].sort_values("date").reset_index(drop=True)


def cross_check(zen: pd.DataFrame, vns: pd.DataFrame) -> dict:
    """Compare Close on the overlapping dates; return discrepancy stats for provenance."""
    m = zen[["date", "close"]].merge(vns[["date", "close"]], on="date", suffixes=("_zen", "_vn"))
    pct = (m["close_zen"] - m["close_vn"]).abs() / m["close_vn"] * 100.0
    worst = m.loc[pct.idxmax()]
    return {"overlap_n": int(len(m)),
            "overlap_start": str(m["date"].min().date()), "overlap_end": str(m["date"].max().date()),
            "close_pct_diff_mean": float(pct.mean()), "close_pct_diff_max": float(pct.max()),
            "n_days_gt_0.1pct": int((pct > 0.1).sum()), "n_days_gt_0.5pct": int((pct > 0.5).sum()),
            "worst_day": str(worst["date"].date()),
            "worst_zenodo_close": float(worst["close_zen"]), "worst_vnstock_close": float(worst["close_vn"])}


def stitch(zen: pd.DataFrame, vns: pd.DataFrame) -> pd.DataFrame:
    """Zenodo for dates < STITCH_BOUNDARY, vnstock from STITCH_BOUNDARY on; tag the volume source."""
    a = zen[zen["date"] < STITCH_BOUNDARY].copy(); a["vol_source"] = "zenodo_K_units"
    b = vns[vns["date"] >= STITCH_BOUNDARY].copy(); b["vol_source"] = "vnstock_raw_shares"
    out = pd.concat([a, b], ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def validate(df: pd.DataFrame) -> None:
    """Fail loud on structural problems: dup/unsorted dates, non-positive or NaN close, HL geometry."""
    assert df["date"].is_monotonic_increasing, "dates not sorted ascending"
    assert not df["date"].duplicated().any(), "duplicate dates"
    assert df["close"].notna().all() and (df["close"] > 0).all(), "close has NaN or non-positive"
    ok_hl = (df["high"] >= df["low"]).fillna(True)
    assert ok_hl.all(), f"{(~ok_hl).sum()} rows with high < low"


def main() -> None:
    zen = parse_zenodo(ZENODO_CSV)
    vns = fetch_vnstock("2015-01-01", "2026-12-31")
    xc = cross_check(zen, vns)
    out = stitch(zen, vns)
    validate(out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    prov = {
        "series": "VNINDEX", "built_from": "two stitched sources with overlap cross-check",
        "output_rows": int(len(out)), "date_start": str(out["date"].min().date()),
        "date_end": str(out["date"].max().date()), "stitch_boundary": str(STITCH_BOUNDARY.date()),
        "sources": [
            {"name": "Zenodo 10.5281/zenodo.21873555 (VN-Index Daily Dataset for Volatility Forecasting 2000-2024)",
             "license": "CC BY 4.0", "covers": "2000-07-31..2024-12-16", "role": "historical OHLC + volume"},
            {"name": "vnstock VCI feed", "license": "vnstock community", "covers": "2024-12-17..latest",
             "role": "recent supplement (community feed ~8-year cap)"},
        ],
        "cross_check_close_zenodo_vs_vnstock": xc,
        "volume_note": "OHLC index points share one scale across sources (verified); volume DISPLAY UNIT "
                       "differs (Zenodo 'K' vs vnstock raw shares), tagged in vol_source. The avg-log-volume "
                       "target must not stitch across the boundary; the headline volatility target is "
                       "Close-based and unaffected.",
    }
    OUT_PROV.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_CSV} rows={len(out)} {prov['date_start']}..{prov['date_end']}")
    print(f"cross-check close: mean {xc['close_pct_diff_mean']:.5f}% max {xc['close_pct_diff_max']:.4f}% "
          f"(overlap {xc['overlap_n']} days, {xc['n_days_gt_0.5pct']} days > 0.5%)")
    print(f"wrote {OUT_PROV}")


if __name__ == "__main__":  # pragma: no cover
    main()
