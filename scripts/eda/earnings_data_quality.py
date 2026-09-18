"""Data-quality check for the crawled earnings-date parquets (regression guard codifying the 2026-09-18
provenance/cleanliness audit).

Earnings dates enter the model on a SEPARATE path from prices: prices go through the ETL pipeline into
``data/processed_enriched`` (Pandera-gated), whereas earnings stay as standalone parquets under
``results/gamma_gbm`` and are joined at runtime by ``full_matrix.panel``. So they never saw the price
data-quality gate; this module fills that gap. ``earnings_quality`` is a pure, unit-tested summary;
``main`` runs it over both markets' parquets and writes a JSON report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA = ["ticker", "earnings_date"]
MIN_HISTORY = 3                      # a ticker needs > MIN_HISTORY events for the cadence feature to form
QUARTERLY_LO, QUARTERLY_HI = 60, 130  # documented "clean quarterly" inter-release gap band (days)


def earnings_quality(df):
    """Return a data-quality summary for one earnings table (columns ticker, earnings_date).

    Hard-invariant fields (must be clean): ``schema_ok``, ``n_null``, ``n_dup_rows``, ``n_dup_pairs``.
    Descriptive fields: row/ticker counts, date range, tickers with too little history for the cadence
    feature, and the fraction of inter-release gaps inside the quarterly band (a market-regularity signal,
    not a pass/fail: HOSE is legitimately irregular because it mixes filing types)."""
    d = df.copy()
    d["earnings_date"] = pd.to_datetime(d["earnings_date"])
    valid = d.dropna(subset=["earnings_date"])
    gaps = []
    counts = {}
    for tk, g in valid.groupby("ticker"):
        dd = np.sort(g["earnings_date"].to_numpy())
        counts[tk] = len(dd)
        if len(dd) >= 2:
            gaps.extend(np.diff(dd).astype("timedelta64[D]").astype(int).tolist())
    gaps = np.asarray(gaps, dtype=float)
    counts = np.asarray(list(counts.values()))
    return {
        "schema_ok": list(df.columns) == SCHEMA,
        "n_rows": int(len(d)),
        "n_tickers": int(d["ticker"].nunique()),
        "n_null": int(d["earnings_date"].isna().sum()),
        "n_dup_rows": int(d.duplicated().sum()),
        "n_dup_pairs": int(d.duplicated(subset=SCHEMA).sum()),
        "date_min": None if valid.empty else str(valid["earnings_date"].min().date()),
        "date_max": None if valid.empty else str(valid["earnings_date"].max().date()),
        "tickers_lt_min_history": int((counts <= MIN_HISTORY).sum()) if len(counts) else 0,
        "pct_gap_quarterly": float(100.0 * np.mean((gaps >= QUARTERLY_LO) & (gaps <= QUARTERLY_HI)))
        if len(gaps) else 0.0,
        "n_gaps": int(len(gaps)),
    }


def _load(market):  # pragma: no cover - thin parquet loader
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    fn = "hose_earnings_combined.parquet" if market == "hose" else "sp500_earnings.parquet"
    return pd.read_parquet(repo / "results" / "gamma_gbm" / fn)


def main():  # pragma: no cover - data-driven driver
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    report = {m: earnings_quality(_load(m)) for m in ("hose", "sp500")}
    for m, r in report.items():
        print(f"{m}: {r}", flush=True)
    (repo / "results" / "gamma_gbm" / "earnings_data_quality.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
