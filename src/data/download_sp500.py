"""Download S&P 500 daily OHLCV from Yahoo Finance (yfinance) into the project's raw schema.

Constituent universe: the maintained datahub CSV
``https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv``
(503 current members; NOTE: current-membership only -> SURVIVORSHIP BIAS, delisted names absent).
Each ticker is written to ``<out_dir>/<TICKER>_ohlcv.csv`` with columns
``date,open,high,low,close,volume`` (plain ``YYYY-MM-DD``), mirroring the VN raw files so the same
``process_parkinson_pipeline`` can consume it. Prices are split/dividend back-adjusted
(``auto_adjust=True``) so the Parkinson high/low ratio is consistent across corporate actions.

Yahoo symbols use ``-`` where the index list uses ``.`` (BRK.B -> BRK-B); the file is named with the
sanitized dash form. Resumable: ``skip_existing`` skips tickers already downloaded.

LICENSING: Yahoo Finance data is for personal use; redistribution is restricted by Yahoo's ToS, so the
raw CSVs should NOT be committed to a public repo (gitignore them) — the same constraint that applied
to the VN Yahoo crawl. Use for local research; publish only derived features if needed.

Run:  python -m src.data.download_sp500 [out_dir] [--limit N] [--sleep S]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
OUT_COLS = ["date", "open", "high", "low", "close", "volume"]


def get_constituents(url: str = CONSTITUENTS_URL) -> list[str]:
    """Return the current S&P 500 ticker symbols (Yahoo dash form)."""
    df = pd.read_csv(url)
    return [str(s).strip().replace(".", "-") for s in df["Symbol"].tolist()]


def normalize(hist: pd.DataFrame) -> pd.DataFrame:
    """yfinance ``history`` frame -> the raw ``date,open,high,low,close,volume`` schema."""
    if hist.empty:
        raise ValueError("empty history frame")
    out = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out.columns = OUT_COLS
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.strftime("%Y-%m-%d")
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out


def fetch_ticker(symbol: str) -> pd.DataFrame:
    """Download one ticker's full daily history (split/dividend adjusted)."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
    return normalize(hist)


def download_universe(tickers: list[str], out_dir: Path, sleep: float = 0.8,
                      skip_existing: bool = True) -> dict[str, dict]:
    """Download each ticker into ``out_dir/<TICKER>_ohlcv.csv``; return per-ticker outcome."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for i, sym in enumerate(tickers, 1):
        path = out_dir / f"{sym}_ohlcv.csv"
        if skip_existing and path.exists():
            results[sym] = {"ok": True, "skipped": True}
            continue
        try:
            df = fetch_ticker(sym)
            df.to_csv(path, index=False)
            results[sym] = {"ok": True, "rows": len(df)}
            print(f"[{i}/{len(tickers)}] {sym}: OK {len(df)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            results[sym] = {"ok": False, "error": str(exc)}
            print(f"[{i}/{len(tickers)}] {sym}: FAIL {exc}", flush=True)
        time.sleep(sleep)
    return results


def main(out_dir: str, limit: int | None, sleep: float) -> int:
    tickers = get_constituents()
    if limit:
        tickers = tickers[:limit]
    print(f"[download_sp500] {len(tickers)} tickers -> {out_dir}", flush=True)
    res = download_universe(tickers, Path(out_dir), sleep=sleep)
    ok = sum(1 for r in res.values() if r["ok"])
    failed = [t for t, r in res.items() if not r["ok"]]
    print(f"[download_sp500] done: {ok}/{len(tickers)} OK" + (f"; failed: {failed}" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser(description="Download S&P 500 daily OHLCV via yfinance.")
    p.add_argument("out_dir", nargs="?", default="data/raw/prices/sp500")
    p.add_argument("--limit", type=int, default=None, help="only the first N tickers (smoke)")
    p.add_argument("--sleep", type=float, default=0.8, help="seconds between tickers")
    a = p.parse_args()
    raise SystemExit(main(a.out_dir, a.limit, a.sleep))
