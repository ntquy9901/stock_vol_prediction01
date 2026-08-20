"""Crawl daily OHLCV from vnstock (a Vietnamese market data source) into raw ``*_ohlcv.csv``.

Motivation: the existing Yahoo-Finance VN100 crawl reported ``volume == 0`` on days where the
price actually moved (feed glitches, later interpolated by ``repair_volume.py``). vnstock exposes
real reported volume from Vietnamese backends (VCI primary, KBS fallback), so the volume feature
becomes trustworthy for cross-stock experiments.

Output schema matches the Yahoo raw files exactly: columns ``date,open,high,low,close,volume``;
``date`` as plain ``YYYY-MM-DD``; OHLC in thousands-VND (VCI/KBS already return adjusted prices in
thousands-VND, verified against Yahoo FPT/VNM — NO division needed); ``volume`` = raw share count.

Anti-IP-lock: crawl SEQUENTIALLY, sleep between tickers, retry with exponential backoff on transient
errors, and rotate the source (VCI <-> KBS) when one backend fails or rate-limits.

CLI:
  python -m src.data.crawl_vnstock data/raw/prices/vn100 data/raw/prices/vn100_vnstock
  # first arg = dir whose *_ohlcv.csv stems define the ticker universe; second = output dir
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

RAW_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_SOURCES = ("VCI", "KBS")
DEFAULT_START = "2009-01-01"
DEFAULT_END = "2026-08-14"


def _default_client_factory(symbol: str, source: str):
    """Build a vnstock stock client for ``symbol`` on ``source`` (imported lazily)."""

    from vnstock import Vnstock

    return Vnstock().stock(symbol=symbol, source=source)


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a vnstock ``quote.history`` frame to the raw ``*_ohlcv.csv`` schema.

    vnstock returns columns ``time,open,high,low,close,volume`` with a datetime ``time`` and prices
    already in thousands-VND. This renames ``time`` -> ``date`` (plain ``YYYY-MM-DD``), drops any
    duplicate/unsorted dates, and coerces ``volume`` to int.
    """

    if df is None or len(df) == 0:
        raise ValueError("empty history frame")
    out = df.rename(columns={"time": "date"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"history missing columns: {missing}")
    out = out[RAW_COLUMNS]
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out = out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return out


def fetch_ticker(
    symbol: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    max_retries: int = 3,
    base_sleep: float = 2.0,
    client_factory: Callable[[str, str], object] = _default_client_factory,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, str]:
    """Fetch one ticker's daily OHLCV, retrying with backoff and rotating ``sources`` on failure.

    Returns ``(normalized_df, source_used)``. Raises ``RuntimeError`` if every source fails.
    """

    last_err: Exception | None = None
    for source in sources:
        for attempt in range(max_retries):
            try:
                client = client_factory(symbol, source)
                df = client.quote.history(start=start, end=end, interval="1D")
                return normalize_history(df), source
            except Exception as err:  # transient/network/rate-limit -> backoff then rotate source
                last_err = err
                if attempt < max_retries - 1:
                    sleeper(base_sleep * (2 ** attempt))
    raise RuntimeError(f"{symbol}: all sources {sources} failed: {last_err!r}")


def read_universe(ticker_dir: str | Path) -> list[str]:
    """Return sorted ticker stems from ``*_ohlcv.csv`` files under ``ticker_dir``."""

    return sorted(p.stem.replace("_ohlcv", "") for p in Path(ticker_dir).glob("*_ohlcv.csv"))


def crawl_universe(
    tickers: list[str],
    out_dir: str | Path,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    polite_sleep: float = 1.5,
    skip_existing: bool = True,
    logger: Callable[[str], None] = print,
    **fetch_kwargs,
) -> dict[str, dict]:
    """Crawl every ticker sequentially into ``out_dir/<TICKER>_ohlcv.csv``; return per-ticker outcome.

    With ``skip_existing`` (default), tickers whose CSV already exists are skipped — this makes the
    crawl resumable after a rate-limit interruption: just re-run and it continues where it stopped.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers, 1):
        if skip_existing and (out_path / f"{ticker}_ohlcv.csv").exists():
            results[ticker] = {"ok": True, "skipped": True}
            continue
        try:
            df, source = fetch_ticker(ticker, start, end, **fetch_kwargs)
            df.to_csv(out_path / f"{ticker}_ohlcv.csv", index=False)
            results[ticker] = {
                "ok": True,
                "source": source,
                "rows": len(df),
                "first": df["date"].iloc[0],
                "last": df["date"].iloc[-1],
            }
            logger(f"[{i}/{len(tickers)}] {ticker}: OK {len(df)} rows via {source} "
                   f"({df['date'].iloc[0]}..{df['date'].iloc[-1]})")
        except Exception as err:
            results[ticker] = {"ok": False, "error": repr(err)[:300]}
            logger(f"[{i}/{len(tickers)}] {ticker}: FAILED {err!r}")
        time.sleep(polite_sleep)  # polite delay between tickers to avoid IP lock
    return results


def main(ticker_dir: str, out_dir: str) -> int:
    tickers = read_universe(ticker_dir)
    if not tickers:
        print(f"[crawl_vnstock] no *_ohlcv.csv under {ticker_dir}")
        return 1
    print(f"[crawl_vnstock] crawling {len(tickers)} tickers -> {out_dir}")
    # vnstock Guest tier caps at 20 requests/minute; ~6s between tickers keeps the rolling rate
    # safely under the cap. Combined with skip_existing, re-running resumes after any interruption.
    results = crawl_universe(tickers, out_dir, polite_sleep=6.0)
    ok = sum(1 for r in results.values() if r["ok"])
    failed = [t for t, r in results.items() if not r["ok"]]
    print(f"[crawl_vnstock] done: {ok}/{len(tickers)} OK" + (f"; failed: {failed}" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    args = sys.argv[1:]
    src_dir = args[0] if args else "data/raw/prices/vn100_vnstock"
    dst_dir = args[1] if len(args) > 1 else "data/raw/prices/vn100_vnstock"
    raise SystemExit(main(src_dir, dst_dir))
