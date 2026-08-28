"""Crawl full daily OHLCV history for EVERY HOSE and HNX equity from vnstock.

Universe comes from vnstock's ``Listing().symbols_by_exchange()`` filtered to ``type == 'stock'`` on
exchanges ``HOSE`` and ``HNX`` (indices/derivatives/bonds/CW/funds excluded). Each exchange is written
to its OWN directory with its OWN log + manifest:

  data/raw/prices/hose_vnstock/<TICKER>_ohlcv.csv   (+ _crawl.log, _manifest.csv)
  data/raw/prices/hnx_vnstock/<TICKER>_ohlcv.csv    (+ _crawl.log, _manifest.csv)

CSV schema matches ``vn100_vnstock`` exactly: ``date,open,high,low,close,volume``; ``date`` as plain
``YYYY-MM-DD``. Fetch/retry/source-rotation/normalisation are reused from ``src.data.crawl_vnstock``.

Robustness: resumable (skips any existing non-empty CSV), polite sleep between tickers, exponential
backoff + source rotation per ticker, and per-ticker checkpoint (CSV written immediately). A single
bad ticker is logged and skipped; it never aborts the crawl.

Run (from repo root, with the vnstock venv):
  .venv_vnstock/Scripts/python.exe scripts/crawl_hose_hnx.py            # both exchanges
  .venv_vnstock/Scripts/python.exe scripts/crawl_hose_hnx.py --exchange HOSE
Re-running the same command resumes where it stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# vnstock prints a Unicode deprecation banner on first client build; the default Windows cp1252
# console raises UnicodeEncodeError on it and that bubbles up as a fetch failure. Force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.crawl_vnstock import RAW_COLUMNS, crawl_universe  # noqa: E402

START = "2000-01-01"
END = "2026-08-22"
POLITE_SLEEP = 7.0  # s between tickers; each fetch ~2 API reqs, guest cap=20/min -> ~8.5s/ticker keeps
#                     the rolling 60s window under ~14 req/min (measured: sleep=4 tripped the cap at #10)
OUT_DIRS = {
    "HOSE": REPO_ROOT / "data" / "raw" / "prices" / "hose_vnstock",
    "HNX": REPO_ROOT / "data" / "raw" / "prices" / "hnx_vnstock",
}


def get_universe() -> dict[str, list[str]]:
    """Return ``{'HOSE': [...], 'HNX': [...]}`` equity tickers from vnstock Listing (deduped, sorted)."""

    from vnstock import Listing

    df = Listing().symbols_by_exchange()
    df = df[df["type"] == "stock"]
    out: dict[str, list[str]] = {}
    for exch in ("HOSE", "HNX"):
        syms = sorted(set(df.loc[df["exchange"] == exch, "symbol"].astype(str)))
        out[exch] = syms
    return out


def build_manifest(exch: str, tickers: list[str], out_dir: Path) -> pd.DataFrame:
    """Read every ticker CSV and produce a manifest row (with basic sanity) for each."""

    rows = []
    for t in tickers:
        fp = out_dir / f"{t}_ohlcv.csv"
        n_rows = 0
        first = last = ""
        status = "missing"
        if fp.exists() and fp.stat().st_size > 0:
            try:
                d = pd.read_csv(fp)
                if list(d.columns) != RAW_COLUMNS:
                    status = "bad_columns"
                elif len(d) == 0:
                    status = "empty"
                else:
                    parsed = pd.to_datetime(d["date"], errors="coerce")
                    if parsed.isna().any():
                        status = "bad_dates"
                    else:
                        n_rows = len(d)
                        first = str(d["date"].iloc[0])
                        last = str(d["date"].iloc[-1])
                        status = "ok"
            except Exception as err:  # unreadable file
                status = f"read_error:{type(err).__name__}"
        rows.append({
            "ticker": t, "exchange": exch, "n_rows": n_rows,
            "first_date": first, "last_date": last, "status": status,
        })
    return pd.DataFrame(rows, columns=["ticker", "exchange", "n_rows", "first_date", "last_date", "status"])


def run_exchange(exch: str, tickers: list[str]) -> None:
    out_dir = OUT_DIRS[exch]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = out_dir / "_crawl.log"

    with log_fp.open("a", encoding="utf-8") as logf:
        def log(msg: str) -> None:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{stamp}] {msg}"
            logf.write(line + "\n")
            logf.flush()
            print(line, flush=True)

        log(f"=== {exch}: crawling {len(tickers)} tickers -> {out_dir} (start={START} end={END}) ===")
        results = crawl_universe(
            tickers, out_dir, start=START, end=END,
            polite_sleep=POLITE_SLEEP, skip_existing=True, logger=log,
        )
        ok = sum(1 for r in results.values() if r["ok"])
        failed = {t: r.get("error", "") for t, r in results.items() if not r["ok"]}
        log(f"=== {exch}: fetch loop done: {ok}/{len(tickers)} ok; {len(failed)} failed ===")
        for t, e in failed.items():
            log(f"    FAIL {t}: {e}")

        manifest = build_manifest(exch, tickers, out_dir)
        manifest.to_csv(out_dir / "_manifest.csv", index=False)
        n_ok = int((manifest["status"] == "ok").sum())
        log(f"=== {exch}: manifest written; {n_ok}/{len(tickers)} files sane -> {out_dir/'_manifest.csv'} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", choices=["HOSE", "HNX", "both"], default="both")
    args = ap.parse_args()

    universe = get_universe()
    print(f"[universe] HOSE={len(universe['HOSE'])} HNX={len(universe['HNX'])}", flush=True)

    exchanges = ["HOSE", "HNX"] if args.exchange == "both" else [args.exchange]
    for exch in exchanges:
        run_exchange(exch, universe[exch])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
