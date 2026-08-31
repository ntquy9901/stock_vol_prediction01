"""CLI for the enriched-processed build (baseline A3).

    python baselines/2026-08-31_enriched_processed/code/cli.py --markets vn30 vn100 --jobs 4

Builds each market to ``data/processed_enriched/<market>/`` and writes the HTML build report.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_HERE))

import enrich  # noqa: E402
import report as report_mod  # noqa: E402


def _map_fn(jobs: int):
    """Return a map function: serial ``map`` for jobs<=1, else a process-pool map across tickers."""
    if jobs <= 1:
        return map

    def _parallel(fn, items):
        items = list(items)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(fn, items))

    return _parallel


def run(markets, out_root=None, jobs: int = 1, html_path=None, limit=None) -> dict:
    """Build the given markets and write the HTML report. Returns ``{market: summary}``."""
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    mfn = _map_fn(jobs)
    reg_dir = enrich.REPO / "data" / "processed"     # VN30 clean-bar regression vs the delivered values
    summaries = {}
    for market in markets:
        rd = reg_dir if market == "vn30" else None
        summaries[market] = enrich.build_market(market, out_root=out_root, map_fn=mfn, limit=limit,
                                                regression_dir=rd)
    if html_path is not None:
        report_mod.build_html_report(summaries, html_path,
                                     regression=summaries.get("vn30", {}).get("regression"))
    return summaries


def main(argv=None) -> int:  # pragma: no cover - entry driver; run()/build_market are unit-tested
    ap = argparse.ArgumentParser(description="ETL-clean + enrich processed data.")
    ap.add_argument("--markets", nargs="+", default=list(enrich.PRICE_DIRS))
    ap.add_argument("--out-root", default=str(enrich.OUT_ROOT))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap tickers per market (smoke)")
    ap.add_argument("--html", default=str(enrich.REPO / "docs" / "reports"
                                          / "2026-08-31_enriched_processed_build.html"))
    a = ap.parse_args(argv)
    summaries = run(a.markets, out_root=a.out_root, jobs=a.jobs, html_path=a.html, limit=a.limit)
    for m, s in summaries.items():
        print(f"[enrich] {m}: {s['n_tickers']} tickers, rows_out={s['rows_out']}, "
              f"dirty_bars={s['n_dirty_bars']}, dropped={s['n_dropped']}", flush=True)
    print(f"[enrich] wrote report {a.html}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
