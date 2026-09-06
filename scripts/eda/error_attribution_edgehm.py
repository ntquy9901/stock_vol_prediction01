"""Error attribution: from a market's walk-forward test set, find WHICH tickers and WHICH dates carry the
worst forecast loss, and link them to data issues (dirty-marker days, variance spikes, near-floor days).

Recomputes the HAR-X out-of-sample predictions per (ticker, date) with the same leakage-free walk-forward as
the paper driver (HAR-X is OLS: fast, deterministic, no GPU). HAR-X loss tracks data hardness, so the worst
ticker-days it flags are where the target is hardest to forecast regardless of model. Each worst entry is
annotated with the enriched dirty flags (zero_range / zero_volume) and whether the target is a spike or a
near-floor value. Writes a Markdown report to docs/reports/.

Run: .venv_gpu_encode/Scripts/python.exe scripts/eda/error_attribution_edgehm.py --market vn100 --horizon 1
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import math  # noqa: E402
import pipeline_config as pc  # noqa: E402
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402


def qlike_loss(y, f):
    """Per-element QLIKE loss y/f - log(y/f) - 1 (both already floored positive); >=0, 0 at y==f."""
    r = y / f
    return r - np.log(r) - 1.0


def per_ticker_stats(entries):
    """entries: list of (ticker, date, y, pred, qlike, sqerr). Return per-ticker aggregates sorted worst-first
    by mean QLIKE."""
    by = {}
    for tk, _d, _y, _p, ql, se in entries:
        s = by.setdefault(tk, {"ticker": tk, "n": 0, "qlike_sum": 0.0, "sqerr_sum": 0.0})
        s["n"] += 1; s["qlike_sum"] += ql; s["sqerr_sum"] += se
    for s in by.values():
        s["qlike_mean"] = s["qlike_sum"] / s["n"]
        s["sqerr_mean"] = s["sqerr_sum"] / s["n"]
    return sorted(by.values(), key=lambda s: s["qlike_mean"], reverse=True)


def per_date_stats(entries):
    """Per forecast date: mean QLIKE across tickers + count. Sorted worst-first."""
    by = {}
    for _tk, d, _y, _p, ql, _se in entries:
        s = by.setdefault(d, {"date": d, "n": 0, "qlike_sum": 0.0})
        s["n"] += 1; s["qlike_sum"] += ql
    for s in by.values():
        s["qlike_mean"] = s["qlike_sum"] / s["n"]
    return sorted(by.values(), key=lambda s: s["qlike_mean"], reverse=True)


def worst_entries(entries, k, key_idx):
    """Top-k entries by the tuple field at key_idx (4=qlike, 5=sqerr)."""
    return sorted(entries, key=lambda e: e[key_idx], reverse=True)[:k]


def classify_target(y, ticker_median, floor):
    """Label a target: 'near-floor' (<=2x floor), 'spike' (>=5x the ticker's median), else 'normal'."""
    if y <= 2.0 * floor:
        return "near-floor"
    if ticker_median > 0 and y >= 5.0 * ticker_median:
        return "spike"
    return "normal"


def _flag_lookup(market):  # pragma: no cover - file I/O
    """Map (ticker, 'YYYY-MM-DD') -> set of dirty flags present in the enriched file."""
    out = {}
    for f in _glob.glob(enriched_glob(market)):
        if "rejection" in f.lower():
            continue
        tk = Path(f).stem
        df = pd.read_csv(f)
        if "date" not in df.columns:
            continue
        cols = [c for c in ("zero_range_flag", "zero_volume_flag", "dirty_flag") if c in df.columns]
        for _, row in df.iterrows():
            fl = {c.replace("_flag", "") for c in cols if bool(row.get(c))}
            if fl:
                out[(tk, str(row["date"])[:10])] = fl
    return out


def _collect_entries(market, horizon, lookback):  # pragma: no cover - panel build + OLS glue (heavy, smoke-tested)
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon)
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    cfg = training_config()
    fl = cfg.qlike_floor
    tick_meds = {}
    entries = []
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        _har, harx = _har_ols_preds(D, fl, nfloor)
        pred = np.maximum(harx["te"], fl); y = np.maximum(D.y_te, fl)
        tm = D.tmask_te.astype(bool)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if not tm[i, j]:
                    continue
                tk = panel.tickers[j]; d = str(D.d_te[i])[:10]
                yy = float(y[i, j]); pp = float(pred[i, j])
                ql = float(qlike_loss(yy, pp)); se = (yy - pp) ** 2
                entries.append((tk, d, yy, pp, ql, se))
                tick_meds.setdefault(tk, []).append(yy)
    tick_meds = {tk: float(np.median(v)) for tk, v in tick_meds.items()}
    return entries, tick_meds, fl


def _report(market, horizon, entries, tick_meds, floor, flags):  # pragma: no cover - formatting/I/O
    tks = per_ticker_stats(entries)
    dts = per_date_stats(entries)
    worst_ql = worst_entries(entries, 30, 4)
    lines = [f"# Error attribution (HAR-X OOS) — {market} h{horizon}", "",
             f"Test ticker-days: {len(entries):,}. QLIKE floor: {floor:.2e}. "
             "HAR-X loss proxies data hardness (model-agnostic).", "",
             "## Worst 15 tickers by mean QLIKE", "",
             "| ticker | mean QLIKE | mean sqerr | n | dirty-day share |", "|---|---|---|---|---|"]
    for s in tks[:15]:
        dd = sum(1 for e in entries if e[0] == s["ticker"] and (e[0], e[1]) in flags)
        lines.append(f"| {s['ticker']} | {s['qlike_mean']:.3f} | {s['sqerr_mean']:.2e} | {s['n']} | "
                     f"{dd}/{s['n']} ({100*dd/s['n']:.0f}%) |")
    lines += ["", "## Worst 30 ticker-days by QLIKE (with data annotation)", "",
              "| ticker | date | target | QLIKE | kind | dirty flags |", "|---|---|---|---|---|---|"]
    for tk, d, y, _p, ql, _se in worst_ql:
        kind = classify_target(y, tick_meds.get(tk, 0.0), floor)
        fl = ",".join(sorted(flags.get((tk, d), set()))) or "-"
        lines.append(f"| {tk} | {d} | {y:.2e} | {ql:.2f} | {kind} | {fl} |")
    lines += ["", "## Worst 15 forecast dates by mean QLIKE (market-wide)", "",
              "| date | mean QLIKE | tickers |", "|---|---|---|"]
    for s in dts[:15]:
        lines.append(f"| {s['date']} | {s['qlike_mean']:.3f} | {s['n']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn100")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--lookback", type=int, default=22)   # match the trained run (driver --lookback default 22)
    a = ap.parse_args(argv)
    entries, tick_meds, floor = _collect_entries(a.market, a.horizon, a.lookback)
    flags = _flag_lookup(a.market)
    out = REPO / "docs" / "reports" / f"2026-09-06_error_attribution_{a.market}_h{a.horizon}.md"
    out.write_text(_report(a.market, a.horizon, entries, tick_meds, floor, flags), encoding="utf-8")
    print(f"[attr] {a.market} h{a.horizon}: {len(entries):,} ticker-days -> wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
