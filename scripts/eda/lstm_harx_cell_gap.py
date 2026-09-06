"""Decompose the LSTM-minus-HAR-X QLIKE gap from per-ticker-per-day cell logs (`--dump-cells` parquet).

Finding (S&P 500): the LSTM's aggregate QLIKE disadvantage vs HAR-X is a ~1% tail of extreme variance-spike
ticker-days where the LSTM under-forecasts catastrophically; on the other ~99% of cells the LSTM is as good or
better. This module holds the pure decomposition so it is testable; the parquet I/O driver is pragma'd.

Run: .venv_gpu_encode/Scripts/python.exe scripts/eda/lstm_harx_cell_gap.py results/edge_hmatched/cells/cells_sp500_clean_h1.parquet
"""
from __future__ import annotations

import sys

import numpy as np

QLIKE_FLOOR = 1e-8  # config-ok: mirrors the pipeline's training_config().qlike_floor for this read-only analysis


def per_cell_qlike(y, f, floor=QLIKE_FLOOR):
    """Per-cell QLIKE = y/f - log(y/f) - 1 with a shared positivity floor on both y and f."""
    y = np.maximum(np.asarray(y, dtype=float), floor)
    f = np.maximum(np.asarray(f, dtype=float), floor)
    r = y / f
    return r - np.log(r) - 1.0


def gap_decomposition(y, harx_f, lstm_f, floor=QLIKE_FLOOR, top_frac=0.01):
    """Decompose the LSTM-minus-HAR-X QLIKE gap. Returns aggregate QLIKEs, the per-cell gap mean/median, the
    share of the total gap contributed by the worst ``top_frac`` of cells, the LSTM under-forecast rate, and
    the aggregate QLIKEs after dropping those worst cells (a robustness view)."""
    qh = per_cell_qlike(y, harx_f, floor)
    ql = per_cell_qlike(y, lstm_f, floor)
    diff = ql - qh
    n = len(diff)
    k = max(1, int(n * top_frac))
    order = np.argsort(-diff)
    worst, keep = order[:k], order[k:]
    total = diff.sum()
    return {
        "n_cells": int(n),
        "qlike_harx": float(qh.mean()), "qlike_lstm": float(ql.mean()),
        "gap": float(ql.mean() - qh.mean()),
        "gap_mean": float(diff.mean()), "gap_median": float(np.median(diff)),
        "top_frac": top_frac,
        "top_share_of_gap": float(diff[worst].sum() / total) if total != 0 else float("nan"),
        "lstm_underforecast_rate": float(np.mean(np.asarray(lstm_f) < np.asarray(y))),
        "qlike_harx_ex_worst": float(qh[keep].mean()) if len(keep) else float("nan"),
        "qlike_lstm_ex_worst": float(ql[keep].mean()) if len(keep) else float("nan"),
    }


def worst_cells(qlike_vals, n):
    """Indices of the ``n`` highest-QLIKE cells, descending (the ticker-days that hurt a model most)."""
    return list(np.argsort(-np.asarray(qlike_vals, dtype=float))[:max(0, n)])


def top_by_group(keys, values, n):
    """Top-``n`` group keys by summed value, as ``[(key, sum), ...]`` descending. Used to rank the worst
    dates / tickers by their total QLIKE contribution."""
    agg = {}
    for k, v in zip(keys, values):
        agg[k] = agg.get(k, 0.0) + float(v)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:n]


def qlike_excluding_top_y(y, f, exclude_frac, floor=QLIKE_FLOOR):
    """Mean QLIKE after excluding the top ``exclude_frac`` of cells by realized value ``y`` (model-agnostic:
    the SAME cells are dropped for every model). exclude_frac=0 -> all cells."""
    y = np.asarray(y, dtype=float); f = np.asarray(f, dtype=float)
    if exclude_frac <= 0:
        keep = np.ones(len(y), dtype=bool)
    else:
        keep = y < np.quantile(y, 1.0 - exclude_frac)
    return float(per_cell_qlike(y[keep], f[keep], floor).mean())


def anchored_forecast(harx_f, z, clip=0.5):
    """HAR-X-anchored residual forecast: yhat = HAR-X * exp(clip(z, -clip, clip)). z=0 -> exactly HAR-X (no
    collapse); the clip bounds the worst-case correction so a spike day cannot fall far below HAR-X."""
    z = np.clip(np.asarray(z, dtype=float), -clip, clip)
    return np.asarray(harx_f, dtype=float) * np.exp(z)


def assert_unique_cells(df):
    """Fail loud if the cell log has duplicate (model, ticker, date) rows, so pivot's first-wins does not
    silently drop records and distort the QLIKE analysis. Returns the number of rows checked."""
    n = len(df)
    if df.duplicated(subset=["model", "ticker", "date"]).any():
        dup = int(df.duplicated(subset=["model", "ticker", "date"]).sum())
        raise ValueError(f"cell log has {dup} duplicate (model,ticker,date) rows -- refusing to pivot silently")
    return n


def _load_test_cells(path):  # pragma: no cover - parquet I/O (large file); pure decomposition is tested
    import pyarrow.parquet as pq
    d = pq.read_table(path, columns=["model", "ticker", "date", "y_true", "y_pred"],
                      filters=[("split", "=", "test")]).to_pandas()
    assert_unique_cells(d)                                  # no silent dedup (aggfunc='first' would hide dups)
    piv = d.pivot_table(index=["ticker", "date"], columns="model", values=["y_true", "y_pred"],
                        aggfunc="first").dropna()
    return (piv["y_true"]["HAR-X"].values, piv["y_pred"]["HAR-X"].values, piv["y_pred"]["LSTM"].values)


def main(argv=None):  # pragma: no cover - CLI entry driver
    path = (argv or sys.argv[1:])[0]
    y, hf, lf = _load_test_cells(path)
    out = gap_decomposition(y, hf, lf)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
