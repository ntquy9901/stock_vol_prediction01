"""FU3b: temporal-stability robustness on the fixed held-out test window.

Splits the test observations into `n_blocks` sequential time blocks (by target date) and runs a
Diebold-Mariano test of FULL vs HAR within each block. This checks whether an advantage is stable
across the test period or driven by one sub-period. It is NOT full rolling recalibration (which would
retrain per window); it reuses the already-computed test dumps, so it is a cheaper robustness check.

Run: python <.../code/subperiod_report.py> <TS_sweep> <horizon> [seeds_csv] [n_blocks]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402


def split_subperiods(dates, n_blocks: int) -> list[np.ndarray]:
    """Boolean masks partitioning observations into `n_blocks` sequential blocks by target date.
    Observations sharing a date always land in the same block."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    dates = np.asarray(dates)
    unique_sorted = np.array(sorted(set(dates.tolist())))
    blocks = np.array_split(unique_sorted, n_blocks)
    return [np.isin(dates, block) for block in blocks]


def run_subperiods(ts: str, horizon: int, seeds, n_blocks: int = 4,
                   comparator: str = "HAR") -> dict[str, Any]:  # pragma: no cover
    """DM FULL vs `comparator` within each sequential time block of the test window."""
    import dm_report as dm
    from diebold_mariano import diebold_mariano

    keys_f, target, pred_full = dm._ensemble(ts, horizon, "FULL", seeds)
    keys_c, tgt_c, pred_cmp = dm._ensemble(ts, horizon, comparator, seeds)
    if keys_f != keys_c or not np.allclose(target, tgt_c):
        raise ValueError("FULL and comparator observations/targets misaligned")
    dates = np.array([k[1] for k in keys_f])
    masks = split_subperiods(dates, n_blocks)
    out: dict[str, Any] = {"horizon": horizon, "n_blocks": n_blocks, "blocks": []}
    for i, sel in enumerate(masks):
        r = diebold_mariano(dm._qlike(target[sel], pred_full[sel]),
                            dm._qlike(target[sel], pred_cmp[sel]), h=horizon)
        favors = "FULL" if r.mean_diff < 0 else ("HAR" if r.mean_diff > 0 else "tie")
        block_dates = sorted(dates[sel].tolist())
        span = (block_dates[0], block_dates[-1])
        out["blocks"].append({"block": i, "span": span, "n": int(sel.sum()),
                              "dm_hln": r.dm_hln, "p_value": r.p_value, "favors": favors})
    return out


def main(ts: str, horizon: int, seeds, n_blocks: int = 4) -> None:  # pragma: no cover
    rep = run_subperiods(ts, horizon, seeds, n_blocks)
    print(f"# Temporal-stability h{horizon} seeds={list(seeds)} blocks={n_blocks} (DM FULL vs HAR, QLIKE)")
    print("| block | span | n | dm_hln | p | favors |")
    print("|---|---|---|---|---|---|")
    for b in rep["blocks"]:
        s = "*" if b["p_value"] < 0.05 else ""
        print(f"| {b['block']} | {b['span'][0]}..{b['span'][1]} | {b['n']} | "
              f"{b['dm_hln']:+.2f} | {b['p_value']:.3f}{s} | {b['favors']} |")


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _h = int(sys.argv[2])
    _seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42, 123, 2026]
    _nb = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    main(_ts, _h, _seeds, _nb)
