"""Aggregate the 3-seed masked knn-8 G0/G1 runs and run Diebold-Mariano tests.

Orchestration only: the DM math (``diebold_mariano``) and per-observation loss extraction
(``paired_losses``) are unit-tested in the baseline test suite. This script loads the seed
run artifacts, prints the 3-seed val-loss/metric table with a paired-t (df=2), and prints the
per-seed DM statistic + p-value for QLIKE and squared-error losses.

Run: python docs/reports/dm_knn8_confirmation.py <seed42_dir> <seed123_dir> <seed2026_dir>
Each dir is the run output-dir (its ``h5/`` subtree holds G0/G1 predictions.json + the
graph_validation_comparison.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_CODE = Path(__file__).resolve().parents[2] / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
sys.path.insert(0, str(_CODE))

from diebold_mariano import diebold_mariano  # noqa: E402
from dm_analysis import load_predictions, paired_losses  # noqa: E402

_HORIZON = 5  # 5-day-ahead forecast -> overlapping MA(h-1) => HAC truncation lag 4
_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["ticker_id"], r["target_date"]))


def _seed_from_dir(path: Path) -> int:
    token = [p for p in path.name.split("_") if p.startswith("seed")][0]
    return int(token.removeprefix("seed"))


def main(dirs: list[str]) -> None:
    seeds: dict[int, dict] = {}
    for raw in dirs:
        base = Path(raw) / "h5"
        comparison = json.loads((base / "graph_validation_comparison.json").read_text(encoding="utf-8"))
        seed = _seed_from_dir(Path(raw))
        g0 = _sorted_rows(load_predictions(base / "G0" / "predictions.json"))
        g1 = _sorted_rows(load_predictions(base / "G1" / "predictions.json"))
        losses = paired_losses(g0, g1)
        dm_qlike = diebold_mariano(losses["qlike_g1"], losses["qlike_g0"], h=_HORIZON)
        dm_se = diebold_mariano(losses["se_g1"], losses["se_g0"], h=_HORIZON)
        seeds[seed] = {"comparison": comparison, "dm_qlike": dm_qlike, "dm_se": dm_se, "n": losses["n"]}

    dir_by_seed = {_seed_from_dir(Path(raw)): Path(raw) for raw in dirs}
    order = sorted(seeds)
    print("\n=== 3-seed knn-8 masked G0/G1 (15 epochs, GPU) ===")
    print(f"{'seed':>6} {'G0 valloss':>12} {'G1 valloss':>12} {'delta(G1-G0)':>14} "
          f"{'G1 QLIKE':>10} {'G1 R2':>8} {'G1 DirAcc':>10} {'nonpos':>7}")
    deltas = []
    for seed in order:
        res = seeds[seed]["comparison"]["results"]
        delta = seeds[seed]["comparison"]["paired_delta"]
        deltas.append(delta)
        g1m = res["G1"]["validation_metrics"]
        g1_json = json.loads(
            (dir_by_seed[seed] / "h5" / "G1" / "results.json").read_text(encoding="utf-8")
        )
        nonpos = g1_json["nonpositive_prediction_rate"]
        print(f"{seed:>6} {res['G0']['validation_loss']:>12.6f} {res['G1']['validation_loss']:>12.6f} "
              f"{delta:>14.6e} {g1m['qlike']:>10.5f} {g1m['r2']:>8.5f} {g1m['directional_accuracy']:>10.4f} "
              f"{nonpos:>7.3f}")

    deltas = np.asarray(deltas, dtype=float)
    n_better = int(np.sum(deltas < 0))
    t_stat, p_two = stats.ttest_1samp(deltas, 0.0)
    print(f"\ndelta mean={deltas.mean():.6e}  std={deltas.std(ddof=1):.6e}  "
          f"G1<G0 in {n_better}/{len(deltas)} seeds")
    print(f"paired-t on 3 deltas vs 0: t={t_stat:.4f} (df={len(deltas) - 1}), "
          f"two-sided p={p_two:.4f}, t_crit(0.05,df=2)=4.303")

    # 6-metric G1-G0 deltas aggregated across seeds
    print("\n=== per-metric G1-G0 delta, mean +/- std over 3 seeds ===")
    for metric in _METRICS:
        vals = np.asarray([
            seeds[s]["comparison"]["results"]["G1"]["validation_metrics"][metric]
            - seeds[s]["comparison"]["results"]["G0"]["validation_metrics"][metric]
            for s in order
        ])
        tval, pval = stats.ttest_1samp(vals, 0.0)
        print(f"{metric:>22}: {vals.mean():+.6e} +/- {vals.std(ddof=1):.2e}   "
              f"t={tval:+.3f} p={pval:.4f}")

    print("\n=== Diebold-Mariano per seed (h=5, Newey-West Bartlett, HLN-corrected, t df=n-1) ===")
    print(f"{'seed':>6} {'n':>7} {'DM_QLIKE':>10} {'p_QLIKE':>10} {'DM_MSE':>10} {'p_MSE':>10}")
    for seed in order:
        dq = seeds[seed]["dm_qlike"]
        ds = seeds[seed]["dm_se"]
        print(f"{seed:>6} {seeds[seed]['n']:>7} {dq.dm_hln:>10.4f} {dq.p_value:>10.4g} "
              f"{ds.dm_hln:>10.4f} {ds.p_value:>10.4g}")
    print("(negative DM => G1 lower loss => graph helps; significant if p < 0.05)")


if __name__ == "__main__":
    main(sys.argv[1:])
