"""Aggregate the consistent-basis ladder runs (P0->P1->P2->P3->G1, seeds 42/123/2026, h5).

Orchestration only. Reuses the unit-tested DM math (``diebold_mariano``) and per-observation loss
extraction (``paired_losses``). Produces the single canonical source of truth the paper + bundle
consume:
  * the 5-rung table (all 6 metrics, VAL + held-out TEST, per-seed + 3-seed mean +/- std),
  * the nesting-verification result (G1-graph-off == P3, determinism + graph-effect magnitude),
  * the G1-vs-P3 Diebold-Mariano test (QLIKE + squared-error) on VAL and TEST per seed,
  * an A/B verdict for the graph effect (A = graph helps; B = null).

Run: python docs/reports/ladder_consistent_dump.py <TS> [out_prefix]
Expects run dirs ``results/ladder_consistent_seed{42,123,2026}_<TS>/h5``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
sys.path.insert(0, str(_CODE))

from diebold_mariano import diebold_mariano  # noqa: E402
from dm_analysis import load_predictions, paired_losses  # noqa: E402

_HORIZON = 5  # 5-day overlapping forecast -> HAC truncation lag 4
_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
_RUNGS = ("P0", "P1", "P2", "P3", "G1")
_SEEDS = (42, 123, 2026)


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["ticker_id"], r["target_date"]))


def _collect(ts: str, horizon: int) -> dict:
    seeds: dict[int, dict] = {}
    for seed in _SEEDS:
        base = _ROOT / "results" / f"ladder_consistent_seed{seed}_{ts}" / f"h{horizon}"
        ladder_path = base / "ladder_metrics.json"
        if not ladder_path.exists():
            continue
        ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        block: dict = {"rungs": ladder["rungs"], "nesting_check": ladder["nesting_check"],
                       "adjacency": ladder.get("adjacency"), "edge_density": ladder.get("edge_density"),
                       "snapshot_count": ladder.get("snapshot_count"), "dm": {}}
        for split, val_file in (("val", "predictions.json"), ("test", "predictions_test.json")):
            p3 = _sorted_rows(load_predictions(base / "P3" / val_file))
            g1 = _sorted_rows(load_predictions(base / "G1" / val_file))
            losses = paired_losses(p3, g1)  # a=P3, b=G1
            dm_q = diebold_mariano(losses["qlike_g1"], losses["qlike_g0"], h=horizon)
            dm_s = diebold_mariano(losses["se_g1"], losses["se_g0"], h=horizon)
            block["dm"][split] = {"dm_qlike": {"stat": dm_q.dm_hln, "p": dm_q.p_value},
                                  "dm_mse": {"stat": dm_s.dm_hln, "p": dm_s.p_value}, "n_obs": losses["n"]}
        seeds[seed] = block
    return seeds


def _rung_table(seeds: dict, split: str) -> dict:
    key = "validation_metrics" if split == "val" else "test_metrics"
    table: dict = {}
    for rung in _RUNGS:
        per_metric: dict = {}
        for metric in _METRICS:
            values = [s["rungs"][rung][key][metric] for s in seeds.values()
                      if rung in s["rungs"] and s["rungs"][rung].get(key)]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            per_metric[metric] = {"mean": float(array.mean()),
                                  "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                                  "per_seed": {str(seed): float(s["rungs"][rung][key][metric])
                                               for seed, s in seeds.items()
                                               if rung in s["rungs"] and s["rungs"][rung].get(key)}}
        table[rung] = per_metric
    return table


def _graph_verdict(seeds: dict, split: str) -> dict:
    """A = graph helps: G1 QLIKE < P3 QLIKE all seeds AND DM-QLIKE significant-negative all seeds."""

    key = "validation_metrics" if split == "val" else "test_metrics"
    dm_q = [s["dm"][split]["dm_qlike"] for s in seeds.values()]
    g1_better = [s["rungs"]["G1"][key]["qlike"] < s["rungs"]["P3"][key]["qlike"] for s in seeds.values()]
    dm_sig_neg = all(d["p"] < 0.05 and d["stat"] < 0 for d in dm_q)
    all_better = all(g1_better)
    diffs = np.asarray([s["rungs"]["G1"][key]["qlike"] - s["rungs"]["P3"][key]["qlike"]
                        for s in seeds.values()], dtype=float)
    t, p = stats.ttest_1samp(diffs, 0.0) if len(diffs) > 1 else (float("nan"), float("nan"))
    return {"verdict": "A" if (all_better and dm_sig_neg) else "B",
            "g1_qlike_better_seeds": f"{int(sum(g1_better))}/{len(g1_better)}",
            "qlike_delta_mean": float(diffs.mean()),
            "qlike_delta_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "paired_t": float(t), "paired_p": float(p), "dm_qlike_all_sig_negative": dm_sig_neg}


def _fmt(value: float, metric: str) -> str:
    return f"{value:.2f}" if metric == "directional_accuracy" else f"{value:.6g}"


def main(ts: str, out_prefix: str, horizon: int = _HORIZON) -> None:
    seeds = _collect(ts, horizon)
    if not seeds:
        raise SystemExit(f"no completed ladder seeds for timestamp {ts}")
    val_table = _rung_table(seeds, "val")
    test_table = _rung_table(seeds, "test")
    verdict = {"val": _graph_verdict(seeds, "val"), "test": _graph_verdict(seeds, "test")}
    nesting = {str(seed): s["nesting_check"] for seed, s in seeds.items()}
    any_seed = next(iter(seeds.values()))
    summary = {"timestamp": ts, "horizon": horizon, "seeds_completed": sorted(seeds),
               "adjacency": any_seed.get("adjacency"), "edge_density": any_seed.get("edge_density"),
               "snapshot_count": any_seed.get("snapshot_count"),
               "basis": "masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, "
                        "positivity floor, identical val/test observations",
               "p3_definition": "the trained G1 model read out with the GAT/message-passing residual "
                                "disabled (replaces the old separate G0 row)",
               "rung_metrics": {"val": val_table, "test": test_table},
               "graph_effect_dm": {str(seed): s["dm"] for seed, s in seeds.items()},
               "graph_effect_verdict": verdict, "nesting_check": nesting}

    lines: list[str] = [
        f"# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (h{horizon})", "",
        f"One basis for every rung: {summary['basis']}. Masked k-NN-8 adjacency, "
        f"seeds {list(_SEEDS)}, horizon {horizon}. "
        f"P3 = {summary['p3_definition']}.", "",
        "## Nesting verification (G1 minus the GAT = P3)", "",
        "The P3 row is the identical trained G1 model with the message-passing residual removed, so "
        "the equality holds by construction; the runtime check confirms the graph-off readout is "
        "deterministic (max abs pred diff on a re-evaluation) and the residual is non-trivial "
        "(mean abs raw pred diff G1 vs P3).", "",
        "| seed | graph-off determinism (max abs diff) | graph effect (mean abs raw diff) | n_val_obs |",
        "|---|---|---|---|",
    ]
    for seed in sorted(seeds):
        n = nesting[str(seed)]
        lines.append(f"| {seed} | {n['graph_off_readout_determinism_max_abs_diff']:.2e} | "
                     f"{n['graph_effect_val_mean_abs_pred_diff_raw']:.3e} | {n['n_val_obs']} |")
    lines.append("")

    for split, table in (("VAL", val_table), ("TEST", test_table)):
        lines += [f"## {split} metrics (3-seed mean +/- std)", "",
                  "| rung | mse | rmse | mae | r2 | qlike | dir_acc |", "|---|---|---|---|---|---|---|"]
        for rung in _RUNGS:
            cells = []
            for metric in _METRICS:
                if metric in table[rung]:
                    entry = table[rung][metric]
                    cells.append(f"{_fmt(entry['mean'], metric)} +/- {_fmt(entry['std'], metric)}")
                else:
                    cells.append("-")
            lines.append(f"| {rung} | " + " | ".join(cells) + " |")
        lines.append("")

    for split in ("val", "test"):
        v = verdict[split]
        lines += [f"## Graph effect (G1 vs P3), {split.upper()}", "",
                  f"**Verdict: {v['verdict']}** "
                  f"(G1 QLIKE < P3 in {v['g1_qlike_better_seeds']} seeds; "
                  f"QLIKE delta mean={v['qlike_delta_mean']:+.3e}; paired-t p={v['paired_p']:.4f}; "
                  f"DM-QLIKE all sig-neg={v['dm_qlike_all_sig_negative']})", "",
                  "| seed | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |", "|---|---|---|---|---|---|"]
        for seed in sorted(seeds):
            dm = seeds[seed]["dm"][split]
            lines.append(f"| {seed} | {dm['dm_qlike']['stat']:+.3f} | {dm['dm_qlike']['p']:.4g} | "
                         f"{dm['dm_mse']['stat']:+.3f} | {dm['dm_mse']['p']:.4g} | {dm['n_obs']} |")
        lines.append("")

    Path(out_prefix + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(out_prefix + ".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWROTE {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    ts_arg = sys.argv[1]
    horizon_arg = int(sys.argv[2]) if len(sys.argv) > 2 else _HORIZON
    prefix = (sys.argv[3] if len(sys.argv) > 3
              else str(_ROOT / "docs" / "reports" / f"ladder_consistent_h{horizon_arg}_{ts_arg}"))
    main(ts_arg, prefix, horizon_arg)
