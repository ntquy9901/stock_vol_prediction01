"""Aggregate the new-backbone masked G0/G1 verdict runs (dense + knn-8, seeds 42/123/2026).

Orchestration only. Reuses the unit-tested DM math (``diebold_mariano``) and per-observation loss
extraction (``paired_losses``). For each adjacency it prints and dumps:
  * per-seed G0/G1 val+test metrics (all 6) + G1-G0 val-loss delta,
  * per-metric G1-G0 mean +/- std over seeds (paired-t df=2),
  * per-seed Diebold-Mariano (QLIKE + squared-error) G1-vs-G0 on the val set,
  * an A/B verdict (A = graph helps: G1<G0 all seeds AND DM significant; else B = noise).

Writes ``<out_prefix>.json`` and ``<out_prefix>.md`` for the paper.

Run: python docs/reports/verdict_paper_dump.py <TS> [out_prefix]
Expects run dirs ``results/verdict_masked_{dense,knn}_seed{42,123,2026}_<TS>/h5``.
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
_SEEDS = (42, 123, 2026)
_ADJ = ("dense", "knn")


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["ticker_id"], r["target_date"]))


def _collect(ts: str) -> dict:
    out: dict = {}
    for adj in _ADJ:
        adj_block: dict = {"seeds": {}}
        for seed in _SEEDS:
            base = _ROOT / "results" / f"verdict_masked_{adj}_seed{seed}_{ts}" / "h5"
            comp_path = base / "graph_validation_comparison.json"
            if not comp_path.exists():
                continue
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            g0 = _sorted_rows(load_predictions(base / "G0" / "predictions.json"))
            g1 = _sorted_rows(load_predictions(base / "G1" / "predictions.json"))
            losses = paired_losses(g0, g1)
            dm_q = diebold_mariano(losses["qlike_g1"], losses["qlike_g0"], h=_HORIZON)
            dm_s = diebold_mariano(losses["se_g1"], losses["se_g0"], h=_HORIZON)
            adj_block["seeds"][seed] = {
                "G0": comp["results"]["G0"], "G1": comp["results"]["G1"],
                "paired_delta_valloss": comp["paired_delta"],
                "dm_qlike": {"stat": dm_q.dm_hln, "p": dm_q.p_value},
                "dm_mse": {"stat": dm_s.dm_hln, "p": dm_s.p_value},
                "n_obs": losses["n"],
            }
        out[adj] = adj_block
    return out


def _verdict(adj_block: dict) -> dict:
    seeds = adj_block["seeds"]
    if not seeds:
        return {"verdict": "N/A", "reason": "no completed seeds"}
    deltas = np.array([s["paired_delta_valloss"] for s in seeds.values()], dtype=float)
    n_better = int((deltas < 0).sum())
    dm_q_sig = all(s["dm_qlike"]["p"] < 0.05 and s["dm_qlike"]["stat"] < 0 for s in seeds.values())
    all_better = n_better == len(deltas)
    verdict = "A" if (all_better and dm_q_sig) else "B"
    t, p = stats.ttest_1samp(deltas, 0.0) if len(deltas) > 1 else (float("nan"), float("nan"))
    return {
        "verdict": verdict,
        "g1_better_seeds": f"{n_better}/{len(deltas)}",
        "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "paired_t": float(t), "paired_p": float(p),
        "dm_qlike_all_sig_negative": dm_q_sig,
    }


def _metric_table(seeds: dict, split: str) -> dict:
    key = "validation_metrics" if split == "val" else "test_metrics"
    table: dict = {}
    for metric in _METRICS:
        g0v, g1v = [], []
        for s in seeds.values():
            if key in s["G0"] and key in s["G1"]:
                g0v.append(s["G0"][key][metric])
                g1v.append(s["G1"][key][metric])
        if not g0v:
            continue
        g0a, g1a = np.array(g0v), np.array(g1v)
        d = g1a - g0a
        table[metric] = {
            "G0_mean": float(g0a.mean()), "G0_std": float(g0a.std(ddof=1)) if len(g0a) > 1 else 0.0,
            "G1_mean": float(g1a.mean()), "G1_std": float(g1a.std(ddof=1)) if len(g1a) > 1 else 0.0,
            "delta_mean": float(d.mean()), "delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        }
    return table


def main(ts: str, out_prefix: str) -> None:
    data = _collect(ts)
    summary: dict = {"timestamp": ts, "horizon": _HORIZON, "adjacencies": {}}
    lines: list[str] = ["# Masked G0/G1 graph verdict (screening-config P3 backbone)", "",
                        f"Backbone: screening P3 (5 epochs, dropout 0.2), leakage-safe graph-bound train set. "
                        f"Masked manifest + positivity floor. Horizon {_HORIZON}, 15 epochs, seeds {list(_SEEDS)}.",
                        ""]
    for adj in _ADJ:
        block = data.get(adj, {"seeds": {}})
        seeds = block["seeds"]
        verdict = _verdict(block)
        val_tab = _metric_table(seeds, "val")
        test_tab = _metric_table(seeds, "test")
        summary["adjacencies"][adj] = {
            "verdict": verdict,
            "per_seed": {
                seed: {
                    "paired_delta_valloss": s["paired_delta_valloss"],
                    "dm_qlike": s["dm_qlike"], "dm_mse": s["dm_mse"], "n_obs": s["n_obs"],
                    "G0_val": s["G0"].get("validation_metrics"), "G1_val": s["G1"].get("validation_metrics"),
                    "G0_test": s["G0"].get("test_metrics"), "G1_test": s["G1"].get("test_metrics"),
                } for seed, s in seeds.items()
            },
            "val_metric_delta": val_tab, "test_metric_delta": test_tab,
        }
        lines += [f"## Adjacency: {adj}", "",
                  f"**Verdict: {verdict['verdict']}** "
                  f"(G1<G0 in {verdict.get('g1_better_seeds','?')} seeds; "
                  f"delta_valloss mean={verdict.get('delta_mean',float('nan')):.3e}; "
                  f"paired-t p={verdict.get('paired_p',float('nan')):.4f}; "
                  f"DM-QLIKE all sig-neg={verdict.get('dm_qlike_all_sig_negative')})", ""]
        lines.append("| seed | G0 valloss | G1 valloss | delta(G1-G0) | DM_QLIKE | p_QLIKE | DM_MSE | p_MSE | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for seed, s in seeds.items():
            dq, ds = s["dm_qlike"], s["dm_mse"]
            lines.append(
                f"| {seed} | {s['G0']['validation_loss']:.6f} | {s['G1']['validation_loss']:.6f} | "
                f"{s['paired_delta_valloss']:+.3e} | {dq['stat']:+.3f} | {dq['p']:.4g} | "
                f"{ds['stat']:+.3f} | {ds['p']:.4g} | {s['n_obs']} |")
        lines.append("")
        for split, tab in (("VAL", val_tab), ("TEST", test_tab)):
            if not tab:
                lines += [f"_{split}: no data_", ""]
                continue
            lines.append(f"### {split} metrics (mean over seeds)")
            lines.append("| metric | G0 | G1 | G1-G0 |")
            lines.append("|---|---|---|---|")
            for m in _METRICS:
                if m in tab:
                    t = tab[m]
                    lines.append(f"| {m} | {t['G0_mean']:.6g} | {t['G1_mean']:.6g} | {t['delta_mean']:+.3g} |")
            lines.append("")

    Path(out_prefix + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(out_prefix + ".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWROTE {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    ts_arg = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else str(_ROOT / "docs" / "reports" / f"verdict_masked_g0g1_{ts_arg}")
    main(ts_arg, prefix)
