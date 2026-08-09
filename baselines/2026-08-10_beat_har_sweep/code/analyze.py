"""Aggregate the beat-HAR sweep vs the HAR bar: per-seed Diebold-Mariano + across-seed paired-t.

Recomputes P0 (pooled HAR) per-observation predictions on the SAME fair basis (deterministic, no
seed), aligns each config's saved per-observation predictions by (ticker_id, target_date), and reports
per-seed DM (QLIKE and squared-error, h=5) plus the across-seed paired-t on seed-mean QLIKE deltas.

Usage (GPU or CPU venv):  python .../code/analyze.py <TS> [device] [config ...]
Writes results/beat_har_sweep_<TS>/analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CODE = str(Path(__file__).resolve().parent)
_PILOT = str(Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code")
_ROOT = str(Path(__file__).resolve().parents[3])
for _p in (_CODE, _PILOT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from diebold_mariano import diebold_mariano  # noqa: E402

HORIZON = 5
EPSILON = 1e-8
SEEDS = (42, 123, 2026)
# Fair-basis HAR reference bar (test set), from the plan / cited JSONs.
BAR = {"P0_qlike": 0.5676, "HARQ_rmse": 0.0022891, "HARQ_r2": 0.76682, "classicalHAR_qlike": 0.5793}


def _qlike_vec(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = np.maximum(pred, EPSILON)
    target = np.maximum(target, EPSILON)
    ratio = target / pred
    return ratio - np.log(ratio) - 1.0


def _se_vec(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (target - pred) ** 2


def compute_p0_predictions(device_name: str, ts: str) -> dict[str, dict[tuple[int, str], tuple[float, float]]]:
    """Recompute P0 pooled-HAR val/test per-observation (target_raw, prediction_raw) on the basis."""

    from ladder_consistent import build_basis, run_har_rung
    from run_pilot import resolve_graph_device

    device = resolve_graph_device(device_name)
    stamp = Path(_ROOT) / "temp" / f"beat_har_p0_{ts}"
    pooled, graph, graph_store, allowed = build_basis(device, stamp)
    out = Path(_ROOT) / "results" / f"beat_har_sweep_{ts}" / "P0"
    run_har_rung(pooled, allowed, graph_store, out)
    result: dict[str, dict[tuple[int, str], tuple[float, float]]] = {}
    for split, sub in (("val", "val"), ("test", "test")):
        payload = json.loads((out / sub / "results.json").read_text(encoding="utf-8"))
        keys = payload["ordered_validation_keys"]
        targets = payload["targets_raw"]
        preds = payload["predictions_raw"]
        result[split] = {
            (int(k["ticker_id"]), str(k["target_date"])): (float(t), float(p))
            for k, t, p in zip(keys, targets, preds, strict=True)
        }
    return result


def _load_config_predictions(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(int(r["ticker_id"]), str(r["target_date"])): (float(r["target_raw"]), float(r["prediction_raw"]))
            for r in rows}


def _aligned_losses(p0: dict, cfg: dict, loss_fn) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(p0) & set(cfg))
    if len(keys) != len(p0) or len(keys) != len(cfg):
        raise ValueError(f"observation sets differ: p0={len(p0)} cfg={len(cfg)} shared={len(keys)}")
    target = np.array([p0[k][0] for k in keys])
    # p0 and cfg targets must match (same fair basis)
    cfg_target = np.array([cfg[k][0] for k in keys])
    if not np.allclose(target, cfg_target, rtol=1e-6, atol=1e-12):
        raise ValueError("aligned targets differ between P0 and config")
    p0_loss = loss_fn(target, np.array([p0[k][1] for k in keys]))
    cfg_loss = loss_fn(target, np.array([cfg[k][1] for k in keys]))
    return cfg_loss, p0_loss


def _analyze_variant(pred_paths: dict[int, Path], p0: dict) -> dict[str, Any]:
    """DM + paired-t for one variant given {seed: predictions_test.json path}."""

    per_seed: dict[int, Any] = {}
    qlike_deltas: list[float] = []
    for seed, pred_path in sorted(pred_paths.items()):
        if not pred_path.exists():
            continue
        cfg = _load_config_predictions(pred_path)
        cfg_q, p0_q = _aligned_losses(p0["test"], cfg, _qlike_vec)
        cfg_se, p0_se = _aligned_losses(p0["test"], cfg, _se_vec)
        dm_q = diebold_mariano(cfg_q, p0_q, h=HORIZON)
        dm_se = diebold_mariano(cfg_se, p0_se, h=HORIZON)
        per_seed[seed] = {
            "test_qlike_config": float(cfg_q.mean()), "test_qlike_p0": float(p0_q.mean()),
            "qlike_delta": float(cfg_q.mean() - p0_q.mean()),
            "dm_qlike_stat": dm_q.dm_hln, "dm_qlike_p": dm_q.p_value,
            "dm_rmse_stat": dm_se.dm_hln, "dm_rmse_p": dm_se.p_value,
            "test_rmse_config": float(np.sqrt(cfg_se.mean())), "test_rmse_p0": float(np.sqrt(p0_se.mean())),
        }
        qlike_deltas.append(float(cfg_q.mean() - p0_q.mean()))
    summary: dict[str, Any] = {"per_seed": per_seed, "n_seeds": len(qlike_deltas)}
    if len(qlike_deltas) >= 2:
        deltas = np.array(qlike_deltas)
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        summary["paired_t_qlike"] = {
            "mean_delta": float(deltas.mean()), "t_stat": float(t_stat), "p_value": float(p_val),
            "all_negative": bool((deltas < 0).all()), "all_positive": bool((deltas > 0).all()),
        }
        dm_ps = [per_seed[s]["dm_qlike_p"] for s in per_seed]
        summary["beats_P0_qlike_dm"] = bool(
            (deltas < 0).all() and all(p < 0.05 for p in dm_ps) and p_val < 0.05)
    return summary


def analyze_config(config_dir: Path, p0: dict, seeds=SEEDS) -> dict[str, Any]:
    """Structure-aware: flat seed dirs, or C5's per-k sub-dirs (reports per-k + best-k)."""

    k_dirs = sorted({p.parent.name for p in config_dir.glob("seed*/k*/predictions_test.json")})
    if k_dirs:
        variants: dict[str, Any] = {}
        for k_name in k_dirs:
            pred_paths = {s: config_dir / f"seed{s}" / k_name / "predictions_test.json" for s in seeds}
            variants[k_name] = _analyze_variant(pred_paths, p0)
        # best-k by mean test QLIKE across seeds
        def _mean_q(v):
            ps = v["per_seed"].values()
            return np.mean([x["test_qlike_config"] for x in ps]) if ps else float("inf")
        best = min(variants, key=lambda k: _mean_q(variants[k]))
        return {"k_sweep": variants, "best_k": best, **variants[best]}

    per_seed: dict[int, Any] = {}
    qlike_deltas: list[float] = []
    for seed in seeds:
        pred_path = config_dir / f"seed{seed}" / "predictions_test.json"
        if not pred_path.exists():
            continue
        cfg = _load_config_predictions(pred_path)
        cfg_q, p0_q = _aligned_losses(p0["test"], cfg, _qlike_vec)
        cfg_se, p0_se = _aligned_losses(p0["test"], cfg, _se_vec)
        dm_q = diebold_mariano(cfg_q, p0_q, h=HORIZON)
        dm_se = diebold_mariano(cfg_se, p0_se, h=HORIZON)
        per_seed[seed] = {
            "test_qlike_config": float(cfg_q.mean()), "test_qlike_p0": float(p0_q.mean()),
            "qlike_delta": float(cfg_q.mean() - p0_q.mean()),
            "dm_qlike_stat": dm_q.dm_hln, "dm_qlike_p": dm_q.p_value,
            "dm_rmse_stat": dm_se.dm_hln, "dm_rmse_p": dm_se.p_value,
            "test_rmse_config": float(np.sqrt(cfg_se.mean())), "test_rmse_p0": float(np.sqrt(p0_se.mean())),
        }
        qlike_deltas.append(float(cfg_q.mean() - p0_q.mean()))
    summary: dict[str, Any] = {"per_seed": per_seed, "n_seeds": len(qlike_deltas)}
    if len(qlike_deltas) >= 2:
        deltas = np.array(qlike_deltas)
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        summary["paired_t_qlike"] = {
            "mean_delta": float(deltas.mean()), "t_stat": float(t_stat), "p_value": float(p_val),
            "all_negative": bool((deltas < 0).all()), "all_positive": bool((deltas > 0).all()),
        }
        # verdict vs P0 on QLIKE (partial-win primary target)
        dm_ps = [per_seed[s]["dm_qlike_p"] for s in per_seed]
        beats_p0 = bool((deltas < 0).all() and all(p < 0.05 for p in dm_ps) and p_val < 0.05)
        summary["beats_P0_qlike_dm"] = beats_p0
    return summary


def main(ts: str, device_name: str = "cuda", configs: list[str] | None = None) -> Path:
    configs = configs or ["C1", "C2", "C3", "C5", "C6"]
    p0 = compute_p0_predictions(device_name, ts)
    results_root = Path(_ROOT) / "results"
    analysis: dict[str, Any] = {"ts": ts, "bar": BAR, "configs": {}}
    for config in configs:
        matches = sorted(results_root.glob(f"beat_har_{config}_{ts}*"))
        if not matches:
            analysis["configs"][config] = {"status": "no_results"}
            continue
        analysis["configs"][config] = analyze_config(matches[0], p0)
    out = results_root / f"beat_har_sweep_{ts}" / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ts = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    cfgs = sys.argv[3:] if len(sys.argv) > 3 else None
    main(ts, device, cfgs)
