import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import aggregate_seeds as ag  # noqa: E402

_SIX = {"mse": 1e-6, "rmse": 1e-3, "mae": 1e-4, "r2": 0.7, "qlike": 0.5, "directional_accuracy": 48.0}


def _write_seed(root, ts, h, seed, full_qlike):
    run = root / "results" / f"volatility_ablation_h{h}_seed{seed}_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    rungs = {}
    for r in ("HAR", "FULL", "minus_graph", "minus_gate", "minus_news"):
        m = dict(_SIX)
        if r == "FULL":
            m = {**_SIX, "qlike": full_qlike}
        rungs[r] = {"test_metrics": m, "validation_metrics": m}
    (run / "ladder_metrics.json").write_text(json.dumps({"rungs": rungs}), encoding="utf-8")
    (run / "lstm_only_metrics.json").write_text(
        json.dumps({"metrics": {"test_metrics": _SIX, "validation_metrics": _SIX}}), encoding="utf-8")


def test_aggregate_mean_std_and_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "ROOT", tmp_path)
    ts, h = "T", 5
    # three seeds with FULL qlike 0.50, 0.52, 0.54 -> mean 0.52; minus_graph qlike = 0.50 (const)
    for seed, fq in ((42, 0.50), (123, 0.52), (2026, 0.54)):
        _write_seed(tmp_path, ts, h, seed, fq)
    a = ag.aggregate(ts, [42, 123, 2026], h)
    assert abs(a["FULL"]["test_metrics"]["qlike"]["mean"] - 0.52) < 1e-9
    assert a["FULL"]["test_metrics"]["qlike"]["std"] > 0
    # effect(graph) = QLIKE(FULL)mean - QLIKE(minus_graph)mean = 0.52 - 0.50 = +0.02
    assert abs(a["effects_qlike_test"]["graph"] - 0.02) < 1e-9
    assert set(a) >= set(ag.RUNGS)
