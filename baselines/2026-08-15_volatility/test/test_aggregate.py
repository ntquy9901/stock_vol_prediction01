import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import aggregate as agg  # noqa: E402

_TS = "2026-01-01_000000"
_SIX = {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5, "qlike": 0.5, "directional_accuracy": 48.0}


def _write_dump(path, targets, preds):
    rows = [{"ticker_id": i, "target_date": f"2025-01-{i + 1:02d}",
             "target_raw": float(t), "prediction_raw": float(p)}
            for i, (t, p) in enumerate(zip(targets, preds, strict=True))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_aggregate_dm_and_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agg, "ROOT", tmp_path)
    rng = np.random.default_rng(0)
    targets = np.abs(rng.normal(1.0, 0.2, 40)) + 0.1
    seeds = (42, 123, 2026)
    for seed in seeds:
        run = tmp_path / "results" / f"volatility_gat_seed{seed}_{_TS}"
        # GNN far from target, NODE/HAR near -> DM favours B (NODE/HAR) over GNN
        _write_dump(run / "GNN" / "predictions_test.json", targets, targets + rng.normal(0.4, 0.15, 40))
        _write_dump(run / "NODE" / "predictions_test.json", targets, targets + rng.normal(0.0, 0.03, 40))
        _write_dump(run / "P0" / "predictions_test.json", targets, targets + rng.normal(0.0, 0.05, 40))
        ladder = {"rungs": {r: {"validation_metrics": _SIX, "test_metrics": _SIX} for r in agg.RUNGS}}
        (run / "ladder_metrics.json").write_text(json.dumps(ladder), encoding="utf-8")

    agg.main(_TS, seeds=seeds)
    summary = json.loads((tmp_path / "results" / f"volatility_gat_{_TS}_summary.json").read_text())
    dm = summary["diebold_mariano"]
    assert set(dm) == {"GNN_vs_NODE", "GNN_vs_HAR", "NODE_vs_HAR"}
    assert dm["GNN_vs_NODE"]["qlike"]["favors"] == "B"   # NODE better than GNN
    assert dm["GNN_vs_NODE"]["n"] == 40
    out = capsys.readouterr().out
    for rung in ("HAR", "NODE", "GNN"):
        assert rung in out


def test_aggregate_degenerate_pair_recorded_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "ROOT", tmp_path)
    rng = np.random.default_rng(1)
    targets = np.abs(rng.normal(1.0, 0.2, 30)) + 0.1
    seeds = (42, 123, 2026)
    for seed in seeds:
        run = tmp_path / "results" / f"volatility_gat_seed{seed}_{_TS}"
        _write_dump(run / "GNN" / "predictions_test.json", targets, targets + rng.normal(0.3, 0.1, 30))
        _write_dump(run / "NODE" / "predictions_test.json", targets, targets + 0.02)   # NODE == HAR
        _write_dump(run / "P0" / "predictions_test.json", targets, targets + 0.02)
        ladder = {"rungs": {r: {"validation_metrics": _SIX, "test_metrics": _SIX} for r in agg.RUNGS}}
        (run / "ladder_metrics.json").write_text(json.dumps(ladder), encoding="utf-8")
    agg.main(_TS, seeds=seeds)   # NODE_vs_HAR is zero-variance -> recorded as error, no crash
    dm = json.loads((tmp_path / "results" / f"volatility_gat_{_TS}_summary.json").read_text())["diebold_mariano"]
    assert "error" in dm["NODE_vs_HAR"]["qlike"]
    assert "favors" in dm["GNN_vs_HAR"]["qlike"]
