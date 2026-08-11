"""Test the 3-seed aggregation + Diebold-Mariano driver on a synthetic results tree."""

import json
from pathlib import Path

import numpy as np

import aggregate

_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _metrics(rmse: float) -> dict[str, float]:
    return {"mse": rmse ** 2, "rmse": rmse, "mae": rmse * 0.3, "r2": 0.77,
            "qlike": 0.57, "directional_accuracy": 48.5}


def _write_run(root: Path, ts: str, seed: int, obs: list[tuple[int, str]]) -> None:
    run = root / "results" / f"eda_gnn_seed{seed}_{ts}" / "h5"
    rung_rmse = {"E0": 0.00229, "E1": 0.002285, "E2": 0.00226, "E3": 0.002267,
                 "E3off": 0.002256, "G1corr": 0.002264}
    ladder = {"seed": seed, "horizon": 5, "epochs": 20, "vol2pk_top_k": 5, "corr_top_k": 8,
              "vol2pk_edges": 198, "rungs": {}}
    rng = np.random.default_rng(seed)
    for rung_key, rmse in rung_rmse.items():
        ladder["rungs"][rung_key] = {"validation_metrics": _metrics(rmse * 0.65),
                                     "test_metrics": _metrics(rmse)}
        dump_dir = run / ("E3_off" if rung_key == "E3off" else rung_key)
        dump_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for ticker_id, date in obs:
            target = 1e-3 + 1e-4 * ((ticker_id + hash(date)) % 5)
            prediction = target * (1.0 + rmse * 100) + rng.normal(0, 1e-5)
            rows.append({"ticker_id": ticker_id, "target_date": date,
                         "target_raw": float(target), "prediction_raw": float(abs(prediction) + 1e-6)})
        (dump_dir / "predictions_test.json").write_text(json.dumps(rows), encoding="utf-8")
    (run / "ladder_metrics.json").write_text(json.dumps(ladder), encoding="utf-8")


def test_aggregate_writes_summary_and_dm(tmp_path, monkeypatch, capsys):
    ts = "2099-01-01_000000"
    obs = [(t, f"2025-0{1 + d % 9}-1{d % 9}") for t in range(6) for d in range(9)]
    monkeypatch.setattr(aggregate, "R", str(tmp_path))
    for seed in aggregate.SEEDS:
        _write_run(tmp_path, ts, seed, obs)

    aggregate.main(ts)

    summary = json.loads((tmp_path / "results" / f"eda_gnn_{ts}_summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == list(aggregate.SEEDS)
    for rung in aggregate.RUNGS:
        for split in ("test_metrics", "validation_metrics"):
            for key in _METRICS:
                assert key in summary["metrics_mean_std"][rung][split]
    for comparison in ("E1_vs_E0", "E2_vs_E0", "E3_vs_E0", "E3_vs_G1corr", "E3_vs_E3off"):
        result = summary["diebold_mariano"][comparison]
        assert result["n"] == len(obs)
        for metric in ("qlike", "se"):
            assert "p_value" in result[metric] and result[metric]["favors"] in ("A", "B")
    captured = capsys.readouterr().out
    assert "Diebold-Mariano" in captured and "E3_vs_G1corr" in captured


def test_aggregate_rejects_target_disagreement(tmp_path, monkeypatch):
    ts = "2099-02-02_000000"
    obs = [(t, f"2025-01-1{d}") for t in range(4) for d in range(9)]
    monkeypatch.setattr(aggregate, "R", str(tmp_path))
    for seed in aggregate.SEEDS:
        _write_run(tmp_path, ts, seed, obs)
    # Corrupt one seed's E0 target so the cross-seed target check must fire.
    bad = tmp_path / "results" / f"eda_gnn_seed123_{ts}" / "h5" / "E0" / "predictions_test.json"
    rows = json.loads(bad.read_text(encoding="utf-8"))
    rows[0]["target_raw"] = 9.99
    bad.write_text(json.dumps(rows), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="disagree on raw targets"):
        aggregate.main(ts)
