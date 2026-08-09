"""Aggregator (``docs/reports/ladder_consistent_dump.py``) horizon parametrization.

The dump reads per-seed run dirs ``results/ladder_consistent_seed{seed}_{ts}/h{horizon}`` and stamps
the horizon into the canonical summary; the Diebold-Mariano test is run at the requested horizon
(HAC truncation lag ``horizon - 1``). This test fabricates a minimal 2-seed tree and asserts the
subdir routing + horizon stamping for a non-default horizon (22).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _load_dump():
    path = _ROOT / "docs" / "reports" / "ladder_consistent_dump.py"
    spec = importlib.util.spec_from_file_location("ladder_consistent_dump", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _metric_block(scale: float) -> dict[str, float]:
    return {m: float(scale + i * 0.01) for i, m in enumerate(_METRICS)}


def _rung(scale: float) -> dict:
    return {"validation_metrics": _metric_block(scale), "test_metrics": _metric_block(scale + 0.5)}


def _predictions(offset: float) -> list[dict]:
    rows = []
    for tid in (0, 1):
        for day in range(1, 6):
            rows.append({"ticker_id": tid, "target_date": f"2021-01-0{day}",
                         "prediction_raw": 0.02 + offset + 0.001 * day,
                         "target_raw": 0.02 + 0.0015 * day})
    return rows


def _write_seed(base: Path, seed: int, horizon: int) -> None:
    run = base / "results" / f"ladder_consistent_seed{seed}_dumpts" / f"h{horizon}"
    ladder_metrics = {
        "seed": seed, "horizon": horizon,
        "rungs": {rung: _rung(0.1 * i + 0.001 * seed * (i + 1))
                  for i, rung in enumerate(("P0", "P1", "P2", "P3", "G1"))},
        "nesting_check": {"graph_off_readout_determinism_max_abs_diff": 0.0,
                          "graph_effect_val_mean_abs_pred_diff_raw": 1e-5,
                          "graph_effect_val_max_abs_pred_diff_raw": 1e-3, "n_val_obs": 10},
        "adjacency": {"mode": "knn", "top_k": 8},
        "edge_density": {"avg_offdiag_nonzeros_per_present_row": 5.8},
        "snapshot_count": 3,
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "ladder_metrics.json").write_text(json.dumps(ladder_metrics), encoding="utf-8")
    for rung, offset in (("P3", 0.002), ("G1", 0.0015)):
        (run / rung).mkdir(parents=True, exist_ok=True)
        (run / rung / "predictions.json").write_text(json.dumps(_predictions(offset)), encoding="utf-8")
        (run / rung / "predictions_test.json").write_text(json.dumps(_predictions(offset + 0.5)), encoding="utf-8")


def test_dump_reads_horizon_subdir_and_stamps_horizon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = _load_dump()
    monkeypatch.setattr(dump, "_ROOT", tmp_path)
    for seed in (42, 123):
        _write_seed(tmp_path, seed, horizon=22)

    prefix = tmp_path / "ladder_consistent_h22_dumpts"
    dump.main("dumpts", str(prefix), horizon=22)

    summary = json.loads((prefix.with_suffix(".json")).read_text(encoding="utf-8"))
    assert summary["horizon"] == 22
    assert summary["seeds_completed"] == [42, 123]
    assert set(summary["rung_metrics"]["val"]) == {"P0", "P1", "P2", "P3", "G1"}
    md = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert "(h22)" in md
    assert "horizon 22" in md
