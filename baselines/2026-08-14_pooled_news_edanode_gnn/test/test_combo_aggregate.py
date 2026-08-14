"""Unit test for the combo DM aggregator on synthetic per-observation dumps (fast, CPU)."""

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import combo_aggregate as agg  # noqa: E402

_TS = "2026-01-01_000000"
_SIX = {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5, "qlike": 0.5, "directional_accuracy": 48.0}


def _write_dump(path: Path, targets, preds):
    rows = [{"ticker_id": i, "target_date": f"2025-01-{i + 1:02d}",
             "target_raw": float(t), "prediction_raw": float(p)}
            for i, (t, p) in enumerate(zip(targets, preds, strict=True))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_aggregate_dm_favours_the_more_accurate_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agg, "ROOT", tmp_path)
    rng = np.random.default_rng(0)
    targets = np.abs(rng.normal(1.0, 0.2, 40)) + 0.1
    for seed in agg.SEEDS:
        run = tmp_path / "results" / f"pooled_news_edanode_seed{seed}_{_TS}" / f"h{agg.HORIZON}"
        # noisy predictions (positive loss-diff variance so DM is defined); G1 far, P0/P3 near
        _write_dump(run / "G1" / "predictions_test.json", targets, targets + rng.normal(0.4, 0.15, 40))
        _write_dump(run / "P0" / "predictions_test.json", targets, targets + rng.normal(0.0, 0.03, 40))
        _write_dump(run / "P3" / "predictions_test.json", targets, targets + rng.normal(0.0, 0.05, 40))
        ladder = {"rungs": {r: {"validation_metrics": _SIX, "test_metrics": _SIX}
                            for r in agg.RUNGS}}
        (run / "ladder_metrics.json").write_text(json.dumps(ladder), encoding="utf-8")

    agg.main(_TS)
    summary = json.loads((tmp_path / "results" / f"pooled_news_edanode_{_TS}_summary.json").read_text())
    dm = summary["diebold_mariano"]
    assert set(dm) == {"G1_vs_P0", "G1_vs_P3", "P3_vs_P0"}
    # G1 (A) is worse than P0 (B) -> favours B on both losses, n = 40
    assert dm["G1_vs_P0"]["qlike"]["favors"] == "B"
    assert dm["G1_vs_P0"]["se"]["favors"] == "B"
    assert dm["G1_vs_P0"]["n"] == 40
    # metric table for all five rungs is emitted
    out = capsys.readouterr().out
    for rung_label in ("P0 HAR", "P1 price5", "P2 +news", "G1 combo"):
        assert rung_label in out


def test_aggregate_records_degenerate_dm_pair_without_aborting(tmp_path, monkeypatch):
    """A zero-variance loss differential must be recorded as an error, not crash the summary."""

    monkeypatch.setattr(agg, "ROOT", tmp_path)
    rng = np.random.default_rng(1)
    targets = np.abs(rng.normal(1.0, 0.2, 30)) + 0.1
    for seed in agg.SEEDS:
        run = tmp_path / "results" / f"pooled_news_edanode_seed{seed}_{_TS}" / f"h{agg.HORIZON}"
        # P0 == P3 (identical predictions) -> P3_vs_P0 loss diff is 0 everywhere -> DM undefined
        _write_dump(run / "G1" / "predictions_test.json", targets, targets + rng.normal(0.3, 0.1, 30))
        _write_dump(run / "P0" / "predictions_test.json", targets, targets + 0.02)
        _write_dump(run / "P3" / "predictions_test.json", targets, targets + 0.02)
        ladder = {"rungs": {r: {"validation_metrics": _SIX, "test_metrics": _SIX}
                            for r in agg.RUNGS}}
        (run / "ladder_metrics.json").write_text(json.dumps(ladder), encoding="utf-8")

    agg.main(_TS)  # must not raise
    dm = json.loads((tmp_path / "results" / f"pooled_news_edanode_{_TS}_summary.json").read_text())[
        "diebold_mariano"]
    assert "error" in dm["P3_vs_P0"]["qlike"] and "error" in dm["P3_vs_P0"]["se"]
    assert "favors" in dm["G1_vs_P0"]["qlike"]  # the well-defined pair still computed
