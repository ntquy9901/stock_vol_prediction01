import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import dm_report as dm  # noqa: E402


def _write_dump(path, targets, preds):
    rows = [{"ticker_id": i, "target_date": f"2025-01-{i + 1:02d}",
             "target_raw": float(t), "prediction_raw": float(p)}
            for i, (t, p) in enumerate(zip(targets, preds, strict=True))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_dm_pair_favors_closer_model(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "RESULTS", tmp_path)
    rng = np.random.default_rng(0)
    targets = np.abs(rng.normal(1.0, 0.2, 40)) + 0.1
    ts, h, seeds = "T", 5, [42]
    run = tmp_path / f"volatility_ablation_h{h}_seed42_{ts}"
    # FULL close to target, HAR far -> DM favours A=FULL (negative mean_diff)
    _write_dump(run / "FULL" / "predictions_test.json", targets, targets + rng.normal(0.0, 0.02, 40))
    _write_dump(run / "P0" / "predictions_test.json", targets, targets + rng.normal(0.3, 0.1, 40))
    r = dm.dm_pair(ts, h, "FULL", "HAR", seeds)
    assert r["n"] == 40
    assert r["favors"] == "A"          # FULL closer -> lower QLIKE
    assert r["mean_diff"] < 0
    # all-metric support: FULL is closer on squared error and absolute error too
    for loss in ("se", "ae"):
        rl = dm.dm_pair(ts, h, "FULL", "HAR", seeds, loss=loss)
        assert rl["favors"] == "A" and rl["n"] == 40


def test_ensemble_averages_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "RESULTS", tmp_path)
    targets = np.array([1.0, 2.0, 3.0])
    ts, h = "T", 5
    for s, bias in ((42, 0.0), (123, 0.4)):
        run = tmp_path / f"volatility_ablation_h{h}_seed{s}_{ts}"
        _write_dump(run / "FULL" / "predictions_test.json", targets, targets + bias)
    keys, tgt, pred = dm._ensemble(ts, h, "FULL", [42, 123])
    assert len(keys) == 3
    assert np.allclose(tgt, targets)
    assert np.allclose(pred, targets + 0.2)   # mean of +0.0 and +0.4
