import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import dm_report as dm  # noqa: E402
import mz_report as mz  # noqa: E402


def _write_dump(path, targets, preds):
    rows = [{"ticker_id": i, "target_date": f"2025-01-{i + 1:02d}",
             "target_raw": float(t), "prediction_raw": float(p)}
            for i, (t, p) in enumerate(zip(targets, preds, strict=True))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_mz_recovers_perfect_line(tmp_path, monkeypatch):
    # y = 0.5 + 1.0*x exactly -> b~1, r2~1, but a!=0 so the joint test should reject.
    monkeypatch.setattr(dm, "RESULTS", tmp_path)
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(1.0, 0.3, 60)) + 0.1
    y = 0.5 + 1.0 * x
    ts, h, seeds = "T", 5, [42]
    run = tmp_path / f"trackA_ablation_h{h}_seed42_{ts}"
    _write_dump(run / "FULL" / "predictions_test.json", y, x)
    r = mz.mz(ts, h, "FULL", seeds)
    assert r["n"] == 60
    assert abs(r["b"] - 1.0) < 1e-6
    assert abs(r["a"] - 0.5) < 1e-6
    assert r["r2"] > 0.999999
