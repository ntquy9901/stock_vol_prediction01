"""P5 runner: dm_glasso must align the glasso dump with the reused vol->PK/HAR/graph-removed dumps
and report a correct DM verdict per comparator."""
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_glasso_edge as rg  # noqa: E402


def _dump(base: Path, subdir, rung, rows):
    d = base / "results" / subdir / rung          # dm_glasso reads ROOT/"results"/...
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions_test.json").write_text(json.dumps(rows), encoding="utf-8")


def test_dm_glasso_aligns_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    h, seeds = 1, [42]
    keys = [(t % 3, f"2020-01-{t + 1:02d}") for t in range(40)]
    rng = np.random.default_rng(0)
    target = rng.uniform(0.5, 1.5, 40)

    def rows(pred):
        return [{"ticker_id": k[0], "target_date": k[1], "target_raw": float(t), "prediction_raw": float(p)}
                for k, t, p in zip(keys, target, pred)]

    glasso_pred = target + rng.normal(0, 0.02, 40)                 # glasso ~ accurate
    vol2pk_pred = target + rng.normal(0, 0.20, 40)                 # vol2pk worse
    _dump(tmp_path, "volatility_glasso_h1_seed42_GTS", "FULL", rows(glasso_pred))
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "FULL", rows(vol2pk_pred))
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "P0", rows(target + rng.normal(0, 0.15, 40)))
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "minus_graph", rows(target + rng.normal(0, 0.15, 40)))

    res = rg.dm_glasso("GTS", "VTS", h, seeds)
    by = {r["vs"]: r for r in res}
    assert set(by) == {"vol2pk_FULL", "HAR", "minus_graph"}
    assert "error" not in by["vol2pk_FULL"]
    assert by["vol2pk_FULL"]["favors"] == "glasso"                 # glasso lower QLIKE -> favors glasso


def test_dm_glasso_flags_misalignment(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    g = [{"ticker_id": 0, "target_date": "2020-01-01", "target_raw": 1.0, "prediction_raw": 1.0}]
    v = [{"ticker_id": 9, "target_date": "2021-09-09", "target_raw": 2.0, "prediction_raw": 2.0}]
    _dump(tmp_path, "volatility_glasso_h1_seed42_GTS", "FULL", g)
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "FULL", v)
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "P0", v)
    _dump(tmp_path, "volatility_ablation_h1_seed42_VTS", "minus_graph", v)
    res = {r["vs"]: r for r in rg.dm_glasso("GTS", "VTS", 1, [42])}
    assert res["vol2pk_FULL"].get("error") == "misaligned"
