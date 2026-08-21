"""TDD: row-aligned prediction export + feature-availability manifest (plan sections 21, 26)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import io_preds  # noqa: E402


def test_export_roundtrip(tmp_path):
    rows = {
        "ticker": np.array(["FPT", "VNM", "FPT"]),
        "target_date": np.array(["2024-01-05", "2024-01-05", "2024-01-08"]),
        "horizon": np.array([5, 5, 5]),
        "y_true": np.array([0.02, 0.03, 0.021]),
        "pred_HAR": np.array([0.019, 0.028, 0.022]),
        "pred_E8": np.array([0.0195, 0.029, 0.0215]),
        "alpha": np.array([np.nan, np.nan, np.nan]),
        "fold": np.array([0, 0, 0]),
        "seed": np.array([42, 42, 42]),
    }
    p = tmp_path / "preds.csv"
    io_preds.export_predictions(p, rows)
    back = io_preds.load_predictions(p)
    assert list(back["ticker"]) == ["FPT", "VNM", "FPT"]
    assert np.allclose(back["pred_E8"].astype(float), rows["pred_E8"])
    assert len(back) == 3


def test_export_requires_equal_length(tmp_path):
    rows = {"ticker": np.array(["A", "B"]), "y_true": np.array([1.0])}
    try:
        io_preds.export_predictions(tmp_path / "x.csv", rows)
        assert False, "expected ValueError on ragged columns"
    except ValueError:
        pass


def test_manifest_written(tmp_path):
    entries = [
        {"feature": "har_daily", "event_time": "t", "available_time": "close(t)",
         "lookback": 1, "fitted_on": "n/a", "used_for": "HAR,LSTM,GAT"},
        {"feature": "glasso_edges", "event_time": "<=train_end", "available_time": "train",
         "lookback": "train", "fitted_on": "train", "used_for": "GAT"},
    ]
    p = tmp_path / "manifest.csv"
    io_preds.write_manifest(p, entries)
    text = p.read_text(encoding="utf-8")
    assert "feature,event_time,available_time,lookback,fitted_on,used_for" in text
    assert "glasso_edges" in text and "har_daily" in text
