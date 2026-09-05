"""Unit tests for the edge_hmatched -> LaTeX table builder."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_edgehm_tables as B  # noqa: E402


def _res(qlike_volga=0.40):
    return {
        "metrics": {
            "HAR": {"mse": 2e-4, "rmse": 0.014, "mae": 0.01, "qlike": 0.50, "r2": 0.30},
            "HAR-X": {"mse": 1.9e-4, "rmse": 0.0138, "mae": 0.0098, "qlike": 0.48, "r2": 0.32},
            "LSTM": {"mse": 1.8e-4, "rmse": 0.0134, "mae": 0.0095, "qlike": 0.45, "r2": 0.35},
            "VolGA": {"mse": 1.7e-4, "rmse": 0.013, "mae": 0.0092, "qlike": qlike_volga, "r2": 0.38},
        },
        "dm_date_clustered": {
            "VolGA_vs_LSTM": {"qlike": {"p_value": 0.008, "favors": "VolGA"}},
            "VolGA_vs_HAR-X": {"qlike": {"p_value": 0.001, "favors": "VolGA"}},
        },
        "fit_diagnostics": {"LSTM": {"status": "ok"}, "VolGA": {"status": "ok"}},
    }


def test_load_results_skips_missing(tmp_path):
    (tmp_path / "edgehm_vn100_h1.json").write_text(json.dumps(_res()), encoding="utf-8")
    by_h = B.load_results("vn100", [1, 5], results_dir=tmp_path)
    assert set(by_h) == {1}          # h5 file absent -> skipped


def test_fmt_and_best_value():
    assert B._fmt(None) == "--"
    assert B._fmt(0.12345) == "0.1234" or B._fmt(0.12345) == "0.1235"
    by_h = {1: _res(qlike_volga=0.40)}
    assert B._best_value(by_h, 1, "qlike", B.MODELS) == 0.40   # lower better
    assert B._best_value(by_h, 1, "r2", B.MODELS) == 0.38      # higher better


def test_latex_metric_table_bolds_best():
    by_h = {1: _res(qlike_volga=0.40)}
    tex = B.latex_metric_table("vn100", by_h, [1, 5], metric="qlike")
    assert "\\begin{tabular}" in tex and "\\toprule" in tex and "$h1$" in tex
    assert "\\textbf{0.4000}" in tex          # VolGA is best QLIKE -> bold
    assert "$h5$" not in tex                   # h5 not present -> not a column


def test_handles_missing_model():
    r = _res(); del r["metrics"]["LSTM"]          # a model absent from metrics
    tex = B.latex_metric_table("vn100", {1: r}, [1], metric="qlike")
    assert "LSTM & --" in tex                       # missing model row shows '--'
    assert B._best_value({1: {"metrics": {}}}, 1, "qlike", B.MODELS) is None   # empty -> None


def test_dm_and_fit_summary():
    by_h = {1: _res()}
    dm = B.dm_summary("vn100", by_h, [1, 5])
    assert "VolGA_vs_LSTM: p=0.008 (VolGA)" in dm
    fit = B.fit_summary("vn100", by_h, [1, 5])
    assert "LSTM=ok, VolGA=ok" in fit
