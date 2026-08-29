"""Tests for build_hnx_tables.render_panel: model/metric subsetting + no-R2 header + bolding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hnx_tables as B  # noqa: E402


def _table():
    # two models at h1; VolGA lower QLIKE, HAR-X lower MSE
    return {"rows": [
        {"panel": "hnx", "horizon": 1, "model": "HAR-X",
         "cells": {"mse": {"value": 1.0e-6, "std": None}, "qlike": {"value": 1.90, "std": None},
                   "rmse": {"value": 1.1e-3, "std": None}, "mae": {"value": 6.5e-4, "std": None},
                   "r2": {"value": 0.21, "std": None}}},
        {"panel": "hnx", "horizon": 1, "model": "LSTM_wGAT_vol2pk",
         "cells": {"mse": {"value": 1.2e-6, "std": None}, "qlike": {"value": 1.81, "std": 0.004},
                   "rmse": {"value": 1.2e-3, "std": None}, "mae": {"value": 6.4e-4, "std": None},
                   "r2": {"value": 0.23, "std": None}}},
    ]}


def test_header_drops_r2():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    assert "MSE & RMSE & MAE & QLIKE" in tex
    assert "R^2" not in tex and "R2" not in tex


def test_column_spec_matches_four_metrics():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    assert r"\begin{tabular}{llcccc}" in tex  # 2 label + 4 metric columns


def test_model_subset_includes_only_requested():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    assert "HAR-X" in tex and "VolGA" in tex
    # GARCH not in this fixture -> row absent, no crash
    assert "LSTM &" not in tex  # plain no-graph LSTM not requested by MAIN_MODELS


def test_bold_marks_best_per_metric():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    # VolGA has the lower QLIKE (1.81) -> bolded; HAR-X lower MSE (scaled 10.000) -> bolded
    assert r"\textbf{1.8100" in tex
    assert r"\textbf{10.000}" in tex


def test_qlike_std_rendered():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    assert r"\,$\pm$.004" in tex


def test_absent_panel_yields_empty_body():
    tex = B.render_panel(_table(), "vn30", B.MAIN_MODELS)
    assert r"\toprule" in tex and "HAR-X" not in tex  # header only, no rows


def test_est_models_drops_no_graph_lstm():
    # estimator tables compare only HAR-X and VolGA -- the plain LSTM row must be absent
    t = _table()
    t["rows"].append({"panel": "hnx", "horizon": 1, "model": "LSTM",
                      "cells": {"mse": {"value": 1.3e-6, "std": 0.01}, "qlike": {"value": 1.85, "std": 0.01},
                                "rmse": {"value": 1.15e-3, "std": None}, "mae": {"value": 6.45e-4, "std": None}}})
    tex = B.render_panel(t, "hnx", B.EST_MODELS)
    assert "HAR-X" in tex and "VolGA" in tex
    assert "& LSTM &" not in tex  # the no-graph LSTM row is excluded


def test_multi_horizon_inserts_midrule_between_blocks():
    t = _table()
    # add an h5 block so the second block triggers the between-block \midrule
    t["rows"].append({"panel": "hnx", "horizon": 5, "model": "HAR-X",
                      "cells": {"mse": {"value": 1.5e-6, "std": None}, "qlike": {"value": 1.94, "std": None},
                                "rmse": {"value": 1.2e-3, "std": None}, "mae": {"value": 7.1e-4, "std": None}}})
    tex = B.render_panel(t, "hnx", B.MAIN_MODELS, horizons=(1, 5))
    assert tex.count(r"\midrule") == 2  # header \midrule + one between-block \midrule
    assert "5 & HAR-X" in tex


def test_missing_metric_renders_dash():
    # a model whose cell lacks 'mse' -> that column renders '-'
    t = {"rows": [{"panel": "hnx", "horizon": 1, "model": "HAR-X",
                   "cells": {"qlike": {"value": 1.9, "std": None}, "rmse": {"value": 1.1e-3, "std": None},
                             "mae": {"value": 6.5e-4, "std": None}}}]}
    tex = B.render_panel(t, "hnx", (("HAR-X", "HAR-X"),))
    assert "1 & HAR-X & - &" in tex  # MSE column is a dash


def test_fmt_p_dm():
    assert B._fmt_p(1e-10) == r"$<$0.001"
    assert B._fmt_p(0.03) == "0.030"
    assert B._fmt_p(None) == "-"


def test_render_dm_voltg_hnx_first_and_voltg_vs_harx(tmp_path):
    import json
    def _cell(favq, pq, fava, pa):
        d = {"dm_date_clustered": {"wGAT_vol2pk_vs_HARX": {
            "qlike": {"favors": favq, "p_value": pq}, "ae": {"favors": fava, "p_value": pa}}}}
        return json.dumps(d)
    for panel in ("hnx", "vn30"):
        (tmp_path / f"{panel}_h1").mkdir(parents=True)
        (tmp_path / f"{panel}_h1" / "result.json").write_text(_cell("A", 1e-10, "B", 0.2))
    tex = B.render_dm_voltg(tmp_path, panels=("hnx", "vn30"), horizons=(1,))
    assert "Panel & $h$ & QLIKE & MAE" in tex
    # HNX must appear before VN30
    assert tex.index("HNX") < tex.index("VN30")
    assert r"\textbf{$<$0.001 (VolGA)}" in tex   # QLIKE favours VolGA, significant
    assert "0.200 (HAR-X)" in tex                 # MAE favours HAR-X, not significant


def test_render_dm_voltg_covers_branches(tmp_path):
    import json
    (tmp_path / "hnx_h1").mkdir(parents=True)
    (tmp_path / "hnx_h1" / "result.json").write_text(json.dumps(
        {"dm_date_clustered": {"wGAT_vol2pk_vs_HARX": {
            "qlike": {"favors": "A", "p_value": 1e-10}, "ae": {"favors": "B", "p_value": 0.2}}}}))
    (tmp_path / "hnx_h5").mkdir()
    (tmp_path / "hnx_h5" / "result.json").write_text(json.dumps(
        {"dm_date_clustered": {"wGAT_vol2pk_vs_HARX": {
            "qlike": {"favors": "A", "p_value": 0.03}, "ae": {"favors": "A", "p_value": None}}}}))
    # vn30 has NO result files -> panel produces no rows and is skipped
    tex = B.render_dm_voltg(tmp_path, panels=("hnx", "vn30"), horizons=(1, 5))
    assert "HNX" in tex and "VN30" not in tex          # empty panel skipped (missing files + no-rows branches)
    assert "\n & 5 &" in tex                            # continuation row for the 2nd horizon
    assert "& - \\\\" in tex                            # missing ae p-value rendered as '-'


def _mkrow(panel, h, model, mse, qlike, std=None):
    return {"panel": panel, "horizon": h, "model": model,
            "cells": {"mse": {"value": mse, "std": None}, "rmse": {"value": mse * 10, "std": None},
                      "mae": {"value": mse * 5, "std": None}, "qlike": {"value": qlike, "std": std}}}


def test_render_est_allmarkets_hnx_first_four_metrics_and_labels():
    t = {"rows": [
        _mkrow("hnx", 1, "HAR-X", 1.4e-6, 1.87), _mkrow("hnx", 1, "LSTM_wGAT_vol2pk", 1.37e-6, 1.81, std=0.004),
        _mkrow("vn30", 1, "HAR-X", 1.0e-6, 0.51), _mkrow("vn30", 1, "LSTM_wGAT_vol2pk", 1.2e-6, 0.55),
    ]}
    tex = B.render_est_allmarkets(t, panels=("hnx", "vn30"), horizons=(1,))
    assert "$h$ & Model & MSE & RMSE & MAE & QLIKE" in tex   # all four metrics
    assert r"\multicolumn{6}{l}{\textbf{HNX}}" in tex        # market label row
    assert tex.index("HNX") < tex.index("VN30")             # HNX first
    assert r"\textbf{1.8100" in tex                          # VolGA lower QLIKE on HNX -> bold
    assert r"\,$\pm$.004" in tex                             # per-seed std kept


def test_render_est_allmarkets_skips_absent_market():
    t = {"rows": [_mkrow("hnx", 1, "HAR-X", 1e-6, 1.0), _mkrow("hnx", 1, "LSTM_wGAT_vol2pk", 1e-6, 0.9)]}
    tex = B.render_est_allmarkets(t, panels=("hnx", "sp500"), horizons=(1,))
    assert "HNX" in tex and "S\&P 500" not in tex           # sp500 has no rows -> block skipped


def test_render_est_allmarkets_missing_metric_dash():
    # a model row missing 'mse' -> that column renders '-'
    t = {"rows": [
        {"panel": "hnx", "horizon": 1, "model": "HAR-X",
         "cells": {"rmse": {"value": 1.1e-3, "std": None}, "mae": {"value": 6.5e-4, "std": None},
                   "qlike": {"value": 1.9, "std": None}}},
        {"panel": "hnx", "horizon": 1, "model": "LSTM_wGAT_vol2pk",
         "cells": {"mse": {"value": 1.0e-6, "std": None}, "rmse": {"value": 1.2e-3, "std": None},
                   "mae": {"value": 6.4e-4, "std": None}, "qlike": {"value": 1.8, "std": None}}},
    ]}
    tex = B.render_est_allmarkets(t, panels=("hnx",), horizons=(1,))
    assert "1 & HAR-X & - &" in tex   # missing MSE -> dash
