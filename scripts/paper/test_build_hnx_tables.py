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


def test_render_est_qlike_bolds_lower_and_groups():
    def _tab(panel, h, har_q, vga_q):
        return {"rows": [
            {"panel": panel, "horizon": h, "model": "HAR-X", "cells": {"qlike": {"value": har_q, "std": None}}},
            {"panel": panel, "horizon": h, "model": "LSTM_wGAT_vol2pk", "cells": {"qlike": {"value": vga_q, "std": None}}},
        ]}
    sources = [("Parkinson", _tab("vn100", 1, 0.51, 0.55)),   # HAR-X lower
               ("Rogers--Satchell", _tab("vn100", 1, 3.85, 3.66))]  # VolGA lower
    tex = B.render_est_qlike(sources, "vn100", horizons=(1,))
    assert "Estimator & $h$ & HAR-X & VolGA" in tex
    assert r"\textbf{0.5100}" in tex   # Parkinson: HAR-X lower -> bold
    assert r"\textbf{3.6600}" in tex   # Rogers: VolGA lower -> bold
    assert tex.index("Parkinson") < tex.index("Rogers--Satchell")


def test_render_est_qlike_skips_missing_and_continuation():
    def _row(panel, h, model, q):
        return {"panel": panel, "horizon": h, "model": model, "cells": {"qlike": {"value": q, "std": None}}}
    t = {"rows": [_row("vn30", 1, "HAR-X", 0.5), _row("vn30", 1, "LSTM_wGAT_vol2pk", 0.6),
                  _row("vn30", 5, "HAR-X", 0.7), _row("vn30", 5, "LSTM_wGAT_vol2pk", 0.65)]}
    # sp500 absent from this table -> that estimator/panel yields no rows (skipped)
    tex = B.render_est_qlike([("Parkinson", t)], "vn30", horizons=(1, 5))
    assert "\n & 5 &" in tex          # continuation row for 2nd horizon
    empty = B.render_est_qlike([("Parkinson", t)], "sp500", horizons=(1,))
    assert "Parkinson" not in empty   # no rows for sp500 -> estimator block skipped
