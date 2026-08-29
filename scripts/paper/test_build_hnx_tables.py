"""Tests for build_hnx_tables.render_panel: model/metric subsetting + no-R2 header + bolding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hnx_tables as B  # noqa: E402


def _table():
    # two models at h1; LSTM+GAT lower QLIKE, HAR-X lower MSE
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
    assert "HAR-X" in tex and "LSTM+GAT" in tex
    # GARCH not in this fixture -> row absent, no crash
    assert "LSTM &" not in tex  # plain no-graph LSTM not requested by MAIN_MODELS


def test_bold_marks_best_per_metric():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    # LSTM+GAT has the lower QLIKE (1.81) -> bolded; HAR-X lower MSE (scaled 10.000) -> bolded
    assert r"\textbf{1.8100" in tex
    assert r"\textbf{10.000}" in tex


def test_qlike_std_rendered():
    tex = B.render_panel(_table(), "hnx", B.MAIN_MODELS)
    assert r"\,$\pm$.004" in tex


def test_absent_panel_yields_empty_body():
    tex = B.render_panel(_table(), "vn30", B.MAIN_MODELS)
    assert r"\toprule" in tex and "HAR-X" not in tex  # header only, no rows


def test_est_models_drops_no_graph_lstm():
    # estimator tables compare only HAR-X and LSTM+GAT -- the plain LSTM row must be absent
    t = _table()
    t["rows"].append({"panel": "hnx", "horizon": 1, "model": "LSTM",
                      "cells": {"mse": {"value": 1.3e-6, "std": 0.01}, "qlike": {"value": 1.85, "std": 0.01},
                                "rmse": {"value": 1.15e-3, "std": None}, "mae": {"value": 6.45e-4, "std": None}}})
    tex = B.render_panel(t, "hnx", B.EST_MODELS)
    assert "HAR-X" in tex and "LSTM+GAT" in tex
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
