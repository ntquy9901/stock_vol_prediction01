"""Tests for collate_dm_pvalues: DM p-value extraction + paper-style formatting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collate_dm_pvalues as C  # noqa: E402


def _cell(p_lstm, favors_lstm, p_gat, favors_gat, p_graph, favors_graph):
    return {
        "metrics": {"HAR-X": {"qlike": 1.0}},
        "metrics_per_seed": {"LSTM": {"qlike": 0.9}, "LSTM_wGAT_vol2pk": {"qlike": 0.95}},
        "n_test_obs": 100,
        "n_test_dates": 20,
        "dm_date_clustered": {
            "LSTM_vs_HARX": {"qlike": {"p_value": p_lstm, "mean_diff": -0.1, "favors": favors_lstm}},
            "wGAT_vol2pk_vs_HARX": {"qlike": {"p_value": p_gat, "mean_diff": -0.05, "favors": favors_gat}},
            "wGAT_vol2pk_vs_LSTM": {"qlike": {"p_value": p_graph, "mean_diff": 0.05, "favors": favors_graph}},
        },
    }


def test_format_p_paper_style():
    assert C.format_p(1e-10) == "<0.001"
    assert C.format_p(0.0005) == "<0.001"
    assert C.format_p(0.03) == "0.030"
    assert C.format_p(0.2456) == "0.246"
    assert C.format_p(None) == "n/a"


def test_format_p_no_scientific_notation():
    # paper-writing-style: never emit e-notation
    for p in (1e-12, 3.3e-6, 0.00099):
        assert "e" not in C.format_p(p).lower()


def test_favored_label_translates_A_and_B():
    assert C._favored_label("LSTM_vs_HARX", "A") == "LSTM"
    assert C._favored_label("LSTM_vs_HARX", "B") == "HAR-X"
    assert C._favored_label("wGAT_vol2pk_vs_HARX", "A") == "LSTM+GAT"
    assert C._favored_label("wGAT_vol2pk_vs_LSTM", "B") == "LSTM"
    assert C._favored_label("LSTM_vs_HARX", "?") == "?"


def test_summarize_cell_extracts_levels_and_pvalues():
    r = C.summarize_cell(_cell(1e-10, "A", 0.04, "A", 0.5, "B"))
    assert r["qlike_HAR-X"] == 1.0
    assert r["qlike_LSTM"] == 0.9
    assert r["qlike_LSTM+GAT"] == 0.95
    assert r["n_test_obs"] == 100 and r["n_test_dates"] == 20
    assert r["p_LSTM_vs_HARX"] == 1e-10
    assert r["win_LSTM_vs_HARX"] == "LSTM"
    assert r["win_wGAT_vol2pk_vs_HARX"] == "LSTM+GAT"
    assert r["win_wGAT_vol2pk_vs_LSTM"] == "LSTM"  # favors B -> LSTM


def test_summarize_cell_handles_missing_dm():
    r = C.summarize_cell({"metrics": {}, "metrics_per_seed": {}})
    assert r["p_LSTM_vs_HARX"] is None
    assert r["win_LSTM_vs_HARX"] is None
    assert r["qlike_HAR-X"] is None


def test_collect_rows_reads_and_sorts(tmp_path):
    # two cells: parkinson/vn30_h1 and yang_zhang/hnx_h5 -> yang_zhang sorts first
    for est, cell in (("parkinson", "vn30_h1"), ("yang_zhang", "hnx_h5")):
        d = tmp_path / est / cell
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            __import__("json").dumps(_cell(0.01, "A", 0.2, "B", 0.3, "A"))
        )
    rows = C.collect_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["estimator"] == "yang_zhang"  # estimator order puts yz first
    assert rows[0]["panel"] == "hnx" and rows[0]["horizon"] == 5
    assert rows[1]["estimator"] == "parkinson"


def test_collect_rows_excludes_proxy_by_default(tmp_path):
    # yz_daily is a per-day proxy -> excluded from the paper table by default
    for est in ("yang_zhang", "yz_daily"):
        d = tmp_path / est / "vn30_h1"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            __import__("json").dumps(_cell(0.01, "A", 0.2, "B", 0.3, "A"))
        )
    default_rows = C.collect_rows(tmp_path)
    assert [r["estimator"] for r in default_rows] == ["yang_zhang"]
    all_rows = C.collect_rows(tmp_path, include_all=True)
    assert {r["estimator"] for r in all_rows} == {"yang_zhang", "yz_daily"}


def test_collect_rows_skips_corrupt_json(tmp_path):
    d = tmp_path / "parkinson" / "vn30_h1"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{ not valid json")
    assert C.collect_rows(tmp_path) == []


def test_render_markdown_groups_and_formats(tmp_path):
    d = tmp_path / "rogers_satchell" / "hnx_h1"
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        __import__("json").dumps(_cell(1e-10, "A", 0.04, "A", 0.5, "B"))
    )
    rows = C.collect_rows(tmp_path)
    md = C.render_markdown(rows)
    assert "## rogers_satchell" in md
    assert "<0.001 (LSTM)" in md
    assert "0.040 (LSTM+GAT)" in md
    assert "Total cells: 1" in md
    assert "e-0" not in md and "e-1" not in md  # no scientific notation leaked


def test_render_and_fmt_handle_missing_values(tmp_path):
    # a cell with no metrics and no dm -> levels/p-values render as 'n/a'
    d = tmp_path / "garman_klass" / "vn30_h1"
    d.mkdir(parents=True)
    (d / "result.json").write_text(__import__("json").dumps({"metrics": {}, "metrics_per_seed": {}}))
    rows = C.collect_rows(tmp_path)
    md = C.render_markdown(rows)
    assert "n/a" in md
    assert C._fmt_q(None) == "n/a"
    assert C._fmt_q(1.0) == "1.0000"


def test_write_csv_roundtrip(tmp_path):
    d = tmp_path / "parkinson" / "vn30_h1"
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        __import__("json").dumps(_cell(0.01, "B", 0.2, "B", 0.3, "A"))
    )
    rows = C.collect_rows(tmp_path)
    out = tmp_path / "dm.csv"
    C.write_csv(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "estimator,panel,horizon" in text
    assert "parkinson,vn30,1" in text
