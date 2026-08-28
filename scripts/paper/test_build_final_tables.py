"""Consumer-level tests for the authoritative paper-table generator (external review F-03)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_tables as BT  # noqa: E402


def _res():
    # ensemble `metrics` and per-seed `metrics_per_seed` deliberately DIFFER so the source is testable
    return {
        "metrics": {
            "HAR": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.90, "r2": 0.10},
            "HAR-X": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.85, "r2": 0.12},
            "LSTM": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.64, "r2": 0.20},          # ensemble (lower)
            "LSTM_wGAT_vol2pk": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.60, "r2": 0.21},
            "GARCH": {"mse": 2.0, "rmse": 1.4, "mae": 1.3, "qlike": 3.5, "r2": -0.5},
        },
        "metrics_per_seed": {
            "LSTM": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.70, "qlike_std": 0.05, "r2": 0.18},
            "LSTM_wGAT_vol2pk": {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "qlike": 0.66, "qlike_std": 0.06, "r2": 0.19},
        },
    }


def test_learned_uses_per_seed_deterministic_uses_ensemble():
    res = _res()
    lstm = BT.authoritative_cell(res, "LSTM", "qlike")
    assert lstm["value"] == 0.70 and lstm["std"] == 0.05 and lstm["source"] == "per_seed"   # NOT 0.64 ensemble
    har = BT.authoritative_cell(res, "HAR", "qlike")
    assert har["value"] == 0.90 and har["std"] is None and har["source"] == "ensemble"
    garch = BT.authoritative_cell(res, "GARCH", "qlike")
    assert garch["value"] == 3.5 and garch["source"] == "ensemble"


def test_learned_missing_per_seed_raises_not_silent_fallback():
    res = _res(); del res["metrics_per_seed"]["LSTM"]     # ensemble present, per-seed gone
    with pytest.raises(KeyError, match="metrics_per_seed"):
        BT.authoritative_cell(res, "LSTM", "qlike")


def test_absent_model_returns_none():
    res = {"metrics": {"HAR": {"qlike": 0.9, "r2": 0.1}}}
    assert BT.authoritative_cell(res, "GARCH", "qlike") is None
    assert BT.authoritative_cell(res, "LSTM", "qlike") is None   # absent learned + no per-seed -> None
    # learned present in per-seed but this metric absent -> None (not a crash)
    assert BT.authoritative_cell(_res(), "LSTM", "mae") is not None
    r = _res(); del r["metrics_per_seed"]["LSTM"]["r2"]
    assert BT.authoritative_cell(r, "LSTM", "r2") is None


def test_build_tables_reads_and_hashes(tmp_path):
    for panel, h in (("vn30", 1), ("vn30", 5)):
        d = tmp_path / f"{panel}_h{h}"; d.mkdir()
        (d / "result.json").write_text(json.dumps(_res()), encoding="utf-8")
    table = BT.build_tables(tmp_path, panels=("vn30",), horizons=(1, 5, 10))   # h10 dir absent -> skipped
    assert set(table["provenance"]) == {"vn30_h1", "vn30_h5"}                   # h10 not present
    assert all(len(v) == 16 for v in table["provenance"].values())             # sha256[:16]
    lstm_rows = [r for r in table["rows"] if r["model"] == "LSTM"]
    assert lstm_rows and lstm_rows[0]["cells"]["qlike"]["value"] == 0.70        # per-seed, authoritative


def test_build_tables_skips_all_none_model_and_renders_dash(tmp_path):
    r = _res(); del r["metrics"]["GARCH"]                 # GARCH entirely absent -> all-None -> row skipped
    del r["metrics"]["HAR-X"]["qlike"]                    # HAR-X missing one metric -> that cell renders "-"
    d = tmp_path / "vn30_h1"; d.mkdir()
    (d / "result.json").write_text(json.dumps(r), encoding="utf-8")
    table = BT.build_tables(tmp_path, panels=("vn30",), horizons=(1,))
    assert not any(row["model"] == "GARCH" for row in table["rows"])   # all-None model skipped (line 73)
    assert BT._fmt(None) == "-"                                         # dash formatter (line 80)
    md = BT.render_markdown(table)
    assert "| - |" in md or " - " in md                                # HAR-X missing-qlike cell shown as "-"


def test_render_markdown_and_latex_carry_provenance_and_per_seed(tmp_path):
    d = tmp_path / "vn30_h1"; d.mkdir()
    (d / "result.json").write_text(json.dumps(_res()), encoding="utf-8")
    table = BT.build_tables(tmp_path, panels=("vn30",), horizons=(1,))
    md = BT.render_markdown(table)
    assert "Provenance" in md and "0.7000 (0.050)" in md and "0.9000" in md   # per-seed LSTM + HAR ensemble
    tex = BT.render_latex(table)
    assert "0.7000 (0.050)" in tex and "\\\\" in tex
