"""TDD: report-builder tables + hypothesis decisions (date-clustered DM + paired graph attribution)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import build_report as BR  # noqa: E402


def _result(h, alpha, e7_q, e6_q, e5_q, e2_q, p_dc):
    """Synthetic result.json-like dict. p_dc = date-clustered DM p-value used for E6/E7/E3 vs HAR."""
    def dm(p):
        return {"dm_qlike": {"p_value": 1e-9}, "date_clustered": {"p_value": p}}
    return {
        "horizon": h, "num_nodes": 33, "n_test": 4000, "alpha_E3": alpha, "dataset": "vn30",
        "lambda_E9_static": 0.0, "mcs": {"mcs_set": ["E0_HAR", "E6", "E7"]},
        "metrics": {
            "E0_HAR": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
            "E1": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.44, "rel_r2_vs_har": -0.1},
            "E2": {"mae": 3e-4, "rmse": 5e-4, "qlike": e2_q, "rel_r2_vs_har": -0.2},
            "E3_blend": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.39, "rel_r2_vs_har": 0.02},
            "E5": {"mae": 3e-4, "rmse": 5e-4, "qlike": e5_q, "rel_r2_vs_har": 0.0},
            "E6": {"mae": 3e-4, "rmse": 5e-4, "qlike": e6_q, "rel_r2_vs_har": 0.03},
            "E7": {"mae": 3e-4, "rmse": 5e-4, "qlike": e7_q, "rel_r2_vs_har": 0.01},
            "E8": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.399, "rel_r2_vs_har": 0.0},
            "E9_gate_static": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
            "E10_gate_dyn": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
        },
        "dm_vs_har": {"E6": dm(p_dc), "E7": dm(p_dc), "E3_blend": dm(p_dc), "E2": dm(0.0)},
        "diagnostics": {"residual_r2_oos": {"E5": -0.001, "E6": 0.03, "E7": 0.02},
                        "error_complementarity_har_vs_E2": {"pearson": 0.9}},
    }


def test_overall_table_and_delta():
    r = _result(5, 0.5, 0.38, 0.37, 0.40, 0.49, 0.01)
    tbl = BR._overall_table(r)
    assert "E0_HAR" in tbl and "E6" in tbl
    assert BR._delta_qlike(r, "E6") > 0                            # E6 better than HAR


def test_dmp_reads_date_clustered():
    r = _result(5, 0.5, 0.38, 0.37, 0.40, 0.49, 0.023)
    assert BR._dmp(r, "E6") == 0.023                              # date-clustered, not the row-level 1e-9


def test_h4_accept_when_e6_beats_e5_paired_and_har(monkeypatch):
    # E6 (0.37) beats E5 (0.40) and HAR (0.40); paired E6-vs-E5 significant; vs-HAR date-clustered sig
    monkeypatch.setattr(BR, "paired_dm_from_rows",
                        lambda ds, h, a, b, floor=1e-8: {"p_value": 0.01, "favors": a, "mean_diff": -0.01})
    dec = BR._decisions([_result(22, 0.3, 0.37, 0.36, 0.40, 0.49, 0.001)])
    assert "H4 cross-sectional/graph incremental value | ACCEPT" in dec
    assert "H3 residual learnability (any resid R2_OOS>0) | ACCEPT" in dec


def test_h4_reject_when_e6_not_better_than_e5(monkeypatch):
    # paired E6-vs-E5 favors E5 (graph no better than no-graph) -> H4 REJECT even if vs-HAR sig
    monkeypatch.setattr(BR, "paired_dm_from_rows",
                        lambda ds, h, a, b, floor=1e-8: {"p_value": 0.9, "favors": b, "mean_diff": 0.01})
    dec = BR._decisions([_result(1, 0.86, 0.399, 0.399, 0.399, 0.49, 0.001)])
    assert "H4 cross-sectional/graph incremental value | REJECT" in dec
