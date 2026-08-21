"""TDD: report-builder table + hypothesis-decision logic on a synthetic result dict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import build_report as BR  # noqa: E402


def _result(h, alpha, e7_q, e5_q, e2_q, e7_p, resid_e7):
    return {
        "horizon": h, "num_nodes": 33, "n_test": 4000, "alpha_E3": alpha,
        "lambda_E9_static": 0.0, "mcs": {"mcs_set": ["E0_HAR", "E7"]},
        "metrics": {
            "E0_HAR": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
            "E1": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.44, "rel_r2_vs_har": -0.1},
            "E2": {"mae": 3e-4, "rmse": 5e-4, "qlike": e2_q, "rel_r2_vs_har": -0.2},
            "E3_blend": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.402, "rel_r2_vs_har": -0.005},
            "E5": {"mae": 3e-4, "rmse": 5e-4, "qlike": e5_q, "rel_r2_vs_har": 0.0},
            "E7": {"mae": 3e-4, "rmse": 5e-4, "qlike": e7_q, "rel_r2_vs_har": 0.01},
            "E8": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.399, "rel_r2_vs_har": 0.0},
            "E9_gate_static": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
            "E10_gate_dyn": {"mae": 3e-4, "rmse": 5e-4, "qlike": 0.40, "rel_r2_vs_har": 0.0},
        },
        "dm_vs_har": {"E7": {"dm_qlike": {"p_value": e7_p}}, "E2": {"dm_qlike": {"p_value": 0.0}},
                      "E3_blend": {"dm_qlike": {"p_value": 0.3}}},
        "diagnostics": {"residual_r2_oos": {"E5": -0.001, "E6": -0.002, "E7": resid_e7},
                        "error_complementarity_har_vs_E2": {"pearson": 0.9}},
    }


def test_overall_table_and_delta():
    r = _result(5, 0.5, 0.38, 0.40, 0.49, 0.01, 0.02)
    tbl = BR._overall_table(r)
    assert "E0_HAR" in tbl and "E7" in tbl and "GARCH" not in tbl  # GARCH absent from this synthetic
    assert BR._delta_qlike(r, "E7") > 0                            # E7 better than HAR


def test_decisions_accept_graph_when_e7_beats_e5_and_har_sig():
    # E7 (0.38) beats E5 (0.40) and HAR (0.40) with p<0.05, resid R2>0 -> H3, H4 ACCEPT; H6 ACCEPT
    results = [_result(5, 0.5, 0.38, 0.40, 0.49, 0.01, 0.02)]
    dec = BR._decisions(results)
    assert "H4 cross-sectional/graph incremental value | ACCEPT" in dec
    assert "H3 residual learnability (any resid R2_OOS>0) | ACCEPT" in dec
    assert "H6 safe anchoring (residual <= full neural) | ACCEPT" in dec


def test_decisions_reject_graph_when_e7_ties_har():
    # E7 ties HAR (0.40), not significant (p=0.7) -> H4 REJECT
    results = [_result(1, 0.86, 0.3999, 0.3999, 0.49, 0.70, -0.0001)]
    dec = BR._decisions(results)
    assert "H4 cross-sectional/graph incremental value | REJECT" in dec
