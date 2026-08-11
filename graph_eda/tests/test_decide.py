"""Unit tests for the verdict decision logic (all four conclusion branches)."""

from graph_eda.run_eda import _decide


def _ev(**over):
    base = {
        "market_adj_retained_frac": 0.23,
        "gate6_C_beats_A": False,
        "gate7_C_beats_B": False,
        "pk_null_obs": 0.38,
        "pk_null_p95": 0.02,
        "neighbor_jaccard_k5": 0.39,
        "asym_abs_mean": 0.02,
        "pk_leadlag1_offdiag_max": 0.49,
    }
    base.update(over)
    return base


def test_conclusion_c_market_dominates():
    d = _decide(_ev())  # retained < 0.35 and not gate7
    assert d["conclusion"] == "C" and d["use_gnn"] is False


def test_conclusion_d_weak_no_null_separation():
    d = _decide(_ev(pk_null_obs=0.01, pk_null_p95=0.02, market_adj_retained_frac=0.5))
    assert d["conclusion"] == "D" and d["use_gnn"] is False


def test_conclusion_a_dynamic_gnn():
    d = _decide(
        _ev(gate7_C_beats_B=True, market_adj_retained_frac=0.6, neighbor_jaccard_k5=0.4)
    )
    assert d["conclusion"] == "A" and d["use_gnn"] is True
    assert d["graph_type"] in {"dynamic", "dynamic_directed"}


def test_conclusion_b_static_gnn():
    d = _decide(
        _ev(gate7_C_beats_B=True, market_adj_retained_frac=0.6, neighbor_jaccard_k5=0.9)
    )
    assert d["conclusion"] == "B" and d["graph_type"] == "static"


def test_fallthrough_structure_but_no_oos_value():
    # gates fail but structure survives market adjustment (>=0.35) -> C, medium
    d = _decide(_ev(market_adj_retained_frac=0.5))
    assert d["conclusion"] == "C" and d["confidence"] == "medium"
