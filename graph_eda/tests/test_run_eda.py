"""Integration/smoke test for the run_eda orchestrator (CLAUDE.md: test run_*() runners).

Runs the full pipeline on the real 33-ticker universe into a tmp dir with reduced
permutation counts, and asserts the deliverables and a coherent verdict are produced.
"""

import json

import pytest

from graph_eda import run_eda


@pytest.mark.smoke
def test_run_eda_end_to_end(tmp_path):
    out = run_eda.main(
        price_dir="data/raw/prices",
        out_base=tmp_path,
        n_perm_corr=5,
        n_perm_ll=5,
    )
    ev, verdict = out["ev"], out["verdict"]

    # verdict is well-formed and consistent
    assert verdict["conclusion"] in {"A", "B", "C", "D"}
    assert isinstance(verdict["use_gnn"], bool)

    # deliverables written
    assert (tmp_path / "reports" / "EDA_GRAPH_REPORT.md").exists()
    assert (tmp_path / "tables" / "pk_corr_pearson.csv").exists()
    assert (tmp_path / "tables" / "predictive_baselines_h1.csv").exists()
    assert (tmp_path / "tables" / "node_feature_ranking_h1.csv").exists()
    assert (tmp_path / "tables" / "edge_definition_ranking_h1.csv").exists()
    assert (tmp_path / "figures" / "16_raw_vs_market_adjusted_pk_corr.png").exists()
    rec = json.loads((tmp_path / "graph_recommendation.json").read_text())
    assert rec["use_gnn"] == verdict["use_gnn"]

    # a concrete GNN config is always produced (user will build + DM-test it)
    cfg = rec["recommended_config"]
    assert cfg["top_k"] == 5
    assert len(cfg["node_features"]) >= 3          # HAR core at minimum
    assert len(cfg["edge_features"]) >= 1
    assert isinstance(cfg["likely_beats_har_under_dm"], bool)

    # core evidence keys present and in-range
    assert 0.0 <= ev["market_adj_retained_frac"] <= 1.5
    assert ev["pred_h1_n_stocks"] == 33
    for key in ("gate6_C_beats_A", "gate7_C_beats_B"):
        assert isinstance(ev[key], bool)
