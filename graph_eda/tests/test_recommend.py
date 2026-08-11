"""Unit tests for _recommend_config lift-evidence branches."""

import pandas as pd

from graph_eda.run_eda import _recommend_config

_EV = {"neighbor_jaccard_k5": 0.4}


def _edge(gain_market, sign_p, winrate, name="edge_vol2pk_dir"):
    return pd.DataFrame(
        [{
            "edge_definition": name,
            "gain_over_har_pct": gain_market + 0.1,
            "gain_over_market_pct": gain_market,
            "winrate_over_market": winrate,
            "sign_p_over_market": sign_p,
        }]
    )


def _node():
    return pd.DataFrame(
        [{"node_feature": "volume_zscore_20", "rmse_gain_pct": 0.5, "win_rate": 0.9, "sign_p": 1e-6}]
    )


def test_recommend_beats_when_significant():
    cfg = _recommend_config(_EV, _node(), _edge(0.5, 0.01, 0.7))
    assert cfg["lift_evidence"] == "beats"
    assert cfg["likely_beats_har_under_dm"] is True
    assert "volume_zscore_20" in cfg["node_features"]
    assert cfg["directed"] is True


def test_recommend_suggestive_when_positive_but_not_significant():
    cfg = _recommend_config(_EV, _node(), _edge(0.2, 0.08, 0.67))
    assert cfg["lift_evidence"] == "suggestive"
    assert cfg["likely_beats_har_under_dm"] is False


def test_recommend_none_when_no_gain():
    cfg = _recommend_config(_EV, _node(), _edge(-1.0, 0.02, 0.3))
    assert cfg["lift_evidence"] == "none"


def test_recommend_handles_empty_edge_rank():
    cfg = _recommend_config(_EV, pd.DataFrame(), pd.DataFrame())
    assert cfg["edge_type"] == "edge_pkcorr_abs"
    assert cfg["node_features"] == ["pk_daily", "pk_weekly", "pk_monthly"]
