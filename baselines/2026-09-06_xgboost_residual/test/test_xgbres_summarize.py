"""Pure table builders + GO/NO-GO tally over loaded result dicts."""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import summarize_xgb as S  # noqa: E402


def _fake(qbase, p_base, fav_base="A"):
    def mm(q):
        return {"qlike": q, "qlike_nonlock": q - 0.01, "qlike_lockonly": 10.0,
                "lock_qlike_share": 0.07, "n_lock": 50, "qlike_excl_top_dates": q - 0.02}
    return {
        "metrics": {"HAR": mm(0.49), "HAR-X": mm(0.48), "XGB_direct": mm(0.70),
                    "XGB_resid_base": mm(qbase), "XGB_resid_market": mm(qbase - 0.001),
                    "XGB_resid_graph": mm(qbase)},
        "dm_date_clustered": {
            "XGB_resid_base_vs_HAR-X": {"qlike": {"p_value": p_base, "favors": fav_base}},
            "XGB_resid_market_vs_base": {"qlike": {"p_value": 0.3, "favors": "A"}},
            "XGB_resid_graph_vs_market": {"qlike": {"p_value": 0.5, "favors": "B"}}},
        "win_rates_vs_harx": {"XGB_resid_base": {"ticker_win_rate": 0.4, "date_win_rate": 0.3}},
        "reference_models": {"VolGA_qlike": 0.47}}


def test_load_results(tmp_path):
    (tmp_path / "xgb_vn30_h1.json").write_text(json.dumps(_fake(0.47, 0.9)))
    r = S.load_results(tmp_path, markets=("vn30",), horizons=(1, 5))
    assert set(r) == {("vn30", 1)}                       # h5 absent -> skipped


def test_summary_and_ablation_and_lock_tables_render():
    res = {("vn30", 1): _fake(0.479, 0.79), ("vn100", 1): _fake(0.49, 0.4)}
    s = S.summary_table(res)
    assert "Panel" in s and "vn30" in s and "0.4790" in s and "VolGA(ref)" in s
    a = S.ablation_table(res)
    assert "market vs base" in a
    lk = S.lock_table(res)
    assert "lock QLIKE share" in lk and "0.070" in lk


def test_go_no_go_negative_when_not_significant():
    res = {("vn30", 1): _fake(0.479, 0.79), ("vn100", 1): _fake(0.49, 0.4)}   # lower once, never sig
    g = S.go_no_go(res)
    assert g["verdict"] == "NO-GO" and g["n_dm_significant_better"] == 0


def test_go_no_go_go_when_lower_and_significant():
    res = {("vn30", 1): _fake(0.47, 0.01, "A"), ("vn100", 1): _fake(0.46, 0.02, "A")}
    g = S.go_no_go(res)
    assert g["verdict"] == "GO" and g["n_lower_qlike_than_harx"] == 2 and g["n_dm_significant_better"] == 2
