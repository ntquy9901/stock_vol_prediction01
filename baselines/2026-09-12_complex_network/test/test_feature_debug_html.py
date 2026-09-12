"""Tests for the feature-debug HTML builder's data-shaping core (keep/drop collapse + JSON loading)."""
import json

import build_feature_debug_html as B


def _screen(verdicts_by_feat):
    """Build a minimal screen dict: verdicts_by_feat maps feature -> list of per-horizon verdicts."""
    hs = ["h1", "h5"]
    horizons = {}
    for i, h in enumerate(hs):
        horizons[h] = {"features": {f: {"verdict": v[i], "group": "graph" if f == "betw" else "har"}
                                    for f, v in verdicts_by_feat.items()}}
    return {"horizons": horizons, "thresholds": {}, "train_start": "2015-01-01", "mi_subsample_cap": 40000}


def test_keep_drop_always_drop_is_dropped():
    rows = B.keep_drop_summary(_screen({"betw": ["drop (no signal)", "drop (no signal)"]}))
    assert rows[0][3] == "DROP"


def test_keep_drop_kept_if_informative_anywhere():
    rows = B.keep_drop_summary(_screen({"har_daily": ["keep", "drop (no signal)"]}))
    assert dict((r[0], r[3]) for r in rows)["har_daily"] == "KEEP"


def test_keep_drop_review_when_redundant_never_kept():
    rows = B.keep_drop_summary(_screen({"har_weekly": ["review (redundant)", "drop (no signal)"]}))
    assert dict((r[0], r[3]) for r in rows)["har_weekly"] == "REVIEW"


def test_keep_drop_orders_graph_last():
    scr = _screen({"har_daily": ["keep", "keep"], "betw": ["keep", "keep"]})
    rows = B.keep_drop_summary(scr)
    assert rows[-1][0] == "betw"            # graph group sorted last


def test_load_reads_present_files(tmp_path, monkeypatch):
    res = tmp_path
    (res / "complex_network_hose.json").write_text(json.dumps({"h1": {"GBM": 1.0}}))
    (res / "complex_network_hose_screen.json").write_text(json.dumps({"horizons": {}}))
    monkeypatch.setattr(B, "RES", res)
    hh, scr, idx = B._load("hose")
    assert hh["h1"]["GBM"] == 1.0 and scr == {"horizons": {}} and idx is None
