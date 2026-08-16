"""Tests for the graph/news contribution HTML viz (pure helpers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_news_contribution as g  # noqa: E402


def test_verdict_class_helps_hurts_ns():
    # negative dm + significant => component helps (removing it worsens) -> "helps"
    assert g.verdict("graph", dm=-3.0, p=0.01) == "helps"
    # positive dm + significant => removing improves => component hurts
    assert g.verdict("news", dm=+4.0, p=0.001) == "hurts"
    # not significant => "none"
    assert g.verdict("gate", dm=+1.2, p=0.25) == "none"
    assert g.verdict("graph", dm=-3.0, p=0.20) == "none"


def test_bar_geometry_center_and_direction():
    # delta 0 -> zero-width bar at center; negative -> left(better), positive -> right(worse)
    z = g.bar_geometry(0.0, scale=0.05)
    assert z["width_pct"] == 0.0
    left = g.bar_geometry(-0.02, scale=0.05)
    assert left["side"] == "better" and 0 < left["width_pct"] <= 100
    right = g.bar_geometry(+0.10, scale=0.05)   # beyond scale -> clamped to 100%
    assert right["side"] == "worse" and right["width_pct"] == 100.0


def test_render_html_smoke(tmp_path):
    data = {
        "horizons": [1, 5],
        "rungs": ["FULL", "minus_graph", "minus_news"],
        "qlike": {"HAR": {1: 0.46, 5: 0.55}, "FULL": {1: 0.46, 5: 0.55},
                  "minus_graph": {1: 0.46, 5: 0.55}, "minus_news": {1: 0.45, 5: 0.54}},
        "loo": {"graph": {1: (-0.2, 0.83), 5: (-0.6, 0.56)}, "news": {1: (9.4, 0.0), 5: (6.6, 0.0)},
                "gate": {1: (3.7, 0.0), 5: (-0.3, 0.78)}},
        "robustness": [{"cell": "x", "three": "p=0.05", "five": "p=0.30", "outcome": "collapsed",
                        "note": "example"}],
    }
    html = g.render_html(data)
    assert "<html" in html.lower() and "QLIKE" in html
    out = tmp_path / "v.html"
    g.write_html(data, out)
    assert out.exists() and out.stat().st_size > 500
