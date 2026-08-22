"""Tiny smoke test for the masked-panel teaching diagram generator."""
import draw_masked_panel as dmp


def test_svg_smoke():
    svg = dmp.build_svg()
    assert svg.startswith("<?xml")
    assert "<svg" in svg and svg.rstrip().endswith("</svg>")
    assert len(svg) > 5000
    # key concepts present as real text
    for token in ["masked union panel", "node_mask", "target_mask",
                  "Common-date intersection", "Snapshot t5", "Snapshot t9"]:
        assert token in svg, token


def test_mask_logic_matches_example():
    # concrete example the diagram claims
    assert dmp.valid_tickers(5) == ["A", "B", "C", "D"]
    assert dmp.valid_tickers(9) == ["A", "B", "C", "D", "E", "F"]
    assert dmp.INTERSECTION_FROM == 8         # only t8..t12 survive intersection
    assert dmp.cell_state("F", 8) == "warmup"
    assert dmp.cell_state("E", 5) == "missing"
