"""Gate the SP500 Colab notebook so the 'cell source has no newlines -> all code rendered as one commented
line -> nothing runs' bug (fixed 2026-09-14) cannot regress. Notebooks are not executed by pytest, so this
static check runs inside the pre-push gate: every code cell must contain real newlines when it has multiple
source lines, and must compile as Python once Colab magics/shell lines are stripped."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[3] / "notebooks" / "gnnhar_sp500_colab.ipynb"


def _code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_exists_and_has_code_cells():
    assert NB.exists(), f"notebook missing: {NB}"
    assert len(_code_cells()) >= 5


def test_code_cells_have_newlines_and_compile():
    for i, c in enumerate(_code_cells()):
        src = "".join(c["source"])
        if len([ln for ln in c["source"] if ln.strip()]) > 1:
            assert "\n" in src, f"code cell {i} has multiple lines but no newline (would render as one comment)"
        code = "\n".join(ln for ln in src.split("\n")
                         if not ln.strip().startswith(("!", "%")) and "get_ipython" not in ln)
        compile(code, f"cell{i}", "exec")                 # SyntaxError here = malformed cell


def test_no_nonascii_mojibake():
    for i, c in enumerate(_code_cells()):
        bad = [ch for ch in "".join(c["source"]) if ord(ch) > 127]
        assert not bad, f"code cell {i} has non-ASCII/mojibake chars: {bad}"


def test_runs_sp500_arg_and_writes_result():
    """Acceptance: the notebook drives run_gnnhar.py with the sp500 arg and inspects gnnhar_sp500.json."""
    text = "".join("".join(c["source"]) for c in _code_cells())
    assert "run_gnnhar.py" in text and "sp500" in text
    assert "gnnhar_sp500.json" in text                    # reads the result it produced
    assert "_commit_once" in text and "180" in text        # disconnect-proof background committer
