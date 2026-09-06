"""Tests for the pragma-logic guard (catches new logic hidden in # pragma: no cover functions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pragma_logic_guard as G  # noqa: E402


def test_added_line_nums_parses_unified0():
    diff = ("@@ -0,0 +5,2 @@\n+a = 1\n+b = 2\n"
            "@@ -10,1 +12,1 @@\n-old\n+new\n")
    assert G.added_line_nums(diff) == {5, 6, 12}


def test_pragma_functions_detects_marked_def_only():
    src = ("def clean(x):\n"
           "    return x + 1\n\n"
           "def run(x):  # pragma: no cover\n"
           "    y = x + 1\n"
           "    return y\n")
    funcs = G.pragma_functions(src)
    names = [f[0] for f in funcs]
    assert names == ["run"]                     # only the pragma-marked function
    name, start, end, doc_end = funcs[0]
    assert start == 5 and end == 6 and doc_end == 0


def test_pragma_functions_skips_docstring_lines():
    src = ('def run(x):  # pragma: no cover\n'
           '    """doc\n'
           '    line2"""\n'
           '    y = x\n'
           '    return y\n')
    name, start, end, doc_end = G.pragma_functions(src)[0]
    assert doc_end == 3                          # docstring ends line 3 -> excluded from logic count


def test_analyze_counts_logic_added_under_pragma_only():
    src = ("def clean(x):\n"                      # 1  (not pragma -> ignored)
           "    return x + 1\n"                   # 2
           "def run(x):  # pragma: no cover\n"    # 3
           '    """doc"""\n'                      # 4  docstring -> not counted
           "    # a comment\n"                    # 5  comment -> not counted
           "    a = x + 1\n"                      # 6  logic
           "    b = a * 2\n"                      # 7  logic
           "    return b\n")                      # 8  logic
    # pretend lines 2,4,5,6,7,8 were added
    findings = G.analyze(src, {2, 4, 5, 6, 7, 8})
    assert findings == [("run", 3)]              # only 6,7,8 counted (docstring/comment/other-func excluded)


def test_analyze_empty_when_no_added_in_pragma():
    src = ("def run(x):  # pragma: no cover\n"
           "    return x\n")
    assert G.analyze(src, set()) == []           # nothing added -> no finding
