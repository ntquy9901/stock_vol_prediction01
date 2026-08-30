"""Unit tests for the config-hardcode pre-push scanner (heuristic WARN/BLOCK)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_config_hardcode as C  # noqa: E402


# --------------------------- classify_line: BLOCK ---------------------------

def test_hardcoded_rolling_window_blocks():
    v = C.classify_line("    z = s.rolling(20).mean()")
    assert v is not None and v[0] == "BLOCK" and "rolling" in v[1]


def test_hardcoded_topk_kwarg_blocks():
    v = C.classify_line("    adj = _vshock_adjacency(x, top_k=5)")
    assert v is not None and v[0] == "BLOCK" and "Top-K" in v[1]


def test_named_tunable_assignment_blocks():
    assert C.classify_line("_VOL_WIN = 20")[0] == "BLOCK"
    assert C.classify_line("EDGE_TOP_K = 5")[0] == "BLOCK"
    assert C.classify_line("patience: int = 15")[0] == "BLOCK"
    assert C.classify_line("weight_decay = 1e-5")[0] == "BLOCK"


# --------------------------- classify_line: not flagged ---------------------------

def test_config_sourced_line_not_flagged():
    assert C.classify_line("_VOL_WIN = pc.VOLUME_ZSCORE_WINDOW") is None
    assert C.classify_line("min_common = config.MIN_COMMON") is None
    assert C.classify_line("fl = cfg.qlike_floor") is None


def test_trivial_zero_one_literals_not_flagged():
    assert C.classify_line("idx = 0") is None            # index / identity
    assert C.classify_line("np.fill_diagonal(A, 1)") is None
    assert C.classify_line("w = s.rolling(1).mean()") is None   # rolling(1) trivial -> not flagged
    assert C.classify_line("top_k=1") is None            # trivial top_k -> not flagged


def test_non_tunable_assignment_not_flagged():
    assert C.classify_line("count = 5") is None          # 'count' is not a tunable name


def test_noqa_and_config_ok_exceptions():
    assert C.classify_line("_EPS = 1e-12  # config-ok") is None
    assert C.classify_line("win = 22  # noqa") is None


def test_comment_and_blank_ignored():
    assert C.classify_line("   ") is None
    assert C.classify_line("# window = 20 historical note") is None


# --------------------------- classify_line: WARN ---------------------------

def test_bare_scientific_literal_warns():
    v = C.classify_line("    resid = np.log(y + 1e-8)")   # bare 1e-8, no tunable NAME= -> WARN
    assert v is not None and v[0] == "WARN"


def test_clean_line_returns_none():
    assert C.classify_line("    return net(x, adj)") is None


# --------------------------- helpers ---------------------------

def test_name_is_tunable_substr_token_and_negative():
    assert C._name_is_tunable("weight_decay")            # substring
    assert C._name_is_tunable("EPOCHS")                  # token
    assert C._name_is_tunable("MIN_TRAIN_ROWS")          # substring
    assert not C._name_is_tunable("customer_id")         # neither


def test_is_trivial_number_handles_bad_string():
    assert C._is_trivial_number("0")
    assert C._is_trivial_number("1.0")
    assert not C._is_trivial_number("22")
    assert not C._is_trivial_number("1__2bad")           # unparseable -> ValueError -> False


def test_is_excluded_path():
    assert C.is_excluded_path("submission/soict_lstm_gat/pipeline_config.py")
    assert C.is_excluded_path("baselines/x/code/tests/test_x.py")
    assert C.is_excluded_path("baselines/x/code/test_masked_rich.py")
    assert C.is_excluded_path("archive/old/thing.py")
    assert C.is_excluded_path("baselines/x/code/conftest.py")
    assert not C.is_excluded_path("baselines/x/code/masked_rich.py")


# --------------------------- scan_added + diff parsing ---------------------------

def test_scan_added_excluded_file_yields_nothing():
    assert C.scan_added("submission/soict_lstm_gat/pipeline_config.py", [(1, "WIN = 20")]) == []


def test_scan_added_reports_block_with_metadata():
    findings = C.scan_added("baselines/x/code/masked_rich.py",
                            [(34, "_VOL_WIN = 20"), (35, "return x")])
    assert len(findings) == 1
    f = findings[0]
    assert f.path.endswith("masked_rich.py") and f.lineno == 34 and f.severity == "BLOCK"


def test_added_lines_from_diff_parses_hunks():
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -10,0 +11,2 @@\n"
        "+WIN = 20\n"
        "+x = 1\n"
        "@@ -20,1 +30,1 @@\n"
        " context_line\n"
        "-removed_line\n"
        "+top_k=5\n"
        "\\ No newline at end of file\n"
    )
    added = C.added_lines_from_diff(diff)
    texts = [t for _, t in added]
    assert "WIN = 20" in texts and "x = 1" in texts and "top_k=5" in texts
    # line numbers track the NEW file side
    nums = {t: ln for ln, t in added}
    assert nums["WIN = 20"] == 11 and nums["x = 1"] == 12
