import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_ablation  # noqa: E402


def test_rungs_include_full_leaveoneout_and_lstm_only():
    # _RUNGS = (name, use_news, use_gate, use_graph)
    spec = {r[0]: r[1:] for r in run_ablation._RUNGS}
    # full leave-one-out set
    assert spec["FULL"] == (True, True, True)
    assert spec["minus_graph"] == (True, True, False)
    assert spec["minus_gate"] == (True, False, True)
    assert spec["minus_news"] == (False, False, True)
    # lstm_only = price-only backbone: news, gate, graph ALL removed (merged from run_lstm_only)
    assert "lstm_only" in spec
    assert spec["lstm_only"] == (False, False, False)


def test_lstm_only_dump_dir_matches_dm_report():
    # dm_report expects the price-only dumps under 'lstm_only' (DUMP_DIR['LSTM_only'])
    import dm_report
    assert dm_report.DUMP_DIR["LSTM_only"] == "lstm_only"
    assert "lstm_only" in {r[0] for r in run_ablation._RUNGS}
