"""Unit tests for the error-attribution pure logic."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import error_attribution_edgehm as EA  # noqa: E402


def test_qlike_loss_zero_at_equal_and_positive():
    assert abs(EA.qlike_loss(np.array([1.0]), np.array([1.0]))[0]) < 1e-12
    assert EA.qlike_loss(np.array([4.0]), np.array([1.0]))[0] > 0   # mismatch -> positive loss


def _entries():
    # (ticker, date, y, pred, qlike, sqerr)
    return [
        ("AAA", "2020-01-01", 1e-3, 1e-3, 0.0, 0.0),
        ("AAA", "2020-01-02", 4e-3, 1e-3, 2.6, 9e-6),
        ("BBB", "2020-01-01", 2e-3, 2e-3, 0.1, 1e-8),
    ]


def test_per_ticker_stats_sorted_worst_first():
    s = EA.per_ticker_stats(_entries())
    assert s[0]["ticker"] == "AAA"                      # highest mean QLIKE
    assert s[0]["n"] == 2 and abs(s[0]["qlike_mean"] - 1.3) < 1e-9


def test_per_date_stats():
    s = EA.per_date_stats(_entries())
    top = s[0]
    assert top["date"] == "2020-01-02" and top["n"] == 1   # only AAA that day, QLIKE 2.6


def test_worst_entries_by_field():
    w = EA.worst_entries(_entries(), 1, 4)                  # worst by QLIKE (idx 4)
    assert w[0][0] == "AAA" and w[0][1] == "2020-01-02"
    w2 = EA.worst_entries(_entries(), 1, 5)                 # worst by sqerr (idx 5)
    assert w2[0][1] == "2020-01-02"


def test_classify_target():
    floor = 1e-3
    assert EA.classify_target(1.5e-3, 5e-3, floor) == "near-floor"   # <= 2x floor
    assert EA.classify_target(3e-2, 5e-3, floor) == "spike"          # >= 5x median
    assert EA.classify_target(6e-3, 5e-3, floor) == "normal"
    assert EA.classify_target(1.0, 0.0, floor) == "normal"          # median 0 -> no spike test
