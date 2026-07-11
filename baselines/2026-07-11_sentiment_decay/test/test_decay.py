"""Tests for compute_decay_state correctness.

Run: pytest baselines/2026-07-11_sentiment_decay/test/test_decay.py -v
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compute_decay import compute_decay_state


def test_news_day_sets_state_to_score():
    s = compute_decay_state([0.8], [1], decay=0.9)
    assert s == [0.8]


def test_no_news_decays_previous():
    """news (0.8) then 2 no-news days: 0.8 → 0.72 → 0.648."""
    s = compute_decay_state([0.8, 0.0, 0.0], [1, 0, 0], decay=0.9)
    assert abs(s[0] - 0.8) < 1e-9
    assert abs(s[1] - 0.72) < 1e-9    # 0.8 × 0.9
    assert abs(s[2] - 0.648) < 1e-9   # 0.72 × 0.9


def test_new_news_resets_state():
    """decay then fresh news resets to the new score."""
    s = compute_decay_state([0.8, 0.0, -0.5], [1, 0, 1], decay=0.9)
    assert abs(s[2] - (-0.5)) < 1e-9


def test_all_no_news_stays_zero():
    s = compute_decay_state([0.0, 0.0, 0.0], [0, 0, 0], decay=0.9)
    assert s == [0.0, 0.0, 0.0]


def test_negative_sentiment_decays():
    """negative news decays toward 0 (magnitude shrinks)."""
    s = compute_decay_state([-0.6, 0.0], [1, 0], decay=0.5)
    assert abs(s[0] - (-0.6)) < 1e-9
    assert abs(s[1] - (-0.3)) < 1e-9   # -0.6 × 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} decay tests passed.")
