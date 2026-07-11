"""Integration test for compute_decay.main(): tiny fixture -> run -> assert output.

Covers main() I/O (read CSV, sort, decay, write) — the part NOT covered by test_decay.py
(which only tests the pure compute_decay_state function).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

import compute_decay


def test_main_preserves_count_and_decays(tmp_path, monkeypatch):
    """main() reads sentiment CSV, decays, writes — preserving original news_count (HIGH-1)."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    # day 1: 3 articles, score 0.8 (has news); days 2-3: no news
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "sentiment_1d": ["0.8", "0.0", "0.0"],
        "news_count_1d": ["3", "0", "0"],
        "news_titles": ["a", "", ""],
    }).to_csv(in_dir / "VCB_sentiment.csv", index=False)

    monkeypatch.setattr("sys.argv",
                        ["compute_decay", "--in_dir", str(in_dir), "--out_dir", str(out_dir), "--decay", "0.9"])
    compute_decay.main()

    out = pd.read_csv(out_dir / "VCB_sentiment.csv")
    # decayed state: 0.8 -> 0.72 -> 0.648
    assert abs(float(out.loc[0, "sentiment_1d"]) - 0.8) < 1e-6
    assert abs(float(out.loc[1, "sentiment_1d"]) - 0.72) < 1e-6
    assert abs(float(out.loc[2, "sentiment_1d"]) - 0.648) < 1e-6
    # [HIGH-1] news_count_1d preserved (NOT overwritten with 0/1 mask)
    assert list(out["news_count_1d"].astype(int)) == [3, 0, 0]


def test_main_skips_file_missing_columns(tmp_path, monkeypatch):
    """A sentiment file missing required columns is skipped (not crash)."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    # valid file
    pd.DataFrame({"date": ["2024-01-01"], "sentiment_1d": ["0.5"],
                  "news_count_1d": ["1"]}).to_csv(in_dir / "VCB_sentiment.csv", index=False)
    # invalid file (no sentiment_1d)
    pd.DataFrame({"date": ["2024-01-01"], "x": ["y"]}).to_csv(in_dir / "BAD_sentiment.csv", index=False)

    monkeypatch.setattr("sys.argv",
                        ["compute_decay", "--in_dir", str(in_dir), "--out_dir", str(out_dir)])
    compute_decay.main()

    # valid file processed, bad file skipped (VCB out exists, BAD does not)
    assert (out_dir / "VCB_sentiment.csv").exists()
    assert not (out_dir / "BAD_sentiment.csv").exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
