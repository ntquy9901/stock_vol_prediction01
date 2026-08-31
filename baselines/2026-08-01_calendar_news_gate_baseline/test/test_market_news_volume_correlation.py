"""Tests for analyze_market_news_volume_correlation.py.

Covers: pure aggregation logic (synthetic, known ground truth) and a real-data-sample smoke test
(actual `data/processed/*_processed.csv` + `dual_group_news_panel.parquet`, per CLAUDE.md
"data-pipeline test phai co real-data-sample smoke").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_market_news_volume_correlation import (  # noqa: E402
    build_joined_frame,
    correlate,
    load_news_volume,
    market_avg_change,
    run,
    NEWS_PANEL_PATH,
    PROCESSED_DIR,
)


def test_market_avg_change_known_values(tmp_path):
    """2 tickers, 3 days, hand-computed cross-sectional mean of day-over-day change."""
    long_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        "ticker": ["A"] * 3 + ["B"] * 3,
        "parkinson_variance": [0.01, 0.02, 0.015, 0.03, 0.033, 0.028],
    })
    result = market_avg_change(long_df)
    # day1: NaN (no prior day). day2: mean(0.02-0.01, 0.033-0.03) = mean(0.01, 0.003) = 0.0065
    # day3: mean(0.015-0.02, 0.028-0.033) = mean(-0.005, -0.005) = -0.005
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.0065)
    assert result.iloc[2] == pytest.approx(-0.005)


def test_load_news_volume_sums_topic_columns_across_tickers(tmp_path):
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({
        "ticker": ["A", "B", "A"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "kq_topic_earnings_count": [1.0, 2.0, 0.0],
        "th_topic_macro_count": [0.0, 1.0, 3.0],
        "kq_emb_0": [0.1, 0.2, 0.3],  # non-topic column must be ignored
    }).to_parquet(panel)
    vol = load_news_volume(panel)
    assert vol[pd.Timestamp("2024-01-01")] == pytest.approx(1.0 + 2.0 + 0.0 + 1.0)
    assert vol[pd.Timestamp("2024-01-02")] == pytest.approx(0.0 + 3.0)


def test_build_joined_frame_shifts_next_day_correctly():
    change = pd.Series(
        [np.nan, 1.0, -2.0, 3.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    news = pd.Series([5.0, 0.0, 2.0, 1.0], index=change.index)
    joined = build_joined_frame(change, news)
    # day1 dropped (change is NaN); day4 dropped (next-day shift has nothing after it)
    assert list(joined.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert joined.loc[pd.Timestamp("2024-01-02"), "market_avg_change_next"] == pytest.approx(-2.0)
    assert joined.loc[pd.Timestamp("2024-01-02"), "market_avg_change_abs_next"] == pytest.approx(2.0)


def test_correlate_perfect_correlation_detected():
    n = 40
    news = pd.Series(np.arange(n, dtype=float))
    joined = pd.DataFrame({
        "news_volume": news,
        "market_avg_change": news * 0.01,
        "market_avg_change_next": news * 0.02,
        "market_avg_change_abs_next": news.abs() * 0.02,
    })
    result = correlate(joined)
    assert result["pearson_same_day"]["r"] == pytest.approx(1.0, abs=1e-6)
    assert result["pearson_next_day"]["r"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_days"] == n


@pytest.mark.smoke
def test_real_data_smoke_runs_and_produces_finite_output(tmp_path):
    """Real-data-sample smoke: actual processed CSVs + actual news panel, not synthetic."""
    if not PROCESSED_DIR.exists() or not NEWS_PANEL_PATH.exists():
        pytest.skip("real data not available in this environment")
    result = run(processed_dir=PROCESSED_DIR, panel_path=NEWS_PANEL_PATH, out_dir=tmp_path)
    assert result["n_days"] > 1000  # ~20 years of VN30 trading days expected
    assert np.isfinite(result["pearson_same_day"]["r"])
    assert np.isfinite(result["pearson_next_day"]["r"])
    assert -1.0 <= result["pearson_next_day"]["r"] <= 1.0
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "scatter.png").exists()
