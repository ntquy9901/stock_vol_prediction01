"""Tests for the P1.2 cross-stock date-alignment fix in dataset_with_graph_method.py.

Covers the module-level functions actually used by the live news-fusion training
pipeline (_load_raw_stock_data, remove_outliers, _split_raw_data_by_date,
_reindex_to_common_dates) — NOT the older MultiStockDatasetWithGraphMethod class,
which has its own separate (unfixed, not used by any live baseline) data-loading
path exercised by test_dataset_data_leakage.py / test_dataset_edge_cases.py.

Background: independent per-ticker outlier removal used to drop different rows
per ticker, then downstream code sliced every ticker at the same *positional*
index assuming that position meant the same calendar date across tickers —
false whenever tickers have different listing dates or any outlier-driven
misalignment. Fixed by winsorizing (not dropping) outliers and explicitly
reindexing every ticker to the intersection of available trading dates before
any positional split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lstm_gat_hybrid.dataset_with_graph_method import (
    remove_outliers,
    _reindex_to_common_dates,
    _split_raw_data_by_date,
    _load_raw_stock_data,
)

pytestmark = pytest.mark.smoke


def _make_ticker_df(dates, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": dates,
        "parkinson_variance": np.abs(rng.normal(0.01, 0.002, size=len(dates))),
    })


class TestRemoveOutliersWinsorizes:
    def test_keeps_row_count_unchanged(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="B").strftime("%Y-%m-%d")
        df = _make_ticker_df(dates, seed=1)
        df.loc[5, "parkinson_variance"] = 10.0  # inject an extreme outlier
        out = remove_outliers(df, n_std=3.0)
        assert len(out) == len(df), "winsorize must never drop rows"
        assert set(out["date"]) == set(df["date"])

    def test_clips_extreme_value_within_bounds(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="B").strftime("%Y-%m-%d")
        df = _make_ticker_df(dates, seed=2)
        df.loc[5, "parkinson_variance"] = 10.0
        out = remove_outliers(df, n_std=3.0)
        assert out.loc[5, "parkinson_variance"] < 10.0
        assert out.loc[5, "parkinson_variance"] > 0

    def test_no_outliers_is_a_noop(self):
        # Fixed, tightly-clustered values (not sampled) so no z-score can exceed
        # n_std regardless of RNG state — a random draw can legitimately produce
        # a borderline outlier by chance, which isn't what this test is for.
        dates = pd.date_range("2024-01-01", periods=20, freq="B").strftime("%Y-%m-%d")
        values = 0.01 + np.array([0.0001 * (i % 5) for i in range(20)])
        df = pd.DataFrame({"date": dates, "parkinson_variance": values})
        out = remove_outliers(df, n_std=3.0)
        pd.testing.assert_series_equal(out["parkinson_variance"], df["parkinson_variance"])


class TestReindexToCommonDates:
    def test_aligns_tickers_with_different_listing_dates(self):
        """The core P1.2 scenario: one ticker listed years later than another."""
        old_dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d")
        new_dates = pd.date_range("2020-06-01", periods=100, freq="B").strftime("%Y-%m-%d")
        stock_data = {
            "OLD": _make_ticker_df(old_dates, seed=1),
            "NEW": _make_ticker_df(new_dates, seed=2),
        }
        aligned = _reindex_to_common_dates(stock_data)

        lengths = {len(df) for df in aligned.values()}
        assert len(lengths) == 1, "every ticker must end up the same length"

        old_dates_out = list(aligned["OLD"]["date"])
        new_dates_out = list(aligned["NEW"]["date"])
        assert old_dates_out == new_dates_out, (
            "position i must be the same calendar date for every ticker after alignment"
        )
        # The common window is exactly NEW's range, since OLD fully contains it.
        assert old_dates_out[0] == new_dates[0]
        assert old_dates_out[-1] == new_dates[-1]

    def test_drops_dates_not_shared_by_every_ticker(self):
        dates_a = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
        dates_b = list(dates_a[:8])  # ticker B is missing the last 2 dates (e.g. a trading halt)
        stock_data = {
            "A": _make_ticker_df(dates_a, seed=1),
            "B": _make_ticker_df(dates_b, seed=2),
        }
        aligned = _reindex_to_common_dates(stock_data)
        assert len(aligned["A"]) == 8
        assert len(aligned["B"]) == 8
        assert list(aligned["A"]["date"]) == list(aligned["B"]["date"])

    def test_raises_on_duplicate_dates_within_a_ticker(self):
        dates = list(pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"))
        dates[3] = dates[2]  # inject a duplicate date
        stock_data = {
            "A": _make_ticker_df(dates, seed=1),
            "B": _make_ticker_df(pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"), seed=2),
        }
        with pytest.raises(ValueError, match="mismatched lengths"):
            _reindex_to_common_dates(stock_data)


class TestSplitRawDataByDateGuardsAlignment:
    def test_raises_on_mismatched_ticker_lengths(self):
        """_split_raw_data_by_date must refuse to silently mis-split unaligned input."""
        stock_data = {
            "A": _make_ticker_df(pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d"), seed=1),
            "B": _make_ticker_df(pd.date_range("2024-01-01", periods=90, freq="B").strftime("%Y-%m-%d"), seed=2),
        }
        with pytest.raises(ValueError, match="mismatched per-ticker lengths"):
            _split_raw_data_by_date(stock_data)

    def test_splits_aligned_data_by_position_correctly(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
        stock_data = {
            "A": _make_ticker_df(dates, seed=1),
            "B": _make_ticker_df(dates, seed=2),
        }
        train, val, test, train_end, val_end, min_len = _split_raw_data_by_date(
            stock_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        assert min_len == 100
        assert len(train["A"]) == train_end == 70
        assert len(val["A"]) == val_end - train_end == 15
        assert len(test["A"]) == min_len - val_end == 15
        # Same calendar dates for every ticker in every split.
        assert list(train["A"]["date"]) == list(train["B"]["date"])
        assert list(test["A"]["date"]) == list(test["B"]["date"])
        # No overlap between splits.
        assert set(train["A"]["date"]).isdisjoint(set(val["A"]["date"]))
        assert set(val["A"]["date"]).isdisjoint(set(test["A"]["date"]))


class TestRealDataSmoke:
    """Real-data-sample smoke test per CLAUDE.md testing rules — catches encoding/
    format issues synthetic fixtures miss (e.g. the VPB/VRE tz-aware date bug)."""

    def test_load_and_split_current_universe(self):
        data_dir = _ROOT / "data" / "processed"
        if not data_dir.exists():
            pytest.skip("data/processed/ not present in this environment")

        stock_data = _load_raw_stock_data(data_dir=str(data_dir), remove_outliers=True, n_std=3.0)
        assert len(stock_data) >= 30, "expected the full VN30-ish universe to load"

        lengths = {len(df) for df in stock_data.values()}
        assert len(lengths) == 1, "all tickers must be date-aligned after loading"

        train, val, test, train_end, val_end, min_len = _split_raw_data_by_date(stock_data)
        assert train_end > 0 and val_end > train_end and min_len > val_end
        # Every ticker's train split must span the exact same date range.
        first_ticker = next(iter(train))
        expected_dates = list(train[first_ticker]["date"])
        for ticker, df in train.items():
            assert list(df["date"]) == expected_dates, f"{ticker} train dates diverge from {first_ticker}"
