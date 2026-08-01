"""Tests for src/common/multi_ticker_dataset.py — per-ticker windowing/scaling bug fix.

Written test-first: these assert the FIX (windows never cross ticker boundaries, scalers
are fit once on train and reused for val/test). Before the module exists, all tests here
fail on import — that is the expected "red" state before implementation.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def _make_ticker_df(start_date, n_rows, value_low, value_high, feature_cols, target_col):
    """Synthetic per-ticker df with values confined to [value_low, value_high]."""
    dates = pd.date_range(start_date, periods=n_rows, freq="B")
    rng = np.random.RandomState(0)
    data = {"date": dates}
    for col in feature_cols:
        data[col] = rng.uniform(value_low, value_high, n_rows)
    data[target_col] = rng.uniform(value_low, value_high, n_rows)
    return pd.DataFrame(data)


class TestBoundarySafety:
    """Windows must never mix rows from two different tickers."""

    def test_windows_never_cross_ticker_boundary(self):
        from src.common.multi_ticker_dataset import build_per_ticker_datasets

        feature_cols = ["f1", "f2"]
        target_col = "target"
        seq_length = 5

        # Ticker A: small-magnitude values. Ticker B: large-magnitude values.
        # If a window ever mixed rows from both tickers, its raw values would span
        # both ranges — which we can detect by checking the fitted scaler's per-ticker
        # mean/std stay within each ticker's own known range.
        ticker_dfs = {
            "TICKA": _make_ticker_df("2020-01-01", 60, 0.01, 0.02, feature_cols, target_col),
            "TICKB": _make_ticker_df("2020-01-01", 60, 100.0, 200.0, feature_cols, target_col),
        }

        result = build_per_ticker_datasets(
            ticker_dfs, feature_cols, target_col, seq_length=seq_length,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, date_column="date",
        )

        for ticker, raw_low, raw_high in [("TICKA", 0.01, 0.02), ("TICKB", 100.0, 200.0)]:
            scaler = result[ticker]["target_scaler"]
            # The scaler was fit on this ticker's own train split only — its mean must
            # fall inside this ticker's own known value range. If windowing/splitting had
            # leaked rows from the other ticker, the fitted mean would be pulled far
            # outside [raw_low, raw_high] (the other ticker's values differ by 4+ orders
            # of magnitude).
            fitted_mean = scaler.mean_[0]
            assert raw_low <= fitted_mean <= raw_high, (
                f"{ticker} target_scaler mean {fitted_mean} outside own range "
                f"[{raw_low}, {raw_high}] — suggests cross-ticker contamination"
            )

        # Every window's raw feature values (inverse-transformed) must fall within
        # that ticker's own range across all 3 splits.
        for ticker, raw_low, raw_high in [("TICKA", 0.01, 0.02), ("TICKB", 100.0, 200.0)]:
            feature_scaler = result[ticker]["feature_scaler"]
            for split in ["train", "val", "test"]:
                ds = result[ticker][split]
                for i in range(len(ds)):
                    x, _ = ds[i]
                    raw_x = feature_scaler.inverse_transform(x.numpy())
                    assert raw_x.min() >= raw_low - 1e-6, f"{ticker}/{split} window {i} below own range"
                    assert raw_x.max() <= raw_high + 1e-6, f"{ticker}/{split} window {i} above own range"

    def test_old_concat_then_window_logic_would_mix_tickers(self):
        """Demonstrates the bug being fixed: concatenating tickers row-wise then
        sliding a window across the combined frame produces boundary-crossing windows."""
        feature_cols = ["f1"]
        n_a, n_b, seq_length = 10, 10, 5
        df_a = _make_ticker_df("2020-01-01", n_a, 0.01, 0.02, feature_cols, "target")
        df_b = _make_ticker_df("2020-01-01", n_b, 100.0, 200.0, feature_cols, "target")
        combined = pd.concat([df_a, df_b], ignore_index=True)

        # OLD (buggy) approach: one global sliding window over the concatenated frame.
        values = combined["f1"].values
        boundary_idx = n_a  # first row of ticker B
        crossing_windows = [
            (i, values[i:i + seq_length])
            for i in range(len(values) - seq_length)
            if i < boundary_idx < i + seq_length
        ]
        assert len(crossing_windows) > 0, "sanity check: boundary should be crossable in old logic"
        for _, window in crossing_windows:
            assert window.min() < 1.0 and window.max() > 1.0, (
                "boundary-crossing window should mix ticker A's small values with "
                "ticker B's large values — this is the bug being fixed"
            )


class TestScalerReuse:
    """Val/test must be transformed with the scaler fit on TRAIN only, never refit."""

    def test_val_test_use_train_fitted_scaler(self):
        from src.common.multi_ticker_dataset import build_per_ticker_datasets

        feature_cols = ["f1"]
        target_col = "target"
        n_rows = 100
        dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        # Deterministic ramp so train/val/test have different, known means.
        values = np.arange(n_rows, dtype=np.float64)
        df = pd.DataFrame({"date": dates, "f1": values, target_col: values})
        ticker_dfs = {"TICKA": df}

        result = build_per_ticker_datasets(
            ticker_dfs, feature_cols, target_col, seq_length=3,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, date_column="date",
        )

        train_end = int(n_rows * 0.7)
        train_mean = values[:train_end].mean()
        train_std = values[:train_end].std()

        feature_scaler = result["TICKA"]["feature_scaler"]
        assert abs(feature_scaler.mean_[0] - train_mean) < 1e-6
        assert abs(feature_scaler.scale_[0] - train_std) < 1e-6

        # Manually transform a known val-split raw value with the TRAIN scaler and
        # compare against what the val dataset actually stored.
        val_start = train_end + int(n_rows * 0.15)
        val_start = train_end  # val split begins right after train
        raw_val_first = values[train_end]
        expected_scaled = (raw_val_first - train_mean) / train_std

        val_ds = result["TICKA"]["val"]
        x0, _ = val_ds[0]
        actual_scaled = x0.numpy()[0, 0]
        assert abs(actual_scaled - expected_scaled) < 1e-4, (
            f"val split value not scaled with TRAIN scaler: expected {expected_scaled}, "
            f"got {actual_scaled} (would differ if val's own mean/std were used instead)"
        )


class TestPerTickerSplit:
    """Each ticker's train/val/test boundary must come from its OWN row count."""

    def test_split_boundary_independent_per_ticker(self):
        from src.common.multi_ticker_dataset import build_per_ticker_datasets

        feature_cols = ["f1"]
        target_col = "target"
        # Two tickers with DIFFERENT total row counts.
        df_short = _make_ticker_df("2020-01-01", 50, 1.0, 2.0, feature_cols, target_col)
        df_long = _make_ticker_df("2020-01-01", 200, 1.0, 2.0, feature_cols, target_col)
        ticker_dfs = {"SHORT": df_short, "LONG": df_long}

        result = build_per_ticker_datasets(
            ticker_dfs, feature_cols, target_col, seq_length=5,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, date_column="date",
        )

        # SHORT: 50 rows -> train=35, val=7, test=8 -> windows = len - seq_length
        assert len(result["SHORT"]["train"]) == max(0, 35 - 5)
        assert len(result["SHORT"]["val"]) == max(0, 7 - 5)
        # LONG: 200 rows -> train=140, val=30, test=30
        assert len(result["LONG"]["train"]) == max(0, 140 - 5)
        assert len(result["LONG"]["val"]) == max(0, 30 - 5)


class TestEmptySplit:
    """A ticker's split can round down to 0 rows (e.g. test_ratio=0.0, or a short
    history combined with a small ratio). Must not crash StandardScaler.transform()."""

    def test_zero_row_test_split_does_not_crash(self):
        from src.common.multi_ticker_dataset import build_per_ticker_datasets

        feature_cols = ["f1"]
        target_col = "target"
        df = _make_ticker_df("2020-01-01", 40, 1.0, 2.0, feature_cols, target_col)
        ticker_dfs = {"TICKA": df}

        # test_ratio=0.0 forces an empty test split.
        result = build_per_ticker_datasets(
            ticker_dfs, feature_cols, target_col, seq_length=5,
            train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, date_column="date",
        )

        assert len(result["TICKA"]["test"]) == 0
