"""
Per-Ticker Windowed Dataset Builder

Fixes 3 structural bugs found in the S&P 500 multi-ticker training scripts
(train_enhanced.py, cross_market_experiment.py):

1. Sliding windows built over a row-wise concatenation of multiple tickers can span
   two unrelated tickers' data at the boundary between them.
2. StandardScaler fit on the pooled multi-ticker split blends different stocks'
   volatility scales together.
3. Val/test splits fit their OWN independent scaler instead of reusing the one fit
   on train, causing a train/inference distribution mismatch.

This module builds one windowed dataset PER TICKER (never mixing rows across
tickers), fits scalers once per ticker on that ticker's train split, and reuses
(.transform(), not refit) for that ticker's val/test splits.

Author: Stock Volatility Prediction Team
Date: 2026-08-01
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.common.temporal_split import temporal_split_dataframe


class WindowedSeriesDataset(Dataset):
    """Sliding-window dataset over ONE already-scaled series (one ticker, one split).

    Never spans more than one ticker because the caller only ever passes one
    ticker's array.
    """

    def __init__(self, features_scaled: np.ndarray, target_scaled: np.ndarray, seq_length: int):
        self.features = features_scaled.astype(np.float32)
        self.target = target_scaled.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.target[idx + self.seq_length - 1]
        return torch.tensor(x), torch.tensor(y)


def _scale_and_window(df, feature_cols, target_col, seq_length, feature_scaler, target_scaler, fit):
    """Scale one split's columns and build its windowed dataset.

    If fit=True, the scalers are fit on `df` first (use for the TRAIN split only).
    Otherwise `df` is transformed with the already-fit scalers (val/test/full-series).
    """
    if fit:
        feature_scaler.fit(df[feature_cols].values.astype(np.float32))
        target_scaler.fit(df[[target_col]].values.astype(np.float32))

    if len(df) == 0:
        # A short ticker's split ratio can round down to 0 rows (e.g. short history
        # + small val_ratio). StandardScaler.transform() rejects 0-sample arrays, so
        # short-circuit to an empty windowed dataset instead of erroring. Warn since
        # this silently drops the ticker from whichever split hit 0 rows.
        print(f"[WARN] multi_ticker_dataset: 0-row split (target_col={target_col}) "
              f"— this ticker contributes 0 samples to this split")
        empty_features = np.empty((0, len(feature_cols)), dtype=np.float32)
        empty_target = np.empty((0,), dtype=np.float32)
        return WindowedSeriesDataset(empty_features, empty_target, seq_length)

    features_scaled = feature_scaler.transform(df[feature_cols].values.astype(np.float32))
    target_scaled = target_scaler.transform(df[[target_col]].values.astype(np.float32)).flatten()
    return WindowedSeriesDataset(features_scaled, target_scaled, seq_length)


def build_per_ticker_datasets(
    ticker_dfs: dict,
    feature_cols: list,
    target_col: str,
    seq_length: int = 22,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_column: str = "date",
):
    """Build per-ticker train/val/test windowed datasets with correct scaling.

    For each ticker: split chronologically (via temporal_split_dataframe, applied
    to this ticker's own rows only), fit feature_scaler + target_scaler on the
    train split only, transform (not refit) val/test, then build a
    WindowedSeriesDataset per split.

    Args:
        ticker_dfs: dict mapping ticker -> DataFrame (one ticker's own rows, NOT
            concatenated with other tickers). Must contain date_column, all
            feature_cols, and target_col.
        feature_cols: list of feature column names.
        target_col: target column name.
        seq_length: sliding window length.
        train_ratio, val_ratio, test_ratio: temporal split ratios (must sum to 1.0).
        date_column: name of the date column used for chronological sorting.

    Returns:
        dict mapping ticker -> {
            "train": WindowedSeriesDataset, "val": WindowedSeriesDataset,
            "test": WindowedSeriesDataset, "feature_scaler": StandardScaler,
            "target_scaler": StandardScaler,
        }
    """
    result = {}
    for ticker, df in ticker_dfs.items():
        train_df, val_df, test_df = temporal_split_dataframe(
            df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            date_column=date_column,
        )

        feature_scaler, target_scaler = StandardScaler(), StandardScaler()
        result[ticker] = {
            "train": _scale_and_window(train_df, feature_cols, target_col, seq_length,
                                        feature_scaler, target_scaler, fit=True),
            "val": _scale_and_window(val_df, feature_cols, target_col, seq_length,
                                      feature_scaler, target_scaler, fit=False),
            "test": _scale_and_window(test_df, feature_cols, target_col, seq_length,
                                       feature_scaler, target_scaler, fit=False),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }

    return result


def build_full_series_datasets(
    ticker_dfs: dict,
    feature_cols: list,
    target_col: str,
    seq_length: int = 22,
    date_column: str = "date",
):
    """One windowed dataset per ticker over its ENTIRE series, scaler fit on that
    same full series.

    For use when there is no separate train split to fit a scaler on — e.g.
    cross-market evaluation, where the test market's tickers never appear in the
    training market's data, so there is no leakage risk in fitting on the test
    market's own full series (there is simply no other reference available).

    Returns:
        dict mapping ticker -> {
            "data": WindowedSeriesDataset, "feature_scaler": StandardScaler,
            "target_scaler": StandardScaler,
        }
    """
    result = {}
    for ticker, df in ticker_dfs.items():
        df = df.sort_values(date_column).reset_index(drop=True)
        feature_scaler, target_scaler = StandardScaler(), StandardScaler()
        result[ticker] = {
            "data": _scale_and_window(df, feature_cols, target_col, seq_length,
                                       feature_scaler, target_scaler, fit=True),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }
    return result
