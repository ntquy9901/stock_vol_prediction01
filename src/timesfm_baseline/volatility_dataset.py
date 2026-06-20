"""
PyTorch Dataset Classes for TimesFM 2.5 LoRA Fine-Tuning

This module provides dataset classes for TimesFM 2.5 fine-tuning with random
window sampling and temporal split integration.

Based on the TimesFM fine-tuning approach from:
https://github.com/UberGuidoZ/timesfm-google/tree/master/timesfm-forecasting/examples/finetuning

Key Features:
- Random Window Sampling: More data-efficient than fixed windows
- Temporal Split Integration: Maintains chronological order (70/15/15)
- No External Normalization: TimesFM applies RevIN internally
- Input Validation: Validates OHLCV data integrity
- Temporal Integrity Checks: Validates chronological ordering

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

from src.common.temporal_split import temporal_split_dataframe
from src.common.parkinson_utils import calculate_parkinson_volatility

logger = logging.getLogger(__name__)


def validate_ohlcv_data(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate OHLCV data integrity before processing.

    Args:
        df: Input OHLCV DataFrame
        required_cols: List of required column names (default: ['high', 'low'])

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails with specific error message
    """
    if required_cols is None:
        required_cols = ['high', 'low']

    # Check DataFrame is not empty
    if df is None or len(df) == 0:
        raise ValueError("OHLCV data is empty")

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for NaN values in critical columns
    if df[required_cols].isnull().any().any():
        raise ValueError(f"OHLCV data contains NaN values in columns: {required_cols}")

    # Check for infinite values
    if np.isinf(df[required_cols].values).any():
        raise ValueError(f"OHLCV data contains infinite values in columns: {required_cols}")

    # Validate high >= low for volatility calculation (only if columns exist)
    if 'high' in df.columns and 'low' in df.columns:
        if (df['high'] < df['low']).any():
            raise ValueError("OHLCV data has high < low (invalid for volatility calculation)")

        # Check all values are positive (only if columns exist)
        if (df[['high', 'low']] < 0).any().any():
            raise ValueError("OHLCV data has negative values in high/low columns")

    logger.info("OHLCV data validation passed")
    return df


def validate_temporal_integrity(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    date_column: str = 'date'
) -> None:
    """
    Validate that temporal split maintains chronological integrity.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        date_column: Name of date column

    Raises:
        ValueError: If temporal integrity is violated (data leakage)
    """
    # Check date column exists
    if date_column not in train_data.columns:
        logger.warning(f"Date column '{date_column}' not found - skipping temporal validation")
        return

    # Check dataframes are not empty
    if len(train_data) == 0:
        raise ValueError("Train dataframe is empty (0 rows). Cannot validate temporal integrity.")
    if len(val_data) == 0:
        raise ValueError("Validation dataframe is empty (0 rows). Cannot validate temporal integrity.")
    if len(test_data) == 0:
        raise ValueError("Test dataframe is empty (0 rows). Cannot validate temporal integrity.")

    try:
        train_dates = pd.to_datetime(train_data[date_column])
        val_dates = pd.to_datetime(val_data[date_column])
        test_dates = pd.to_datetime(test_data[date_column])
    except Exception as e:
        raise ValueError(
            f"Failed to parse dates from column '{date_column}': {e}. "
            f"Check date format is valid (e.g., YYYY-MM-DD)."
        )

    # Check no overlap: train max < val min
    if train_dates.max() >= val_dates.min():
        raise ValueError(
            f"Temporal integrity violated: Train and Val overlap. "
            f"Train max: {train_dates.max()}, Val min: {val_dates.min()}"
        )

    # Check no overlap: val max < test min
    if val_dates.max() >= test_dates.min():
        raise ValueError(
            f"Temporal integrity violated: Val and Test overlap. "
            f"Val max: {val_dates.max()}, Test min: {test_dates.min()}"
        )

    logger.info("Temporal integrity validated: train < val < test ✓")


class TimeSeriesRandomWindowDataset(Dataset):
    """
    Random-window dataset for TimesFM 2.5 fine-tuning.

    Pre-samples random (series, split-point) windows similar to Chronos-2's
    random slicing approach. Each window has a full context_len context
    (no zero-padding) to avoid corrupting TimesFM's internal RevIN
    normalization statistics.

    NO external normalization is needed — TimesFM handles instance
    normalization internally via RevIN. The loss is computed in the
    original data scale.

    Attributes:
        series_list (List[np.ndarray]): List of 1D time series arrays
        context_len (int): Length of context window (default: 64 for TimesFM 2.5)
        horizon_len (int): Length of forecast horizon (default: 5 for 5-day forecasts)
        samples (List[Tuple[int, int]]): List of (series_idx, start_point) pairs

    Example:
        >>> series_list = [np.array([0.1, 0.2, 0.3, ...]), ...]
        >>> dataset = TimeSeriesRandomWindowDataset(
        ...     series_list, context_len=64, horizon_len=5, num_samples=1000
        ... )
        >>> context, target = dataset[0]
        >>> print(context.shape)  # torch.Size([64])
        >>> print(target.shape)   # torch.Size([5])
    """

    def __init__(
        self,
        series_list: List[np.ndarray],
        context_len: int = 64,
        horizon_len: int = 5,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        """
        Initialize random window dataset with pre-sampled windows.

        Args:
            series_list: List of 1D numpy arrays (one per time series)
            context_len: Length of context window (must be multiple of 32 for TimesFM)
            horizon_len: Length of forecast horizon
            num_samples: Number of random windows to pre-sample
            seed: Random seed for reproducibility

        Raises:
            ValueError: If no series long enough for context + horizon
        """
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: List[Tuple[int, int]] = []

        # Use numpy random generator for reproducibility
        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len

        # Filter series that are long enough
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        if not valid:
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len}. Shortest series: "
                f"{min(len(s) for s in series_list)}"
            )

        logger.info(
            f"Creating random window dataset: {len(valid)} valid series "
            f"(need >= {min_len} points), {num_samples} random samples"
        )

        # Pre-sample random (series_idx, start_point) pairs
        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))

    def __len__(self) -> int:
        """Return number of pre-sampled windows."""
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (context, target) window.

        Args:
            i: Index of pre-sampled window

        Returns:
            Tuple of (context, target) tensors:
                - context: torch.Tensor of shape (context_len,)
                - target: torch.Tensor of shape (horizon_len,)

        Raises:
            IndexError: If i is out of bounds
        """
        if i < 0 or i >= len(self.samples):
            raise IndexError(f"Index {i} out of bounds for dataset of size {len(self.samples)}")

        idx, start = self.samples[i]

        # Bounds check for idx and start
        if idx < 0 or idx >= len(self.series_list):
            raise IndexError(f"Series index {idx} out of bounds for {len(self.series_list)} series")

        series = self.series_list[idx]
        if start < 0 or start >= len(series):
            raise ValueError(f"Start position {start} invalid for series of length {len(series)}")

        end = start + self.context_len + self.horizon_len
        if end > len(series):
            raise ValueError(f"Window end {end} exceeds series length {len(series)}")

        # Extract context and target windows
        context = torch.from_numpy(
            series[start : start + self.context_len]
        ).float()
        target = torch.from_numpy(
            series[start + self.context_len : end]
        ).float()

        return context, target


class TimeSeriesLastWindowDataset(Dataset):
    """
    Validation dataset using the last window of each time series.

    This dataset uses only the final (context, horizon) window from each
    series, simulating real-world forecasting where we predict the
    most recent future period.

    Creates tensors on-the-fly to avoid memory overhead (unlike pre-creating
    all tensors in __init__ which wastes memory for large datasets).

    Attributes:
        series_list (List[np.ndarray]): List of 1D time series arrays
        context_len (int): Length of context window
        horizon_len (int): Length of forecast horizon

    Example:
        >>> series_list = [np.array([0.1, 0.2, 0.3, ...]), ...]
        >>> val_dataset = TimeSeriesLastWindowDataset(
        ...     series_list, context_len=64, horizon_len=5
        ... )
        >>> context, target = val_dataset[0]
        >>> print(context.shape)  # torch.Size([64])
        >>> print(target.shape)   # torch.Size([5])
    """

    def __init__(
        self,
        series_list: List[np.ndarray],
        context_len: int = 64,
        horizon_len: int = 5,
    ):
        """
        Initialize last-window validation dataset.

        Args:
            series_list: List of 1D numpy arrays (one per time series)
            context_len: Length of context window
            horizon_len: Length of forecast horizon
        """
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.valid_indices: List[int] = []
        min_len = context_len + horizon_len

        # Store indices of series that are long enough
        for idx, s in enumerate(series_list):
            if len(s) >= min_len:
                self.valid_indices.append(idx)

        logger.info(
            f"Created last-window dataset: {len(self.valid_indices)} validation samples "
            f"(from {len(series_list)} series)"
        )

    def __len__(self) -> int:
        """Return number of validation windows."""
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single validation window.

        Args:
            i: Index of validation window

        Returns:
            Tuple of (context, target) tensors:
                - context: torch.Tensor of shape (context_len,)
                - target: torch.Tensor of shape (horizon_len,)

        Raises:
            IndexError: If i is out of bounds
        """
        if i < 0 or i >= len(self.valid_indices):
            raise IndexError(f"Index {i} out of bounds for dataset of size {len(self.valid_indices)}")

        series_idx = self.valid_indices[i]
        series = self.series_list[series_idx]

        # Calculate start position (last window)
        start = len(series) - self.context_len - self.horizon_len

        # Create tensors on-the-fly (memory efficient)
        context = torch.from_numpy(series[start : start + self.context_len]).float()
        target = torch.from_numpy(series[start + self.context_len : start + self.context_len + self.horizon_len]).float()

        return context, target


def create_volatility_datasets(
    ohlcv_data: pd.DataFrame,
    context_len: int = 64,
    horizon_len: int = 5,
    num_train_samples: int = 5000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset, TimeSeriesLastWindowDataset]:
    """
    Create TimesFM 2.5 datasets from OHLCV data with temporal split.

    This function:
    1. Calculates Parkinson volatility from OHLCV data
    2. Applies chronological temporal split (70/15/15)
    3. Creates train/val/test datasets with appropriate sampling

    Args:
        ohlcv_data: DataFrame with columns [date, open, high, low, close, volume]
        context_len: Length of context window (default: 64 for TimesFM 2.5)
        horizon_len: Length of forecast horizon (default: 5 for 5-day forecasts)
        num_train_samples: Number of random windows to sample for training
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> ohlcv_df = pd.read_csv('stock_ohlcv.csv')
        >>> train_ds, val_ds, test_ds = create_volatility_datasets(ohlcv_df)
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    logger.info("Creating volatility datasets from OHLCV data...")

    # Step 0: Validate input data
    ohlcv_data = validate_ohlcv_data(ohlcv_data, required_cols=['high', 'low'])

    # Step 1: Calculate Parkinson volatility (returns pd.Series)
    logger.info("Step 1: Calculating Parkinson volatility...")
    parkinson_series = calculate_parkinson_volatility(ohlcv_data).values.astype(np.float32)

    # Step 2: Create DataFrame for temporal_split with actual dates from OHLCV data
    logger.info("Step 2: Creating DataFrame with actual dates from OHLCV data...")
    if 'date' in ohlcv_data.columns:
        dates = pd.to_datetime(ohlcv_data['date'])
        # Ensure dates length matches parkinson_series length
        if len(dates) < len(parkinson_series):
            raise ValueError(
                f"Date array ({len(dates)}) is shorter than parkinson series "
                f"({len(parkinson_series)}). Cannot maintain temporal integrity."
            )
        dates = dates[:len(parkinson_series)]
    else:
        logger.warning("No 'date' column in OHLCV data - creating synthetic date range starting from today")
        dates = pd.date_range(start=pd.Timestamp.today(), periods=len(parkinson_series), freq='D')

    parkinson_df = pd.DataFrame({
        'date': dates,
        'parkinson_volatility': parkinson_series
    })

    # Step 3: Apply temporal split (chronological 70/15/15)
    logger.info("Step 3: Applying temporal split (70/15/15)...")
    train_data, val_data, test_data = temporal_split_dataframe(
        parkinson_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # Step 3.5: Validate temporal integrity
    logger.info("Step 3.5: Validating temporal integrity...")
    validate_temporal_integrity(train_data, val_data, test_data, date_column='date')

    train_series = train_data['parkinson_volatility'].values.astype(np.float32)
    val_series = val_data['parkinson_volatility'].values.astype(np.float32)
    test_series = test_data['parkinson_volatility'].values.astype(np.float32)

    logger.info(
        f"Temporal split complete: "
        f"Train={len(train_series)}, Val={len(val_series)}, Test={len(test_series)}"
    )

    # Step 4: Create datasets
    logger.info("Step 4: Creating TimesFM 2.5 datasets...")

    # Training: Random window sampling
    train_dataset = TimeSeriesRandomWindowDataset(
        [train_series],  # Wrap in list (single series for now)
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=num_train_samples,
        seed=seed
    )

    # Validation: Last window only
    val_dataset = TimeSeriesLastWindowDataset(
        [val_series],
        context_len=context_len,
        horizon_len=horizon_len
    )

    # Test: Last window only
    test_dataset = TimeSeriesLastWindowDataset(
        [test_series],
        context_len=context_len,
        horizon_len=horizon_len
    )

    logger.info(
        f"Datasets created: "
        f"Train={len(train_dataset)} samples, "
        f"Val={len(val_dataset)} samples, "
        f"Test={len(test_dataset)} samples"
    )

    return train_dataset, val_dataset, test_dataset


def create_multi_stock_volatility_datasets(
    ohlcv_data_dict: dict,
    context_len: int = 64,
    horizon_len: int = 5,
    num_train_samples: int = 5000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset, TimeSeriesLastWindowDataset]:
    """
    Create TimesFM 2.5 datasets for multiple stocks with temporal split.

    This function processes multiple stocks, applies temporal split to each,
    and creates combined datasets for training/val/test.

    Args:
        ohlcv_data_dict: Dictionary mapping stock_id to OHLCV DataFrame
        context_len: Length of context window (default: 64)
        horizon_len: Length of forecast horizon (default: 5)
        num_train_samples: Number of random windows per epoch (default: 5000)
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> stocks = {
        ...     'ACB': pd.read_csv('ACB_ohlcv.csv'),
        ...     'FPT': pd.read_csv('FPT_ohlcv.csv'),
        ... }
        >>> train_ds, val_ds, test_ds = create_multi_stock_volatility_datasets(stocks)
    """
    logger.info(f"Creating multi-stock volatility datasets for {len(ohlcv_data_dict)} stocks...")

    all_train_series = []
    all_val_series = []
    all_test_series = []

    # Process each stock separately
    for stock_id, ohlcv_data in ohlcv_data_dict.items():
        logger.info(f"Processing stock {stock_id}...")

        # Calculate Parkinson volatility (returns pd.Series)
        parkinson_series = calculate_parkinson_volatility(ohlcv_data).values.astype(np.float32)

        # Create DataFrame for temporal_split with date column
        # Use dates from original data if available, otherwise create range
        if 'date' in ohlcv_data.columns:
            dates = pd.to_datetime(ohlcv_data['date'])
            # Ensure dates length matches parkinson_series length
            if len(dates) < len(parkinson_series):
                logger.warning(
                    f"Stock {stock_id}: Date array ({len(dates)}) is shorter than "
                    f"parkinson series ({len(parkinson_series)}). Using available dates only."
                )
        else:
            logger.warning("No 'date' column in OHLCV data - creating synthetic date range starting from today")
            dates = pd.date_range(start=pd.Timestamp.today(), periods=len(parkinson_series), freq='D')

        parkinson_df = pd.DataFrame({
            'date': dates[:len(parkinson_series)],  # Ensure lengths match
            'parkinson_volatility': parkinson_series
        })

        # Apply temporal split
        train_data, val_data, test_data = temporal_split_dataframe(
            parkinson_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # Collect series
        all_train_series.append(train_data['parkinson_volatility'].values.astype(np.float32))
        all_val_series.append(val_data['parkinson_volatility'].values.astype(np.float32))
        all_test_series.append(test_data['parkinson_volatility'].values.astype(np.float32))

    logger.info(
        f"Processed {len(all_train_series)} stocks: "
        f"Train series avg={np.mean([len(s) for s in all_train_series]):.0f} points"
    )

    # Create datasets with all series
    train_dataset = TimeSeriesRandomWindowDataset(
        all_train_series,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=num_train_samples,
        seed=seed
    )

    val_dataset = TimeSeriesLastWindowDataset(
        all_val_series,
        context_len=context_len,
        horizon_len=horizon_len
    )

    test_dataset = TimeSeriesLastWindowDataset(
        all_test_series,
        context_len=context_len,
        horizon_len=horizon_len
    )

    logger.info(
        f"Multi-stock datasets created: "
        f"Train={len(train_dataset)} samples, "
        f"Val={len(val_dataset)} samples, "
        f"Test={len(test_dataset)} samples"
    )

    return train_dataset, val_dataset, test_dataset
