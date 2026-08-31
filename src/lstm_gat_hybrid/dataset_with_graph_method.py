"""
LEGACY / NON-AUTHORITATIVE (external senior review HIGH-02, 2026-08-29). This news-fusion-lineage pipeline
computes outlier statistics over the COMPLETE ticker history (``remove_outliers``) and, in the
``*_with_graph_method`` entry, can apply that processing BEFORE the chronological split — a look-ahead
leakage path (mean/std/clip bounds may see validation/test rows). It is NOT the pipeline behind any published
result. The AUTHORITATIVE, leakage-safe path for the paper is the masked-rich runner
``baselines/2026-08-21_har_anchored_residual/code/{masked_rich.py,run_masked_rich.py}`` (train-only scalers +
train-only graph edges, regression-tested by ``test_train_only_invariance_no_leakage``). Do NOT use this
module for reproducing paper numbers; treat its outputs as non-reproducible legacy.

Split-first data-loading pipeline for the news-fusion baseline lineage.

Provides remove_outliers/_reindex_to_common_dates (the P1.2 cross-stock
date-alignment fix), _load_raw_stock_data, _split_raw_data_by_date,
_generate_har_for_split, and create_multi_stock_dataloaders_with_graph_method_fixed
— the split-first pipeline used by baselines/*/code/dataset_*.py
(e.g. dataset_dual_news.py) via MultiStockDatasetWithPreSplitData
(dataset_presplit.py).

The original MultiStockDatasetWithGraphMethod class and its non-"_fixed"
wrapper (create_multi_stock_dataloaders_with_graph_method) were archived
2026-08-02 (confirmed dead: not used by any live baseline, only by their
own now-archived tests/debug scripts, and superseded by the split-first
pipeline below) — see archive/lstm_gat_hybrid_legacy/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from scipy import stats
from torch.utils.data import DataLoader

from src.common.har_features import generate_har_features
from src.common.data_normalization import VolatilityNormalizer


def remove_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Winsorize outliers using z-score method (clip values, keep every row).

    Previously dropped outlier rows entirely. Outlier detection runs independently
    per ticker, so dropping rows desynced each ticker's row position from calendar
    date relative to every other ticker (P1.2, the cross-stock date-misalignment
    bug: position i stopped meaning "the same trading day" across tickers once
    different rows had been removed from each). Clipping in place preserves row
    count and date alignment — see _reindex_to_common_dates() for the remaining
    alignment step (per-ticker date gaps unrelated to outliers, e.g. trading
    halts) that this alone doesn't cover.
    """
    if len(df) == 0 or 'parkinson_variance' not in df.columns:
        return df

    volatility_values = df['parkinson_variance'].values

    if np.std(volatility_values) == 0:
        return df

    z_scores = stats.zscore(volatility_values)
    outlier_mask = np.abs(z_scores) >= n_std
    outlier_count = int(outlier_mask.sum())

    if outlier_count > 0:
        mean = volatility_values.mean()
        std = volatility_values.std()
        lower, upper = mean - n_std * std, mean + n_std * std
        df = df.copy()
        df.loc[outlier_mask, 'parkinson_variance'] = np.clip(
            volatility_values[outlier_mask], lower, upper)
        print(f"    [Outlier Winsorize] Clipped {outlier_count} outliers "
              f"({outlier_count/len(df)*100:.2f}%) to [{lower:.6f}, {upper:.6f}]")

    return df


def _reindex_to_common_dates(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align every ticker's DataFrame to the same trading-date axis (P1.2 fix).

    Restricts to the intersection of dates present in EVERY ticker, so row
    position i is guaranteed to be the same calendar date for every ticker —
    a precondition the rest of this pipeline's positional indexing
    (_split_raw_data_by_date, _create_sequences-style window construction)
    silently assumed but never enforced. Without this, tickers with different
    listing dates or sporadic gaps (trading halts) end up compared at the same
    "position" while actually representing different real-world dates.

    Args:
        stock_data: ticker -> DataFrame with a 'date' column (any per-ticker
            date range/gaps; not required to already be aligned)

    Returns:
        ticker -> DataFrame restricted to the common date index, sorted
        chronologically, all with identical length and date sequence.
    """
    if not stock_data:
        return stock_data

    date_sets = [set(df['date']) for df in stock_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    print(f"\n[_reindex_to_common_dates] Common trading dates across "
          f"{len(stock_data)} tickers: {len(common_dates)}")
    if common_dates:
        print(f"  Range: {common_dates[0]} -> {common_dates[-1]}")

    aligned = {}
    for ticker, df in stock_data.items():
        df_sorted = df.sort_values('date').reset_index(drop=True)
        aligned[ticker] = df_sorted[df_sorted['date'].isin(common_dates)].reset_index(drop=True)

    lengths = {len(df) for df in aligned.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"_reindex_to_common_dates produced mismatched lengths {lengths} — "
            "a ticker likely has duplicate dates; investigate before proceeding.")

    return aligned


# ========================================================================
# CRITICAL FIX #4: Split-First Data Loading (HAR Features Leakage Fix)
# ========================================================================

def _load_raw_stock_data(
    data_dir: str,
    remove_outliers: bool = True,
    n_std: float = 3.0
) -> Dict[str, pd.DataFrame]:
    """
    Load raw stock data WITHOUT generating HAR features yet.

    This is the first step in the split-first approach to prevent HAR leakage.

    Args:
        data_dir: Directory containing processed CSV files
        remove_outliers: Whether to remove outliers
        n_std: Number of standard deviations for outlier threshold

    Returns:
        Dictionary mapping stock names to raw DataFrames
    """
    # Import the remove_outliers function to avoid name collision
    # The parameter remove_outliers (bool) shadows the function name
    import sys
    module = sys.modules[__name__]
    remove_outliers_func = module.remove_outliers
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    stock_data = {}
    loaded_count = 0

    for csv_file in csv_files:
        stock_name = csv_file.stem.replace('_processed', '')
        df = pd.read_csv(csv_file)

        if 'date' not in df.columns or 'parkinson_variance' not in df.columns:
            print(f"[Warning] Skipping {stock_name}: missing required columns")
            continue

        if len(df) < 30:
            print(f"[Warning] Skipping {stock_name}: insufficient data ({len(df)} rows)")
            continue

        # Calculate returns
        df['returns'] = df['parkinson_variance'].pct_change()
        df['returns'] = df['returns'].fillna(0)

        # Apply outlier winsorization (call the module-level function). This clips
        # extreme values in place rather than dropping rows, so it never changes
        # len(df) or which dates are present — see remove_outliers()'s docstring
        # for why row-dropping here used to desync cross-ticker date alignment.
        if remove_outliers:
            df = remove_outliers_func(df, n_std=n_std)

        stock_data[stock_name] = df
        loaded_count += 1

    print(f"[_load_raw_stock_data] Successfully loaded {loaded_count} stocks")

    if loaded_count == 0:
        raise ValueError("No valid stock data loaded")

    # P1.2 fix: align every ticker to the same trading-date axis before any
    # downstream code does positional indexing (_split_raw_data_by_date,
    # sequence-window construction) that silently assumed — but never
    # enforced — that position i means the same calendar date for every ticker.
    stock_data = _reindex_to_common_dates(stock_data)

    return stock_data


def _split_raw_data_by_date(
    stock_data: Dict[str, pd.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Dict, Dict, Dict, int, int, int]:
    """
    Split raw stock data chronologically BEFORE generating HAR features.

    This prevents HAR rolling means from leaking future information.

    Args:
        stock_data: Dictionary of raw DataFrames (no HAR features yet)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test ratio

    Returns:
        train_raw, val_raw, test_raw, train_end_idx, val_end_idx, min_length
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Positional slicing below is only a valid DATE split if every ticker is
    # already aligned to the same date axis (see _reindex_to_common_dates,
    # the P1.2 fix) — assert it rather than silently mis-splitting again.
    lengths = {len(df) for df in stock_data.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"_split_raw_data_by_date got mismatched per-ticker lengths {lengths} — "
            "stock_data must be reindexed to a common date axis first "
            "(_reindex_to_common_dates), otherwise this positional split silently "
            "compares different calendar dates across tickers (P1.2).")

    # Find minimum length across all stocks
    min_length = min(len(df) for df in stock_data.values())

    # Calculate split points by DATE index (not sequence index)
    train_end_idx = int(min_length * train_ratio)
    val_end_idx = int(min_length * (train_ratio + val_ratio))

    print(f"\n[_split_raw_data_by_date] Chronological split by DATE index:")
    print(f"  Min length across stocks: {min_length}")
    print(f"  Train: [0, {train_end_idx})")
    print(f"  Val:   [{train_end_idx}, {val_end_idx})")
    print(f"  Test:  [{val_end_idx}, {min_length})")

    # Split raw data
    train_raw = {}
    val_raw = {}
    test_raw = {}

    for stock_name, df in stock_data.items():
        # Ensure data is sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        # Split by date index
        train_raw[stock_name] = df.iloc[:train_end_idx].copy()
        val_raw[stock_name] = df.iloc[train_end_idx:val_end_idx].copy()
        test_raw[stock_name] = df.iloc[val_end_idx:min_length].copy()

    print(f"[_split_raw_data_by_date] Split complete:")
    print(f"  Train samples: {len(list(train_raw.values())[0])}")
    print(f"  Val samples:   {len(list(val_raw.values())[0])}")
    print(f"  Test samples:  {len(list(test_raw.values())[0])}")

    return train_raw, val_raw, test_raw, train_end_idx, val_end_idx, min_length


def _generate_har_for_split(
    raw_split: Dict[str, pd.DataFrame],
    split_name: str
) -> Dict[str, pd.DataFrame]:
    """
    Generate HAR features for a single split (train/val/test).

    This ensures HAR features are computed ONLY on data from that split,
    preventing leakage from future splits.

    Args:
        raw_split: Dictionary of raw DataFrames for this split
        split_name: Name of split ('train', 'val', 'test')

    Returns:
        Dictionary of DataFrames with HAR features added
    """
    print(f"\n[_generate_har_for_split] Generating HAR features for {split_name} split...")

    split_with_har = {}

    for stock_name, df in raw_split.items():
        df_copy = df.copy()

        # Skip if insufficient data for HAR features
        if len(df_copy) < 23:  # Need at least 22 + 1 for monthly window
            print(f"  [Warning] Skipping {stock_name} in {split_name}: insufficient data ({len(df_copy)} rows)")
            continue

        try:
            df_har = generate_har_features(df_copy)
            df_har['parkinson_variance'] = df_copy['parkinson_variance'].values
            split_with_har[stock_name] = df_har
        except ValueError as e:
            # Handle edge case: constant volatility (zero variance)
            if 'all zeros' in str(e) or 'all NaN' in str(e):
                print(f"  [Warning] {stock_name} in {split_name}: constant volatility, using raw features")
                # Use raw volatility as fallback
                df_copy['har_daily_vol'] = df_copy['parkinson_variance'].values
                df_copy['har_weekly_vol'] = df_copy['parkinson_variance'].values
                df_copy['har_monthly_vol'] = df_copy['parkinson_variance'].values
                split_with_har[stock_name] = df_copy
            else:
                raise e

    print(f"[_generate_har_for_split] Generated HAR features for {len(split_with_har)} stocks ({split_name})")

    return split_with_har


def create_multi_stock_dataloaders_with_graph_method_fixed(
    data_dir: str,
    seq_length: int = 22,
    forecast_horizon: int = 5,
    graph_method: str = 'correlation',
    graph_threshold: float = 0.7,
    k_neighbors: int = 8,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    normalize: bool = True,
    remove_outliers: bool = True,
    n_std: float = 3.0,
    data_augmentation: bool = True,
    augmentation_prob: float = 0.3,
    augmentation_factor: float = 0.1
) -> Tuple:
    """
    Create train, validation, and test dataloaders with CRITICAL FIX #4 applied.

    FIX #4: Split raw data FIRST, then generate HAR features separately.
    This prevents HAR rolling means from leaking future information into training.

    OLD (WRONG): Load all data → Generate HAR → Create sequences → Split sequences
    NEW (CORRECT): Load raw data → Split raw data → Generate HAR separately → Create sequences separately

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Input sequence length
        forecast_horizon: Forecast horizon
        graph_method: 'correlation' or 'knn'
        graph_threshold: Correlation threshold for 'correlation' method
        k_neighbors: k neighbors for 'knn' method
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test ratio
        num_workers: Number of workers for DataLoader
        normalize: Whether to normalize data
        remove_outliers: Whether to remove outliers
        n_std: Number of standard deviations for outlier threshold
        data_augmentation: Whether to apply data augmentation
        augmentation_prob: Probability of applying augmentation
        augmentation_factor: Strength of augmentation

    Returns:
        train_loader, val_loader, test_loader, datasets (tuple of 4 datasets)
    """
    print(f"\n[create_multi_stock_dataloaders_with_graph_method_fixed]")
    print(f"  Graph Method: {graph_method}")
    if graph_method == 'correlation':
        print(f"  Correlation Threshold: {graph_threshold}")
    elif graph_method == 'knn':
        print(f"  k-Neighbors: {k_neighbors}")
    print(f"  *** CRITICAL FIX #4 ACTIVE: Split-first approach to prevent HAR leakage ***")

    # ========================================================================
    # Step 1: Load RAW data (no HAR features yet)
    # ========================================================================
    print("\n[Step 1] Loading raw stock data...")
    stock_data_raw = _load_raw_stock_data(
        data_dir=data_dir,
        remove_outliers=remove_outliers,
        n_std=n_std
    )

    # ========================================================================
    # Step 2: Split RAW data chronologically by DATE index
    # ========================================================================
    print("\n[Step 2] Splitting raw data chronologically...")
    train_raw, val_raw, test_raw, train_end_idx, val_end_idx, min_length = \
        _split_raw_data_by_date(stock_data_raw, train_ratio, val_ratio, test_ratio)

    # ========================================================================
    # Step 3: Generate HAR features SEPARATELY for each split
    # ========================================================================
    print("\n[Step 3] Generating HAR features separately for each split...")
    train_har = _generate_har_for_split(train_raw, 'train')
    val_har = _generate_har_for_split(val_raw, 'val')
    test_har = _generate_har_for_split(test_raw, 'test')

    # ========================================================================
    # Step 4: Create datasets from pre-split HAR data
    # ========================================================================
    print("\n[Step 4] Creating datasets from pre-split HAR data...")

    # Get common stock names - MUST be in train, then check val/test availability
    train_stocks = set(train_har.keys())
    val_stocks = set(val_har.keys())
    test_stocks = set(test_har.keys())

    # Only require stocks to be in training
    common_stocks = train_stocks
    common_stocks = sorted(common_stocks)

    print(f"[Step 4] Stocks in training: {len(common_stocks)}")
    print(f"  Stocks also in val: {len(train_stocks & val_stocks)}")
    print(f"  Stocks also in test: {len(train_stocks & test_stocks)}")

    # Filter HAR data to only common stocks (train only requirement)
    train_har_common = {k: v for k, v in train_har.items() if k in common_stocks}

    # For val/test, only include stocks that exist in those splits
    val_har_common = {k: v for k, v in val_har.items() if k in common_stocks if k in val_stocks}
    test_har_common = {k: v for k, v in test_har.items() if k in common_stocks if k in test_stocks}

    # Filter raw data similarly
    train_raw_common = {k: v for k, v in train_raw.items() if k in common_stocks}
    val_raw_common = {k: v for k, v in val_raw.items() if k in common_stocks if k in val_raw}
    test_raw_common = {k: v for k, v in test_raw.items() if k in common_stocks if k in test_raw}

    print(f"  Final stocks - Train: {len(train_har_common)}, Val: {len(val_har_common)}, Test: {len(test_har_common)}")

    # Import the lightweight dataset class
    from .dataset_presplit import MultiStockDatasetWithPreSplitData
    from .config import LSTMGATConfig
    config = LSTMGATConfig()

    # Create training dataset with PRE-SPLIT data (NO loading from directory)
    print("\n[Step 4] Creating datasets from PRE-SPLIT HAR data (NO leakage)...")
    train_dataset = MultiStockDatasetWithPreSplitData(
        stock_data=train_raw_common,
        stock_data_with_har=train_har_common,
        stock_names=common_stocks,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        graph_threshold=graph_threshold,
        k_neighbors=k_neighbors,
        normalize=normalize,  # ✅ FIX: Use parameter instead of hardcoded False
        train_mode=True,
        config=config
    )

    # Create validation dataset with PRE-SPLIT data
    val_dataset = MultiStockDatasetWithPreSplitData(
        stock_data=val_raw_common,
        stock_data_with_har=val_har_common,
        stock_names=common_stocks,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        graph_threshold=graph_threshold,
        k_neighbors=k_neighbors,
        normalize=normalize,  # ✅ FIX: Use parameter instead of hardcoded False
        train_mode=False,
        config=config
    )

    # Create test dataset with PRE-SPLIT data
    test_dataset = MultiStockDatasetWithPreSplitData(
        stock_data=test_raw_common,
        stock_data_with_har=test_har_common,
        stock_names=common_stocks,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        graph_threshold=graph_threshold,
        k_neighbors=k_neighbors,
        normalize=normalize,  # ✅ FIX: Use parameter instead of hardcoded False
        train_mode=False,
        config=config
    )

    print(f"\n[Step 4] Dataset creation complete:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(test_dataset)} sequences")

    # ========================================================================
    # Step 5: Fit normalizers on TRAINING HAR data only
    # ========================================================================
    if normalize:
        print("\n[Step 5] Fitting normalizers on TRAINING HAR data only...")

        # Re-initialize normalizers for common stocks only
        for stock_name in common_stocks:
            train_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            train_dataset.target_normalizers[stock_name] = VolatilityNormalizer()
            val_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            val_dataset.target_normalizers[stock_name] = VolatilityNormalizer()
            test_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            test_dataset.target_normalizers[stock_name] = VolatilityNormalizer()

        for stock_idx, stock_name in enumerate(train_dataset.stock_names):
            # Collect features and targets from TRAINING sequences only
            train_features = []
            train_targets = []

            for seq in train_dataset.sequences:
                x, adj_matrix, y = seq
                train_features.append(x[:, stock_idx, :])
                train_targets.append(y[stock_idx])

            # Concatenate all training samples
            train_features = np.concatenate(train_features, axis=0)
            train_targets = np.array(train_targets)

            # Fit normalizers on TRAINING data only
            train_dataset.feature_normalizers[stock_name].fit(train_features)
            train_dataset.target_normalizers[stock_name].fit(train_targets.reshape(-1, 1))

            # Copy fitted normalizers to val and test
            val_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            val_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]
            test_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            test_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]

            print(f"  [Fitting] {stock_name}: features={train_features.shape}, targets={train_targets.shape}")

        print("[Step 5] Normalizers fitted on TRAINING HAR data and copied to val/test")

    # ========================================================================
    # Step 6: Create dataloaders
    # ========================================================================
    print("\n[Step 6] Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: Don't shuffle time series!
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\n[create_multi_stock_dataloaders_with_graph_method_fixed] COMPLETE:")
    print(f"  - NO HAR leakage (features computed separately for each split)")
    print(f"  - NO normalization leakage (fitted on training only)")
    print(f"  - NO graph leakage (per-sequence construction)")
    print(f"  - Train: {len(train_dataset)} sequences")
    print(f"  - Val:   {len(val_dataset)} sequences")
    print(f"  - Test:  {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
