"""
Enhanced Multi-Stock Dataset with Graph Method Selection

Supports both graph construction methods:
1. Correlation-based (from Sonani et al. 2025 paper)
2. k-NN sparse graph (current method)

Allows comparison to determine optimal graph construction for volatility prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime
from scipy import stats

from src.common.har_features import generate_har_features
from src.common.data_normalization import VolatilityNormalizer
from .config import LSTMGATConfig
from .graph_utils_fixed import DynamicGraphBuilder  # For k-NN method
from .graph_correlation import construct_correlation_graph  # For correlation method


class MultiStockDatasetWithGraphMethod(Dataset):
    """
    PyTorch Dataset for LSTM-GAT with selectable graph construction method

    Graph Methods:
    - 'correlation': Pearson correlation threshold (|corr| > 0.7) - from paper
    - 'knn': k-NN sparse graph - current method
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        graph_method: str = 'correlation',
        graph_threshold: float = 0.7,
        k_neighbors: int = 8,
        normalize: bool = True,
        remove_outliers: bool = True,
        n_std: float = 3.0,
        data_augmentation: bool = True,
        augmentation_prob: float = 0.3,
        augmentation_factor: float = 0.1,
        train_mode: bool = False
    ):
        """
        Initialize multi-stock dataset with graph method selection

        Args:
            data_dir: Directory containing processed CSV files
            seq_length: Input sequence length
            forecast_horizon: Forecast horizon (5-day ahead)
            graph_method: 'correlation' (paper) or 'knn' (current)
            graph_threshold: Correlation threshold for 'correlation' method (default: 0.7)
            k_neighbors: Number of neighbors for 'knn' method (default: 8)
            normalize: Whether to normalize data
            remove_outliers: Whether to remove outliers (Phase 1)
            n_std: Number of standard deviations for outlier threshold
            data_augmentation: Whether to apply data augmentation (Phase 1)
            augmentation_prob: Probability of applying augmentation (0.3 = 30%)
            augmentation_factor: Strength of augmentation (default: 0.1)
            train_mode: If True, apply data augmentation during training
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.graph_method = graph_method
        self.graph_threshold = graph_threshold
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.n_std = n_std
        self.data_augmentation = data_augmentation
        self.augmentation_prob = augmentation_prob
        self.augmentation_factor = augmentation_factor
        self.train_mode = train_mode
        self.config = LSTMGATConfig()

        print(f"[MultiStockDatasetWithGraphMethod] Configuration:")
        print(f"  - Graph Method: {graph_method}")
        if graph_method == 'correlation':
            print(f"  - Correlation Threshold: {graph_threshold} (from paper)")
        elif graph_method == 'knn':
            print(f"  - k-Neighbors: {k_neighbors}")
        print(f"  - Outlier Removal: {remove_outliers} (n_std={n_std})")
        print(f"  - Data Augmentation: {data_augmentation} (prob={augmentation_prob}, factor={augmentation_factor})")
        print(f"  - Train Mode: {train_mode}")

        # Load and process data
        self.stock_data = self._load_multi_stock_data(data_dir)
        self.stock_names = sorted(self.stock_data.keys())

        print(f"[MultiStockDatasetWithGraphMethod] Loaded {len(self.stock_names)} stocks: {self.stock_names[:5]}...")

        # Generate HAR features for each stock
        self.stock_data_with_har = self._generate_features_for_all_stocks()

        # Initialize normalizers (DO NOT fit yet - will fit on training data only)
        self.feature_normalizers = {}
        self.target_normalizers = {}
        if normalize:
            self._initialize_normalizers(fit=False)  # Only initialize, DO NOT fit

        # Create sequences with selected graph method
        self.sequences = self._create_sequences()

        print(f"[MultiStockDatasetWithGraphMethod] Created {len(self.sequences)} sequences")

    def _load_multi_stock_data(self, data_dir: str) -> Dict:
        """Load data for all stocks with outlier removal"""
        data_path = Path(data_dir)
        csv_files = sorted(data_path.glob('*.csv'))

        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        stock_data = {}
        loaded_count = 0
        total_outliers_removed = 0

        for csv_file in csv_files:
            stock_name = csv_file.stem.replace('_processed', '')
            df = pd.read_csv(csv_file)

            if 'date' not in df.columns or 'parkinson_volatility' not in df.columns:
                print(f"[Warning] Skipping {stock_name}: missing required columns")
                continue

            if len(df) < 30:
                print(f"[Warning] Skipping {stock_name}: insufficient data ({len(df)} rows)")
                continue

            # Calculate returns
            df['returns'] = df['parkinson_volatility'].pct_change()
            df['returns'] = df['returns'].fillna(0)

            # Apply outlier removal
            if self.remove_outliers:
                original_len = len(df)
                from src.lstm_gat_hybrid.dataset_with_graph_method import remove_outliers as remove_outliers_func
                df = remove_outliers_func(df, n_std=self.n_std)
                outliers_removed = original_len - len(df)
                total_outliers_removed += outliers_removed
                if len(df) < 30:
                    print(f"[Warning] Skipping {stock_name}: insufficient data after outlier removal ({len(df)} rows)")
                    continue

            stock_data[stock_name] = df
            loaded_count += 1

        print(f"[_load_multi_stock_data] Successfully loaded {loaded_count} stocks")
        if self.remove_outliers:
            print(f"[_load_multi_stock_data] Total outliers removed: {total_outliers_removed}")

        if loaded_count == 0:
            raise ValueError("No valid stock data loaded")

        return stock_data

    def _generate_features_for_all_stocks(self) -> Dict:
        """Generate HAR features for all stocks"""
        stock_features = {}

        for stock_name, df in self.stock_data.items():
            df_copy = df.copy()
            df_har = generate_har_features(df_copy)
            df_har['parkinson_volatility'] = df_copy['parkinson_volatility'].values
            stock_features[stock_name] = df_har

        return stock_features

    def _initialize_normalizers(self, fit=False):
        """
        Initialize normalizers for each stock

        Args:
            fit: If True, fit normalizers on data. If False, only initialize.
                 IMPORTANT: Should only fit on TRAINING data to avoid data leakage!
        """
        for stock_name in self.stock_names:
            features = self.stock_data_with_har[stock_name][['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].values
            targets = self.stock_data_with_har[stock_name]['parkinson_volatility'].values

            self.feature_normalizers[stock_name] = VolatilityNormalizer()
            self.target_normalizers[stock_name] = VolatilityNormalizer()

            if fit:
                # ONLY fit on training data (called after temporal split)
                self.feature_normalizers[stock_name].fit(features)
                self.target_normalizers[stock_name].fit(targets.reshape(-1, 1))

    def fit_normalizers_on_subset(self, subset_indices):
        """
        Fit normalizers on a subset of data (e.g., training data only)

        This should be called AFTER temporal split to avoid data leakage.

        Args:
            subset_indices: List of indices to fit normalizers on (e.g., training indices)
        """
        print(f"[fit_normalizers_on_subset] Fitting normalizers on {len(subset_indices)} samples")

        for stock_idx, stock_name in enumerate(self.stock_names):
            # Collect features and targets from subset
            subset_features = []
            subset_targets = []

            for idx in subset_indices:
                if idx < len(self.sequences):
                    x, adj_matrix, y = self.sequences[idx]
                    # x: [seq_len, num_stocks, num_features]
                    # y: [num_stocks]
                    subset_features.append(x[:, stock_idx, :])  # All timesteps for this stock
                    subset_targets.append(y[stock_idx])

            # Concatenate all samples
            subset_features = np.concatenate(subset_features, axis=0)  # [num_samples * seq_len, num_features]
            subset_targets = np.array(subset_targets)  # [num_samples]

            # Fit normalizers
            self.feature_normalizers[stock_name].fit(subset_features)
            self.target_normalizers[stock_name].fit(subset_targets.reshape(-1, 1))

            print(f"  [fit_normalizers_on_subset] {stock_name}: features shape={subset_features.shape}, targets shape={subset_targets.shape}")

    def _create_sequences(self) -> list:
        """Create multi-stock sequences with PER-SEQUENCE graph construction (NO data leakage)"""
        sequences = []

        min_length = min(len(df) for df in self.stock_data_with_har.values())

        print(f"[_create_sequences] Min length: {min_length}")
        print(f"[_create_sequences] Creating sequences with graph_method={self.graph_method}")
        print(f"[_create_sequences] PER-SEQUENCE graph construction - NO data leakage")

        # Prepare all volatility data for per-sequence graph construction
        # Truncate all stocks to minimum length (they have different lengths)
        all_volatility_list = []

        for stock in self.stock_names:
            vol_data = self.stock_data_with_har[stock]['parkinson_volatility'].values
            # Truncate to minimum length
            vol_data_truncated = vol_data[:min_length]
            all_volatility_list.append(vol_data_truncated)

        all_volatility = np.stack(all_volatility_list, axis=1)  # [min_length, num_stocks]

        if len(all_volatility) != min_length or all_volatility.shape[1] != len(self.stock_names):
            raise ValueError(f"all_volatility shape mismatch: {all_volatility.shape} vs expected ({min_length}, {len(self.stock_names)})")

        # Initialize graph builder for k-NN method
        if self.graph_method == 'knn':
            self.graph_builder = DynamicGraphBuilder(self.config)

        # Create sequences with PER-SEQUENCE graph construction
        for i in range(min_length - self.seq_length - self.forecast_horizon):
            # =====================================================================
            # BUILD GRAPH USING ONLY THIS SEQUENCE'S DATA WINDOW
            # =====================================================================
            # CRITICAL FIX: Use ONLY data from [i, i+seq_length], NOT cumulative [0, i+seq_length]
            # This ensures NO future information leaks into training
            # Seq 0: [0:22], Seq 1: [1:23], Seq 1000: [1000:1022] <- No lookahead!
            sequence_volatility = all_volatility[i:i+self.seq_length]

            if self.graph_method == 'correlation':
                # Paper's correlation-based method (applied to this sequence only)
                adj_matrix = construct_correlation_graph(
                    sequence_volatility,
                    corr_threshold=self.graph_threshold
                )

            elif self.graph_method == 'knn':
                # k-NN graph built from this sequence only
                graph_data = {
                    'volatility': sequence_volatility,
                    'returns': sequence_volatility
                }
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            else:
                raise ValueError(f"Unknown graph_method: {self.graph_method}")

            # =====================================================================
            # CREATE SEQUENCE FEATURES
            # =====================================================================
            x_all_stocks = []
            y_all_stocks = []

            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                x_seq = stock_feats[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values
                x_all_stocks.append(x_seq)

                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_target = stock_feats['parkinson_volatility'].iloc[target_idx]
                y_all_stocks.append(y_target)

            x = np.stack(x_all_stocks, axis=1)  # [seq_len, num_stocks, num_features]
            y = np.array(y_all_stocks)  # [num_stocks]

            sequences.append((x, adj_matrix, y))

        print(f"[_create_sequences] Created {len(sequences)} sequences")
        print(f"[_create_sequences] Each sequence has its OWN graph built from HISTORICAL data only")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sequence with optional data augmentation"""
        x, adj_matrix, y = self.sequences[idx]

        # Apply data augmentation if enabled
        if self.data_augmentation and self.train_mode and np.random.random() < self.augmentation_prob:
            from .dataset import augment_sequence
            x, y = augment_sequence(x, y, self.augmentation_factor)

        # Normalize features and targets (if enabled)
        if self.normalize:
            # Normalize features (per-stock normalization)
            x_normalized = np.zeros_like(x)
            for stock_idx in range(x.shape[1]):
                stock_name = self.stock_names[stock_idx]
                if stock_name in self.feature_normalizers:
                    # Normalize each feature for this stock
                    for feat_idx in range(x.shape[2]):
                        x_normalized[:, stock_idx, feat_idx] = \
                            self.feature_normalizers[stock_name].transform(
                                x[:, stock_idx, feat_idx:feat_idx+1]
                            ).flatten()
                else:
                    x_normalized[:, stock_idx, :] = x[:, stock_idx, :]

            x = x_normalized

            # Normalize targets (per-stock normalization)
            y_normalized = np.zeros_like(y)
            for stock_idx, stock_name in enumerate(self.stock_names):
                if stock_name in self.target_normalizers:
                    y_normalized[stock_idx] = \
                        self.target_normalizers[stock_name].transform(
                            y[stock_idx:stock_idx+1].reshape(1, -1)
                        ).flatten()[0]
                else:
                    y_normalized[stock_idx] = y[stock_idx]

            y = y_normalized

        x = torch.FloatTensor(x)
        adj_matrix = torch.FloatTensor(adj_matrix)
        y = torch.FloatTensor(y)

        # Create placeholder graph_data for compatibility
        graph_data = {}

        return x, adj_matrix, y, graph_data


def create_multi_stock_dataloaders_with_graph_method(
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
    Create train, validation, and test dataloaders with graph method selection

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
        test_ratio: Test set ratio
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
    print(f"\n[create_multi_stock_dataloaders_with_graph_method]")
    print(f"  Graph Method: {graph_method}")
    if graph_method == 'correlation':
        print(f"  Correlation Threshold: {graph_threshold}")
    elif graph_method == 'knn':
        print(f"  k-Neighbors: {k_neighbors}")

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # ========================================================================
    # CRITICAL FIX #1: Create ONE full dataset, then split into Subsets
    # ========================================================================
    # This ensures sequences are created BEFORE split, not after
    full_dataset = MultiStockDatasetWithGraphMethod(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        graph_threshold=graph_threshold,
        k_neighbors=k_neighbors,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for full dataset
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False
    )

    # Calculate split indices (TEMPORAL split)
    n = len(full_dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print(f"[create_multi_stock_dataloaders_with_graph_method] Temporal split:")
    print(f"  Train: 0 to {train_end} ({train_end} sequences)")
    print(f"  Val:   {train_end} to {val_end} ({val_end - train_end} sequences)")
    print(f"  Test:  {val_end} to {n} ({n - val_end} sequences)")

    # ========================================================================
    # CRITICAL FIX #2: Fit normalizers on TRAINING SEQUENCES only
    # ========================================================================
    if normalize:
        print(f"[create_multi_stock_dataloaders_with_graph_method] Fitting normalizers on TRAINING sequences only")

        # Extract training sequences ONLY
        train_sequences = [full_dataset.sequences[i] for i in range(train_end)]

        # Fit normalizers on training sequences
        for stock_idx, stock_name in enumerate(full_dataset.stock_names):
            # Collect features and targets from training sequences
            train_features = []
            train_targets = []

            for seq in train_sequences:
                x, adj_matrix, y = seq
                # x: [seq_len, num_stocks, num_features]
                # y: [num_stocks]
                train_features.append(x[:, stock_idx, :])  # All timesteps for this stock
                train_targets.append(y[stock_idx])

            # Concatenate all training samples
            train_features = np.concatenate(train_features, axis=0)  # [num_samples * seq_len, num_features]
            train_targets = np.array(train_targets)  # [num_samples]

            # Fit normalizers on TRAINING data only
            full_dataset.feature_normalizers[stock_name].fit(train_features)
            full_dataset.target_normalizers[stock_name].fit(train_targets.reshape(-1, 1))

            print(f"  [Fitting] {stock_name}: features shape={train_features.shape}, targets shape={train_targets.shape}")

        print(f"[create_multi_stock_dataloaders_with_graph_method] Normalizers fitted on TRAINING sequences only")

    # ========================================================================
    # Create Subsets for train/val/test (AFTER normalizers fitted)
    # ========================================================================
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(full_dataset, range(val_end, n))

    # Create dataloaders
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

    print(f"[create_multi_stock_dataloaders_with_graph_method] Split complete:")
    print(f"  - Train: {len(train_dataset)} sequences")
    print(f"  - Val:   {len(val_dataset)} sequences")
    print(f"  - Test:  {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)


def remove_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Remove outliers using z-score method"""
    if len(df) == 0 or 'parkinson_volatility' not in df.columns:
        return df

    volatility_values = df['parkinson_volatility'].values

    if np.std(volatility_values) == 0:
        return df

    z_scores = np.abs(stats.zscore(volatility_values))
    outlier_mask = z_scores < n_std
    df_clean = df[outlier_mask].copy()

    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"    [Outlier Removal] Removed {removed_count} outliers ({removed_count/len(df)*100:.2f}%)")

    return df_clean


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
    total_outliers_removed = 0

    for csv_file in csv_files:
        stock_name = csv_file.stem.replace('_processed', '')
        df = pd.read_csv(csv_file)

        if 'date' not in df.columns or 'parkinson_volatility' not in df.columns:
            print(f"[Warning] Skipping {stock_name}: missing required columns")
            continue

        if len(df) < 30:
            print(f"[Warning] Skipping {stock_name}: insufficient data ({len(df)} rows)")
            continue

        # Calculate returns
        df['returns'] = df['parkinson_volatility'].pct_change()
        df['returns'] = df['returns'].fillna(0)

        # Apply outlier removal (call the module-level function)
        if remove_outliers:
            original_len = len(df)
            # Call the function defined at the end of this file
            df = remove_outliers_func(df, n_std=n_std)
            outliers_removed = original_len - len(df)
            total_outliers_removed += outliers_removed
            if len(df) < 30:
                print(f"[Warning] Skipping {stock_name}: insufficient data after outlier removal ({len(df)} rows)")
                continue

        stock_data[stock_name] = df
        loaded_count += 1

    print(f"[_load_raw_stock_data] Successfully loaded {loaded_count} stocks")
    if remove_outliers:
        print(f"[_load_raw_stock_data] Total outliers removed: {total_outliers_removed}")

    if loaded_count == 0:
        raise ValueError("No valid stock data loaded")

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
            df_har['parkinson_volatility'] = df_copy['parkinson_volatility'].values
            split_with_har[stock_name] = df_har
        except ValueError as e:
            # Handle edge case: constant volatility (zero variance)
            if 'all zeros' in str(e) or 'all NaN' in str(e):
                print(f"  [Warning] {stock_name} in {split_name}: constant volatility, using raw features")
                # Use raw volatility as fallback
                df_copy['har_daily_vol'] = df_copy['parkinson_volatility'].values
                df_copy['har_weekly_vol'] = df_copy['parkinson_volatility'].values
                df_copy['har_monthly_vol'] = df_copy['parkinson_volatility'].values
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
