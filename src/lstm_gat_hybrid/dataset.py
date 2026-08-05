"""
Multi-Stock Dataset for LSTM-GAT Hybrid

Processes all 30 VN30 stocks simultaneously for graph-based modeling.
Handles data loading, graph construction, and temporal splitting.

Phase 1 Improvements:
- Outlier removal (n_std=3)
- Data augmentation (jittering, scaling)
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
from .graph_utils_fixed import DynamicGraphBuilder  # Use fixed graph utils


def remove_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method

    Args:
        df: DataFrame with 'parkinson_volatility' column
        n_std: Number of standard deviations for outlier threshold

    Returns:
        Cleaned DataFrame with outliers removed
    """
    if len(df) == 0 or 'parkinson_volatility' not in df.columns:
        return df

    # Calculate z-scores for volatility
    volatility_values = df['parkinson_volatility'].values

    # Handle cases where all values are the same (std = 0)
    if np.std(volatility_values) == 0:
        return df

    z_scores = np.abs(stats.zscore(volatility_values))

    # Filter outliers
    outlier_mask = z_scores < n_std
    df_clean = df[outlier_mask].copy()

    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"    [Outlier Removal] Removed {removed_count} outliers ({removed_count/len(df)*100:.2f}%)")

    return df_clean


def augment_sequence(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    augmentation_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply time series augmentation for training

    Techniques:
    - Jittering: Add small Gaussian noise
    - Scaling: Random scaling of the sequence

    Args:
        x_seq: Input sequence [seq_len, num_stocks, num_features]
        y_seq: Target sequence [num_stocks]
        augmentation_factor: Strength of augmentation (default: 0.1)

    Returns:
        Augmented (x_seq, y_seq)
    """
    # Jittering: add small Gaussian noise
    noise = np.random.normal(0, augmentation_factor * x_seq.std(), x_seq.shape)
    x_aug = x_seq + noise

    # Scaling: random scaling factor
    scale = np.random.uniform(1 - augmentation_factor, 1 + augmentation_factor)
    x_aug = x_aug * scale

    # Keep target unchanged (we want to augment features, not targets)
    return x_aug, y_seq


class MultiStockDataset(Dataset):
    """
    PyTorch Dataset for LSTM-GAT multi-stock volatility prediction

    Features:
    - Loads all 30 VN30 stocks simultaneously
    - Constructs dynamic graphs for cross-stock relationships
    - Handles temporal splitting (70/15/15)
    - Normalizes features and targets

    Returns:
        x: Input features [seq_len, num_stocks, num_features]
        adj_matrix: Adjacency matrix [num_stocks, num_stocks]
        y: Target volatility [num_stocks]
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        graph_method: str = 'hybrid',
        normalize: bool = True,
        remove_outliers: bool = True,
        n_std: float = 3.0,
        data_augmentation: bool = True,
        augmentation_prob: float = 0.3,
        augmentation_factor: float = 0.1,
        train_mode: bool = False,
        precomputed_raw_data: Optional[Dict] = None,
        precomputed_har_data: Optional[Dict] = None
    ):
        """
        Initialize multi-stock dataset with anti-overfitting techniques

        Args:
            data_dir: Directory containing processed CSV files (legacy direct-load
                path). Ignored when precomputed_raw_data/precomputed_har_data given.
            seq_length: Input sequence length
            forecast_horizon: Forecast horizon (5-day ahead)
            graph_method: Graph construction method ('correlation', 'spillover', 'hybrid')
            normalize: Whether to normalize data
            remove_outliers: Whether to remove outliers (Phase 1)
            n_std: Number of standard deviations for outlier threshold
            data_augmentation: Whether to apply data augmentation (Phase 1)
            augmentation_prob: Probability of applying augmentation (0.3 = 30%)
            augmentation_factor: Strength of augmentation (default: 0.1)
            train_mode: If True, apply data augmentation during training
            precomputed_raw_data: Pre-split raw stock data dict (ticker -> DataFrame).
                When both precomputed_* args are given, disk loading and full-series
                HAR/normalizer computation are skipped (split-first pipeline path);
                the caller (create_multi_stock_dataloaders) fits normalizers on the
                train split only. Prevents the HAR/normalizer leakage of the legacy
                full-series construction path.
            precomputed_har_data: Pre-split HAR feature data dict (ticker -> DataFrame).
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.graph_method = graph_method
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.n_std = n_std
        self.data_augmentation = data_augmentation
        self.augmentation_prob = augmentation_prob
        self.augmentation_factor = augmentation_factor
        self.train_mode = train_mode
        self.config = LSTMGATConfig()

        print(f"[MultiStockDataset] Anti-Overfitting Configuration:")
        print(f"  - Outlier Removal: {remove_outliers} (n_std={n_std})")
        print(f"  - Data Augmentation: {data_augmentation} (prob={augmentation_prob}, factor={augmentation_factor})")
        print(f"  - Train Mode: {train_mode}")

        # Initialize graph builder (needed by _create_sequences in both paths)
        self.graph_builder = DynamicGraphBuilder(self.config)

        # Initialize normalizers (populated below or by the caller in the pre-split path)
        self.feature_normalizers = {}
        self.target_normalizers = {}

        if precomputed_raw_data is not None and precomputed_har_data is not None:
            # Pre-split path: raw + HAR already computed per split by the caller.
            # Skip disk load, full-series HAR generation, and normalizer fitting —
            # the caller fits normalizers on the train split only (no leakage).
            self.stock_data = precomputed_raw_data
            self.stock_data_with_har = precomputed_har_data
            self.stock_names = sorted(precomputed_har_data.keys())
        else:
            # Legacy direct-load path (still used by test_phase1_implementation.py):
            # load full series, generate HAR over full series, fit normalizers.
            self.stock_data = self._load_multi_stock_data(data_dir)
            self.stock_names = sorted(self.stock_data.keys())

            print(f"[MultiStockDataset] Loaded {len(self.stock_names)} stocks: {self.stock_names[:5]}...")

            # Generate HAR features for each stock
            self.stock_data_with_har = self._generate_features_for_all_stocks()

            if normalize:
                self._initialize_normalizers()

        # Create sequences
        self.sequences = self._create_sequences()

        print(f"[MultiStockDataset] Created {len(self.sequences)} sequences")
        print(f"  - Each sequence: {len(self.stock_names)} stocks × {seq_length} timesteps × {self.config.num_features_per_stock} features")

    def _load_multi_stock_data(self, data_dir: str) -> Dict:
        """
        Load data for all stocks with outlier removal

        Args:
            data_dir: Directory containing CSV files

        Returns:
            Dictionary mapping stock names to DataFrames
        """
        data_path = Path(data_dir)
        csv_files = sorted(data_path.glob('*.csv'))

        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        stock_data = {}
        loaded_count = 0
        total_outliers_removed = 0

        for csv_file in csv_files:
            # Extract stock name from filename
            stock_name = csv_file.stem.replace('_processed', '')
            df = pd.read_csv(csv_file)

            # Ensure required columns exist
            if 'date' not in df.columns or 'parkinson_volatility' not in df.columns:
                print(f"[Warning] Skipping {stock_name}: missing required columns")
                continue

            # Check if we have enough data
            if len(df) < 30:  # Need at least 30 rows for HAR features
                print(f"[Warning] Skipping {stock_name}: insufficient data ({len(df)} rows)")
                continue

            # Calculate returns from parkinson_volatility (percentage change)
            df['returns'] = df['parkinson_volatility'].pct_change()

            # Fill NaN returns (first row)
            df['returns'] = df['returns'].fillna(0)

            # Apply outlier removal if enabled
            if self.remove_outliers:
                original_len = len(df)
                df = remove_outliers(df, n_std=self.n_std)
                outliers_removed = original_len - len(df)
                total_outliers_removed += outliers_removed
                if len(df) < 30:  # Check again after outlier removal
                    print(f"[Warning] Skipping {stock_name}: insufficient data after outlier removal ({len(df)} rows)")
                    continue

            stock_data[stock_name] = df
            loaded_count += 1

        print(f"[_load_multi_stock_data] Successfully loaded {loaded_count} stocks")
        if self.remove_outliers:
            print(f"[_load_multi_stock_data] Total outliers removed: {total_outliers_removed}")

        if loaded_count == 0:
            raise ValueError("No valid stock data loaded. Check CSV files and format.")

        return stock_data

    def _generate_features_for_all_stocks(self) -> Dict:
        """
        Generate HAR features for all stocks

        Returns:
            Dictionary mapping stock names to feature DataFrames with both HAR and raw volatility
        """
        stock_features = {}

        for stock_name, df in self.stock_data.items():
            # Keep original data
            df_copy = df.copy()

            # Generate HAR features
            df_har = generate_har_features(df_copy)

            # Combine raw volatility with HAR features
            df_har['parkinson_volatility'] = df_copy['parkinson_volatility'].values

            stock_features[stock_name] = df_har

        return stock_features

    def _initialize_normalizers(self):
        """Initialize normalizers for each stock"""
        for stock_name in self.stock_names:
            features = self.stock_data_with_har[stock_name][['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].values
            targets = self.stock_data_with_har[stock_name]['parkinson_volatility'].values

            self.feature_normalizers[stock_name] = VolatilityNormalizer()
            self.target_normalizers[stock_name] = VolatilityNormalizer()

            self.feature_normalizers[stock_name].fit(features)
            self.target_normalizers[stock_name].fit(targets.reshape(-1, 1))

    def _create_sequences(self) -> list:
        """
        Create multi-stock sequences

        Returns:
            List of (x, adj_matrix, y, metadata) tuples
        """
        sequences = []

        # Find minimum length across all stocks
        min_length = min(len(df) for df in self.stock_data_with_har.values())

        print(f"[_create_sequences] Min length across stocks: {min_length}")
        print(f"[_create_sequences] Creating sequences for {min_length - self.seq_length - self.forecast_horizon} time points...")

        # Create sequences (temporal alignment)
        for i in range(min_length - self.seq_length - self.forecast_horizon):
            # Extract features for all stocks at this time point
            x_all_stocks = []
            y_all_stocks = []
            returns_all_stocks = []
            volatility_all_stocks = []

            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]

                # Input features: [seq_len, num_features]
                x_seq = stock_feats[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values
                x_all_stocks.append(x_seq)

                # Target: scalar
                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_target = stock_feats['parkinson_volatility'].iloc[target_idx]
                y_all_stocks.append(y_target)

                # Additional data for graph construction
                returns_all_stocks.append(stock_feats['har_daily_vol'].iloc[i:i+self.seq_length].values)
                volatility_all_stocks.append(stock_feats['parkinson_volatility'].iloc[i:i+self.seq_length].values)

            # Stack arrays: convert lists to [num_stocks, ...]
            x = np.stack(x_all_stocks, axis=1)  # [seq_len, num_stocks, num_features]
            returns = np.stack(returns_all_stocks, axis=1)  # [seq_len, num_stocks]
            volatility = np.stack(volatility_all_stocks, axis=1)  # [seq_len, num_stocks]
            y = np.array(y_all_stocks)  # [num_stocks]

            # Build graph
            graph_data = {'returns': returns, 'volatility': volatility}
            adj_matrix = self.graph_builder.build_graph_from_data(graph_data, self.graph_method)

            sequences.append((x, adj_matrix, y, graph_data))

        print(f"[_create_sequences] Created {len(sequences)} sequences")
        return sequences

    def __len__(self):
        """Return number of sequences"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sequence with optional data augmentation

        Args:
            idx: Index of the sequence

        Returns:
            x: Input features [seq_len, num_stocks, num_features]
            adj_matrix: Adjacency matrix [num_stocks, num_stocks]
            y: Target volatility [num_stocks]
            graph_data: Additional data for graph updates
        """
        x, adj_matrix, y, graph_data = self.sequences[idx]

        # Apply data augmentation if enabled and in training mode
        if self.data_augmentation and self.train_mode and np.random.random() < self.augmentation_prob:
            x, y = augment_sequence(x, y, self.augmentation_factor)

        # Apply per-stock normalization (mirrors MultiStockDatasetWithPreSplitData.
        # __getitem__ in dataset_presplit.py). Previously the fitted normalizers were
        # never applied here, so training/eval ran on raw ~1e-3 volatility values —
        # the exact "fit a scaler then forget to use it" failure documented in
        # CLAUDE.md (LSTM-GNN Normalization Failure). Falls back to raw (with a
        # warning) for any stock missing from the normalizer dict.
        if self.normalize:
            x_normalized = np.zeros_like(x)
            for stock_idx in range(x.shape[1]):
                stock_name = self.stock_names[stock_idx]
                if stock_name in self.feature_normalizers:
                    for feat_idx in range(x.shape[2]):
                        x_normalized[:, stock_idx, feat_idx] = \
                            self.feature_normalizers[stock_name].transform(
                                x[:, stock_idx, feat_idx:feat_idx+1]
                            ).flatten()
                else:
                    x_normalized[:, stock_idx, :] = x[:, stock_idx, :]
                    print(f"[MultiStockDataset __getitem__] WARNING: {stock_name} not in feature_normalizers!")
            x = x_normalized

            y_normalized = np.zeros_like(y)
            for stock_idx, stock_name in enumerate(self.stock_names):
                if stock_name in self.target_normalizers:
                    y_normalized[stock_idx] = \
                        self.target_normalizers[stock_name].transform(
                            y[stock_idx:stock_idx+1].reshape(1, -1)
                        ).flatten()[0]
                else:
                    y_normalized[stock_idx] = y[stock_idx]
                    print(f"[MultiStockDataset __getitem__] WARNING: {stock_name} not in target_normalizers!")
            # Clip normalized targets to guard against distribution-shift outliers
            # (val/test volatility far outside the training range → huge z-scores).
            y = np.clip(y_normalized, -10.0, 10.0)

        # Convert to tensors
        x = torch.FloatTensor(x)
        adj_matrix = torch.FloatTensor(adj_matrix)
        y = torch.FloatTensor(y)

        return x, adj_matrix, y, graph_data


def create_multi_stock_dataloaders(
    data_dir: str,
    seq_length: int = 22,
    forecast_horizon: int = 5,
    graph_method: str = 'hybrid',
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
    Create train, validation, and test dataloaders for multi-stock data with anti-overfitting

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Input sequence length
        forecast_horizon: Forecast horizon
        graph_method: Graph construction method
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of workers for DataLoader
        normalize: Whether to normalize data
        remove_outliers: Whether to remove outliers (Phase 1)
        n_std: Number of standard deviations for outlier threshold
        data_augmentation: Whether to apply data augmentation (Phase 1)
        augmentation_prob: Probability of applying augmentation
        augmentation_factor: Strength of augmentation

    Returns:
        train_loader, val_loader, test_loader, datasets (tuple of 4 datasets)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    print(f"[create_multi_stock_dataloaders] Anti-Overfitting Configuration:")
    print(f"  - Outlier Removal: {remove_outliers} (n_std={n_std})")
    print(f"  - Data Augmentation: {data_augmentation} (prob={augmentation_prob}, factor={augmentation_factor})")

    # Split-first pipeline (reuses the already-tested helpers from the news-fusion
    # lineage): load RAW data → split RAW chronologically by date → generate HAR
    # SEPARATELY per split → fit normalizers on the TRAIN split only. This fixes
    # both the HAR rolling-window leakage and the normalizer-fit-on-full-series
    # leakage of the previous "build full dataset then positionally Subset" flow.
    from .dataset_with_graph_method import (
        _load_raw_stock_data,
        _split_raw_data_by_date,
        _generate_har_for_split,
    )

    # Step 1: load raw data (no HAR yet), date-aligned across tickers (P1.2 fix)
    stock_data_raw = _load_raw_stock_data(
        data_dir=data_dir, remove_outliers=remove_outliers, n_std=n_std
    )

    # Step 2: split raw data chronologically BEFORE generating HAR features
    train_raw, val_raw, test_raw, _, _, _ = _split_raw_data_by_date(
        stock_data_raw, train_ratio, val_ratio, test_ratio
    )

    # Step 3: generate HAR features separately per split (no cross-split leakage)
    train_har = _generate_har_for_split(train_raw, 'train')
    val_har = _generate_har_for_split(val_raw, 'val')
    test_har = _generate_har_for_split(test_raw, 'test')

    # Tickers present in ALL splits (need HAR features in each to build sequences)
    common_stocks = sorted(set(train_har) & set(val_har) & set(test_har))
    if not common_stocks:
        raise ValueError("No stock is present in all of train/val/test HAR splits.")

    train_har_c = {k: v for k, v in train_har.items() if k in common_stocks}
    val_har_c = {k: v for k, v in val_har.items() if k in common_stocks}
    test_har_c = {k: v for k, v in test_har.items() if k in common_stocks}
    train_raw_c = {k: v for k, v in train_raw.items() if k in common_stocks}
    val_raw_c = {k: v for k, v in val_raw.items() if k in common_stocks}
    test_raw_c = {k: v for k, v in test_raw.items() if k in common_stocks}

    print(f"[create_multi_stock_dataloaders] Creating pre-split datasets "
          f"({len(common_stocks)} common stocks)...")

    # Step 4: construct the 3 dataset instances via the pre-split path
    train_dataset = MultiStockDataset(
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=data_augmentation,
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=True,  # Enable augmentation for training
        precomputed_raw_data=train_raw_c,
        precomputed_har_data=train_har_c,
    )
    val_dataset = MultiStockDataset(
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for validation
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False,
        precomputed_raw_data=val_raw_c,
        precomputed_har_data=val_har_c,
    )
    test_dataset = MultiStockDataset(
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for testing
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False,
        precomputed_raw_data=test_raw_c,
        precomputed_har_data=test_har_c,
    )

    # Step 5: fit normalizers on TRAINING sequences only, then copy (not refit) the
    # fitted objects into val/test — mirrors create_multi_stock_dataloaders_with_
    # graph_method_fixed's Step 5. Fitting on train only is the leakage fix.
    if normalize:
        print(f"[create_multi_stock_dataloaders] Fitting normalizers on TRAIN split only...")
        for stock_name in common_stocks:
            train_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            train_dataset.target_normalizers[stock_name] = VolatilityNormalizer()

        for stock_idx, stock_name in enumerate(train_dataset.stock_names):
            train_features = []
            train_targets = []
            for seq in train_dataset.sequences:
                x, _adj, y, _graph = seq
                train_features.append(x[:, stock_idx, :])
                train_targets.append(y[stock_idx])
            train_features = np.concatenate(train_features, axis=0)
            train_targets = np.array(train_targets)

            train_dataset.feature_normalizers[stock_name].fit(train_features)
            train_dataset.target_normalizers[stock_name].fit(train_targets.reshape(-1, 1))

            # Copy the SAME fitted objects to val/test (no independent refit)
            val_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            val_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]
            test_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            test_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]

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

    print(f"[create_multi_stock_dataloaders] Split complete:")
    print(f"  - Train: {len(train_dataset)} sequences")
    print(f"  - Val:   {len(val_dataset)} sequences")
    print(f"  - Test:  {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
