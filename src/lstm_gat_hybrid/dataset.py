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
        data_dir: str,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        graph_method: str = 'hybrid',
        normalize: bool = True,
        remove_outliers: bool = True,
        n_std: float = 3.0,
        data_augmentation: bool = True,
        augmentation_prob: float = 0.3,
        augmentation_factor: float = 0.1,
        train_mode: bool = False
    ):
        """
        Initialize multi-stock dataset with anti-overfitting techniques

        Args:
            data_dir: Directory containing processed CSV files
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

        # Load and process data
        self.stock_data = self._load_multi_stock_data(data_dir)
        self.stock_names = sorted(self.stock_data.keys())

        print(f"[MultiStockDataset] Loaded {len(self.stock_names)} stocks: {self.stock_names[:5]}...")

        # Generate HAR features for each stock
        self.stock_data_with_har = self._generate_features_for_all_stocks()

        # Initialize graph builder
        self.graph_builder = DynamicGraphBuilder(self.config)

        # Initialize normalizers
        self.feature_normalizers = {}
        self.target_normalizers = {}
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

    # Create full dataset with anti-overfitting (train_mode=False for validation split)
    full_dataset = MultiStockDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for full dataset
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False  # Not training mode during split
    )

    # Calculate split indices (TEMPORAL split - no shuffling!)
    n = len(full_dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Create separate dataset instances for train/val/test with different train_mode settings
    # This allows data augmentation only for training set
    print(f"[create_multi_stock_dataloaders] Creating separate datasets with different train_mode settings...")

    # Training dataset (with augmentation enabled)
    train_dataset = MultiStockDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=data_augmentation,
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=True  # Enable augmentation for training
    )

    # Validation dataset (no augmentation)
    val_dataset = MultiStockDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for validation
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False
    )

    # Test dataset (no augmentation)
    test_dataset = MultiStockDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        graph_method=graph_method,
        normalize=normalize,
        remove_outliers=remove_outliers,
        n_std=n_std,
        data_augmentation=False,  # No augmentation for testing
        augmentation_prob=augmentation_prob,
        augmentation_factor=augmentation_factor,
        train_mode=False
    )

    # Apply temporal split using Subset
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(val_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(test_dataset, range(val_end, n))

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
