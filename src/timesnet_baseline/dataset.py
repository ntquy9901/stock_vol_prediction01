"""
TimesNet Dataset for Volatility Prediction

Adapts HAR features and temporal information for TimesNet architecture.
Handles 1D→2D data transformation and temporal feature extraction.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from src.common.har_features import generate_har_features
from src.common.data_normalization import VolatilityNormalizer


class VolatilityTimesNetDataset(Dataset):
    """
    PyTorch Dataset for TimesNet volatility prediction

    Features:
    - HAR features (daily, weekly, monthly volatility)
    - Temporal features (month, day, weekday)
    - Parkinson volatility as target

    Returns:
        x_har: HAR features [seq_len, 3]
        x_temporal: Temporal features [seq_len, 3]
        y: Target volatility [1]
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        normalize: bool = True
    ):
        """
        Initialize TimesNet dataset

        Args:
            data_dir: Directory containing processed CSV files
            seq_length: Input sequence length (default: 22 trading days)
            forecast_horizon: Forecast horizon (default: 5 days)
            normalize: Whether to normalize data (default: True)
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize

        # Load and process data
        self.df = self._load_data(data_dir)

        # Generate HAR features
        self.df = generate_har_features(self.df)

        # Extract temporal features
        self.temporal_features = self._extract_temporal_features()

        # Validate data
        self._validate_data()

        # Initialize normalizers
        self.feature_normalizer = None
        self.target_normalizer = None
        if normalize:
            self._initialize_normalizers()

        # Create sequences
        self.sequences = self._create_sequences()

        print(f"[VolatilityTimesNetDataset] Loaded {len(self.sequences)} sequences")
        print(f"  - HAR features shape: {self.sequences[0][0].shape}")
        print(f"  - Temporal features shape: {self.sequences[0][1].shape}")

    def _load_data(self, data_dir: str) -> pd.DataFrame:
        """Load data from CSV file"""
        data_path = Path(data_dir)

        # Find CSV file
        csv_files = list(data_path.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        if len(csv_files) > 1:
            print(f"[Warning] Multiple CSV files found. Using: {csv_files[0]}")

        df = pd.read_csv(csv_files[0])
        print(f"[VolatilityTimesNetDataset] Loaded data from {csv_files[0].name}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")

        return df

    def _extract_temporal_features(self) -> np.ndarray:
        """
        Extract temporal features from date column

        Returns:
            Array of shape [n_samples, 3] with (month, day, weekday)
        """
        if 'date' not in self.df.columns:
            raise ValueError("DataFrame must have 'date' column")

        # Convert date strings to datetime
        dates = pd.to_datetime(self.df['date'])

        # Extract temporal features
        temporal_df = pd.DataFrame({
            'month': dates.dt.month,
            'day': dates.dt.day,
            'weekday': dates.dt.weekday
        })

        return temporal_df.values

    def _validate_data(self):
        """Validate that required columns exist"""
        required_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol', 'parkinson_volatility']

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Check for NaN values
        if self.df[required_cols].isna().any().any():
            raise ValueError(f"NaN values found in required columns")

        print(f"[VolatilityTimesNetDataset] Data validation passed")

    def _initialize_normalizers(self):
        """Initialize feature and target normalizers"""
        # HAR features
        har_features = self.df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].values
        self.feature_normalizer = VolatilityNormalizer()
        self.feature_normalizer.fit(har_features)

        # Target (parkinson_volatility)
        target = self.df['parkinson_volatility'].values
        self.target_normalizer = VolatilityNormalizer()
        self.target_normalizer.fit(target)

        print(f"[VolatilityTimesNetDataset] Normalizers initialized")

    def _create_sequences(self) -> list:
        """
        Create input-output sequences

        Returns:
            List of tuples (x_har, x_temporal, y)
        """
        sequences = []
        df = self.df.copy()
        temporal_features = self.temporal_features.copy()

        # Normalize if requested
        if self.normalize:
            df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']] = (
                self.feature_normalizer.transform(df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].values)
            )
            df['parkinson_volatility'] = (
                self.target_normalizer.transform(df['parkinson_volatility'].values.reshape(-1, 1)).flatten()
            )

        # Create sequences
        for i in range(len(df) - self.seq_length - self.forecast_horizon):
            # HAR features: [seq_len, 3]
            x_har = df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values

            # Temporal features: [seq_len, 3]
            x_temporal = temporal_features[i:i+self.seq_length]

            # Target: scalar (5-day ahead volatility)
            target_idx = i + self.seq_length + self.forecast_horizon - 1
            y = df['parkinson_volatility'].iloc[target_idx]

            sequences.append((x_har, x_temporal, y))

        return sequences

    def __len__(self):
        """Return number of sequences"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sequence

        Returns:
            x_har: HAR features [seq_len, 3]
            x_temporal: Temporal features [seq_len, 3]
            y: Target volatility [1]
        """
        x_har, x_temporal, y = self.sequences[idx]

        # Convert to tensors
        x_har = torch.FloatTensor(x_har)
        x_temporal = torch.FloatTensor(x_temporal)
        y = torch.FloatTensor([y])

        return x_har, x_temporal, y

    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions back to original scale

        Args:
            predictions: Normalized predictions

        Returns:
            Denormalized predictions
        """
        if self.target_normalizer is None:
            return predictions

        return self.target_normalizer.inverse_transform(predictions.reshape(-1, 1)).flatten()


def create_timesnet_dataloaders(
    data_dir: str,
    seq_length: int = 22,
    forecast_horizon: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple:
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Input sequence length
        forecast_horizon: Forecast horizon
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of workers for DataLoader
        normalize: Whether to normalize data

    Returns:
        train_loader, val_loader, test_loader, datasets (tuple of 3 datasets)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Create full dataset
    full_dataset = VolatilityTimesNetDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        normalize=normalize
    )

    # Calculate split indices
    n = len(full_dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split datasets (TEMPORAL split - no shuffling!)
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(full_dataset, range(val_end, n))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: Don't shuffle time series!
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"[create_timesnet_dataloaders] Split complete:")
    print(f"  - Train: {len(train_dataset)} sequences")
    print(f"  - Val:   {len(val_dataset)} sequences")
    print(f"  - Test:  {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
