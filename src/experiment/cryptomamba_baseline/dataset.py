"""
CryptoMamba Dataset for Volatility Prediction

Compatible with project's existing data pipeline:
- Loads processed data from data/processed/
- Uses HAR features + additional inputs
- Creates sequences for CryptoMamba
- Target: 5-day ahead Parkinson volatility

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import glob


class CryptoMambaDataset(Dataset):
    """
    Dataset for CryptoMamba volatility prediction.

    Input: HAR features + volume (if available)
    Output: 5-day ahead Parkinson volatility (scaled)

    Compatible with project's data format and preprocessing.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        use_volume: bool = False,  # Changed: default False (no volume in processed data)
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing processed CSV files
            seq_length: Number of past days to use as input
            forecast_horizon: Days ahead to predict
            use_volume: Include volume as feature (if available)
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.use_volume = use_volume

        # Load all processed CSV files
        csv_files = glob.glob(os.path.join(data_dir, "*_processed.csv"))
        if not csv_files:
            raise ValueError(f"No processed CSV files found in {data_dir}")

        print(f"Loading {len(csv_files)} CSV files from {data_dir}...")

        # Load and combine all data
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)

        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(self.data)} total rows from {len(csv_files)} files")

        # Check required columns (adapted for current processed data)
        required_cols = ['parkinson_volatility']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Found columns: {self.data.columns.tolist()}")

        # Prepare features (adapted for current data)
        self._prepare_features()

        # Create sequences
        self._create_sequences()

        # Print dataset info
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {len(self)}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Forecast horizon: {self.forecast_horizon} days")
        print(f"  Features: {self.feature_names}")
        print(f"  Target mean: {self.target_mean:.6f}")
        print(f"  Target std: {self.target_std:.6f}")

    def _prepare_features(self):
        """Prepare and scale features - adapted for current processed data."""

        # For Phase 1: Use raw Parkinson volatility + rolling windows as features
        # This is similar to Simple LSTM approach (1 feature)
        # But we create multiple windowed features for SSM

        vol = self.data['parkinson_volatility']

        # Create features using rolling windows (HAR-style)
        self.data['har_daily_vol'] = vol  # Daily (same as raw)
        self.data['har_weekly_vol'] = vol.rolling(5).mean().fillna(method='bfill').fillna(0)
        self.data['har_monthly_vol'] = vol.rolling(22).mean().fillna(method='bfill').fillna(0)

        # Forward fill target
        self.data['target_5d'] = self.data['parkinson_volatility'].shift(-self.forecast_horizon).fillna(method='bfill').fillna(0)

        # Select features
        base_features = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']

        if self.use_volume and 'volume' in self.data.columns:
            # Try to find volume column
            vol_cols = [col for col in self.data.columns if 'volume' in col.lower()]
            if vol_cols:
                vol_col = vol_cols[0]
                vol_ma = self.data[vol_col].rolling(20).mean()
                self.data['volume_ratio'] = self.data[vol_col] / vol_ma
                self.data['volume_ratio'].fillna(1.0, inplace=True)
                base_features = base_features + ['volume_ratio']
                print("Using volume feature")
            else:
                print("Volume column not found, proceeding without volume")
                self.use_volume = False
        else:
            if self.use_volume:
                print("Volume requested but not available in data, proceeding without volume")
                self.use_volume = False

        self.feature_names = base_features

        # Extract features and target
        self.features = self.data[self.feature_names].values
        self.targets = self.data['target_5d'].values

        # Scale features
        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(self.features)

        # Scale targets
        self.target_scaler = StandardScaler()
        self.targets_scaled = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()

        # Store target statistics
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std()

    def _create_sequences(self):
        """Create sequences for CryptoMamba."""

        X_sequences = []
        y_sequences = []

        # Create sequences with rolling window
        for i in range(len(self.features) - self.seq_length - self.forecast_horizon + 1):
            # Input: seq_length consecutive days
            X_seq = self.features[i:i + self.seq_length]

            # Output: target at horizon days ahead
            target_idx = i + self.seq_length + self.forecast_horizon - 1
            y_seq = self.targets_scaled[target_idx]

            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

        # Convert to numpy arrays
        self.X = np.array(X_sequences, dtype=np.float32)
        self.y = np.array(y_sequences, dtype=np.float32)

        print(f"Created {len(self.X)} sequences")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            X: (seq_length, num_features)
            y: (1,) - scaled volatility
        """
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(-1)

        return X, y

    def inverse_transform_targets(self, y_scaled):
        """
        Convert scaled predictions back to original scale.

        Args:
            y_scaled: Scaled volatility predictions

        Returns:
            Volatility in original scale
        """
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.cpu().numpy()

        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        y_original = self.target_scaler.inverse_transform(y_scaled)
        return y_original.flatten()


def create_dataloader(
    data_dir: str,
    seq_length: int = 22,
    forecast_horizon: int = 5,
    batch_size: int = 32,
    use_volume: bool = True,
    shuffle: bool = True,
):
    """
    Create dataloader for CryptoMamba training.

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Input sequence length
        forecast_horizon: Forecast horizon in days
        batch_size: Batch size
        use_volume: Include volume feature
        shuffle: Shuffle data

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = CryptoMambaDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        use_volume=use_volume,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows: use 0 for stability
        pin_memory=False,
    )

    return dataloader, dataset
