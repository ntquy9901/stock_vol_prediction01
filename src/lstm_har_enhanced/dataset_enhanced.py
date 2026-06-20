"""
Enhanced LSTM-HAR Dataset with Raw + HAR Features

This module extends the HARVolatilityDataset by including both
raw Parkinson volatility AND HAR features as input.

Architecture:
    Input: (batch, seq_len, 4) - [raw, daily, weekly, monthly]
    LSTM: Processes combined features
    Output: (batch, 1) - Volatility prediction

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class EnhancedHARDataset(Dataset):
    """
    Enhanced LSTM-HAR Dataset with Raw Volatility + HAR Features.

    Features (NO REDUNDANCY):
        1. Raw Parkinson volatility (current day) - removes daily redundancy
        2. HAR weekly volatility (5-day avg)
        3. HAR monthly volatility (22-day avg)

    Note: HAR daily (rolling(1).mean()) removed because it's identical to raw volatility.
          This prevents redundancy and overfitting.

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Sequence length for LSTM input
        forecast_horizon: Days ahead to forecast (default: 5)
        feature_scaler: Optional fitted scaler for features
        target_scaler: Optional fitted scaler for targets
    """

    def __init__(self,
                 data_dir: str,
                 seq_length: int = 22,
                 forecast_horizon: int = 5,
                 feature_scaler=None,
                 target_scaler=None):

        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        # Load all data and create sequences
        self.sequences = []
        self.targets = []
        self.metadata = []

        self._load_all_data(data_dir)

        # Convert to numpy arrays
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

        print(f"Loaded {len(self.sequences)} enhanced HAR sequences from {len(set(m[0] for m in self.metadata))} stocks")

        # Fit feature scaler or use provided scaler
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            # Reshape for fitting: (n_samples * seq_length, n_features)
            all_features = self.sequences.reshape(-1, self.sequences.shape[-1])
            self.feature_scaler.fit(all_features)
        else:
            self.feature_scaler = feature_scaler

        # Fit target scaler or use provided scaler
        if target_scaler is None:
            self.target_scaler = StandardScaler()
            all_targets = np.array(self.targets).reshape(-1, 1)
            self.target_scaler.fit(all_targets)
        else:
            self.target_scaler = target_scaler

    def _load_all_data(self, data_dir):
        """Load and create enhanced HAR sequences from all processed CSV files."""
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'processed' in f]

        for csv_file in csv_files:
            # Extract ticker name
            ticker = csv_file.replace('_processed.csv', '').replace('.csv', '')

            # Load processed data
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)

            if 'parkinson_volatility' not in df.columns:
                print(f"Warning: {csv_file} missing parkinson_volatility column")
                continue

            # Extract parkinson volatility
            parkinson = df['parkinson_volatility'].values

            # Remove NaN values
            valid_mask = ~np.isnan(parkinson)
            parkinson = parkinson[valid_mask]

            # Skip if insufficient data
            min_required = self.seq_length + self.forecast_horizon + 22
            if len(parkinson) < min_required:
                print(f"Warning: {ticker} insufficient data ({len(parkinson)} records)")
                continue

            # Create HAR features (weekly and monthly only - daily is redundant with raw)
            har_weekly = pd.Series(parkinson).rolling(5).mean().values
            har_monthly = pd.Series(parkinson).rolling(22).mean().values

            # Create sequences with raw + HAR features
            for i in range(self.seq_length, len(parkinson) - self.forecast_horizon):
                # Check if we have valid target
                y_target = parkinson[i + self.forecast_horizon]

                if np.isnan(y_target) or y_target == 0:
                    continue

                # Extract sequence for each feature
                # Feature 1: Raw volatility (direct from parkinson)
                raw_seq = parkinson[i-self.seq_length:i]

                # Feature 2-3: HAR features (weekly and monthly only)
                weekly_seq = har_weekly[i-self.seq_length:i]
                monthly_seq = har_monthly[i-self.seq_length:i]

                # Combine into single sequence: (seq_length, 3)
                # Each row: [raw, weekly, monthly]
                # NO daily - it's redundant with raw!
                X_seq = np.column_stack([raw_seq, weekly_seq, monthly_seq])

                # Verify no NaN in sequence
                if np.isnan(X_seq).any():
                    continue

                self.sequences.append(X_seq)
                self.targets.append(y_target)
                self.metadata.append((ticker, i))

        print(f"Created {len(self.sequences)} total enhanced HAR sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq = self.sequences[idx]
        y_target = self.targets[idx]

        # Scale input sequence (4 features)
        X_scaled = self.feature_scaler.transform(X_seq)

        # Scale target
        y_scaled = self.target_scaler.transform([[y_target]])[0, 0]

        # Convert to tensors - shape (seq_length, 4)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor([y_scaled])

        return X_tensor, y_tensor


if __name__ == "__main__":
    # Test the enhanced dataset
    print("Testing Enhanced HAR Dataset...")

    dataset = EnhancedHARDataset(
        data_dir='data/processed',
        seq_length=22,
        forecast_horizon=5
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    X, y = dataset[0]
    print(f"\nSample shape:")
    print(f"  X: {X.shape} (seq_length, n_features)")
    print(f"  y: {y.shape}")

    print(f"\nFeature breakdown (3 features - NO REDUNDANCY):")
    print(f"  Feature 0 (raw): mean={X[:, 0].mean():.6f}, std={X[:, 0].std():.6f}")
    print(f"  Feature 1 (weekly): mean={X[:, 1].mean():.6f}, std={X[:, 1].std():.6f}")
    print(f"  Feature 2 (monthly): mean={X[:, 2].mean():.6f}, std={X[:, 2].std():.6f}")

    print(f"\nTarget: {y.item():.6f}")
    print("\nNote: HAR daily removed - redundant with raw volatility")
