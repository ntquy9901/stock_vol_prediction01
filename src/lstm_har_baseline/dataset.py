"""
HAR Dataset for LSTM-HAR Volatility Prediction

This module contains the dataset class for training LSTM on HAR
features from all stocks.

Features:
    - har_daily_vol: 1-day average volatility
    - har_weekly_vol: 5-day average volatility
    - har_monthly_vol: 22-day average volatility

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from .features import create_har_features, create_har_dataset_with_targets


class HARVolatilityDataset(Dataset):
    """
    Dataset for HAR-based volatility prediction across all stocks.

    Uses HAR features (daily, weekly, monthly) instead of raw volatility.
    Input shape: (batch, seq_length, 3) - 3 HAR features

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Window size (default: 22)
        forecast_horizon: Days ahead to predict (default: 5)
        feature_scaler: Optional pre-fitted scaler for features
        target_scaler: Optional pre-fitted scaler for targets
        split: None keeps the original pooled/unsplit behavior. 'train'/'val'/'test'
            restricts each ticker's windows to its chronological fraction (per-ticker
            temporal split — sequences are stock-blocked, not date-interleaved).
            Pass the already-fitted train feature_scaler/target_scaler into the
            val/test instances so scalers are never fit on val/test data.
    """

    def __init__(self, data_dir: str, seq_length: int = 22,
                 forecast_horizon: int = 5, feature_scaler=None, target_scaler=None,
                 split=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Load all processed data
        self.sequences = []
        self.targets = []
        self.metadata = []

        self._load_all_data()

        # Fit feature scaler or use provided scaler
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            all_data = np.concatenate(self.sequences)
            self.feature_scaler.fit(all_data.reshape(-1, 3))
        else:
            self.feature_scaler = feature_scaler

        # Fit target scaler or use provided scaler
        if target_scaler is None:
            self.target_scaler = StandardScaler()
            all_targets = np.array(self.targets).reshape(-1, 1)
            self.target_scaler.fit(all_targets)
        else:
            self.target_scaler = target_scaler

        num_stocks = len(set(m[0] for m in self.metadata))
        print(f"Loaded {len(self.sequences)} HAR sequences from {num_stocks} stocks")

    def _load_all_data(self):
        """Load and create HAR sequences from all processed CSV files."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and 'processed' in f]

        for csv_file in csv_files:
            # Extract ticker name
            ticker = csv_file.replace('_processed.csv', '').replace('.csv', '')

            # Load processed data
            file_path = os.path.join(self.data_dir, csv_file)
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
            if len(parkinson) < self.seq_length + self.forecast_horizon + 22:
                print(f"Warning: {ticker} insufficient data ({len(parkometer)} records)")
                continue

            # Create HAR features
            parkinson_series = pd.Series(parkinson)
            har_features = create_har_features(parkinson_series)

            # Create dataset with targets
            har_dataset = create_har_dataset_with_targets(har_features, self.forecast_horizon)

            # Extract HAR features and targets
            har_data = har_dataset[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].values
            targets = har_dataset['target'].values

            # Create sliding windows (chronological: i increases with time)
            ticker_sequences, ticker_targets, ticker_metadata = [], [], []
            for i in range(len(har_data) - self.seq_length):
                # Input sequence: past seq_length days with 3 HAR features
                X_seq = har_data[i:i + self.seq_length]

                # Target: volatility at t + forecast_horizon
                y_target = targets[i + self.seq_length - 1]

                # Skip if target is NaN or zero
                if np.isnan(y_target) or y_target == 0:
                    continue

                ticker_sequences.append(X_seq)
                ticker_targets.append(y_target)
                ticker_metadata.append((ticker, i))

            if self.split is not None:
                # Per-ticker temporal split: windows are already chronological
                # within this ticker (increasing i), so a positional cut here
                # is a true chronological cut, unlike a global random_split.
                n = len(ticker_sequences)
                train_end = int(n * self.train_ratio)
                val_end = int(n * (self.train_ratio + self.val_ratio))
                if self.split == 'train':
                    lo, hi = 0, train_end
                elif self.split == 'val':
                    lo, hi = train_end, val_end
                elif self.split == 'test':
                    lo, hi = val_end, n
                else:
                    raise ValueError(f"Unknown split={self.split!r}, expected 'train'/'val'/'test'")
                ticker_sequences = ticker_sequences[lo:hi]
                ticker_targets = ticker_targets[lo:hi]
                ticker_metadata = ticker_metadata[lo:hi]

            self.sequences.extend(ticker_sequences)
            self.targets.extend(ticker_targets)
            self.metadata.extend(ticker_metadata)

        print(f"Created {len(self.sequences)} total HAR sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq = self.sequences[idx]
        y_target = self.targets[idx]

        # Scale input sequence (3 features)
        X_scaled = self.feature_scaler.transform(X_seq)

        # Scale target
        y_scaled = self.target_scaler.transform([[y_target]])[0, 0]

        # Convert to tensors - shape (seq_length, 3)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor([y_scaled])

        return X_tensor, y_tensor
