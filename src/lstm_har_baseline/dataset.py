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
from typing import List, Tuple
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
    """

    def __init__(self, data_dir: str, seq_length: int = 22,
                 forecast_horizon: int = 5, feature_scaler=None, target_scaler=None):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

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

            # Create sliding windows
            for i in range(len(har_data) - self.seq_length):
                # Input sequence: past seq_length days with 3 HAR features
                X_seq = har_data[i:i + self.seq_length]

                # Target: volatility at t + forecast_horizon
                y_target = targets[i + self.seq_length - 1]

                # Skip if target is NaN or zero
                if np.isnan(y_target) or y_target == 0:
                    continue

                self.sequences.append(X_seq)
                self.targets.append(y_target)
                self.metadata.append((ticker, i))

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