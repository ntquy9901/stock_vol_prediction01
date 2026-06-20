"""
Pooled Dataset for LSTM Volatility Prediction

This module contains the dataset class for training LSTM on pooled
volatility data from all stocks.

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


class PooledVolatilityDataset(Dataset):
    """
    Dataset for pooled volatility prediction across all stocks.

    Creates sliding windows from all stocks and combines them into
    a single training set.

    Args:
        data_dir: Directory containing processed CSV files
        seq_length: Window size (default: 22)
        forecast_horizon: Days ahead to predict (default: 5)
        scaler: Optional pre-fitted scaler
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
            self.feature_scaler.fit(all_data.reshape(-1, 1))
        else:
            self.feature_scaler = feature_scaler

        # Fit target scaler or use provided scaler - CRITICAL FIX
        if target_scaler is None:
            self.target_scaler = StandardScaler()
            all_targets = np.array(self.targets).reshape(-1, 1)
            self.target_scaler.fit(all_targets)
        else:
            self.target_scaler = target_scaler

        print(f"Loaded {len(self.sequences)} sequences from {len(set(m[0] for m in self.metadata))} stocks")

    def _load_all_data(self):
        """Load and create sequences from all processed CSV files."""
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
            if len(parkinson) < self.seq_length + self.forecast_horizon:
                continue

            # Create sliding windows
            for i in range(len(parkinson) - self.seq_length - self.forecast_horizon + 1):
                # Input sequence: past 22 days
                X_seq = parkinson[i:i + self.seq_length]

                # Target: parkinson_volatility at t + 5
                y_target = parkinson[i + self.seq_length + self.forecast_horizon - 1]

                # Skip if target is NaN or zero
                if np.isnan(y_target) or y_target == 0:
                    continue

                self.sequences.append(X_seq)
                self.targets.append(y_target)
                self.metadata.append((ticker, i))

        print(f"Created {len(self.sequences)} total sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq = self.sequences[idx]
        y_target = self.targets[idx]

        # Scale input sequence
        X_scaled = self.feature_scaler.transform(X_seq.reshape(-1, 1)).flatten()

        # Scale target - CRITICAL FIX to avoid scaling mismatch
        y_scaled = self.target_scaler.transform([[y_target]])[0, 0]

        # Convert to tensors - ensure 3D shape (seq_length, 1)
        X_tensor = torch.FloatTensor(X_scaled).reshape(self.seq_length, 1)
        y_tensor = torch.FloatTensor([y_scaled])

        return X_tensor, y_tensor
