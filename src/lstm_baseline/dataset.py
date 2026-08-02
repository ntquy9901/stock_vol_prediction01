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
        split: None keeps the original pooled/unsplit behavior (all callers that
            predate the temporal-split fix). 'train'/'val'/'test' restricts each
            ticker's windows to its chronological fraction (per-ticker temporal
            split, since sequences are stock-blocked, not date-interleaved) —
            use this with train_ratio/val_ratio/test_ratio, and pass the
            already-fitted train feature_scaler/target_scaler into the val/test
            instances so scalers are never fit on val/test data.
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

            # Create sliding windows (chronological: i increases with time)
            ticker_sequences, ticker_targets, ticker_metadata = [], [], []
            for i in range(len(parkinson) - self.seq_length - self.forecast_horizon + 1):
                # Input sequence: past 22 days
                X_seq = parkinson[i:i + self.seq_length]

                # Target: parkinson_volatility at t + 5
                y_target = parkinson[i + self.seq_length + self.forecast_horizon - 1]

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
