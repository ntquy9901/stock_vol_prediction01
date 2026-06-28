"""
Test Temporal Encoder (LSTM-HAR) Component

Tests the LSTM-HAR temporal encoder with real VN30 data.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from src.lstm_har_gat_hybrid.temporal_encoder import TemporalLSTM, EnhancedTemporalLSTM

print("\n" + "="*60)
print("TEMPORAL ENCODER - COMPONENT TEST")
print("="*60)

# Load sample data
data_dir = 'data/processed'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')][:5]  # Test with 5 stocks

print(f"\n[1] Loading data from {len(csv_files)} stocks...")

sequences = []
stock_names = []

for csv_file in sorted(csv_files):
    ticker = csv_file.replace('_processed.csv', '')
    stock_names.append(ticker)

    df = pd.read_csv(os.path.join(data_dir, csv_file))

    if 'parkinson_volatility' in df.columns:
        parkinson = df['parkinson_volatility'].dropna().values

        # Create HAR features
        har_weekly = pd.Series(parkinson).rolling(5).mean().values
        har_monthly = pd.Series(parkinson).rolling(22).mean().values

        # Create sequences
        seq_len = 22
        for i in range(seq_len, len(parkinson) - 5):
            if np.isnan(har_weekly[i]) or np.isnan(har_monthly[i]):
                continue

            raw_seq = parkinson[i-seq_len:i]
            weekly_seq = har_weekly[i-seq_len:i]
            monthly_seq = har_monthly[i-seq_len:i]

            X_seq = np.column_stack([raw_seq, weekly_seq, monthly_seq])

            if np.isnan(X_seq).any():
                continue

            sequences.append(X_seq)

print(f"  Loaded {len(sequences)} sequences from {len(stock_names)} stocks")

# Convert to tensors
sequences = np.array(sequences)
X = torch.FloatTensor(sequences)

print(f"  Input shape: {X.shape} (num_samples, seq_len, num_features)")

# Test EnhancedTemporalLSTM
print(f"\n[2] Testing EnhancedTemporalLSTM...")

encoder = EnhancedTemporalLSTM(
    seq_length=22,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

# Reshape for multi-stock processing
batch_size = 4
num_stocks = len(stock_names)
seq_length = 22
num_features = 3

# Create batch input (batch, stocks, seq_len, features)
X_batch = X[:batch_size * num_stocks].reshape(batch_size, num_stocks, seq_length, num_features)

print(f"  Batch input shape: {X_batch.shape}")

# Forward pass
output = encoder(X_batch)

print(f"  Output shape: {output.shape}")
print(f"  Expected: ({batch_size}, {num_stocks}, 64)")

assert output.shape == (batch_size, num_stocks, 64), f"Shape mismatch: {output.shape}"

print(f"  [OK] EnhancedTemporalLSTM test passed!")

# Test output statistics
print(f"\n[3] Output Statistics:")
print(f"  Mean: {output.mean().item():.6f}")
print(f"  Std: {output.std().item():.6f}")
print(f"  Min: {output.min().item():.6f}")
print(f"  Max: {output.max().item():.6f}")

print(f"\n[4] Memory Check:")
param_count = sum(p.numel() for p in encoder.parameters())
print(f"  Parameters: {param_count:,}")
print(f"  Memory (approx): {param_count * 4 / 1024 / 1024:.2f} MB (float32)")

print("\n" + "="*60)
print("TEMPORAL ENCODER TEST PASSED!")
print("="*60)
