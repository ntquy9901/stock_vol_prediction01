"""
Test Fusion Layer Component

Tests the temporal-spatial fusion layer with real VN30 data.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from src.lstm_har_gat_hybrid.fusion_layer import FusionLayer, AttentionFusionLayer
from src.lstm_har_gat_hybrid.temporal_encoder import EnhancedTemporalLSTM
from src.lstm_har_gat_hybrid.spatial_encoder import BatchSpatialGAT
from src.common.graph_utils import build_correlation_graph

print("\n" + "="*60)
print("FUSION LAYER - COMPONENT TEST")
print("="*60)

# Load sample data
data_dir = 'data/processed'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')][:5]

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
            raw_seq = parkinson[i-seq_len:i]
            weekly_seq = har_weekly[i-seq_len:i]
            monthly_seq = har_monthly[i-seq_len:i]

            X_seq = np.column_stack([raw_seq, weekly_seq, monthly_seq])

            if np.isnan(X_seq).any():
                continue

            sequences.append(X_seq)

print(f"  Loaded {len(sequences)} sequences")

# Build graph
volatility_list = []
for csv_file in sorted(csv_files):
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    if 'parkinson_volatility' in df.columns:
        volatility = df['parkinson_volatility'].dropna().values[:500]
        volatility_list.append(volatility)

max_len = max(len(v) for v in volatility_list)
volatility_matrix = np.zeros((len(stock_names), max_len))
for i, vol in enumerate(volatility_list):
    volatility_matrix[i, :len(vol)] = vol

edge_index, edge_weight, graph_info = build_correlation_graph(
    volatility_matrix, stock_names, threshold=0.5
)

print(f"  Graph: {graph_info['num_nodes']} nodes, {graph_info['num_edges']} edges")

# Test FusionLayer
print(f"\n[2] Testing FusionLayer...")

batch_size = 4
num_stocks = len(stock_names)
hidden_dim = 64

fusion = FusionLayer(
    temporal_dim=hidden_dim,
    spatial_dim=hidden_dim,
    hidden_dim=128,
    num_layers=3,
    dropout=0.2
)

# Create dummy features
temporal_features = torch.randn(batch_size, num_stocks, hidden_dim)
spatial_features = torch.randn(batch_size, num_stocks, hidden_dim)

print(f"  Temporal features shape: {temporal_features.shape}")
print(f"  Spatial features shape: {spatial_features.shape}")

# Forward pass
predictions = fusion(temporal_features, spatial_features)

print(f"  Predictions shape: {predictions.shape}")
print(f"  Expected: ({batch_size}, {num_stocks}, 1)")

assert predictions.shape == (batch_size, num_stocks, 1), f"Shape mismatch: {predictions.shape}"

print(f"  [OK] FusionLayer test passed!")

# Test AttentionFusionLayer
print(f"\n[3] Testing AttentionFusionLayer...")

attention_fusion = AttentionFusionLayer(
    temporal_dim=hidden_dim,
    spatial_dim=hidden_dim,
    hidden_dim=128,
    dropout=0.2
)

attention_predictions = attention_fusion(temporal_features, spatial_features)

print(f"  Predictions shape: {attention_predictions.shape}")
print(f"  Expected: ({batch_size}, {num_stocks}, 1)")

assert attention_predictions.shape == (batch_size, num_stocks, 1), f"Shape mismatch: {attention_predictions.shape}"

print(f"  [OK] AttentionFusionLayer test passed!")

# Test output statistics
print(f"\n[4] Output Statistics:")
print(f"  Fusion Layer - Mean: {predictions.mean().item():.6f}, Std: {predictions.std().item():.6f}")
print(f"  Attention Fusion - Mean: {attention_predictions.mean().item():.6f}, Std: {attention_predictions.std().item():.6f}")

# Memory check
print(f"\n[5] Memory Check:")
param_count_fusion = sum(p.numel() for p in fusion.parameters())
param_count_attention = sum(p.numel() for p in attention_fusion.parameters())
print(f"  Fusion Layer Parameters: {param_count_fusion:,}")
print(f"  Attention Fusion Parameters: {param_count_attention:,}")
print(f"  Total Memory (approx): {(param_count_fusion + param_count_attention) * 4 / 1024 / 1024:.2f} MB")

# Test complete pipeline
print(f"\n[6] Testing Complete Pipeline (Temporal + Spatial + Fusion)...")

# Process sequences
sequences = np.array(sequences)
X = torch.FloatTensor(sequences)

# Create batch
X_batch = X[:batch_size * num_stocks].reshape(batch_size, num_stocks, 22, 3)

# Temporal encoding
temporal_encoder = EnhancedTemporalLSTM(seq_length=22, hidden_size=hidden_dim)
temporal_output = temporal_encoder(X_batch)

# Spatial encoding
spatial_encoder = BatchSpatialGAT(in_channels=hidden_dim, out_channels=hidden_dim, heads=4)
spatial_input = torch.randn(batch_size, num_stocks, hidden_dim)  # Simplified
spatial_output = spatial_encoder(spatial_input, edge_index, edge_weight)

# Fusion
final_predictions = fusion(temporal_output, spatial_output)

print(f"  Final predictions shape: {final_predictions.shape}")
print(f"  Expected: ({batch_size}, {num_stocks}, 1)")

assert final_predictions.shape == (batch_size, num_stocks, 1), f"Shape mismatch: {final_predictions.shape}"

print(f"  [OK] Complete pipeline test passed!")

print("\n" + "="*60)
print("FUSION LAYER TEST PASSED!")
print("="*60)
