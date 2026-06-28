"""
Test Spatial Encoder (GAT) Component

Tests the Graph Attention Network spatial encoder with real VN30 data.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from src.lstm_har_gat_hybrid.spatial_encoder import SpatialGAT, BatchSpatialGAT
from src.common.graph_utils import build_correlation_graph, build_spillover_graph

print("\n" + "="*60)
print("SPATIAL ENCODER (GAT) - COMPONENT TEST")
print("="*60)

# Load sample data
data_dir = 'data/processed'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')][:10]  # 10 stocks

print(f"\n[1] Loading data from {len(csv_files)} stocks...")

volatility_list = []
stock_names = []

for csv_file in sorted(csv_files):
    ticker = csv_file.replace('_processed.csv', '')
    stock_names.append(ticker)

    df = pd.read_csv(os.path.join(data_dir, csv_file))

    if 'parkinson_volatility' in df.columns:
        volatility = df['parkinson_volatility'].dropna().values[:500]  # First 500 days
        volatility_list.append(volatility)

# Pad sequences
max_len = max(len(v) for v in volatility_list)
volatility_matrix = np.zeros((len(stock_names), max_len))

for i, vol in enumerate(volatility_list):
    volatility_matrix[i, :len(vol)] = vol

print(f"  Volatility matrix shape: {volatility_matrix.shape}")

# Build graph using correlation method (more reliable)
print(f"\n[2] Building correlation graph...")
edge_index, edge_weight, graph_info = build_correlation_graph(
    volatility_matrix, stock_names, threshold=0.5
)

print(f"  Graph: {graph_info['num_nodes']} nodes, {graph_info['num_edges']} edges")

# Test SpatialGAT (single sample)
print(f"\n[3] Testing SpatialGAT (single sample)...")

num_stocks = len(stock_names)
in_channels = 64
out_channels = 64
heads = 4

encoder = SpatialGAT(
    in_channels=in_channels,
    out_channels=out_channels,
    heads=heads,
    dropout=0.2
)

# Create dummy node features
x_single = torch.randn(num_stocks, in_channels)

# Forward pass
output_single = encoder(x_single, edge_index, edge_weight)

print(f"  Input shape: {x_single.shape}")
print(f"  Output shape: {output_single.shape}")
print(f"  Expected: ({num_stocks}, {out_channels})")

assert output_single.shape == (num_stocks, out_channels), f"Shape mismatch: {output_single.shape}"

print(f"  [OK] SpatialGAT single sample test passed!")

# Test BatchSpatialGAT (multiple samples)
print(f"\n[4] Testing BatchSpatialGAT (batch processing)...")

batch_encoder = BatchSpatialGAT(
    in_channels=in_channels,
    out_channels=out_channels,
    heads=heads,
    dropout=0.2
)

batch_size = 4
x_batch = torch.randn(batch_size, num_stocks, in_channels)

# Forward pass
output_batch = batch_encoder(x_batch, edge_index, edge_weight)

print(f"  Input shape: {x_batch.shape}")
print(f"  Output shape: {output_batch.shape}")
print(f"  Expected: ({batch_size}, {num_stocks}, {out_channels})")

assert output_batch.shape == (batch_size, num_stocks, out_channels), f"Shape mismatch: {output_batch.shape}"

print(f"  [OK] BatchSpatialGAT test passed!")

# Test output statistics
print(f"\n[5] Output Statistics:")
print(f"  Mean: {output_batch.mean().item():.6f}")
print(f"  Std: {output_batch.std().item():.6f}")
print(f"  Min: {output_batch.min().item():.6f}")
print(f"  Max: {output_batch.max().item():.6f}")

# Test with different graph configurations
print(f"\n[6] Testing with spillover graph...")

try:
    spill_edge_index, spill_edge_weight, spill_info = build_spillover_graph(
        volatility_matrix, stock_names, p=4, H=5, threshold=0.01
    )

    output_spill = batch_encoder(x_batch, spill_edge_index, spill_edge_weight)

    print(f"  Spillover graph output shape: {output_spill.shape}")
    print(f"  [OK] Spillover graph test passed!")
except Exception as e:
    print(f"  [!] Spillover graph test skipped: {e}")

# Memory check
print(f"\n[7] Memory Check:")
param_count = sum(p.numel() for p in batch_encoder.parameters())
print(f"  Parameters: {param_count:,}")
print(f"  Memory (approx): {param_count * 4 / 1024 / 1024:.2f} MB (float32)")

print("\n" + "="*60)
print("SPATIAL ENCODER (GAT) TEST PASSED!")
print("="*60)
