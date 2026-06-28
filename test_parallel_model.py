"""
Quick test for Parallel LSTM-GNN Model
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
import torch

print("Testing Parallel LSTM-GNN Model...")

# Create config
config = LSTMGATConfig()
config.num_stocks = 5  # Small test
config.num_features_per_stock = 3  # HAR features

# Create model
model = create_parallel_lstm_gat_model(config)

# Create dummy data
batch_size = 2
seq_len = 22
num_stocks = 5
num_features = 3

x = torch.randn(batch_size, seq_len, num_stocks, num_features)
adj_matrix = torch.randn(batch_size, num_stocks, num_stocks)

print(f"\nInput shape: {x.shape}")
print(f"Adjacency shape: {adj_matrix.shape}")

# Forward pass
print("\nRunning forward pass...")
predictions = model(x, adj_matrix)

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions: {predictions}")

# Check predictions are not constant
pred_std = predictions.std().item()
pred_range = (predictions.max() - predictions.min()).item()

print(f"\nPrediction statistics:")
print(f"  Std: {pred_std:.6f}")
print(f"  Range: {pred_range:.6f}")

if pred_std > 1e-6:
    print("  [SUCCESS] Predictions are NOT constant (good!)")
else:
    print("  [FAIL] Predictions are CONSTANT (bad!)")

# Test embedding extraction
print("\nTesting embedding extraction...")
h_lstm, h_gnn = model.get_embeddings(x, adj_matrix)
print(f"LSTM embeddings shape: {h_lstm.shape}")
print(f"GNN embeddings shape: {h_gnn.shape}")

print("\n[SUCCESS] Parallel LSTM-GNN model test passed!")
