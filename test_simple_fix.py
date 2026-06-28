"""
Simple test to verify data leakage fixes are working
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Quick check - do sequences have different graphs?
print("Creating dataset...")
from src.lstm_gat_hybrid.dataset_with_graph_method import MultiStockDatasetWithGraphMethod

dataset = MultiStockDatasetWithGraphMethod(
    data_dir='data/processed/vn30_only',
    seq_length=22,
    forecast_horizon=5,
    graph_method='knn',
    k_neighbors=8,
    normalize=False,
    remove_outliers=False,
    data_augmentation=False,
    train_mode=False
)

print(f"Dataset created: {len(dataset)} sequences")

# Check first 3 sequences
print("\nChecking first 3 sequences...")
for i in range(min(3, len(dataset))):
    x, adj_matrix, y, graph_data = dataset[i]
    print(f"Sequence {i}: adj_matrix shape = {adj_matrix.shape}, sum = {adj_matrix.sum():.4f}")

# Check if graphs are different
print("\nChecking if graphs are different...")
x0, adj0, y0, _ = dataset[0]
x1, adj1, y1, _ = dataset[1]

diff = ((adj0 - adj1)**2).sum().item() if hasattr(adj0, 'sum') else ((adj0 - adj1)**2).sum()
print(f"Difference between seq 0 and seq 1: {diff:.6f}")

if diff > 1e-6:
    print("[PASS] GRAPHS ARE DIFFERENT - Fix is working!")
else:
    print("[FAIL] GRAPHS ARE THE SAME - Data leakage still present!")

print("\nDone!")
