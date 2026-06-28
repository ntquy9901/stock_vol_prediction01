"""
Test improved correlation-based graph construction

Verify that graphs now use the entire temporal sequence instead of single-point similarity.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.graph_utils_fixed import DynamicGraphBuilder

def test_improved_graph_construction():
    """Test that improved graph construction uses full sequence"""
    print("="*80)
    print("TESTING IMPROVED GRAPH CONSTRUCTION")
    print("="*80)

    # Create config
    config = LSTMGATConfig()
    config.num_stocks = 5
    config.top_k_neighbors = 3

    # Create graph builder
    builder = DynamicGraphBuilder(config)

    # Create synthetic data with known patterns
    seq_len = 22
    num_stocks = 5

    # Create synthetic HAR daily volatility data
    # Pattern: Stocks 0,1,2 are correlated; Stocks 3,4 are correlated
    np.random.seed(42)
    base_signal = np.random.randn(seq_len)

    returns = np.zeros((seq_len, num_stocks))
    returns[:, 0] = base_signal + 0.1 * np.random.randn(seq_len)  # Group 1
    returns[:, 1] = base_signal + 0.1 * np.random.randn(seq_len)  # Group 1
    returns[:, 2] = base_signal + 0.1 * np.random.randn(seq_len)  # Group 1
    returns[:, 3] = np.random.randn(seq_len)  # Group 2 (uncorrelated)
    returns[:, 4] = np.random.randn(seq_len)  # Group 2 (uncorrelated)

    volatility = returns.copy()  # Simplified

    print(f"\nData shape: {returns.shape}")
    print(f"Expected pattern: Stocks 0-2 correlated, Stocks 3-4 different")

    # Test correlation graph
    print("\n" + "-"*80)
    print("TESTING CORRELATION GRAPH")
    print("-"*80)

    corr_graph = builder.build_correlation_graph(returns, 'pearson')

    print(f"Correlation graph shape: {corr_graph.shape}")
    print(f"Graph density: {(corr_graph > 0).sum() / (num_stocks * num_stocks):.4f}")

    print(f"\nCorrelation adjacency matrix:")
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i != j and corr_graph[i, j] > 0:
                print(f"  Stock {i} -> Stock {j}: {corr_graph[i, j]:.4f}")

    # Test spillover graph
    print("\n" + "-"*80)
    print("TESTING SPILLOVER GRAPH")
    print("-"*80)

    spill_graph = builder.build_spillover_graph(volatility, returns)

    print(f"Spillover graph shape: {spill_graph.shape}")
    print(f"Graph density: {(spill_graph > 0).sum() / (num_stocks * num_stocks):.4f}")

    print(f"\nSpillover adjacency matrix:")
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i != j and spill_graph[i, j] > 0:
                print(f"  Stock {i} -> Stock {j}: {spill_graph[i, j]:.4f}")

    # Test hybrid graph
    print("\n" + "-"*80)
    print("TESTING HYBRID GRAPH")
    print("-"*80)

    hybrid_graph = builder.build_hybrid_graph(returns, volatility, alpha=0.5)

    print(f"Hybrid graph shape: {hybrid_graph.shape}")
    print(f"Graph density: {(hybrid_graph > 0).sum() / (num_stocks * num_stocks):.4f}")

    print(f"\nHybrid adjacency matrix:")
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i != j and hybrid_graph[i, j] > 0:
                print(f"  Stock {i} -> Stock {j}: {hybrid_graph[i, j]:.4f}")

    # Verify expected patterns
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Check that stocks 0-2 are connected
    connections_group1 = 0
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j and hybrid_graph[i, j] > 0:
                connections_group1 += 1

    print(f"\nExpected: Stocks 0-2 should be interconnected")
    print(f"Actual connections among group 1: {connections_group1}/6")

    if connections_group1 >= 3:
        print("[PASS] Group 1 stocks are connected")
    else:
        print("[FAIL] Group 1 stocks not properly connected")

    # Check sparsity
    total_connections = (hybrid_graph > 0).sum()
    max_connections = num_stocks * (num_stocks - 1)  # No self-loops
    sparsity = total_connections / max_connections

    print(f"\nExpected: Sparse graph (k-NN with k={config.top_k_neighbors})")
    print(f"Actual density: {sparsity:.4f}")
    print(f"Expected density: ~{2 * config.top_k_neighbors / num_stocks:.4f}")

    if sparsity < 0.8:
        print("[PASS] Graph is sparse")
    else:
        print("[FAIL] Graph is too dense")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    test_improved_graph_construction()