"""
Correlation-Based Graph Construction (from Sonani et al. 2025)

Methodology from paper:
1. Calculate Pearson correlation between stock volatilities
2. Create edges if |correlation| > threshold (default: 0.7)
3. Weight edges by correlation strength
4. Symmetrize adjacency matrix

Comparison:
- Paper: Correlation threshold (|corr| > 0.7)
- Current: k-NN sparse graph (k=8)

Benefits:
- Captures linear relationships between stocks
- Threshold based on statistical significance
- Weights reflect relationship strength
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple
import networkx as nx


def construct_correlation_graph(
    volatility_data: np.ndarray,
    corr_threshold: float = 0.7,
    min_data_points: int = 10  # ✅ FIX: Reduced from 100 to 10 (seq_length is 22)
) -> np.ndarray:
    """
    Construct stock relationship graph using Pearson correlation (from paper)

    Method from Sonani et al. (2025):
    - Calculate Pearson correlation coefficient between each pair of stocks
    - Create edge if |correlation| > threshold
    - Weight edge by correlation strength

    Args:
        volatility_data: Volatility data [num_days, num_stocks]
        corr_threshold: Correlation threshold for edge creation (default: 0.7 from paper)
        min_data_points: Minimum data points required for correlation

    Returns:
        adj_matrix: Adjacency matrix [num_stocks, num_stocks]
                   (symmetric, weighted by correlation strength)
    """
    num_days, num_stocks = volatility_data.shape

    # Check data sufficiency
    if num_days < min_data_points:
        print(f"[Warning] Only {num_days} data points (< {min_data_points}), "
              f"correlation may be unreliable")

    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_stocks, num_stocks))

    # Calculate Pearson correlation for each pair of stocks
    edges_created = 0
    total_pairs = 0

    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):  # Only upper triangle (avoid duplicates)
            total_pairs += 1

            # Extract volatility series for both stocks
            vol_i = volatility_data[:, i]
            vol_j = volatility_data[:, j]

            # Check for sufficient data and variance
            if len(vol_i) < min_data_points or len(vol_j) < min_data_points:
                continue

            if np.std(vol_i) == 0 or np.std(vol_j) == 0:
                # No variance, cannot calculate correlation
                continue

            try:
                # Calculate Pearson correlation coefficient
                # Equation 3 from paper:
                # ρ_ij = Σ(t=1 to n) [(r_i,t - r̄_i) × (r_j,t - r̄_j)] /
                #        [√(Σ(r_i,t - r̄_i)²) × √(Σ(r_j,t - r̄_j)²)]
                corr, p_value = pearsonr(vol_i, vol_j)

                # Check if correlation is significant and strong enough
                if not np.isnan(corr) and abs(corr) > corr_threshold:
                    # Create edge with weight = absolute correlation
                    weight = abs(corr)

                    # Symmetric matrix (undirected graph)
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight

                    edges_created += 1

            except Exception as e:
                # Skip this pair if correlation calculation fails
                print(f"[Warning] Failed to calculate correlation for stocks {i}-{j}: {e}")
                continue

    # Print graph statistics
    print(f"\n[Correlation Graph Statistics]")
    print(f"  Total possible pairs: {total_pairs}")
    print(f"  Edges created: {edges_created}")
    print(f"  Graph density: {edges_created / total_pairs:.2%}")
    print(f"  Correlation threshold: {corr_threshold}")

    # Calculate degree distribution
    degrees = np.sum(adj_matrix > 0, axis=1)
    print(f"  Average degree: {np.mean(degrees):.2f}")
    print(f"  Min degree: {np.min(degrees)}")
    print(f"  Max degree: {np.max(degrees)}")

    return adj_matrix


def construct_knn_graph(
    volatility_data: np.ndarray,
    k: int = 8,
    window: int = 22
) -> np.ndarray:
    """
    Construct k-NN sparse graph (current method)

    Uses temporal correlation over sliding window to find k nearest neighbors.

    Args:
        volatility_data: Volatility data [num_days, num_stocks]
        k: Number of nearest neighbors
        window: Window for correlation calculation

    Returns:
        adj_matrix: Adjacency matrix [num_stocks, num_stocks]
    """
    num_days, num_stocks = volatility_data.shape

    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_stocks, num_stocks))

    # For each stock, find its k nearest neighbors
    for i in range(num_stocks):
        correlations = []

        for j in range(num_stocks):
            if i == j:
                correlations.append((j, 0.0))
                continue

            # Calculate correlation over entire period
            vol_i = volatility_data[:, i]
            vol_j = volatility_data[:, j]

            if len(vol_i) < 50 or len(vol_j) < 50:
                correlations.append((j, 0.0))
                continue

            if np.std(vol_i) == 0 or np.std(vol_j) == 0:
                correlations.append((j, 0.0))
                continue

            try:
                corr, _ = pearsonr(vol_i, vol_j)
                if np.isnan(corr):
                    corr = 0.0
                correlations.append((j, abs(corr)))
            except:
                correlations.append((j, 0.0))

        # Sort by correlation and select top-k
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_k = correlations[:k]

        # Create edges to top-k neighbors
        for j, weight in top_k:
            if weight > 0:  # Only create edge if there's positive correlation
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Symmetric

    print(f"\n[k-NN Graph Statistics]")
    print(f"  k (neighbors per node): {k}")
    degrees = np.sum(adj_matrix > 0, axis=1)
    print(f"  Average degree: {np.mean(degrees):.2f}")
    print(f"  Min degree: {np.min(degrees)}")
    print(f"  Max degree: {np.max(degrees)}")

    return adj_matrix


def compare_graph_methods(
    volatility_data: np.ndarray,
    corr_threshold: float = 0.7,
    k: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare correlation-based vs k-NN graph construction

    Args:
        volatility_data: Volatility data [num_days, num_stocks]
        corr_threshold: Correlation threshold
        k: Number of neighbors for k-NN

    Returns:
        adj_corr: Correlation-based adjacency matrix
        adj_knn: k-NN adjacency matrix
    """
    print("\n" + "="*80)
    print("GRAPH CONSTRUCTION METHOD COMPARISON")
    print("="*80)

    # Method 1: Correlation-based (from paper)
    print("\n[Method 1] Correlation-Based Graph (Sonani et al. 2025)")
    adj_corr = construct_correlation_graph(volatility_data, corr_threshold=corr_threshold)

    # Method 2: k-NN (current)
    print("\n[Method 2] k-NN Sparse Graph (Current)")
    adj_knn = construct_knn_graph(volatility_data, k=k)

    # Comparison
    print("\n[Comparison]")
    density_corr = np.sum(adj_corr > 0) / (adj_corr.shape[0] ** 2)
    density_knn = np.sum(adj_knn > 0) / (adj_knn.shape[0] ** 2)

    print(f"  Correlation graph density: {density_corr:.2%}")
    print(f"  k-NN graph density: {density_knn:.2%}")
    print(f"  Difference: {(density_knn - density_corr):.2%}")

    return adj_corr, adj_knn


def visualize_graph_comparison(adj_corr: np.ndarray, adj_knn: np.ndarray, stock_names: list, save_path: str = None):
    """
    Visualize and compare both graph constructions

    Args:
        adj_corr: Correlation-based adjacency matrix
        adj_knn: k-NN adjacency matrix
        stock_names: List of stock names
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot correlation-based graph
    G_corr = nx.from_numpy_array(adj_corr)
    pos_corr = nx.spring_layout(G_corr, seed=42)

    axes[0].set_title('Correlation-Based Graph (|corr| > 0.7)', fontsize=14, fontweight='bold')
    nx.draw(G_corr, pos_corr, with_labels=True, node_color='lightblue',
            node_size=500, font_size=8, ax=axes[0], edge_color='gray',
            width=[w * 2 for w in nx.get_edge_attributes(G_corr, 'weight').values()])

    # Plot k-NN graph
    G_knn = nx.from_numpy_array(adj_knn)
    pos_knn = nx.spring_layout(G_knn, seed=42)

    axes[1].set_title('k-NN Sparse Graph (k=8)', fontsize=14, fontweight='bold')
    nx.draw(G_knn, pos_knn, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=8, ax=axes[1], edge_color='gray',
            width=[w * 2 for w in nx.get_edge_attributes(G_knn, 'weight').values()])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Graph comparison saved: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing correlation-based graph construction...")

    # Create synthetic volatility data for 5 stocks
    np.random.seed(42)
    num_days = 500
    num_stocks = 5

    # Generate correlated volatility series
    vol_data = np.random.randn(num_days, num_stocks)

    # Add some correlations manually
    vol_data[:, 1] = vol_data[:, 0] * 0.8 + np.random.randn(num_days) * 0.2  # Stock 1 correlated with Stock 0
    vol_data[:, 2] = vol_data[:, 0] * 0.7 + np.random.randn(num_days) * 0.3  # Stock 2 correlated with Stock 0
    vol_data[:, 3] = vol_data[:, 4] * 0.9 + np.random.randn(num_days) * 0.1  # Stock 3 correlated with Stock 4

    print(f"\nSynthetic data shape: {vol_data.shape}")
    print(f"Expected correlations: (0,1), (0,2), (3,4)")

    # Compare both methods
    adj_corr, adj_knn = compare_graph_methods(vol_data, corr_threshold=0.7, k=2)

    # Visualize
    stock_names = ['Stock0', 'Stock1', 'Stock2', 'Stock3', 'Stock4']
    visualize_graph_comparison(adj_corr, adj_knn, stock_names, 'graph_comparison_test.png')

    print("\n[SUCCESS] Graph construction test complete!")
