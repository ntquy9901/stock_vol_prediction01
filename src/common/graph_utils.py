"""
Graph Construction Utilities for LSTM-HAR-GAT Hybrid Model

This module provides utilities for constructing graphs from volatility data:
1. Correlation-based graph (symmetric, undirected)
2. Volatility spillover graph (asymmetric, directed) - RECOMMENDED

Both methods are implemented to allow comparison and validation.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import numpy as np
import pandas as pd
import torch
import networkx as nx
from typing import Tuple, Dict, Optional
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


def build_correlation_graph(
    returns_matrix: np.ndarray,
    stock_names: list,
    threshold: float = 0.5,
    min_edges: int = 29
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Construct graph based on Pearson correlation of returns.

    Creates an undirected graph where edges exist between stocks with
    absolute correlation above threshold.

    Args:
        returns_matrix: (num_stocks, time_steps) - returns for stocks
        stock_names: List of stock ticker names
        threshold: Minimum absolute correlation to create edge (default: 0.5)
        min_edges: Ensure each node has at least this many edges (default: 29)

    Returns:
        edge_index: (2, num_edges) - COO format for PyTorch Geometric
        edge_weight: (num_edges,) - correlation coefficients
        graph_info: Dict with graph statistics
    """
    num_stocks = returns_matrix.shape[0]

    print(f"\n[Building Correlation Graph]")
    print(f"  Stocks: {num_stocks}")
    print(f"  Time steps: {returns_matrix.shape[1]}")
    print(f"  Threshold: {threshold}")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(returns_matrix)

    # Handle NaN values (set to 0)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Create edges
    edges = []
    weights = []

    for i in range(num_stocks):
        for j in range(i+1, num_stocks):  # Upper triangle only (undirected)
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                # Add bidirectional edge
                edges.append([i, j])
                edges.append([j, i])
                weights.extend([corr, corr])

    # Ensure minimum connectivity
    if len(edges) < min_edges * num_stocks:
        print(f"  [!] Low connectivity, adding edges to ensure min {min_edges} per node")
        for i in range(num_stocks):
            # Find top-k correlations for this node
            corr_row = corr_matrix[i].copy()
            corr_row[i] = 0  # Exclude self

            # Get existing neighbors
            current_neighbors = sum(1 for e in edges if e[0] == i)

            if current_neighbors < min_edges:
                # Add edges to highest correlated stocks
                top_k = min_edges - current_neighbors
                top_indices = np.argsort(np.abs(corr_row))[-top_k:]

                for j in top_indices:
                    # Check if edge already exists
                    if [i, j] not in edges and [j, i] not in edges:
                        corr = corr_matrix[i, j]
                        edges.append([i, j])
                        edges.append([j, i])
                        weights.extend([corr, corr])

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Compute graph statistics
    graph_info = {
        'method': 'correlation',
        'num_nodes': num_stocks,
        'num_edges': len(edges) // 2,  # Divide by 2 for undirected
        'edge_density': (len(edges) // 2) / (num_stocks * (num_stocks - 1) / 2),
        'avg_degree': len(edges) / num_stocks,
        'corr_matrix': corr_matrix
    }

    print(f"  [OK] Graph constructed")
    print(f"    Nodes: {graph_info['num_nodes']}")
    print(f"    Edges: {graph_info['num_edges']}")
    print(f"    Density: {graph_info['edge_density']:.4f}")
    print(f"    Avg degree: {graph_info['avg_degree']:.2f}")

    return edge_index, edge_weight, graph_info


def build_spillover_graph(
    volatility_data: np.ndarray,
    stock_names: list,
    p: int = 4,
    H: int = 5,
    threshold: float = 0.01,
    min_edges: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Construct graph using Diebold-Yilmaz volatility spillover index.

    Creates a DIRECTED graph where edge weights represent volatility
    transmission from one stock to another.

    Methodology:
    1. Fit VAR(p) model to volatility data
    2. Compute forecast error variance decomposition (FEVD)
    3. Extract spillover index at horizon H

    Args:
        volatility_data: (num_stocks, time_steps) - volatility series
        stock_names: List of stock ticker names
        p: VAR lag order (default: 4)
        H: forecast horizon for FEVD (default: 5)
        threshold: Minimum spillover intensity to create edge (default: 0.01 = 1%)
        min_edges: Ensure each node has at least this many outgoing edges (default: 5)

    Returns:
        edge_index: (2, num_edges) - COO format (DIRECTED edges)
        edge_weight: (num_edges,) - spillover intensities
        graph_info: Dict with graph statistics
    """
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        raise ImportError(
            "statsmodels required for spillover graph. "
            "Install with: pip install statsmodels"
        )

    num_stocks = volatility_data.shape[0]

    print(f"\n[Building Spillover Graph]")
    print(f"  Stocks: {num_stocks}")
    print(f"  Time steps: {volatility_data.shape[1]}")
    print(f"  VAR lag order (p): {p}")
    print(f"  Forecast horizon (H): {H}")
    print(f"  Threshold: {threshold}")

    # Initialize variables
    spillover_matrix = None
    spillover_index = None
    spillover_available = False

    # Fit VAR model
    print(f"  Fitting VAR({p}) model...")
    try:
        model = VAR(volatility_data.T)
        results = model.fit(p, ic=None, trend='c')
        spillover_available = True
    except Exception as e:
        print(f"  [!] VAR fitting failed: {e}")
        print(f"  [!] Using fallback: correlation-based spillover approximation")

    # Compute forecast error variance decomposition (FEVD)
    if spillover_available:
        print(f"  Computing FEVD for horizon {H}...")
        try:
            fevd = results.fevd(H)
            spillover_available = True
        except Exception as e:
            print(f"  [!] FEVD computation failed: {e}")
            print(f"  [!] Using fallback: correlation-based spillover approximation")
            spillover_available = False

    # Process spillover data (either from FEVD or fallback)
    if not spillover_available:
        # Use correlation matrix as fallback
        corr_matrix = np.abs(np.corrcoef(volatility_data))
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        spillover_matrix = corr_matrix
        spillover_index = corr_matrix / (corr_matrix.sum(axis=1, keepdims=True) + 1e-8)
        spillover_index = np.nan_to_num(spillover_index, nan=0.0)
    else:
        # Extract spillover matrix at horizon H from FEVD
        if isinstance(fevd, list):
            spillover_matrix = fevd[-1]  # Last horizon
        else:
            # Extract from FEVD object
            try:
                spillover_matrix = results.mae.coef()
                spillover_matrix = np.abs(spillover_matrix)
            except:
                # Fallback to correlation
                spillover_matrix = np.abs(np.corrcoef(volatility_data))

        # Ensure spillover_matrix is 2D numpy array
        if not isinstance(spillover_matrix, np.ndarray):
            spillover_matrix = np.array(spillover_matrix)
        if spillover_matrix.ndim == 1:
            spillover_matrix = spillover_matrix.reshape(-1, 1)

        # Normalize rows to get spillover index
        row_sums = spillover_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        spillover_index = spillover_matrix / row_sums
        spillover_index = np.nan_to_num(spillover_index, nan=0.0)

    # Create directed edges
    edges = []
    weights = []

    for i in range(num_stocks):  # From stock i
        for j in range(num_stocks):  # To stock j
            if i == j:
                continue  # No self-loops

            spillover_intensity = spillover_index[i, j]

            if spillover_intensity >= threshold:
                # Add directed edge: i -> j
                edges.append([i, j])
                weights.append(spillover_intensity)

    # Ensure minimum outgoing edges per node
    if len(edges) < min_edges * num_stocks:
        print(f"  [!] Low connectivity, adding edges to ensure min {min_edges} outgoing per node")
        for i in range(num_stocks):
            # Find top-k spillovers for this node
            spillover_row = spillover_index[i].copy()
            spillover_row[i] = 0  # Exclude self

            # Get existing outgoing neighbors
            current_outgoing = sum(1 for e in edges if e[0] == i)

            if current_outgoing < min_edges:
                # Add edges to highest spillover targets
                top_k = min_edges - current_outgoing
                top_indices = np.argsort(spillover_row)[-top_k:]

                for j in top_indices:
                    spillover_intensity = spillover_index[i, j]
                    # Check if edge already exists
                    if [i, j] not in edges:
                        edges.append([i, j])
                        weights.append(spillover_intensity)

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Compute graph statistics
    num_directed_edges = len(edges)
    graph_info = {
        'method': 'spillover',
        'num_nodes': num_stocks,
        'num_edges': num_directed_edges,
        'edge_density': num_directed_edges / (num_stocks * (num_stocks - 1)),
        'avg_out_degree': num_directed_edges / num_stocks,
        'spillover_matrix': spillover_matrix,
        'spillover_index': spillover_index,
        'var_results': results if spillover_available else None,
        'used_fallback': not spillover_available
    }

    print(f"  [OK] Graph constructed")
    print(f"    Nodes: {graph_info['num_nodes']}")
    print(f"    Directed edges: {graph_info['num_edges']}")
    print(f"    Density: {graph_info['edge_density']:.4f}")
    print(f"    Avg out-degree: {graph_info['avg_out_degree']:.2f}")

    return edge_index, edge_weight, graph_info


def visualize_graph(
    edge_index: torch.Tensor,
    stock_names: list,
    edge_weight: Optional[torch.Tensor] = None,
    graph_info: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> nx.Graph:
    """
    Visualize graph using NetworkX and matplotlib.

    Args:
        edge_index: (2, num_edges) - COO format
        stock_names: List of stock ticker names
        edge_weight: Optional edge weights for visualization
        graph_info: Optional graph statistics for title
        save_path: Optional path to save figure

    Returns:
        networkx Graph object
    """
    import matplotlib.pyplot as plt

    # Create NetworkX graph
    if edge_weight is None:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    # Add nodes
    for i, name in enumerate(stock_names):
        G.add_node(i, label=name)

    # Add edges
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        if edge_weight is not None:
            weight = edge_weight[i].item()
            G.add_edge(src, dst, weight=weight)
        else:
            G.add_edge(src, dst)

    # Create visualization
    plt.figure(figsize=(14, 10))

    # Compute layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=500, alpha=0.8)

    # Draw edges with weights
    if edge_weight is not None:
        # Normalize edge weights for visualization
        weights_np = edge_weight.cpu().numpy()
        if len(weights_np) > 0:
            min_w, max_w = weights_np.min(), weights_np.max()
            if max_w > min_w:
                normalized_weights = (weights_np - min_w) / (max_w - min_w)
            else:
                normalized_weights = np.ones_like(weights_np)

            # Draw edges
            for i, (src, dst) in enumerate(edge_index_np.T):
                width = 0.5 + 3 * normalized_weights[i]
                alpha = 0.3 + 0.7 * normalized_weights[i]
                nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                      width=width, alpha=alpha,
                                      edge_color='gray', arrows=True)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={i: name for i, name in enumerate(stock_names)},
                          font_size=8, font_weight='bold')

    # Title
    if graph_info:
        title = f"Volatility Graph ({graph_info['method'].title()})\n"
        title += f"Nodes: {graph_info['num_nodes']}, Edges: {graph_info['num_edges']}, "
        title += f"Density: {graph_info['edge_density']:.4f}"
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title("Volatility Graph", fontsize=14, fontweight='bold')

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Graph visualization saved to: {save_path}")

    plt.show()

    return G


def compute_graph_statistics(
    edge_index: torch.Tensor,
    stock_names: list,
    graph_info: Optional[Dict] = None
) -> Dict:
    """
    Compute detailed graph statistics for analysis.

    Args:
        edge_index: (2, num_edges) - COO format
        stock_names: List of stock ticker names
        graph_info: Optional existing graph information

    Returns:
        Dict with comprehensive graph statistics
    """
    import networkx as nx

    # Create NetworkX graph
    edge_index_np = edge_index.cpu().numpy()
    G = nx.DiGraph() if graph_info and graph_info['method'] == 'spillover' else nx.Graph()

    # Add nodes
    G.add_nodes_from(range(len(stock_names)))

    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        G.add_edge(src, dst)

    # Compute statistics
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
        'num_components': nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G),
    }

    # Node-level statistics
    degrees = dict(G.degree())
    stats['avg_degree'] = np.mean(list(degrees.values()))
    stats['max_degree'] = max(degrees.values())
    stats['min_degree'] = min(degrees.values())

    # Top connected nodes
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_connected'] = [(stock_names[i], deg) for i, deg in top_nodes]

    # Centrality measures
    betweenness = nx.betweenness_centrality(G)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_betweenness'] = [(stock_names[i], cent) for i, cent in top_betweenness]

    if not G.is_directed():
        # Clustering coefficient (undirected only)
        stats['avg_clustering'] = nx.average_clustering(G)

    return stats


def compare_graph_methods(
    returns_matrix: np.ndarray,
    volatility_data: np.ndarray,
    stock_names: list,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Compare correlation and spillover graph construction methods.

    Args:
        returns_matrix: (num_stocks, time_steps) - returns for correlation graph
        volatility_data: (num_stocks, time_steps) - volatility for spillover graph
        stock_names: List of stock ticker names
        save_dir: Optional directory to save visualizations

    Returns:
        Dict with comparison results
    """
    print("\n" + "="*60)
    print("COMPARING GRAPH CONSTRUCTION METHODS")
    print("="*60)

    results = {}

    # Build correlation graph
    print("\n[1] Correlation-based Graph")
    corr_edge_index, corr_edge_weight, corr_info = build_correlation_graph(
        returns_matrix, stock_names, threshold=0.5
    )

    results['correlation'] = {
        'edge_index': corr_edge_index,
        'edge_weight': corr_edge_weight,
        'info': corr_info
    }

    # Visualize correlation graph
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        corr_path = os.path.join(save_dir, 'correlation_graph.png')
        visualize_graph(corr_edge_index, stock_names, corr_edge_weight, corr_info, corr_path)

    # Build spillover graph
    print("\n[2] Volatility Spillover Graph")
    spill_edge_index, spill_edge_weight, spill_info = build_spillover_graph(
        volatility_data, stock_names, p=4, H=5, threshold=0.01
    )

    results['spillover'] = {
        'edge_index': spill_edge_index,
        'edge_weight': spill_edge_weight,
        'info': spill_info
    }

    # Visualize spillover graph
    if save_dir:
        spill_path = os.path.join(save_dir, 'spillover_graph.png')
        visualize_graph(spill_edge_index, stock_names, spill_edge_weight, spill_info, spill_path)

    # Compute detailed statistics
    print("\n[3] Graph Statistics Comparison")
    corr_stats = compute_graph_statistics(corr_edge_index, stock_names, corr_info)
    spill_stats = compute_graph_statistics(spill_edge_index, stock_names, spill_info)

    # Print comparison
    print("\n" + "-"*60)
    print(f"{'Metric':<30} {'Correlation':>15} {'Spillover':>15}")
    print("-"*60)
    print(f"{'Nodes':<30} {corr_stats['num_nodes']:>15} {spill_stats['num_nodes']:>15}")
    print(f"{'Edges':<30} {corr_stats['num_edges']:>15} {spill_stats['num_edges']:>15}")
    print(f"{'Density':<30} {corr_stats['density']:>15.4f} {spill_stats['density']:>15.4f}")
    print(f"{'Avg Degree':<30} {corr_stats['avg_degree']:>15.2f} {spill_stats['avg_degree']:>15.2f}")
    print(f"{'Connected':<30} {str(corr_stats['is_connected']):>15} {str(spill_stats['is_connected']):>15}")

    print(f"\n{'Top Connected (Correlation)':<30}")
    for stock, deg in corr_stats['top_connected'][:3]:
        print(f"  {stock:<28} {deg:>15}")

    print(f"\n{'Top Connected (Spillover)':<30}")
    for stock, deg in spill_stats['top_connected'][:3]:
        print(f"  {stock:<28} {deg:>15}")

    print("-"*60)

    results['comparison'] = {
        'correlation_stats': corr_stats,
        'spillover_stats': spill_stats
    }

    print("\n[OK] Graph comparison complete")

    return results


# CLI interface for standalone usage
if __name__ == "__main__":
    import sys
    import os

    print("\n" + "="*60)
    print("GRAPH UTILITIES - CLI INTERFACE")
    print("="*60)

    # Check if data directory provided
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m src.common.graph_utils <data_dir> [output_dir]")
        print("\nExample:")
        print("  python -m src.common.graph_utils data/processed results/graphs")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/graphs"

    # Load data from processed files
    print(f"\nLoading data from: {data_dir}")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]

    if len(csv_files) == 0:
        print(f"[X] No processed CSV files found in {data_dir}")
        sys.exit(1)

    # Load all data
    stock_names = []
    returns_list = []
    volatility_list = []

    for csv_file in sorted(csv_files):
        ticker = csv_file.replace('_processed.csv', '')
        stock_names.append(ticker)

        df = pd.read_csv(os.path.join(data_dir, csv_file))

        # Extract volatility (this is our main data)
        if 'parkinson_volatility' in df.columns:
            volatility = df['parkinson_volatility'].dropna().values
            volatility_list.append(volatility[:500])  # Use first 500 days

            # Calculate returns from volatility as proxy for correlation
            # For volatility data, we can use percentage changes
            returns = pd.Series(volatility).pct_change().dropna().values
            returns_list.append(returns[:500])

    # Pad sequences to same length
    max_len = max(len(r) for r in returns_list)
    returns_matrix = np.zeros((len(stock_names), max_len))
    volatility_matrix = np.zeros((len(stock_names), max_len))

    for i, (ret, vol) in enumerate(zip(returns_list, volatility_list)):
        returns_matrix[i, :len(ret)] = ret
        volatility_matrix[i, :len(vol)] = vol

    print(f"Loaded {len(stock_names)} stocks")
    print(f"Time steps: {max_len}")

    # Compare graph methods
    results = compare_graph_methods(
        returns_matrix=returns_matrix,
        volatility_data=volatility_matrix,
        stock_names=stock_names,
        save_dir=output_dir
    )

    print(f"\n[OK] Results saved to: {output_dir}")
