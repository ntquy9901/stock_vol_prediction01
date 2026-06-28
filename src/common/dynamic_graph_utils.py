"""
Dynamic Graph Construction with Rolling Windows

Implements time-varying graph construction that updates graph structure
over time to capture changing market conditions and correlations.

Features:
1. Rolling window graph updates (e.g., every 252 trading days = 1 year)
2. Temporal evolution tracking
3. Graph statistics over time
4. Dynamic adjacency matrices

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from statsmodels.tsa.api import VAR
except ImportError:
    VAR = None

from .graph_utils import build_correlation_graph, build_spillover_graph


class DynamicGraphBuilder:
    """
    Dynamic graph builder with rolling window updates.

    Creates time-varying graphs by rebuilding graph structure at regular
    intervals using a rolling window approach.

    Args:
        window_size: Size of rolling window in days (default: 252 = 1 year)
        update_frequency: How often to rebuild graph (default: 63 = quarterly)
        graph_method: Graph construction method ('correlation' or 'spillover')
        graph_params: Additional parameters for graph construction
    """

    def __init__(
        self,
        window_size: int = 252,
        update_frequency: int = 63,
        graph_method: str = 'correlation',
        **graph_params
    ):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.graph_method = graph_method
        self.graph_params = graph_params

        # Track graph evolution
        self.graph_history = []
        self.current_window_start = 0

    def build_graphs_over_time(
        self,
        returns_matrix: np.ndarray,
        stock_names: List[str]
    ) -> List[Dict]:
        """
        Build graphs at multiple time points using rolling windows.

        Args:
            returns_matrix: (num_stocks, total_time_steps) - returns data
            stock_names: List of stock ticker names

        Returns:
            List of graph information dictionaries over time
        """
        num_stocks, total_time_steps = returns_matrix.shape

        print(f"\n[Building Dynamic Graphs]")
        print(f"  Time steps: {total_time_steps}")
        print(f"  Window size: {self.window_size} days")
        print(f"  Update frequency: Every {self.update_frequency} days")
        print(f"  Graph method: {self.graph_method}")

        graphs = []
        update_points = []

        # Calculate update points
        current_start = 0
        while current_start + self.window_size < total_time_steps:
            update_points.append(current_start)
            current_start += self.update_frequency

        print(f"  Update points: {len(update_points)}")

        # Build graphs at each update point
        for i, window_start in enumerate(update_points):
            window_end = min(window_start + self.window_size, total_time_steps)

            # Extract window data
            window_data = returns_matrix[:, window_start:window_end]

            # Build graph
            if self.graph_method == 'correlation':
                edge_index, edge_weight, graph_info = build_correlation_graph(
                    window_data,
                    stock_names,
                    threshold=self.graph_params.get('threshold', 0.5),
                    min_edges=self.graph_params.get('min_edges', 29)
                )
            elif self.graph_method == 'spillover':
                edge_index, edge_weight, graph_info = build_spillover_graph(
                    window_data,
                    stock_names,
                    p=self.graph_params.get('p', 4),
                    H=self.graph_params.get('H', 5),
                    threshold=self.graph_params.get('threshold', 0.01)
                )

            # Add temporal information
            graph_info['window_start'] = window_start
            graph_info['window_end'] = window_end
            graph_info['time_index'] = i
            graph_info['edge_index'] = edge_index
            graph_info['edge_weight'] = edge_weight

            graphs.append(graph_info)

            print(f"  [{i+1}/{len(update_points)}] Window {window_start}-{window_end}: "
                  f"{graph_info['num_edges']} edges")

        self.graph_history = graphs

        return graphs

    def track_graph_evolution(self, graphs: List[Dict]) -> Dict:
        """
        Track graph statistics evolution over time.

        Args:
            graphs: List of graph information dictionaries

        Returns:
            Dictionary with evolution statistics
        """
        print(f"\n[Tracking Graph Evolution]")

        evolution = {
            'time_indices': [],
            'num_edges': [],
            'edge_density': [],
            'avg_degree': [],
            'top_connected': defaultdict(list)
        }

        for graph in graphs:
            time_idx = graph['time_index']
            evolution['time_indices'].append(time_idx)
            evolution['num_edges'].append(graph['num_edges'])
            evolution['edge_density'].append(graph['edge_density'])
            evolution['avg_degree'].append(graph['avg_degree'])

            # Track top connected stocks
            if 'top_connected' in graph:
                for stock, degree in graph['top_connected'][:3]:
                    evolution['top_connected'][stock].append(degree)

        # Convert to numpy arrays
        for key in ['time_indices', 'num_edges', 'edge_density', 'avg_degree']:
            evolution[key] = np.array(evolution[key])

        print(f"  Tracked {len(graphs)} time points")

        return evolution

    def get_most_stable_edges(
        self,
        graphs: List[Dict],
        stability_threshold: float = 0.5
    ) -> Dict:
        """
        Identify most stable edges across time.

        Args:
            graphs: List of graph information dictionaries
            stability_threshold: Minimum frequency to consider edge stable

        Returns:
            Dictionary with stable edges and their frequencies
        """
        print(f"\n[Finding Most Stable Edges]")

        edge_counts = defaultdict(int)

        for graph in graphs:
            edge_index_np = graph['edge_index'].cpu().numpy()

            # Convert to set of tuples for undirected, or ordered pairs for directed
            for i in range(graph['num_edges']):
                src, dst = edge_index_np[:, i]
                edge_counts[(src, dst)] += 1

        total_graphs = len(graphs)
        stable_edges = {
            edge: count / total_graphs
            for edge, count in edge_counts.items()
            if count / total_graphs >= stability_threshold
        }

        print(f"  Found {len(stable_edges)} stable edges (threshold: {stability_threshold})")

        return stable_edges

    def visualize_graph_evolution(
        self,
        evolution: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize graph evolution over time.

        Args:
            evolution: Dictionary with evolution statistics
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Number of edges over time
        axes[0, 0].plot(evolution['time_indices'], evolution['num_edges'], marker='o')
        axes[0, 0].set_title('Number of Edges Over Time')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Number of Edges')
        axes[0, 0].grid(True)

        # Plot 2: Edge density over time
        axes[0, 1].plot(evolution['time_indices'], evolution['edge_density'], marker='o', color='orange')
        axes[0, 1].set_title('Edge Density Over Time')
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('Edge Density')
        axes[0, 1].grid(True)

        # Plot 3: Average degree over time
        axes[1, 0].plot(evolution['time_indices'], evolution['avg_degree'], marker='o', color='green')
        axes[1, 0].set_title('Average Degree Over Time')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Average Degree')
        axes[1, 0].grid(True)

        # Plot 4: Top connected stocks over time
        for stock, degrees in evolution['top_connected'].items():
            if len(degrees) > 1:
                axes[1, 1].plot(evolution['time_indices'], degrees, marker='o', label=stock, alpha=0.7)

        axes[1, 1].set_title('Top Connected Stocks Over Time')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Degree')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Graph evolution visualization saved to: {save_path}")

        plt.show()


def build_dynamic_ensemble_graph(
    returns_matrix: np.ndarray,
    stock_names: List[str],
    num_windows: int = 5,
    method: str = 'correlation'
) -> Tuple[List, Dict]:
    """
    Build ensemble of graphs over different time windows.

    Creates multiple graphs from different time periods and computes
    ensemble statistics.

    Args:
        returns_matrix: (num_stocks, total_time_steps) - returns data
        stock_names: List of stock ticker names
        num_windows: Number of time windows to use
        method: Graph construction method

    Returns:
        Tuple of (list of graphs, ensemble statistics)
    """
    num_stocks, total_time_steps = returns_matrix.shape

    print(f"\n[Building Ensemble Graphs]")
    print(f"  Method: {method}")
    print(f"  Windows: {num_windows}")

    window_size = total_time_steps // num_windows

    graphs = []
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size

        window_data = returns_matrix[:, window_start:window_end]

        if method == 'correlation':
            edge_index, edge_weight, graph_info = build_correlation_graph(
                window_data, stock_names, threshold=0.5
            )
        elif method == 'spillover':
            edge_index, edge_weight, graph_info = build_spillover_graph(
                window_data, stock_names, p=4, H=5, threshold=0.01
            )

        graph_info['window_start'] = window_start
        graph_info['window_end'] = window_end
        graphs.append(graph_info)

        print(f"  Window {i+1}/{num_windows}: {graph_info['num_edges']} edges")

    # Compute ensemble statistics
    ensemble_stats = {
        'avg_num_edges': np.mean([g['num_edges'] for g in graphs]),
        'avg_density': np.mean([g['edge_density'] for g in graphs]),
        'avg_degree': np.mean([g['avg_degree'] for g in graphs])
    }

    print(f"\n  Ensemble Statistics:")
    print(f"    Avg Edges: {ensemble_stats['avg_num_edges']:.2f}")
    print(f"    Avg Density: {ensemble_stats['avg_density']:.4f}")
    print(f"    Avg Degree: {ensemble_stats['avg_degree']:.2f}")

    return graphs, ensemble_stats


# CLI interface for testing
if __name__ == "__main__":
    import sys
    import os

    print("\n" + "="*60)
    print("DYNAMIC GRAPH UTILITIES - TEST")
    print("="*60)

    # Load sample data
    data_dir = 'data/processed'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')][:10]

    volatility_list = []
    stock_names = []

    for csv_file in sorted(csv_files):
        ticker = csv_file.replace('_processed.csv', '')
        stock_names.append(ticker)

        df = pd.read_csv(os.path.join(data_dir, csv_file))
        if 'parkinson_volatility' in df.columns:
            volatility = df['parkinson_volatility'].dropna().values
            # Calculate returns as proxy
            returns = pd.Series(volatility).pct_change().dropna().values
            volatility_list.append(returns)

    max_len = max(len(r) for r in volatility_list)
    returns_matrix = np.zeros((len(stock_names), max_len))

    for i, ret in enumerate(volatility_list):
        returns_matrix[i, :len(ret)] = ret

    print(f"Loaded data: {returns_matrix.shape}")

    # Test dynamic graph builder
    print("\n[1] Testing Dynamic Graph Builder...")

    builder = DynamicGraphBuilder(
        window_size=252,
        update_frequency=63,
        graph_method='correlation',
        threshold=0.5
    )

    graphs = builder.build_graphs_over_time(returns_matrix, stock_names)

    print(f"Built {len(graphs)} dynamic graphs")

    # Track evolution
    evolution = builder.track_graph_evolution(graphs)

    print("\n[2] Testing Ensemble Graphs...")

    ensemble_graphs, ensemble_stats = build_ensemble_graph(
        returns_matrix, stock_names, num_windows=5, method='correlation'
    )

    print("\n" + "="*60)
    print("DYNAMIC GRAPH UTILITIES TEST PASSED!")
    print("="*60)
