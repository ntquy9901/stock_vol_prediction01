"""
Graph Construction Utilities for LSTM-GAT Hybrid

Implements dynamic graph construction methods:
1. Correlation-based graph: Connect stocks with high price correlation
2. Volatility spillover graph: Connect stocks with volatility transmission
3. Hybrid graph: Combine correlation and spillover edges
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .config import LSTMGATConfig


class DynamicGraphBuilder:
    """
    Build dynamic graphs for cross-stock relationship modeling

    Creates adjacency matrices for GAT based on:
    - Price correlations
    - Volatility spillovers
    - Hybrid approach
    """

    def __init__(self, config: LSTMGATConfig):
        """
        Initialize graph builder

        Args:
            config: LSTM-GAT configuration
        """
        self.config = config
        self.num_stocks = config.num_stocks

    def build_correlation_graph(
        self,
        returns: np.ndarray,
        method: str = 'pearson'
    ) -> torch.Tensor:
        """
        Build graph based on return correlations

        Args:
            returns: Array of shape [seq_len, num_stocks] containing returns
            method: Correlation method ('pearson', 'spearman')

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        # Calculate correlation matrix across stocks
        if returns.shape[0] < 2:
            # If insufficient data, use identity matrix
            return torch.eye(self.num_stocks)

        # Calculate correlation for each time step and average
        correlations = []
        for t in range(returns.shape[0]):
            # Get returns for all stocks at time t
            ret_t = returns[t, :]  # [num_stocks]

            # Calculate correlation with previous time step
            if t > 0:
                ret_prev = returns[t-1, :]
                corr = np.corrcoef(ret_t, ret_prev)[0, 1] if not np.isnan(corr) else 0.0
            else:
                corr = 0.0

            correlations.append(corr)

        # Create adjacency matrix from correlations
        adj_matrix = np.zeros((self.num_stocks, self.num_stocks))

        # Fill upper triangle with correlations
        edge_weights = []
        for i in range(self.num_stocks):
            for j in range(i+1, self.num_stocks):
                # Use correlation as edge weight
                if len(correlations) > 0:
                    edge_weight = np.mean([c for c in correlations if not np.isnan(c)])
                else:
                    edge_weight = 0.0

                adj_matrix[i, j] = edge_weight
                adj_matrix[j, i] = edge_weight  # Symmetric

                if edge_weight != 0:
                    edge_weights.append(abs(edge_weight))

        # Use top-k neighbors instead of threshold for sparse graphs
        if len(edge_weights) > 0:
            # Calculate appropriate threshold to keep only top-k connections
            k = min(self.config.top_k_neighbors, self.num_stocks - 1)
            if len(edge_weights) > k:
                # Sort and find threshold for top-k
                edge_weights_sorted = np.sort(edge_weights)
                threshold = edge_weights_sorted[-k]  # Keep only top-k edges
                # Apply threshold - keep edges above threshold
                adj_matrix = np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)
            else:
                # If not enough edges, use higher threshold
                threshold = 0.7  # Much higher threshold for VN30 stocks
                adj_matrix = np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)
        else:
            # Fallback to high threshold
            threshold = 0.7
            adj_matrix = np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)

        # Normalize adjacency matrix
        adj_matrix = self._normalize_adjacency(adj_matrix)

        return torch.FloatTensor(adj_matrix)

    def build_spillover_graph(
        self,
        volatility: np.ndarray,
        returns: np.ndarray
    ) -> torch.Tensor:
        """
        Build graph based on volatility spillover effects

        Uses volatility changes and return correlations to detect spillover

        Args:
            volatility: [seq_len, num_stocks] volatility values
            returns: [seq_len, num_stocks] return values

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        # Calculate volatility changes
        vol_changes = np.diff(volatility, axis=0)  # [seq_len-1, num_stocks]

        # Calculate spillover intensity
        spillover_matrix = np.zeros((self.num_stocks, self.num_stocks))

        for i in range(self.num_stocks):
            for j in range(self.num_stocks):
                if i == j:
                    continue

                # Calculate correlation of volatility changes
                vol_i = vol_changes[:, i]
                vol_j = vol_changes[:, j]

                if len(vol_i) > 1 and len(vol_j) > 1:
                    # Correlation of volatility changes
                    spillover = np.corrcoef(vol_i, vol_j)[0, 1]
                    if not np.isnan(spillover):
                        spillover_matrix[i, j] = abs(spillover)

        # Keep top-k neighbors per stock
        for i in range(self.num_stocks):
            # Get top-k neighbors for stock i
            neighbors = np.argsort(spillover_matrix[i, :])[::-1][:self.config.top_k_neighbors]
            mask = np.zeros(self.num_stocks, dtype=bool)
            mask[neighbors] = True
            spillover_matrix[i, :] = spillover_matrix[i, :] * mask

        # Normalize
        spillover_matrix = self._normalize_adjacency(spillover_matrix)

        return torch.FloatTensor(spillover_matrix)

    def build_hybrid_graph(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Build hybrid graph combining correlation and spillover

        Args:
            returns: [seq_len, num_stocks] return values
            volatility: [seq_len, num_stocks] volatility values
            alpha: Weight for correlation vs spillover (0-1)

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        # Build individual graphs
        corr_graph = self.build_correlation_graph(returns, 'pearson')
        spillover_graph = self.build_spillover_graph(volatility, returns)

        # Combine graphs
        hybrid_graph = alpha * corr_graph + (1 - alpha) * spillover_graph

        return hybrid_graph

    def _normalize_adjacency(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix for GAT while preserving sparsity

        Args:
            adj_matrix: [num_stocks, num_stocks] adjacency matrix

        Returns:
            normalized_adjacency: Normalized adjacency matrix
        """
        # Add self-loops ONLY for non-zero edges
        # This preserves sparsity better
        adj_matrix = adj_matrix + np.eye(self.num_stocks) * 0.1  # Small self-loops

        # Normalize only the non-zero parts to preserve sparsity
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        normalized = adj_matrix / row_sums

        return normalized

    def build_graph_from_data(
        self,
        data: dict,
        method: str = 'hybrid'
    ) -> torch.Tensor:
        """
        Build graph from data dictionary

        Args:
            data: Dictionary containing 'returns' and 'volatility' arrays
            method: Graph construction method ('correlation', 'spillover', 'hybrid')

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        returns = data.get('returns', None)
        volatility = data.get('volatility', None)

        if method == 'correlation':
            return self.build_correlation_graph(returns)
        elif method == 'spillover':
            return self.build_spillover_graph(volatility, returns)
        elif method == 'hybrid':
            return self.build_hybrid_graph(returns, volatility)
        else:
            raise ValueError(f"Unknown graph method: {method}")


def create_stock_mapping(stock_names: list) -> dict:
    """
    Create mapping between stock names and indices

    Args:
        stock_names: List of stock names

    Returns:
        Dictionary mapping stock names to indices and vice versa
    """
    return {
        'name_to_idx': {name: idx for idx, name in enumerate(stock_names)},
        'idx_to_name': {idx: name for idx, name in enumerate(stock_names)}
    }


def visualize_attention_weights(
    attention_weights: torch.Tensor,
    stock_names: list,
    save_path: str
):
    """
    Visualize attention weights between stocks

    Args:
        attention_weights: [num_heads, num_stocks, num_stocks] attention weights
        stock_names: List of stock names
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_heads = attention_weights.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(6*num_heads, 5))

    if num_heads == 1:
        axes = [axes]

    for head in range(num_heads):
        ax = axes[head]

        # Get attention weights for this head
        attn = attention_weights[head].detach().numpy()

        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=stock_names,
            yticklabels=stock_names,
            cmap='YlOrRd',
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Attention Head {head+1}')
        ax.set_xlabel('Target Stocks')
        ax.set_ylabel('Source Stocks')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Graph Utils] Saved attention visualization to {save_path}")
