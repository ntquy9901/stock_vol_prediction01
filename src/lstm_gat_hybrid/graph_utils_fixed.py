"""
Graph Construction Utilities for LSTM-GAT Hybrid

Implements dynamic graph construction methods for multi-stock volatility modeling.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from .config import LSTMGATConfig


class DynamicGraphBuilder:
    """
    Dynamic graph builder for constructing stock relationship graphs

    Supports multiple graph construction methods:
    - correlation: Based on return correlations
    - spillover: Based on volatility spillover effects
    - hybrid: Combination of correlation and spillover
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
        Build k-NN graph based on TRUE correlation over entire sequence

        Args:
            returns: [seq_len, num_stocks] HAR daily volatility values
            method: Correlation method ('pearson', 'spearman')

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        # Handle edge case
        if returns.shape[0] < 2:
            return torch.eye(returns.shape[1])

        # Use k-nearest neighbors based on temporal correlation
        num_stocks = returns.shape[1]
        k = min(self.config.top_k_neighbors, num_stocks - 1)
        correlation_matrix = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            for j in range(num_stocks):
                if i != j:
                    # Calculate correlation over entire temporal sequence
                    series_i = returns[:, i]
                    series_j = returns[:, j]

                    # Calculate correlation
                    if method == 'pearson':
                        corr = np.corrcoef(series_i, series_j)[0, 1]
                    else:  # spearman
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(series_i, series_j)

                    # Handle NaN
                    if np.isnan(corr):
                        corr = 0.0

                    correlation_matrix[i, j] = abs(corr)  # Use absolute correlation

        # Create sparse k-NN adjacency matrix based on correlation strength
        adj_matrix = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            # Get top-k most correlated stocks (by absolute correlation)
            correlations = [(j, correlation_matrix[i, j]) for j in range(num_stocks) if i != j]
            correlations.sort(key=lambda x: x[1], reverse=True)

            # Connect to top-k most correlated stocks
            for j, correlation in correlations[:k]:
                adj_matrix[i, j] = correlation
                adj_matrix[j, i] = correlation  # Symmetric

        # Normalize adjacency matrix
        adj_matrix = self._normalize_adjacency(adj_matrix)

        return torch.FloatTensor(adj_matrix)

    def build_spillover_graph(
        self,
        volatility: np.ndarray,
        returns: np.ndarray
    ) -> torch.Tensor:
        """
        Build graph based on volatility spillover effects using entire sequence

        Uses correlation of volatility changes to detect spillover over time

        Args:
            volatility: [seq_len, num_stocks] volatility values
            returns: [seq_len, num_stocks] return values (HAR daily vol)

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        if volatility.shape[0] < 2:
            return torch.eye(volatility.shape[1])

        # Calculate volatility changes over entire sequence
        volatility_changes = np.diff(volatility, axis=0)  # [seq_len-1, num_stocks]

        # Create k-NN graph based on spillover correlation
        num_stocks = volatility.shape[1]
        k = min(self.config.top_k_neighbors, num_stocks - 1)

        # Calculate spillover correlation matrix
        spillover_matrix = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            for j in range(num_stocks):
                if i != j:
                    # Calculate correlation of volatility changes over entire sequence
                    vol_changes_i = volatility_changes[:, i]
                    vol_changes_j = volatility_changes[:, j]

                    # Calculate correlation
                    if len(vol_changes_i) > 1:
                        spillover = np.corrcoef(vol_changes_i, vol_changes_j)[0, 1]
                        if np.isnan(spillover):
                            spillover = 0.0
                        spillover_matrix[i, j] = abs(spillover)  # Use absolute correlation
                    else:
                        spillover_matrix[i, j] = 0.0

        # Create sparse k-NN adjacency matrix based on spillover strength
        adj_matrix = np.zeros((num_stocks, num_stocks))

        for i in range(num_stocks):
            # Get top-k stocks with strongest spillover
            spillovers = [(j, spillover_matrix[i, j]) for j in range(num_stocks) if i != j]
            spillovers.sort(key=lambda x: x[1], reverse=True)

            # Connect to top-k most connected stocks
            for j, spillover in spillovers[:k]:
                adj_matrix[i, j] = spillover
                adj_matrix[j, i] = spillover  # Symmetric

        # Normalize adjacency matrix
        adj_matrix = self._normalize_adjacency(adj_matrix)

        return torch.FloatTensor(adj_matrix)

    def build_hybrid_graph(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Build hybrid graph combining correlation and spillover

        Args:
            returns: [seq_len, num_stocks] HAR daily volatility values
            volatility: [seq_len, num_stocks] volatility values
            alpha: Weight for correlation vs spillover (0-1)

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        # Build individual graphs (both now use full sequence)
        corr_graph = self.build_correlation_graph(returns, 'pearson')
        spill_graph = self.build_spillover_graph(volatility, returns)

        # Convert to numpy for combination
        if torch.is_tensor(corr_graph):
            corr_graph = corr_graph.cpu().numpy()
        if torch.is_tensor(spill_graph):
            spill_graph = spill_graph.cpu().numpy()

        # Combine graphs with weights
        combined_graph = alpha * corr_graph + (1 - alpha) * spill_graph

        # Normalize adjacency matrix
        combined_graph = self._normalize_adjacency(combined_graph)

        return torch.FloatTensor(combined_graph)

    def _normalize_adjacency(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix for GAT while preserving sparsity

        Args:
            adj_matrix: [num_stocks, num_stocks] adjacency matrix

        Returns:
            normalized_adjacency: Normalized adjacency matrix
        """
        # Get actual number of stocks from the matrix itself
        actual_num_stocks = adj_matrix.shape[0]

        # Add small self-loops (0.1 instead of 1.0 to preserve sparsity)
        adj_matrix = adj_matrix + np.eye(actual_num_stocks) * 0.1

        # Normalize by row sum
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        normalized = adj_matrix / row_sums

        return normalized

    def build_graph_from_data(self, graph_data: Dict, graph_method: str = 'correlation') -> torch.Tensor:
        """
        Build graph from data dictionary

        Args:
            graph_data: Dictionary with 'returns' and 'volatility' arrays
            graph_method: Graph construction method

        Returns:
            adjacency_matrix: [num_stocks, num_stocks] adjacency matrix
        """
        returns = graph_data['returns']
        volatility = graph_data['volatility']

        if graph_method == 'correlation':
            return self.build_correlation_graph(returns)
        elif graph_method == 'spillover':
            return self.build_spillover_graph(volatility, returns)
        elif graph_method == 'hybrid':
            return self.build_hybrid_graph(returns, volatility)
        else:
            raise ValueError(f"Unknown graph method: {graph_method}")
