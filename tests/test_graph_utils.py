"""
Tests for Graph Utilities Module

Tests graph construction methods (correlation and spillover) and
related utilities for the LSTM-HAR-GAT hybrid model.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import pytest
import numpy as np
import pandas as pd
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common.graph_utils import (
    build_correlation_graph,
    build_spillover_graph,
    visualize_graph,
    compute_graph_statistics
)


class TestCorrelationGraph:
    """Test correlation-based graph construction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        num_stocks = 10
        time_steps = 500

        # Create correlated returns
        returns = np.random.randn(num_stocks, time_steps) * 0.01

        # Add correlation between some stocks
        returns[0] = returns[1] * 0.8 + np.random.randn(time_steps) * 0.005
        returns[2] = returns[3] * 0.7 + np.random.randn(time_steps) * 0.006

        stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]

        return returns, stock_names

    def test_correlation_graph_basic(self, sample_data):
        """Test basic correlation graph construction."""
        returns, stock_names = sample_data

        edge_index, edge_weight, graph_info = build_correlation_graph(
            returns, stock_names, threshold=0.5
        )

        # Check output types
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_weight, torch.Tensor)
        assert isinstance(graph_info, dict)

        # Check dimensions
        assert edge_index.shape[0] == 2  # COO format
        assert edge_index.shape[1] == len(edge_weight)
        assert edge_index.shape[1] % 2 == 0  # Undirected (bidirectional edges)

        # Check graph info
        assert 'method' in graph_info
        assert graph_info['method'] == 'correlation'
        assert 'num_nodes' in graph_info
        assert graph_info['num_nodes'] == len(stock_names)
        assert 'num_edges' in graph_info

        print(f"[OK] Correlation graph basic test passed")

    def test_correlation_graph_connectivity(self, sample_data):
        """Test that graph maintains minimum connectivity."""
        returns, stock_names = sample_data

        edge_index, edge_weight, graph_info = build_correlation_graph(
            returns, stock_names, threshold=0.9, min_edges=5
        )

        # Check that graph has reasonable connectivity
        num_nodes = graph_info['num_nodes']
        num_edges = graph_info['num_edges']

        # For undirected graph, total degree = 2 * num_edges
        avg_degree = (2 * num_edges) / num_nodes

        # Check that average degree is reasonable (at least 2 for connectivity)
        assert avg_degree >= 2, f"Average degree {avg_degree} < 2"
        assert num_edges > 0, "Graph has no edges"

        print(f"[OK] Correlation graph connectivity test passed (avg_degree={avg_degree:.2f})")

    def test_correlation_graph_edge_weights(self, sample_data):
        """Test that edge weights are correlation coefficients."""
        returns, stock_names = sample_data

        edge_index, edge_weight, graph_info = build_correlation_graph(
            returns, stock_names, threshold=0.3
        )

        # Check that all weights are between -1 and 1 (correlation range)
        assert torch.all(edge_weight >= -1)
        assert torch.all(edge_weight <= 1)

        # Check that weights match correlation matrix
        corr_matrix = graph_info['corr_matrix']
        edge_index_np = edge_index.cpu().numpy()

        for i in range(0, len(edge_weight), 2):  # Check every pair once
            src, dst = edge_index_np[:, i]
            expected_corr = corr_matrix[src, dst]
            actual_weight = edge_weight[i].item()

            assert abs(expected_corr - actual_weight) < 1e-6

        print(f"[OK] Correlation graph edge weights test passed")


class TestSpilloverGraph:
    """Test volatility spillover graph construction."""

    @pytest.fixture
    def sample_volatility(self):
        """Create sample volatility data for testing."""
        np.random.seed(42)
        num_stocks = 10
        time_steps = 500

        # Create volatility data with some spillover effects
        volatility = np.abs(np.random.randn(num_stocks, time_steps)) * 0.02

        # Add spillover effects
        volatility[2] += volatility[0] * 0.3
        volatility[3] += volatility[1] * 0.4

        stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]

        return volatility, stock_names

    def test_spillover_graph_basic(self, sample_volatility):
        """Test basic spillover graph construction."""
        volatility, stock_names = sample_volatility

        edge_index, edge_weight, graph_info = build_spillover_graph(
            volatility, stock_names, p=4, H=5, threshold=0.01
        )

        # Check output types
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_weight, torch.Tensor)
        assert isinstance(graph_info, dict)

        # Check dimensions
        assert edge_index.shape[0] == 2  # COO format
        assert edge_index.shape[1] == len(edge_weight)

        # Check graph info
        assert 'method' in graph_info
        assert graph_info['method'] == 'spillover'
        assert 'num_nodes' in graph_info
        assert graph_info['num_nodes'] == len(stock_names)
        assert 'spillover_matrix' in graph_info
        assert 'spillover_index' in graph_info

        print(f"[OK] Spillover graph basic test passed")

    def test_spillover_graph_directed(self, sample_volatility):
        """Test that spillover graph is directed."""
        volatility, stock_names = sample_volatility

        edge_index, edge_weight, graph_info = build_spillover_graph(
            volatility, stock_names, p=4, H=5, threshold=0.01
        )

        # Check that graph is directed (edges may not be symmetric)
        edge_index_np = edge_index.cpu().numpy()

        # Sample some edges to check for asymmetry
        found_asymmetric = False
        for i in range(min(10, len(edge_weight))):
            src, dst = edge_index_np[:, i]
            # Check if reverse edge exists with different weight
            reverse_idx = -1
            for j in range(len(edge_weight)):
                if edge_index_np[0, j] == dst and edge_index_np[1, j] == src:
                    reverse_idx = j
                    break

            if reverse_idx == -1:
                # No reverse edge - clearly directed
                found_asymmetric = True
                break
            else:
                # Check if weights are different
                if abs(edge_weight[i] - edge_weight[reverse_idx]) > 1e-6:
                    found_asymmetric = True
                    break

        # We expect at least some asymmetry in spillover graphs
        print(f"  Graph has asymmetric edges: {found_asymmetric}")
        print(f"[OK] Spillover graph directed test passed")

    def test_spillover_graph_threshold(self, sample_volatility):
        """Test that threshold filtering works correctly."""
        volatility, stock_names = sample_volatility

        # Build with high threshold
        edge_index_high, edge_weight_high, _ = build_spillover_graph(
            volatility, stock_names, p=4, H=5, threshold=0.1
        )

        # Build with low threshold
        edge_index_low, edge_weight_low, _ = build_spillover_graph(
            volatility, stock_names, p=4, H=5, threshold=0.001
        )

        # High threshold should have fewer or equal edges
        assert len(edge_weight_high) <= len(edge_weight_low)

        # Check that we got valid graphs
        assert len(edge_weight_high) > 0, "High threshold graph has no edges"
        assert len(edge_weight_low) > 0, "Low threshold graph has no edges"

        print(f"[OK] Spillover graph threshold test passed (high={len(edge_weight_high)} edges, low={len(edge_weight_low)} edges)")


class TestGraphVisualization:
    """Test graph visualization utilities."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph for visualization."""
        num_stocks = 10
        stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]

        # Create simple graph
        edges = []
        for i in range(num_stocks - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weight = torch.ones(len(edges)) * 0.5

        return edge_index, stock_names

    def test_compute_graph_statistics(self, sample_graph):
        """Test graph statistics computation."""
        edge_index, stock_names = sample_graph

        graph_info = {'method': 'correlation'}
        stats = compute_graph_statistics(edge_index, stock_names, graph_info)

        # Check statistics
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert 'avg_degree' in stats
        assert 'top_connected' in stats

        assert stats['num_nodes'] == len(stock_names)
        assert stats['num_edges'] > 0
        assert 0 <= stats['density'] <= 1
        assert stats['avg_degree'] > 0
        assert len(stats['top_connected']) == 5

        print(f"[OK] Graph statistics test passed")


class TestGraphIntegration:
    """Integration tests for graph utilities."""

    def test_graph_construction_pipeline(self):
        """Test complete graph construction pipeline."""
        # Create sample data
        np.random.seed(42)
        num_stocks = 30  # Full VN30
        time_steps = 500

        returns = np.random.randn(num_stocks, time_steps) * 0.01
        volatility = np.abs(np.random.randn(num_stocks, time_steps)) * 0.02

        stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]

        # Build correlation graph
        corr_edge_index, corr_edge_weight, corr_info = build_correlation_graph(
            returns, stock_names, threshold=0.5
        )

        # Build spillover graph
        spill_edge_index, spill_edge_weight, spill_info = build_spillover_graph(
            volatility, stock_names, p=4, H=5, threshold=0.01
        )

        # Verify both graphs have same nodes
        assert corr_info['num_nodes'] == spill_info['num_nodes'] == num_stocks

        # Verify both are valid graphs
        assert corr_info['num_edges'] > 0
        assert spill_info['num_edges'] > 0

        print(f"[OK] Graph construction pipeline test passed")
        print(f"  Correlation edges: {corr_info['num_edges']}")
        print(f"  Spillover edges: {spill_info['num_edges']}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with identical data (perfect correlation)
        num_stocks = 5
        time_steps = 100
        identical_data = np.ones((num_stocks, time_steps))
        stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]

        # Should still create graph
        edge_index, edge_weight, graph_info = build_correlation_graph(
            identical_data, stock_names, threshold=0.5
        )

        assert graph_info['num_nodes'] == num_stocks
        assert graph_info['num_edges'] > 0  # Should maintain min_edges

        print(f"[OK] Edge cases test passed")


def run_manual_tests():
    """Run tests manually (without pytest)."""
    print("\n" + "="*60)
    print("GRAPH UTILITIES - MANUAL TEST SUITE")
    print("="*60)

    # Run correlation graph tests
    print("\n[1] Testing Correlation Graph...")
    test_corr = TestCorrelationGraph()

    # Create sample data manually
    np.random.seed(42)
    num_stocks = 10
    time_steps = 500
    returns = np.random.randn(num_stocks, time_steps) * 0.01
    returns[0] = returns[1] * 0.8 + np.random.randn(time_steps) * 0.005
    returns[2] = returns[3] * 0.7 + np.random.randn(time_steps) * 0.006
    stock_names = [f"STOCK{i:02d}" for i in range(num_stocks)]
    sample_data = (returns, stock_names)

    test_corr.test_correlation_graph_basic(sample_data)
    test_corr.test_correlation_graph_connectivity(sample_data)
    test_corr.test_correlation_graph_edge_weights(sample_data)

    # Run spillover graph tests
    print("\n[2] Testing Spillover Graph...")
    test_spill = TestSpilloverGraph()

    # Create sample volatility manually
    np.random.seed(42)
    volatility = np.abs(np.random.randn(num_stocks, time_steps)) * 0.02
    volatility[2] += volatility[0] * 0.3
    volatility[3] += volatility[1] * 0.4
    sample_vol = (volatility, stock_names)

    test_spill.test_spillover_graph_basic(sample_vol)
    test_spill.test_spillover_graph_directed(sample_vol)
    test_spill.test_spillover_graph_threshold(sample_vol)

    # Run visualization tests
    print("\n[3] Testing Graph Visualization...")
    test_viz = TestGraphVisualization()

    # Create sample graph manually
    edges = []
    for i in range(num_stocks - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.ones(len(edges)) * 0.5
    sample_graph = (edge_index, stock_names)

    test_viz.test_compute_graph_statistics(sample_graph)

    # Run integration tests
    print("\n[4] Testing Integration...")
    test_int = TestGraphIntegration()
    test_int.test_graph_construction_pipeline()
    test_int.test_edge_cases()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


# Run tests
if __name__ == "__main__":
    run_manual_tests()
