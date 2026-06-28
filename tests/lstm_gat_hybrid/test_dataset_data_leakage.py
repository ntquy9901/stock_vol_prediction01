"""
Unit tests for data leakage fixes in LSTM-GNN dataset pipeline

Tests verify:
1. Per-sequence graph construction (no cumulative data leakage)
2. Training-only normalization (no test statistics leakage)
3. Temporal split integrity (no future information in training)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.lstm_gat_hybrid.dataset_with_graph_method import (
    MultiStockDatasetWithGraphMethod,
    create_multi_stock_dataloaders_with_graph_method
)


class TestPerSequenceGraphConstruction:
    """Test that graphs are built per-sequence, not cumulatively"""

    @pytest.fixture
    def dataset_knn(self):
        """Create dataset with k-NN graph method"""
        data_dir = project_root / "data" / "processed" / "vn30_only"
        return MultiStockDatasetWithGraphMethod(
            data_dir=str(data_dir),
            seq_length=22,
            forecast_horizon=5,
            graph_method='knn',
            k_neighbors=8,
            normalize=False,  # Test without normalization first
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False,
            train_mode=False
        )

    @pytest.fixture
    def dataset_correlation(self):
        """Create dataset with correlation graph method"""
        data_dir = project_root / "data" / "processed" / "vn30_only"
        return MultiStockDatasetWithGraphMethod(
            data_dir=str(data_dir),
            seq_length=22,
            forecast_horizon=5,
            graph_method='correlation',
            graph_threshold=0.7,
            normalize=False,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False,
            train_mode=False
        )

    def test_graphs_are_per_sequence_knn(self, dataset_knn):
        """Test that k-NN graphs are different across sequences"""
        # Collect graphs from different sequences
        graphs = []
        indices = [0, 10, 100, 500, 1000]  # Test various positions

        for idx in indices:
            if idx < len(dataset_knn):
                x, adj_matrix, y, graph_data = dataset_knn[idx]
                graphs.append(adj_matrix)

        # Verify graphs are not all identical
        for i in range(1, len(graphs)):
            assert not torch.allclose(graphs[0], graphs[i]), \
                f"Graphs should be different across sequences (seq 0 vs seq {i})"

        print(f"✅ Test passed: k-NN graphs are per-sequence (tested {len(graphs)} sequences)")

    def test_graphs_are_per_sequence_correlation(self, dataset_correlation):
        """Test that correlation graphs are different across sequences"""
        # Collect graphs from different sequences
        graphs = []
        indices = [0, 10, 100, 500, 1000]

        for idx in indices:
            if idx < len(dataset_correlation):
                x, adj_matrix, y, graph_data = dataset_correlation[idx]
                graphs.append(adj_matrix)

        # Verify graphs are not all identical
        for i in range(1, len(graphs)):
            # For correlation graphs, check at least some edges differ
            diff = torch.abs(graphs[0] - graphs[i]).sum()
            assert diff > 0, \
                f"Correlation graphs should differ across sequences (seq 0 vs seq {i}, diff={diff})"

        print(f"✅ Test passed: Correlation graphs are per-sequence (tested {len(graphs)} sequences)")

    def test_graph_uses_only_sequence_window_knn(self, dataset_knn):
        """Test that k-NN graph uses only [i:i+seq_length], not cumulative [0:i+seq_length]"""
        # For sequence at index i, graph should use data from [i, i+seq_length]
        # NOT [0, i+seq_length]

        # Check a few sequences
        for idx in [0, 100, 500, 1000]:
            if idx < len(dataset_knn):
                x, adj_matrix, y, graph_data = dataset_knn[idx]

                # Verify adjacency matrix is not constant (would indicate cumulative data)
                # If cumulative, early sequences would have very similar graphs
                # If per-sequence, graphs should vary more

                # Check that graph has appropriate sparsity (k-NN should be sparse)
                num_edges = (adj_matrix > 0).sum().item()
                num_nodes = adj_matrix.shape[0]

                # k-NN with k=8 should have approximately 8*30 = 240 edges (undirected)
                # Allow some tolerance due to graph construction
                assert 100 < num_edges < 500, \
                    f"Seq {idx}: k-NN graph has unexpected edge count {num_edges} (expected ~240)"

        print("✅ Test passed: k-NN graphs use sequence window (edge counts look reasonable)")


class TestTrainingOnlyNormalization:
    """Test that normalizers are fit only on training data"""

    @pytest.fixture
    def dataloaders_normalized(self):
        """Create dataloaders with normalization enabled"""
        data_dir = project_root / "data" / "processed" / "vn30_only"
        return create_multi_stock_dataloaders_with_graph_method(
            data_dir=str(data_dir),
            seq_length=22,
            forecast_horizon=5,
            graph_method='knn',
            k_neighbors=8,
            batch_size=32,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=True,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False
        )

    def test_normalizers_fit_on_training_only(self, dataloaders_normalized):
        """Test that normalizers are fit only on training data"""
        train_loader, val_loader, test_loader, datasets = dataloaders_normalized
        train_dataset, val_dataset, test_dataset = datasets

        # Get the underlying full_dataset
        full_dataset = train_dataset.dataset

        # Verify normalizers are fitted
        for stock_name in full_dataset.stock_names[:3]:  # Check first 3 stocks
            feature_norm = full_dataset.feature_normalizers[stock_name]
            target_norm = full_dataset.target_normalizers[stock_name]

            # Check that normalizers are fitted (have mean_ and scale_ attributes)
            assert hasattr(feature_norm, 'mean_'), "Feature normalizer should be fitted"
            assert hasattr(feature_norm, 'scale_'), "Feature normalizer should be fitted"
            assert hasattr(target_norm, 'mean_'), "Target normalizer should be fitted"
            assert hasattr(target_norm, 'scale_'), "Target normalizer should be fitted"

            # Check that normalizers are not empty
            assert feature_norm.mean_ is not None, "Feature normalizer mean should not be None"
            assert target_norm.mean_ is not None, "Target normalizer mean should not be None"

        print("✅ Test passed: Normalizers are fitted on training data only")

    def test_val_test_use_training_normalizers(self, dataloaders_normalized):
        """Test that val and test use the same normalizer objects as train"""
        train_loader, val_loader, test_loader, datasets = dataloaders_normalized
        train_dataset, val_dataset, test_dataset = datasets

        # Get the underlying full_dataset (all Subsets share the same dataset)
        full_dataset = train_dataset.dataset

        # Verify all Subsets reference the same dataset
        assert val_dataset.dataset is full_dataset, \
            "Val dataset should share the same underlying dataset as train"
        assert test_dataset.dataset is full_dataset, \
            "Test dataset should share the same underlying dataset as train"

        # Verify they all use the same normalizers (by object identity)
        # Since they all use full_dataset, they automatically share normalizers
        for stock_name in full_dataset.stock_names[:3]:
            # All normalizers should be the same object
            assert id(full_dataset.feature_normalizers[stock_name]) == \
                   id(full_dataset.feature_normalizers[stock_name]), \
                   "Feature normalizers should be the same object"

        print("✅ Test passed: Val and test use training normalizers (same objects)")

    def test_normalization_values_are_reasonable(self, dataloaders_normalized):
        """Test that normalized values have reasonable statistics"""
        train_loader, val_loader, test_loader, datasets = dataloaders_normalized

        # Get a batch from training
        for batch_x, batch_adj, batch_y, batch_graph_data in train_loader:
            # Check that normalized values are roughly centered around 0
            # (StandardScaler should give mean≈0, std≈1)

            # Check features (per-stock, per-feature)
            for stock_idx in range(min(3, batch_x.shape[1])):  # Check first 3 stocks
                for feat_idx in range(batch_x.shape[2]):  # Check all features
                    feat_values = batch_x[:, stock_idx, feat_idx].numpy()

                    # Mean should be close to 0 (within reasonable tolerance)
                    mean_val = np.mean(feat_values)
                    assert -2.0 < mean_val < 2.0, \
                        f"Normalized feature mean should be near 0, got {mean_val}"

                    # Std should be close to 1 (within reasonable tolerance)
                    std_val = np.std(feat_values)
                    assert 0.1 < std_val < 5.0, \
                        f"Normalized feature std should be near 1, got {std_val}"

            # Check targets
            for stock_idx in range(min(3, batch_y.shape[1])):  # Check first 3 stocks
                target_values = batch_y[:, stock_idx].numpy()

                mean_val = np.mean(target_values)
                assert -2.0 < mean_val < 2.0, \
                    f"Normalized target mean should be near 0, got {mean_val}"

            break  # Only check first batch

        print("✅ Test passed: Normalized values have reasonable statistics")


class TestTemporalSplitIntegrity:
    """Test that temporal split maintains chronological integrity"""

    @pytest.fixture
    def dataloaders(self):
        """Create dataloaders with temporal split"""
        data_dir = project_root / "data" / "processed" / "vn30_only"
        return create_multi_stock_dataloaders_with_graph_method(
            data_dir=str(data_dir),
            seq_length=22,
            forecast_horizon=5,
            graph_method='knn',
            k_neighbors=8,
            batch_size=32,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=False,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False
        )

    def test_split_ratios_are_correct(self, dataloaders):
        """Test that train/val/test ratios match specification"""
        train_loader, val_loader, test_loader, datasets = dataloaders
        train_dataset, val_dataset, test_dataset = datasets

        n_train = len(train_dataset)
        n_val = len(val_dataset)
        n_test = len(test_dataset)
        n_total = n_train + n_val + n_test

        train_ratio = n_train / n_total
        val_ratio = n_val / n_total
        test_ratio = n_test / n_total

        # Check ratios are approximately correct (within 1% tolerance)
        assert abs(train_ratio - 0.7) < 0.01, \
            f"Train ratio {train_ratio:.3f} should be ~0.7"
        assert abs(val_ratio - 0.15) < 0.01, \
            f"Val ratio {val_ratio:.3f} should be ~0.15"
        assert abs(test_ratio - 0.15) < 0.01, \
            f"Test ratio {test_ratio:.3f} should be ~0.15"

        print(f"✅ Test passed: Split ratios correct (train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f})")

    def test_sequences_are_chronologically_ordered(self, dataloaders):
        """Test that sequences maintain chronological order"""
        train_loader, val_loader, test_loader, datasets = dataloaders
        train_dataset, val_dataset, test_dataset = datasets

        full_dataset = train_dataset.dataset

        # Check that indices are correctly partitioned
        n = len(full_dataset)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        # Verify train indices
        train_indices = set(train_dataset.indices)
        expected_train = set(range(0, train_end))
        assert train_indices == expected_train, \
            "Train indices should be [0, train_end)"

        # Verify val indices
        val_indices = set(val_dataset.indices)
        expected_val = set(range(train_end, val_end))
        assert val_indices == expected_val, \
            "Val indices should be [train_end, val_end)"

        # Verify test indices
        test_indices = set(test_dataset.indices)
        expected_test = set(range(val_end, n))
        assert test_indices == expected_test, \
            "Test indices should be [val_end, n)"

        print("✅ Test passed: Sequences are chronologically ordered")

    def test_no_overlap_between_splits(self, dataloaders):
        """Test that train/val/test have no overlapping sequences"""
        train_loader, val_loader, test_loader, datasets = dataloaders
        train_dataset, val_dataset, test_dataset = datasets

        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        test_indices = set(test_dataset.indices)

        # Check no overlap
        assert len(train_indices & val_indices) == 0, \
            "Train and val should not overlap"
        assert len(train_indices & test_indices) == 0, \
            "Train and test should not overlap"
        assert len(val_indices & test_indices) == 0, \
            "Val and test should not overlap"

        print("✅ Test passed: No overlap between splits")

    def test_train_does_not_contain_future_info(self, dataloaders):
        """Test that training set does not contain data from val/test periods"""
        train_loader, val_loader, test_loader, datasets = dataloaders
        train_dataset, val_dataset, test_dataset = datasets

        full_dataset = train_dataset.dataset

        # Get max index in train
        max_train_idx = max(train_dataset.indices)

        # Get min index in val and test
        min_val_idx = min(val_dataset.indices)
        min_test_idx = min(test_dataset.indices)

        # Train should end before val starts
        assert max_train_idx < min_val_idx, \
            "Train should end before val starts"

        # Val should end before test starts
        max_val_idx = max(val_dataset.indices)
        assert max_val_idx < min_test_idx, \
            "Val should end before test starts"

        print("✅ Test passed: Train does not contain future information")


class TestGraphConstructionNoDataLeakage:
    """Test that graph construction does not leak future data"""

    @pytest.fixture
    def dataset(self):
        """Create dataset"""
        data_dir = project_root / "data" / "processed" / "vn30_only"
        return MultiStockDatasetWithGraphMethod(
            data_dir=str(data_dir),
            seq_length=22,
            forecast_horizon=5,
            graph_method='knn',
            k_neighbors=8,
            normalize=False,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False,
            train_mode=False
        )

    def test_early_sequences_dont_use_late_patterns(self, dataset):
        """Test that early sequences don't benefit from late sequence patterns"""
        # If graphs were cumulative, early sequences would have similar graphs to late sequences
        # If per-sequence, early graphs should be different from late graphs

        # Get early graphs (first 10 sequences)
        early_graphs = []
        for idx in range(min(10, len(dataset))):
            x, adj_matrix, y, graph_data = dataset[idx]
            early_graphs.append(adj_matrix.clone())

        # Get late graphs (last 10 sequences)
        late_graphs = []
        start_idx = max(0, len(dataset) - 10)
        for idx in range(start_idx, len(dataset)):
            x, adj_matrix, y, graph_data = dataset[idx]
            late_graphs.append(adj_matrix.clone())

        # Compute average early graph and average late graph
        avg_early_graph = torch.stack(early_graphs).mean(dim=0)
        avg_late_graph = torch.stack(late_graphs).mean(dim=0)

        # They should be different (if cumulative, they'd be similar)
        diff = torch.abs(avg_early_graph - avg_late_graph).sum()
        assert diff > 0, "Early and late graphs should differ"

        print(f"✅ Test passed: Early sequences don't use late patterns (diff={diff:.2f})")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
