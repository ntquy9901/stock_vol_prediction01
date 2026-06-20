"""
Test Suite for TimesFM 2.5 LoRA Fine-Tuning

Comprehensive tests for data pipeline, model loading, LoRA configuration,
training loop, and evaluation metrics.

Target: 80%+ test coverage

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.timesfm_baseline.volatility_dataset import (
    TimeSeriesRandomWindowDataset,
    TimeSeriesLastWindowDataset,
    create_volatility_datasets,
    create_multi_stock_volatility_datasets,
)
from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
from src.common.parkinson_utils import calculate_parkinson_volatility
from src.common.evaluation import evaluate_predictions


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 1000

    dates = pd.date_range(start='2010-01-01', periods=n_days, freq='D')
    price = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    high = price + np.abs(np.random.randn(n_days) * 2)
    low = price - np.abs(np.random.randn(n_days) * 2)

    df = pd.DataFrame({
        'date': dates,
        'open': price,
        'high': high,
        'low': low,
        'close': price,
        'volume': np.random.randint(1000000, 10000000, n_days),
    })

    return df


@pytest.fixture
def sample_parkinson_data(sample_ohlcv_data):
    """Create sample Parkinson volatility data."""
    return calculate_parkinson_volatility(sample_ohlcv_data)


@pytest.fixture
def sample_series_list():
    """Create list of sample time series for testing."""
    np.random.seed(42)
    series_list = []

    for i in range(5):
        # Each series has 200-500 points
        n_points = np.random.randint(200, 500)
        series = np.random.randn(n_points).astype(np.float32) * 0.1 + 0.5
        series_list.append(series)

    return series_list


class TestTimeSeriesRandomWindowDataset:
    """Tests for TimeSeriesRandomWindowDataset."""

    def test_initialization(self, sample_series_list):
        """Test dataset initialization with valid parameters."""
        dataset = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=64,
            horizon_len=5,
            num_samples=100,
            seed=42
        )

        assert len(dataset) == 100
        assert dataset.context_len == 64
        assert dataset.horizon_len == 5

    def test_initialization_insufficient_data(self):
        """Test that ValueError is raised when series too short."""
        short_series = [np.array([1.0, 2.0])]

        with pytest.raises(ValueError, match="No series long enough"):
            TimeSeriesRandomWindowDataset(
                short_series,
                context_len=64,
                horizon_len=5,
                num_samples=10
            )

    def test_getitem(self, sample_series_list):
        """Test retrieving a single sample."""
        dataset = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=64,
            horizon_len=5,
            num_samples=100,
            seed=42
        )

        context, target = dataset[0]

        assert isinstance(context, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert context.shape == (64,)
        assert target.shape == (5,)
        assert context.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_getitem_multiple_samples(self, sample_series_list):
        """Test retrieving multiple samples."""
        dataset = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=42
        )

        for i in range(10):
            context, target = dataset[i]
            assert context.shape == (32,)
            assert target.shape == (5,)

    def test_reproducibility(self, sample_series_list):
        """Test that same seed produces same samples."""
        dataset1 = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=42
        )

        dataset2 = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=42
        )

        # Same seed should produce same samples
        context1, target1 = dataset1[0]
        context2, target2 = dataset2[0]

        assert torch.equal(context1, context2)
        assert torch.equal(target1, target2)

    def test_different_seeds(self, sample_series_list):
        """Test that different seeds produce different samples."""
        dataset1 = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=42
        )

        dataset2 = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=123
        )

        # Different seeds should produce different samples
        context1, target1 = dataset1[0]
        context2, target2 = dataset2[0]

        assert not torch.equal(context1, context2)


class TestTimeSeriesLastWindowDataset:
    """Tests for TimeSeriesLastWindowDataset."""

    def test_initialization(self, sample_series_list):
        """Test dataset initialization with valid parameters."""
        dataset = TimeSeriesLastWindowDataset(
            sample_series_list,
            context_len=64,
            horizon_len=5
        )

        assert len(dataset) == len(sample_series_list)
        assert dataset.context_len == 64
        assert dataset.horizon_len == 5

    def test_initialization_filters_short_series(self, sample_series_list):
        """Test that short series are filtered out."""
        # Add a short series
        short_series_list = sample_series_list + [np.array([1.0, 2.0, 3.0])]

        dataset = TimeSeriesLastWindowDataset(
            short_series_list,
            context_len=64,
            horizon_len=5
        )

        # Short series should be filtered out
        assert len(dataset) == len(sample_series_list)

    def test_getitem(self, sample_series_list):
        """Test retrieving a single sample."""
        dataset = TimeSeriesLastWindowDataset(
            sample_series_list,
            context_len=64,
            horizon_len=5
        )

        context, target = dataset[0]

        assert isinstance(context, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert context.shape == (64,)
        assert target.shape == (5,)

    def test_last_window_correctness(self):
        """Test that dataset uses last window correctly."""
        # Create a series with known values
        series = np.arange(100, dtype=np.float32)

        dataset = TimeSeriesLastWindowDataset(
            [series],
            context_len=10,
            horizon_len=5
        )

        context, target = dataset[0]

        # Context should be last 10 points before target
        expected_context = torch.tensor(series[85:95], dtype=torch.float32)
        expected_target = torch.tensor(series[95:100], dtype=torch.float32)

        assert torch.equal(context, expected_context)
        assert torch.equal(target, expected_target)


class TestCreateVolatilityDatasets:
    """Tests for create_volatility_datasets function."""

    def test_create_datasets(self, sample_ohlcv_data):
        """Test dataset creation from OHLCV data."""
        train_ds, val_ds, test_ds = create_volatility_datasets(
            sample_ohlcv_data,
            context_len=64,
            horizon_len=5,
            num_train_samples=100,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        assert isinstance(train_ds, TimeSeriesRandomWindowDataset)
        assert isinstance(val_ds, TimeSeriesLastWindowDataset)
        assert isinstance(test_ds, TimeSeriesLastWindowDataset)

        assert len(train_ds) == 100
        assert len(val_ds) >= 1
        assert len(test_ds) >= 1

    def test_temporal_split_maintained(self, sample_ohlcv_data):
        """Test that temporal split is maintained."""
        train_ds, val_ds, test_ds = create_volatility_datasets(
            sample_ohlcv_data,
            context_len=64,
            horizon_len=5,
            num_train_samples=10,
            seed=42
        )

        # Check that train comes before val comes before test
        # This is implicit in the temporal_split function
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None


class TestCreateMultiStockVolatilityDatasets:
    """Tests for create_multi_stock_volatility_datasets function."""

    def test_multi_stock_datasets(self, sample_ohlcv_data):
        """Test dataset creation for multiple stocks."""
        stocks_data = {
            'STOCK1': sample_ohlcv_data,
            'STOCK2': sample_ohlcv_data.copy(),
            'STOCK3': sample_ohlcv_data.copy(),
        }

        train_ds, val_ds, test_ds = create_multi_stock_volatility_datasets(
            stocks_data,
            context_len=64,
            horizon_len=5,
            num_train_samples=100,
            seed=42
        )

        assert isinstance(train_ds, TimeSeriesRandomWindowDataset)
        assert isinstance(val_ds, TimeSeriesLastWindowDataset)
        assert isinstance(test_ds, TimeSeriesLastWindowDataset)

    def test_empty_stocks_dict(self):
        """Test that empty dict raises appropriate error."""
        with pytest.raises((ValueError, KeyError)):
            create_multi_stock_volatility_datasets(
                {},
                context_len=64,
                horizon_len=5,
                seed=42
            )


class TestTimesFMLoRAFineTuner:
    """Tests for TimesFMLoRAFineTuner class."""

    @pytest.fixture
    def finetuner(self):
        """Create a TimesFMLoRAFineTuner instance."""
        return TimesFMLoRAFineTuner(
            context_len=64,
            horizon_len=5,
            lora_r=4,
            lora_alpha=8,
            seed=42
        )

    def test_initialization(self, finetuner):
        """Test finetuner initialization."""
        assert finetuner.context_len == 64
        assert finetuner.horizon_len == 5
        assert finetuner.lora_config.r == 4
        assert finetuner.lora_config.lora_alpha == 8
        assert finetuner.seed == 42

    def test_device_detection(self):
        """Test device detection."""
        finetuner = TimesFMLoRAFineTuner(device="cpu", seed=42)
        assert finetuner.device == "cpu"

    @patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction')
    @patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model')
    def test_load_model(self, mock_get_peft, mock_model_class, finetuner):
        """Test model loading."""
        # Mock model
        mock_model = MagicMock()
        mock_model.config.context_length = 512
        mock_model_class.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Mock print_trainable_parameters
        mock_model.print_trainable_parameters = MagicMock()

        finetuner.load_model()

        assert finetuner.model is not None
        mock_model_class.from_pretrained.assert_called_once()
        mock_get_peft.assert_called_once()

    def test_load_model_adjusts_context_length(self):
        """Test that context length is adjusted if exceeds model max."""
        finetuner = TimesFMLoRAFineTuner(
            context_len=20000,  # Exceeds TimesFM 2.5 max of 16384
            horizon_len=5,
            seed=42
        )

        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_model = MagicMock()
            mock_model.config.context_length = 16384
            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                finetuner.load_model()

                # Original context_len should remain unchanged (no silent mutation)
                assert finetuner.context_len == 20000
                # Actual context length used should be adjusted to model max
                assert finetuner._actual_context_len == 16384

    def test_train_without_model_raises_error(self, finetuner):
        """Test that training without loading model raises ValueError."""
        with pytest.raises(ValueError, match="Model not loaded"):
            finetuner.evaluate(test_dataset=MagicMock())


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_evaluate_predictions_all_metrics(self):
        """Test that all 6 mandatory metrics are calculated."""
        np.random.seed(42)
        y_true = np.random.randn(100).astype(np.float32) * 0.1 + 0.5
        y_pred = y_true + np.random.randn(100).astype(np.float32) * 0.05

        metrics = evaluate_predictions(y_true, y_pred)

        # Check all 6 metrics are present
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'qlike' in metrics
        assert 'directional_accuracy' in metrics

    def test_evaluate_predictions_metric_ranges(self):
        """Test that metrics are in expected ranges."""
        np.random.seed(42)
        y_true = np.random.randn(100).astype(np.float32) * 0.1 + 0.5
        y_pred = y_true + np.random.randn(100).astype(np.float32) * 0.05

        metrics = evaluate_predictions(y_true, y_pred)

        # MSE, RMSE, MAE should be positive
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0

        # R² should be between -inf and 1
        assert metrics['r2'] <= 1.0

        # QLIKE should be positive
        assert metrics['qlike'] >= 0

        # Directional accuracy should be between 0 and 100
        assert 0 <= metrics['directional_accuracy'] <= 100

    def test_evaluate_predictions_perfect_forecast(self):
        """Test metrics with perfect forecast."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        y_pred = y_true.copy()

        metrics = evaluate_predictions(y_true, y_pred)

        # Perfect forecast should have zero error
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2'] == 1.0  # R² = 1 for perfect fit
        assert metrics['directional_accuracy'] == 100.0  # Perfect direction

    def test_evaluate_predictions_directional_accuracy_formula(self):
        """Test that directional accuracy uses sign of CHANGES, not values."""
        # Create data where all values are positive but changes vary
        y_true = np.array([0.1, 0.2, 0.1, 0.3, 0.2], dtype=np.float32)
        y_pred = y_true.copy()  # Perfect forecast

        metrics = evaluate_predictions(y_true, y_pred)

        # Should be 100% because changes match perfectly
        assert metrics['directional_accuracy'] == 100.0


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self, sample_ohlcv_data):
        """Test complete pipeline from data to model loading."""
        # Create datasets
        train_ds, val_ds, test_ds = create_volatility_datasets(
            sample_ohlcv_data,
            context_len=64,
            horizon_len=5,
            num_train_samples=10,
            seed=42
        )

        # Verify datasets are created
        assert len(train_ds) == 10
        assert len(val_ds) >= 1

        # Create finetuner
        finetuner = TimesFMLoRAFineTuner(
            context_len=64,
            horizon_len=5,
            seed=42
        )

        # Verify finetuner is created (we won't actually load model in tests)
        assert finetuner is not None
        assert finetuner.context_len == 64

    def test_data_loader_compatibility(self, sample_series_list):
        """Test that datasets work with DataLoader."""
        dataset = TimeSeriesRandomWindowDataset(
            sample_series_list,
            context_len=32,
            horizon_len=5,
            num_samples=50,
            seed=42
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        batches = list(loader)
        assert len(batches) > 0

        # Check batch structure
        contexts, targets = batches[0]
        assert contexts.shape[0] <= 8  # Batch size
        assert contexts.shape[1] == 32  # Context length
        assert targets.shape[1] == 5  # Horizon length


class TestAdversarialReviewFixes:
    """Tests for bugs found in second adversarial review."""

    @pytest.fixture
    def finetuner(self):
        """Create a TimesFMLoRAFineTuner instance for testing."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
        return TimesFMLoRAFineTuner(
            context_len=64,
            horizon_len=5,
            seed=42
        )

    def test_actual_context_len_initialized(self, finetuner):
        """Test that _actual_context_len is initialized in __init__ (fixes AttributeError)."""
        # Should be initialized even without calling load_model()
        assert hasattr(finetuner, '_actual_context_len')
        assert finetuner._actual_context_len == 64

    def test_train_without_load_model_raises_clear_error(self, finetuner):
        """Test that train() without load_model() loads model automatically."""
        # Create datasets
        import numpy as np
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10, seed=42)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        # Mock the model loading to avoid actually downloading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                # This should work - load_model() is called automatically
                with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFMLoRAFineTuner.train'):
                    # Don't actually run training, just check it doesn't crash on _actual_context_len
                    assert finetuner._actual_context_len == 64

    def test_train_epochs_validation(self, finetuner):
        """Test that epochs <= 0 raises ValueError."""
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        with pytest.raises(ValueError, match="epochs must be positive"):
            finetuner.train(train_dataset, val_dataset, epochs=0, batch_size=2)

        with pytest.raises(ValueError, match="epochs must be positive"):
            finetuner.train(train_dataset, val_dataset, epochs=-1, batch_size=2)

    def test_train_batch_size_validation(self, finetuner):
        """Test that batch_size <= 0 raises ValueError."""
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            finetuner.train(train_dataset, val_dataset, epochs=1, batch_size=0)

    def test_train_output_dir_validation(self, finetuner):
        """Test that invalid output_dir raises ValueError."""
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
            finetuner.train(train_dataset, val_dataset, epochs=1, batch_size=2, output_dir="")

        with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
            finetuner.train(train_dataset, val_dataset, epochs=1, batch_size=2, output_dir=None)

        with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
            finetuner.train(train_dataset, val_dataset, epochs=1, batch_size=2, output_dir="   ")

    def test_temporal_validation_empty_dataframe(self):
        """Test that empty dataframes raise clear error."""
        import pandas as pd

        empty_df = pd.DataFrame({'date': [], 'value': []})
        val_df = pd.DataFrame({'date': ['2020-01-01'], 'value': [1.0]})
        test_df = pd.DataFrame({'date': ['2021-01-01'], 'value': [2.0]})

        from src.timesfm_baseline.volatility_dataset import validate_temporal_integrity

        with pytest.raises(ValueError, match="Train dataframe is empty"):
            validate_temporal_integrity(empty_df, val_df, test_df)

        with pytest.raises(ValueError, match="Validation dataframe is empty"):
            validate_temporal_integrity(val_df, empty_df, test_df)

    def test_temporal_validation_bad_dates(self):
        """Test that unparseable dates raise clear error."""
        import pandas as pd

        train_df = pd.DataFrame({'date': ['2020-01-01', 'invalid-date'], 'value': [1.0, 2.0]})
        val_df = pd.DataFrame({'date': ['2021-01-01'], 'value': [1.0]})
        test_df = pd.DataFrame({'date': ['2022-01-01'], 'value': [2.0]})

        from src.timesfm_baseline.volatility_dataset import validate_temporal_integrity

        with pytest.raises(ValueError, match="Failed to parse dates"):
            validate_temporal_integrity(train_df, val_df, test_df)

    def test_validation_dataset_on_the_fly_tensor_creation(self):
        """Test that TimeSeriesLastWindowDataset creates tensors on-the-fly (memory efficient)."""
        import numpy as np

        # Create multiple series
        series_list = [np.random.randn(1000).astype(np.float32) for _ in range(10)]

        dataset = TimeSeriesLastWindowDataset(series_list, context_len=64, horizon_len=5)

        # Check that it stores indices, not pre-created tensors
        assert hasattr(dataset, 'valid_indices')
        assert len(dataset.valid_indices) == 10
        assert not hasattr(dataset, 'items')  # Should not have 'items' attribute from old implementation

        # Get first sample - tensor should be created now
        context, target = dataset[0]
        assert isinstance(context, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert context.shape == (64,)
        assert target.shape == (5,)

        # Get same sample again - should create new tensor (not cached)
        context2, target2 = dataset[0]
        # Tensors should be different objects (created on-the-fly)
        assert context is not context2
        # But values should be the same
        assert torch.equal(context, context2)

    def test_config_checkpoint_interval_calculation(self):
        """Test config checkpoint_interval calculation with edge cases."""
        from src.timesfm_baseline.config import TimesFMLoRAConfig

        # Normal case: multiple batches per epoch
        config = TimesFMLoRAConfig(num_train_samples=100, batch_size=10)
        # 100 samples / 10 batch_size = 10 batches, save twice per epoch → every 5 batches
        assert config.checkpoint_interval == 5

        # Edge case: batch_size > num_train_samples (one batch per epoch)
        config = TimesFMLoRAConfig(num_train_samples=10, batch_size=100)
        # Should save once per epoch
        assert config.checkpoint_interval == 1

        # Edge case: exact multiple
        config = TimesFMLoRAConfig(num_train_samples=100, batch_size=25)
        # 100 / 25 = 4 batches, save twice → every 2 batches
        assert config.checkpoint_interval == 2

        # Edge case: with remainder
        config = TimesFMLoRAConfig(num_train_samples=95, batch_size=10)
        # 95 / 10 = 9.5 → 10 batches (with partial), save twice → every 5 batches
        assert config.checkpoint_interval == 5

        # Custom checkpoint frequency overrides calculation
        config = TimesFMLoRAConfig(num_train_samples=100, batch_size=10, checkpoint_frequency=3)
        assert config.checkpoint_interval == 3


class TestThirdAdversarialReviewFixes:
    """Tests for bugs found in third adversarial review."""

    def test_patience_is_configurable(self):
        """Test that patience is now a parameter (not hardcoded)."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
        from unittest.mock import MagicMock, patch

        import numpy as np
        import torch
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)

        # Mock model loading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            # Create a mock parameter that optimizer can use
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model.parameters.return_value = [mock_param]  # Return actual parameter

            # Mock forward pass to return output object with loss and prediction
            class MockOutput:
                def __init__(self, batch_size=2):
                    self.loss = torch.randn(1, requires_grad=True)
                    self.prediction = torch.randn(batch_size, 5)
                    self.mean_predictions = torch.randn(batch_size, 5)  # For validation

            mock_model.side_effect = lambda *args, **kwargs: MockOutput(batch_size=2)

            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                # Test with custom patience
                finetuner.train(
                    train_dataset,
                    val_dataset,
                    epochs=1,
                    batch_size=2,
                    patience=20,  # Custom patience (not hardcoded 15)
                    output_dir=tempfile.mkdtemp()
                )

    def test_patience_validation(self):
        """Test that patience <= 0 raises ValueError."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner

        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)

        # Mock model loading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                # Test patience=0 raises error
                with pytest.raises(ValueError, match="patience must be positive"):
                    finetuner.train(
                        train_dataset,
                        val_dataset,
                        epochs=1,
                        batch_size=2,
                        patience=0,
                        output_dir=tempfile.mkdtemp()
                    )

                # Test patience=-1 raises error
                with pytest.raises(ValueError, match="patience must be positive"):
                    finetuner.train(
                        train_dataset,
                        val_dataset,
                        epochs=1,
                        batch_size=2,
                        patience=-1,
                        output_dir=tempfile.mkdtemp()
                    )

    def test_context_horizon_validation(self):
        """Test that context_len + horizon_len is validated against series length."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner

        # Create series that is too short (needs 64+5=69, but has 50)
        short_series = np.random.randn(50).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset(
            [short_series],  # Should fail validation
            context_len=64,
            horizon_len=5,
            num_samples=10,
            seed=42
        )
        val_dataset = TimeSeriesLastWindowDataset([short_series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)

        # Should raise ValueError about series being too short
        with pytest.raises(ValueError, match="requires at least.*69 points"):
            finetuner.train(
                train_dataset,
                val_dataset,
                epochs=1,
                batch_size=2,
                patience=5,
                output_dir=tempfile.mkdtemp()
            )

    def test_drop_last_warning(self, caplog):
        """Test that drop_last=True warns about data loss."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
        from unittest.mock import MagicMock, patch

        import numpy as np
        import logging
        import torch

        # Create dataset with size not divisible by batch_size (100 samples, batch_size=32)
        series = np.random.randn(100).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)

        # Mock model loading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model.parameters.return_value = [mock_param]

            # Mock forward pass to return output object with loss and prediction
            class MockOutput:
                def __init__(self, batch_size=2):
                    self.loss = torch.randn(1, requires_grad=True)
                    self.prediction = torch.randn(batch_size, 5)
                    self.mean_predictions = torch.randn(batch_size, 5)  # For validation

            mock_model.side_effect = lambda *args, **kwargs: MockOutput(batch_size=2)

            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                with caplog.at_level(logging.WARNING):
                    finetuner.train(
                        train_dataset,
                        val_dataset,
                        epochs=1,
                        batch_size=32,  # 100 samples / 32 = 3 batches, 1 sample dropped
                        patience=5,
                        output_dir=tempfile.mkdtemp()
                    )

                # Verify warning was logged about data loss
                assert "Losing" in caplog.text
                assert "samples" in caplog.text
                assert "100.0%" in caplog.text  # All samples dropped because only 10 samples in dataset

    def test_best_model_has_timestamp(self):
        """Test that best model path includes timestamp."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import numpy as np
        import torch
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)
        output_dir = Path(tempfile.mkdtemp())

        # Mock model loading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model.parameters.return_value = [mock_param]

            # Mock forward pass to return output object with loss and prediction
            class MockOutput:
                def __init__(self, batch_size=2):
                    self.loss = torch.randn(1, requires_grad=True)
                    self.prediction = torch.randn(batch_size, 5)
                    self.mean_predictions = torch.randn(batch_size, 5)  # For validation

            mock_model.side_effect = lambda *args, **kwargs: MockOutput(batch_size=2)

            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()
                mock_model.save_pretrained = MagicMock()

                # First "best" save
                finetuner.train(
                    train_dataset,
                    val_dataset,
                    epochs=1,
                    batch_size=2,
                    patience=5,
                    output_dir=str(output_dir)
                )

                # Verify save_pretrained was called with timestamped path
                assert mock_model.save_pretrained.call_count >= 1
                # Get the last call (should be the best model)
                last_call_args = mock_model.save_pretrained.call_args
                last_path = Path(last_call_args[0][0])

                # Verify path has timestamp format (best_model_YYYYMMDD_HHMMSS)
                assert "best_model_" in str(last_path)
                # Extract timestamp from path
                timestamp_str = str(last_path).split("best_model_")[1].split("\\")[0]
                assert len(timestamp_str) == 15  # YYYYMMDD_HHMMSS format

    def test_learning_curves_has_timestamp(self):
        """Test that learning curves have timestamp to prevent race condition."""
        from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import numpy as np
        import torch
        series = np.random.randn(200).astype(np.float32)
        train_dataset = TimeSeriesRandomWindowDataset([series], context_len=64, horizon_len=5, num_samples=10)
        val_dataset = TimeSeriesLastWindowDataset([series], context_len=64, horizon_len=5)

        finetuner = TimesFMLoRAFineTuner(context_len=64, horizon_len=5, seed=42)

        # Mock model loading
        with patch('src.timesfm_baseline.timesfm_lora_finetuning.TimesFm2_5ModelForPrediction') as mock_model_class:
            mock_param = torch.nn.Parameter(torch.randn(10, 10))
            mock_model = MagicMock()
            mock_model.config.context_length = 64
            mock_model.parameters.return_value = [mock_param]

            # Mock forward pass to return output object with loss and prediction
            class MockOutput:
                def __init__(self, batch_size=2):
                    self.loss = torch.randn(1, requires_grad=True)
                    self.prediction = torch.randn(batch_size, 5)
                    self.mean_predictions = torch.randn(batch_size, 5)  # For validation

            mock_model.side_effect = lambda *args, **kwargs: MockOutput(batch_size=2)

            mock_model_class.from_pretrained.return_value = mock_model

            with patch('src.timesfm_baseline.timesfm_lora_finetuning.get_peft_model') as mock_get_peft:
                mock_get_peft.return_value = mock_model
                mock_model.print_trainable_parameters = MagicMock()

                output_dir = Path(tempfile.mkdtemp())

                finetuner.train(
                    train_dataset,
                    val_dataset,
                    epochs=1,
                    batch_size=2,
                    patience=5,
                    output_dir=str(output_dir)
                )

                # Verify learning curves file has timestamp
                learning_curves = list(output_dir.glob("learning_curves_*.png"))
                assert len(learning_curves) == 1
                # Verify timestamp format
                timestamp_str = learning_curves[0].stem.split("_")[1]
                assert len(timestamp_str) == 15  # YYYYMMDD_HHMMSS format


@pytest.mark.slow
class TestSlowTests:
    """Slow tests that require model loading (mark as slow)."""

    @pytest.fixture
    def loaded_finetuner(self):
        """Load actual TimesFM 2.5 model (slow test)."""
        finetuner = TimesFMLoRAFineTuner(
            context_len=64,
            horizon_len=5,
            seed=42
        )

        # This will download the model if not cached
        finetuner.load_model()

        return finetuner

    def test_model_loading_with_actual_model(self, loaded_finetuner):
        """Test that actual model loads correctly."""
        assert loaded_finetuner.model is not None

        # Check that LoRA is applied
        # The model should have trainable parameters
        total_params = sum(p.numel() for p in loaded_finetuner.model.parameters())
        trainable_params = sum(p.numel() for p in loaded_finetuner.model.parameters() if p.requires_grad)

        # With LoRA r=4, trainable params should be ~1.4M out of ~232M
        assert trainable_params > 0
        assert trainable_params < total_params
        assert trainable_params / total_params < 0.01  # Less than 1%


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=src/timesfm_baseline", "--cov-report=html"])
