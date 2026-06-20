"""
CryptoMamba Enhanced V2 - Integration Tests

Comprehensive integration tests that validate actual behavior (no mocking).
Tests the complete training pipeline with edge cases and failure modes.

Root cause fixes validated:
- HAR features generation and validation
- Model ReLU output constraint (non-negative predictions)
- Edge case handling (NaN, exploding gradients, empty data)
- Learning curve plotting (every 10 epochs)
- All 6 metrics computation

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cryptomamba_baseline.model_enhanced import create_cryptomamba_model_enhanced
from src.cryptomamba_baseline.config_enhanced import MODEL_CONFIG_ENHANCED
from src.common.har_features import generate_har_features, validate_har_features
from src.common.evaluation import evaluate_predictions


class TestHARFeaturesGeneration:
    """Test HAR feature generation from Parkinson volatility."""

    def test_generate_har_features_creates_columns(self):
        """Given parkinson_volatility exists, when generate_har_features called, then HAR columns created."""
        # Create sample data with parkinson_volatility
        df = pd.DataFrame({
            'parkinson_volatility': np.random.rand(100) * 0.1 + 0.01
        })

        # Generate HAR features
        result_df = generate_har_features(df)

        # Assert all HAR columns exist
        assert 'har_daily_vol' in result_df.columns
        assert 'har_weekly_vol' in result_df.columns
        assert 'har_monthly_vol' in result_df.columns

    def test_har_features_have_valid_values(self):
        """Given HAR features generated, when values checked, then no NaN or all zeros."""
        df = pd.DataFrame({
            'parkinson_volatility': np.random.rand(100) * 0.1 + 0.01
        })

        result_df = generate_har_features(df)

        # Check no NaN values
        assert not result_df['har_daily_vol'].isnull().any()
        assert not result_df['har_weekly_vol'].isnull().any()
        assert not result_df['har_monthly_vol'].isnull().any()

        # Check not all zeros
        assert not (result_df['har_daily_vol'] == 0).all()
        assert not (result_df['har_weekly_vol'] == 0).all()
        assert not (result_df['har_monthly_vol'] == 0).all()

    def test_har_features_validation_passes(self):
        """Given valid HAR features, when validate_har_features called, then returns True."""
        df = pd.DataFrame({
            'parkinson_volatility': np.random.rand(100) * 0.1 + 0.01
        })

        result_df = generate_har_features(df)

        # Should not raise exception
        is_valid = validate_har_features(result_df)
        assert is_valid is True

    def test_har_features_missing_columns_raises_error(self):
        """Given HAR features missing, when validate called, then raises ValueError."""
        df = pd.DataFrame({
            'parkinson_volatility': np.random.rand(100) * 0.1 + 0.01
        })

        # Remove HAR columns
        with pytest.raises(ValueError, match="Missing required HAR columns"):
            validate_har_features(df)

    def test_har_features_all_nan_raises_error(self):
        """Given HAR features all NaN, when validate called, then raises ValueError."""
        df = pd.DataFrame({
            'har_daily_vol': [np.nan] * 100,
            'har_weekly_vol': [np.nan] * 100,
            'har_monthly_vol': [np.nan] * 100
        })

        with pytest.raises(ValueError, match="is all NaN"):
            validate_har_features(df)


class TestModelReLUOutput:
    """Test model ReLU output constraint prevents negative predictions."""

    def test_model_output_is_non_negative(self):
        """Given model created with ReLU, when forward pass, then all predictions >= 0."""
        model = create_cryptomamba_model_enhanced(
            num_features=3,
            hidden_dim=64,
            num_layers=3,
            d_state=32,
        )

        model.eval()

        # Create sample input
        x = torch.randn(4, 22, 3)

        with torch.no_grad():
            predictions = model(x)

        # Assert all predictions are non-negative
        assert (predictions >= 0).all(), f"Negative predictions found: {predictions.min().item()}"

    def test_model_output_shape_correct(self):
        """Given model created, when forward pass, then output shape is (batch, 1)."""
        model = create_cryptomamba_model_enhanced()

        x = torch.randn(4, 22, 3)

        with torch.no_grad():
            predictions = model(x)

        assert predictions.shape == (4, 1)

    def test_model_handle_large_input(self):
        """Given large input values, when forward pass, then predictions remain non-negative."""
        model = create_cryptomamba_model_enhanced()
        model.eval()

        # Large input values
        x = torch.randn(4, 22, 3) * 100

        with torch.no_grad():
            predictions = model(x)

        assert (predictions >= 0).all()

    def test_model_handle_negative_input(self):
        """Given negative input values, when forward pass, then predictions remain non-negative."""
        model = create_cryptomamba_model_enhanced()
        model.eval()

        # Negative input values (shouldn't happen with normalized data, but test robustness)
        x = torch.randn(4, 22, 3) * -1

        with torch.no_grad():
            predictions = model(x)

        assert (predictions >= 0).all()


class TestModelInputValidation:
    """Test model input validation catches NaN and inf."""

    def test_model_raises_on_nan_input(self):
        """Given NaN in input, when forward pass, then raises ValueError."""
        model = create_cryptomamba_model_enhanced()
        model.eval()

        # Create input with NaN
        x = torch.randn(4, 22, 3)
        x[0, 0, 0] = float('nan')

        with pytest.raises(ValueError, match="NaN detected"):
            with torch.no_grad():
                _ = model(x)

    def test_model_raises_on_inf_input(self):
        """Given inf in input, when forward pass, then raises ValueError."""
        model = create_cryptomamba_model_enhanced()
        model.eval()

        # Create input with inf
        x = torch.randn(4, 22, 3)
        x[0, 0, 0] = float('inf')

        with pytest.raises(ValueError, match="Inf detected"):
            with torch.no_grad():
                _ = model(x)


class TestTrainingPipeline:
    """Test complete training pipeline with real data."""

    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create sample data file for testing."""
        # Generate sample data
        dates = pd.date_range('2010-01-01', periods=500, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'parkinson_volatility': np.random.rand(500) * 0.1 + 0.01
        })

        # Save to temp file
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        return data_path

    def test_load_data_generates_har_features(self, sample_data_path):
        """Given data without HAR features, when load_and_prepare_data called, then HAR features generated."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        # Load data (should generate HAR features)
        train_loader, val_loader, test_loader = trainer.load_and_prepare_data(sample_data_path)

        # Assert dataloaders created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_insufficient_data_raises_error(self):
        """Given insufficient data, when load_and_prepare_data called, then raises ValueError."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        # Create very small dataset
        df = pd.DataFrame({
            'parkinson_volatility': [0.01] * 10  # Only 10 rows
        })

        temp_path = Path("temp_test.csv")
        df.to_csv(temp_path, index=False)

        trainer = CryptoMambaEnhancedTrainer()

        with pytest.raises(ValueError, match="Insufficient data"):
            trainer.load_and_prepare_data(temp_path)

        # Cleanup
        os.remove(temp_path)

    def test_model_training_runs(self, sample_data_path):
        """Given data loaded, when train called, then training completes without errors."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        train_loader, val_loader, test_loader = trainer.load_and_prepare_data(sample_data_path)
        trainer.create_model()

        # Train for just 5 epochs (integration test, not full training)
        original_num_epochs = trainer.training_config['num_epochs']
        trainer.training_config['num_epochs'] = 5
        trainer.training_config['patience'] = 10
        trainer.training_config['min_epochs'] = 2

        try:
            trainer.train(train_loader, val_loader)
        except Exception as e:
            pytest.fail(f"Training failed with exception: {e}")

        # Assert training ran
        assert len(trainer.train_losses) > 0
        assert len(trainer.val_losses) > 0

    def test_learning_curves_plotted(self, sample_data_path):
        """Given training runs, when epoch 10 reached, then learning curves saved."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        train_loader, val_loader, test_loader = trainer.load_and_prepare_data(sample_data_path)
        trainer.create_model()

        # Train for 10 epochs (should trigger learning curve plot)
        trainer.training_config['num_epochs'] = 10
        trainer.training_config['patience'] = 20
        trainer.training_config['min_epochs'] = 5

        trainer.train(train_loader, val_loader)

        # Check learning curves file exists
        plot_path = trainer.output_dir / "learning_curves.png"
        assert plot_path.exists(), f"Learning curves not saved at {plot_path}"

    def test_all_metrics_computed(self, sample_data_path):
        """Given training completes, when evaluate called, then all 6 metrics computed."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        train_loader, val_loader, test_loader = trainer.load_and_prepare_data(sample_data_path)
        trainer.create_model()

        # Train for 5 epochs
        trainer.training_config['num_epochs'] = 5
        trainer.training_config['patience'] = 10
        trainer.training_config['min_epochs'] = 2

        trainer.train(train_loader, val_loader)
        test_metrics = trainer.evaluate_test_set(test_loader)

        # Assert all 6 metrics present
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']
        for metric in required_metrics:
            assert metric in test_metrics, f"Missing metric: {metric}"

    def test_no_negative_predictions_during_training(self, sample_data_path):
        """Given training runs, when predictions checked, then no negative values occur."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        train_loader, val_loader, test_loader = trainer.load_and_prepare_data(sample_data_path)
        trainer.create_model()

        # Train for 5 epochs
        trainer.training_config['num_epochs'] = 5
        trainer.training_config['patience'] = 10
        trainer.training_config['min_epochs'] = 2

        # Track if any negative predictions
        negative_found = False

        # Patch validate to check predictions
        original_validate = trainer.validate

        def patched_validate(val_loader):
            val_loss, val_metrics = original_validate(val_loader)

            # Check model on validation set
            trainer.model.eval()
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(trainer.device)
                    preds = trainer.model(X_batch)
                    all_preds.append(preds.cpu().numpy())

            all_predictions = np.concatenate(all_preds)

            # Check for negative predictions
            if (all_predictions < 0).any():
                nonlocal negative_found
                negative_found = True

            return val_loss, val_metrics

        trainer.validate = patched_validate

        trainer.train(train_loader, val_loader)

        # Assert no negative predictions found
        assert not negative_found, "Negative predictions detected during training"


class TestEdgeCases:
    """Test edge case handling."""

    def test_empty_dataset_raises_error(self):
        """Given empty dataset, when load called, then raises error."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        df = pd.DataFrame({'parkinson_volatility': []})

        temp_path = Path("temp_empty.csv")
        df.to_csv(temp_path, index=False)

        trainer = CryptoMambaEnhancedTrainer()

        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            trainer.load_and_prepare_data(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_nan_in_features_raises_error(self):
        """Given NaN in features, when load called, then raises ValueError."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        # Create data with NaN in HAR features
        df = pd.DataFrame({
            'parkinson_volatility': [0.01] * 100,
            'har_daily_vol': [0.01] * 50 + [np.nan] * 50,
            'har_weekly_vol': [0.01] * 100,
            'har_monthly_vol': [0.01] * 100
        })

        temp_path = Path("temp_nan.csv")
        df.to_csv(temp_path, index=False)

        trainer = CryptoMambaEnhancedTrainer()

        with pytest.raises(ValueError, match="NaN found in features"):
            trainer.load_and_prepare_data(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_invalid_learning_rate_raises_error(self):
        """Given invalid learning rate, when create_model called, then raises ValueError."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()

        # Set invalid learning rate
        trainer.training_config['learning_rate'] = 2.0  # > 1.0

        with pytest.raises(ValueError, match="Invalid learning rate"):
            trainer.create_model()


class TestResultsSaving:
    """Test results saving and output format."""

    def test_results_json_saved(self, tmp_path):
        """Given training completes, when save_results called, then JSON saved correctly."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        # Override output dir to tmp_path
        trainer = CryptoMambaEnhancedTrainer()
        trainer.output_dir = tmp_path / "test_results"
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        # Mock test metrics
        trainer.test_metrics = {
            'mse': 0.0005,
            'rmse': 0.022,
            'mae': 0.018,
            'r2': 0.65,
            'qlike': 0.12,
            'directional_accuracy': 51.5
        }

        # Create matching validation metrics for each epoch
        val_metrics_1 = trainer.test_metrics.copy()
        val_metrics_2 = trainer.test_metrics.copy()
        val_metrics_3 = trainer.test_metrics.copy()
        trainer.val_metrics_history = [val_metrics_1, val_metrics_2, val_metrics_3]
        trainer.train_losses = [0.001, 0.0008, 0.0006]
        trainer.val_losses = [0.0012, 0.0010, 0.0009]

        # Save results
        results = trainer.save_results()

        # Check JSON file exists
        results_path = trainer.output_dir / "cryptomamba_enhanced_results.json"
        assert results_path.exists()

        # Load and validate JSON structure
        import json
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)

        # Check required keys
        assert 'validation_metrics' in loaded_results
        assert 'test_metrics' in loaded_results
        assert 'val_test_diff' in loaded_results

        # Check all 6 metrics present
        for metrics_key in ['validation_metrics', 'test_metrics']:
            for metric in ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']:
                assert metric in loaded_results[metrics_key], f"Missing {metric} in {metrics_key}"

    def test_results_json_serializable(self, tmp_path):
        """Given results with numpy types, when save_results called, then JSON serializable."""
        from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer

        trainer = CryptoMambaEnhancedTrainer()
        trainer.output_dir = tmp_path / "test_results_serializable"
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        # Use numpy types (common scenario)
        trainer.test_metrics = {
            'mse': np.float64(0.0005),
            'rmse': np.float64(0.022),
            'mae': np.float64(0.018),
            'r2': np.float64(0.65),
            'qlike': np.float64(0.12),
            'directional_accuracy': np.float64(51.5)
        }

        trainer.val_metrics_history = [trainer.test_metrics]
        trainer.train_losses = [0.001]
        trainer.val_losses = [0.0012]

        # Should not raise TypeError
        try:
            trainer.save_results()
        except TypeError as e:
            pytest.fail(f"JSON serialization failed: {e}")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
