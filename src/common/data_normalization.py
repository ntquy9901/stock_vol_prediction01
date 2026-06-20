"""
Data normalization utilities for volatility prediction

Volatility data has very small scale (0.0001-0.01) which causes training instability.
This module provides normalization to standard scale (mean=0, std=1) for training.
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict


class VolatilityNormalizer:
    """
    Normalize/denormalize volatility data for stable training.

    Standard approach: Normalize to mean=0, std=1 for training,
    then denormalize predictions for evaluation.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> 'VolatilityNormalizer':
        """
        Fit normalizer to data statistics.

        Args:
            data: Training data to compute statistics from

        Returns:
            self (fitted normalizer)
        """
        self.mean = np.mean(data)
        self.std = np.std(data)

        # Avoid division by zero
        if self.std < 1e-8:
            self.std = 1.0

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to mean=0, std=1.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        return (data - self.mean) / self.std

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original scale.

        Args:
            normalized_data: Normalized data

        Returns:
            Data in original scale
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        return normalized_data * self.std + self.mean

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def get_params(self) -> Dict[str, float]:
        """Get normalization parameters for saving/loading."""
        return {'mean': self.mean, 'std': self.std}

    def set_params(self, params: Dict[str, float]):
        """Set normalization parameters from saved state."""
        self.mean = params['mean']
        self.std = params['std']


def normalize_for_training(
    X: np.ndarray,
    y: np.ndarray,
    fit_on_training: bool = True
) -> Tuple[np.ndarray, np.ndarray, VolatilityNormalizer, Dict]:
    """
    Normalize features and targets for stable training.

    Args:
        X: Features (sequences)
        y: Targets (volatility values)
        fit_on_training: Whether to fit normalizer on training data

    Returns:
        X_norm: Normalized features (per-feature normalization)
        y_norm: Normalized targets
        target_normalizer: Fitted normalizer for denormalization
        feature_stats: Dictionary of feature statistics
    """
    # Normalize features (per-feature)
    X_norm = np.zeros_like(X)
    feature_stats = {}

    for i in range(X.shape[-1]):  # For each feature
        feature_data = X[:, :, i]
        mean = np.mean(feature_data)
        std = np.std(feature_data)

        if std < 1e-8:
            std = 1.0

        X_norm[:, :, i] = (feature_data - mean) / std
        feature_stats[f'feature_{i}'] = {'mean': mean, 'std': std}

    # Normalize targets
    target_normalizer = VolatilityNormalizer()
    y_norm = target_normalizer.fit_transform(y.flatten()).reshape(-1, 1)

    return X_norm, y_norm, target_normalizer, feature_stats


def denormalize_predictions(
    predictions: np.ndarray,
    normalizer: VolatilityNormalizer
) -> np.ndarray:
    """
    Denormalize predictions back to original volatility scale.

    Args:
        predictions: Normalized predictions
        normalizer: Fitted normalizer

    Returns:
        Predictions in original scale
    """
    return normalizer.inverse_transform(predictions.flatten())


# CLI for testing
if __name__ == "__main__":
    import pandas as pd

    # Test with sample data
    print("Testing volatility normalization...")

    # Create sample data (small scale like real volatility)
    y_train = np.array([0.0001, 0.0002, 0.0005, 0.0010, 0.0015]).reshape(-1, 1)

    # Fit normalizer
    normalizer = VolatilityNormalizer()
    y_norm = normalizer.fit_transform(y_train.flatten())

    print(f"Original y: {y_train.flatten()}")
    print(f"Normalized y: {y_norm}")
    print(f"Mean: {normalizer.mean:.6f}, Std: {normalizer.std:.6f}")

    # Test inverse transform
    y_reconstructed = normalizer.inverse_transform(y_norm).reshape(-1, 1)
    print(f"Reconstructed y: {y_reconstructed.flatten()}")
    print(f"Reconstruction error: {np.max(np.abs(y_train - y_reconstructed)):.10f}")

    # Test with predictions
    predictions_norm = np.array([0.5, -0.3, 1.2, 0.0, -0.8])
    predictions_orig = denormalize_predictions(predictions_norm, normalizer)
    print(f"\nNormalized predictions: {predictions_norm}")
    print(f"Original scale predictions: {predictions_orig}")
