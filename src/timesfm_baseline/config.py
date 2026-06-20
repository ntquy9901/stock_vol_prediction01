"""
TimesFM 2.5 LoRA Fine-Tuning Configuration

This module contains all hyperparameters and configuration constants
to avoid magic numbers throughout the codebase.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimesFMLoRAConfig:
    """Configuration for TimesFM 2.5 LoRA fine-tuning."""

    # Model configuration
    model_id: str = "google/timesfm-2.5-200m-transformers"
    context_len: int = 64  # Must be multiple of 32 for TimesFM patch size
    horizon_len: int = 5   # 5-day ahead forecast

    # LoRA configuration
    lora_r: int = 4               # Rank for low-rank adaptation
    lora_alpha: int = 8           # Scaling factor for LoRA
    lora_dropout: float = 0.05    # Dropout rate for LoRA layers
    lora_target_modules: str = "all-linear"  # Apply LoRA to all linear layers

    # Training configuration
    epochs: int = 70               # Maximum training epochs (standardized across all models)
    batch_size: int = 32          # Training batch size
    lr: float = 1e-4              # Learning rate for AdamW
    weight_decay: float = 0.01    # Weight decay for AdamW
    max_grad_norm: float = 1.0     # Max gradient norm for clipping
    patience: int = 15             # Early stopping patience (standardized across all models)

    # Data configuration
    num_train_samples: int = 5000  # Number of random windows per epoch
    train_ratio: float = 0.7       # Training set ratio
    val_ratio: float = 0.15        # Validation set ratio
    test_ratio: float = 0.15       # Test set ratio

    # System configuration
    seed: int = 42                 # Random seed for reproducibility
    device: Optional[str] = None   # Device for training (cuda/cpu). Auto-detected if None

    # Checkpoint configuration
    checkpoint_frequency: int = None  # Checkpoint save interval (None = twice per epoch)

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate positive parameters
        if self.context_len <= 0:
            raise ValueError(f"context_len must be positive, got {self.context_len}")
        if self.horizon_len <= 0:
            raise ValueError(f"horizon_len must be positive, got {self.horizon_len}")
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.num_train_samples <= 0:
            raise ValueError(f"num_train_samples must be positive, got {self.num_train_samples}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")

        # Validate ranges
        if not (0 <= self.lora_dropout <= 1):
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        # Validate split ratios sum to 1.0
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Validate context_len is multiple of 32 (TimesFM requirement)
        if self.context_len % 32 != 0:
            logger.warning(
                f"context_len={self.context_len} is not a multiple of 32 "
                f"(TimesFM patch size)"
            )

    @property
    def checkpoint_interval(self) -> int:
        """Calculate checkpoint save interval (default: twice per epoch)."""
        if self.checkpoint_frequency is not None:
            return self.checkpoint_frequency

        # Calculate actual number of batches per epoch
        # Handle edge case where batch_size > num_train_samples
        if self.batch_size >= self.num_train_samples:
            # One batch per epoch, save once (at end)
            return 1
        else:
            # Multiple batches per epoch, save twice per epoch
            batches_per_epoch = self.num_train_samples // self.batch_size
            if self.num_train_samples % self.batch_size != 0:
                batches_per_epoch += 1  # Partial final batch
            return max(1, batches_per_epoch // 2)

    @property
    def total_trainable_params(self) -> int:
        """
        Estimate trainable parameters with LoRA.

        For TimesFM 2.5 (232M params) with LoRA rank=4 on all-linear layers:
        - Roughly 1.4M trainable params (~0.6%)
        """
        # TimesFM 2.5 has ~232M parameters
        base_params = 232_000_000
        # LoRA adds: rank * (input_dim + output_dim) per linear layer
        # Rough estimate for all-linear layers with rank=4
        lora_params = int(base_params * 0.006)  # ~0.6%
        return lora_params

    @property
    def training_steps_per_epoch(self) -> int:
        """Calculate training steps per epoch."""
        return max(1, self.num_train_samples // self.batch_size)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model_id": self.model_id,
            "context_len": self.context_len,
            "horizon_len": self.horizon_len,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "patience": self.patience,
            "num_train_samples": self.num_train_samples,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }


# Default configuration instance
DEFAULT_CONFIG = TimesFMLoRAConfig()


# Preset configurations for different scenarios
QUICK_TEST_CONFIG = TimesFMLoRAConfig(
    epochs=2,
    batch_size=16,
    num_train_samples=100,
)

PRODUCTION_CONFIG = TimesFMLoRAConfig(
    epochs=70,
    batch_size=32,
    num_train_samples=10000,
)

SMALL_MODEL_CONFIG = TimesFMLoRAConfig(
    lora_r=2,          # Smaller rank for faster training
    lora_alpha=4,
    batch_size=16,
)

LARGE_MODEL_CONFIG = TimesFMLoRAConfig(
    lora_r=8,          # Higher rank for better adaptation
    lora_alpha=16,
    batch_size=64,
)
