"""
TimesFM 2.5 LoRA Fine-Tuning Core Implementation

This module provides parameter-efficient fine-tuning of TimesFM 2.5 using LoRA
(Low-Rank Adaptation) for Parkinson volatility forecasting.

Based on the fine-tuning workflow from:
https://github.com/UberGuidoZ/timesfm-google/tree/master/timesfm-forecasting/examples/finetuning

Key Features:
- LoRA Adapters: Only ~0.6% trainable parameters (1.4M out of 232M)
- AdamW Optimizer: lr=1e-4, weight_decay=0.01
- Cosine Annealing: Standard Transformers learning rate schedule
- Gradient Clipping: max_norm=1.0 for training stability
- MLflow Tracking: Automatic experiment tracking
- 6-Metric Evaluation: MSE, RMSE, MAE, R², QLIKE, Dir Acc
- Gradient Norm Monitoring: Track gradient norms for stability
- Early Stopping: Prevent overfitting
- Checkpoint Saving: Resume from crashes

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import TimesFm2_5ModelForPrediction

from src.common.evaluation import evaluate_predictions
from .volatility_dataset import TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset

logger = logging.getLogger(__name__)


class TimesFMLoRAFineTuner:
    """
    TimesFM 2.5 LoRA fine-tuning for volatility forecasting.

    This class handles the complete fine-tuning pipeline:
    1. Load TimesFM 2.5 base model from Hugging Face
    2. Apply LoRA adapters (parameter-efficient fine-tuning)
    3. Train with AdamW optimizer and cosine annealing
    4. Validate with 6 mandatory metrics
    5. Log everything to MLflow
    6. Save best LoRA adapter

    Attributes:
        model_id (str): Hugging Face model ID
        context_len (int): Context window length
        horizon_len (int): Forecast horizon length
        lora_config (LoraConfig): LoRA adapter configuration
        device (str): Device for training (cuda/cpu)
        model: TimesFM 2.5 model with LoRA adapters

    Example:
        >>> finetuner = TimesFMLoRAFineTuner(
        ...     context_len=64,
        ...     horizon_len=5,
        ...     lora_r=4,
        ...     lora_alpha=8
        ... )
        >>> finetuner.train(train_dataset, val_dataset, epochs=10)
        >>> metrics = finetuner.evaluate(test_dataset)
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-transformers",
        context_len: int = 64,
        horizon_len: int = 5,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize TimesFM 2.5 LoRA fine-tuner with validation.

        Args:
            model_id: Hugging Face model ID
            context_len: Context window length (must be positive)
            horizon_len: Forecast horizon length (must be positive)
            lora_r: LoRA rank (must be positive integer)
            lora_alpha: LoRA alpha scaling factor (must be positive)
            lora_dropout: LoRA dropout rate (must be 0-1)
            device: Device for training (cuda/cpu). Auto-detected if None
            seed: Random seed for reproducibility

        Raises:
            ValueError: If hyperparameters are invalid
        """
        # Validate hyperparameters
        if context_len <= 0:
            raise ValueError(f"context_len must be positive, got {context_len}")
        if horizon_len <= 0:
            raise ValueError(f"horizon_len must be positive, got {horizon_len}")
        if lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {lora_r}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if not (0 <= lora_dropout <= 1):
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        # Validate context_len is multiple of 32 (TimesFM requirement)
        if context_len % 32 != 0:
            logger.warning(f"context_len={context_len} is not a multiple of 32 (TimesFM patch size)")

        self.model_id = model_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.seed = seed

        # Set device with better error handling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Validate CUDA availability if requested
        if self.device == "cuda":
            try:
                # Test CUDA with a small tensor
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                logger.info("CUDA device validated successfully")
            except Exception as e:
                logger.warning(f"CUDA device validation failed: {e}. Falling back to CPU.")
                self.device = "cpu"

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model and LoRA
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # Initialize _actual_context_len to prevent AttributeError if train() called before load_model()
        self._actual_context_len = context_len  # Will be adjusted in load_model() if needed
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=lora_dropout,
            bias="none",
        )

        logger.info(
            f"TimesFM LoRA FineTuner initialized: "
            f"context_len={context_len}, horizon_len={horizon_len}, "
            f"lora_r={lora_r}, lora_alpha={lora_alpha}"
        )

    def load_model(self) -> None:
        """
        Load TimesFM 2.5 base model and apply LoRA adapters.

        This method:
        1. Loads TimesFM 2.5 from Hugging Face
        2. Applies LoRA adapters to all linear layers
        3. Prints trainable parameter count
        """
        logger.info(f"Loading TimesFM 2.5 model: {self.model_id}")

        # Load base model
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Apply LoRA adapters
        logger.info("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        # Verify context length
        actual_context_len = min(self.context_len, self.model.config.context_length)
        if actual_context_len != self.context_len:
            logger.warning(
                f"Context length adjusted: {self.context_len} → {actual_context_len} "
                f"(model max: {self.model.config.context_length})"
            )
            # Store actual context length separately to avoid confusion
            self._actual_context_len = actual_context_len
        else:
            self._actual_context_len = self.context_len

        logger.info("Model loaded and LoRA adapters applied successfully")

    def train(
        self,
        train_dataset: TimeSeriesRandomWindowDataset,
        val_dataset: TimeSeriesLastWindowDataset,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        output_dir: str = "timesfm2_5-volatility-lora",
        patience: int = 15,  # Early stopping patience (configurable)
        mlflow_experiment_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train TimesFM 2.5 with LoRA fine-tuning.

        Args:
            train_dataset: Training dataset with random window sampling
            val_dataset: Validation dataset (last window only)
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate for AdamW
            weight_decay: Weight decay for AdamW
            max_grad_norm: Max gradient norm for clipping
            output_dir: Directory to save LoRA adapter
            patience: Early stopping patience (default: 15)
            mlflow_experiment_name: MLflow experiment name

        Returns:
            Dictionary of best validation metrics

        Training process:
        1. Setup optimizer (AdamW) and scheduler (cosine annealing)
        2. For each epoch:
           - Train on random windows
           - Validate on last windows
           - Compute 6 metrics (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
           - Log to MLflow
        3. Save best adapter based on validation loss
        """
        if self.model is None:
            self.load_model()

        # Validate training parameters
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if not output_dir or not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"output_dir must be a non-empty string, got: {repr(output_dir)}")

        # Validate dataset size against context + horizon requirements
        min_required_len = self.context_len + self.horizon_len
        if len(train_dataset.series_list) > 0:
            for i, series in enumerate(train_dataset.series_list):
                if len(series) < min_required_len:
                    raise ValueError(
                        f"Training dataset series {i} has length {len(series)} "
                        f"but requires at least {min_required_len} points "
                        f"(context_len={self.context_len} + horizon_len={self.horizon_len}). "
                        f"Use shorter context/horizon or filter short series."
                    )

        # Validate dataset types
        if not isinstance(train_dataset, TimeSeriesRandomWindowDataset):
            raise TypeError(
                f"train_dataset must be TimeSeriesRandomWindowDataset, "
                f"got {type(train_dataset).__name__}"
            )
        if not isinstance(val_dataset, TimeSeriesLastWindowDataset):
            raise TypeError(
                f"val_dataset must be TimeSeriesLastWindowDataset, "
                f"got {type(val_dataset).__name__}"
            )

        # Setup MLflow
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)
            run_name = f"timesfm-lora-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)

            # Log hyperparameters
            mlflow.log_params({
                "model_id": self.model_id,
                "context_len": self.context_len,
                "horizon_len": self.horizon_len,
                "lora_r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "seed": self.seed,
            })

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            pin_memory=True,  # For faster GPU data transfer
            num_workers=4      # For parallel data loading
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            pin_memory=True,
            num_workers=4
        )

        # Warn about data loss from drop_last
        total_train_samples = len(train_dataset)
        train_batches = len(train_loader)
        samples_used = train_batches * batch_size
        samples_dropped = total_train_samples - samples_used
        if samples_dropped > 0:
            drop_pct = samples_dropped / total_train_samples * 100
            logger.warning(
                f"Drop last enabled: losing {samples_dropped} training samples ({drop_pct:.1f}%). "
                f"Consider reducing batch_size or disabling drop_last."
            )

        logger.info(
            f"Data loaders created: "
            f"Train={len(train_dataset)} samples ({len(train_loader)} batches), "
            f"Val={len(val_dataset)} samples ({len(val_loader)} batches)"
        )

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_loader)
        )

        logger.info(
            f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay}), "
            f"Scheduler: CosineAnnealing(T_max={epochs * len(train_loader)})"
        )

        # Training loop
        best_val_loss = float("inf")
        best_val_metrics = {}
        training_losses = []
        validation_losses = []
        patience_counter = 0
        checkpoint_interval = max(1, len(train_loader) // 2)  # Save checkpoint twice per epoch

        try:
            for epoch in range(1, epochs + 1):
                # Training phase
                self.model.train()
                epoch_loss = 0.0
                n_batches = 0
                epoch_grad_norm = 0.0

                for batch_idx, (context, target_vals) in enumerate(train_loader):
                    # Save checkpoint at START of batch (before work) to recover from crashes
                    if batch_idx > 0 and batch_idx % checkpoint_interval == 0:
                        try:
                            checkpoint_path = Path(output_dir) / f"checkpoint_epoch{epoch}_batch{batch_idx}"
                            checkpoint_path.mkdir(parents=True, exist_ok=True)
                            self.model.save_pretrained(checkpoint_path)
                            logger.debug(f"  Checkpoint saved → {checkpoint_path}")
                        except (OSError, IOError) as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                            logger.warning("Continuing training without checkpoint save")

                    # Async CPU→GPU transfer (non-blocking) - reduces CPU memory pressure
                    context = context.to(self.device, non_blocking=True)
                    target_vals = target_vals.to(self.device, non_blocking=True)

                    # Forward pass
                    outputs = self.model(
                        past_values=context,
                        future_values=target_vals,
                        forecast_context_len=self._actual_context_len,
                    )
                    loss = outputs.loss

                    # Backward pass with gradient norm monitoring
                    loss.backward()

                    # Compute gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_grad_norm
                    )
                    epoch_grad_norm += grad_norm.item()

                    # Step optimizer with explicit gradient cleanup
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Memory leak fix
                    self.scheduler.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                # Check training loop processed data
                if n_batches == 0:
                    raise RuntimeError(
                        f"Training loop processed no batches in epoch {epoch}. "
                        f"Check train_dataset is not empty and batch_size is appropriate."
                    )

                avg_train_loss = epoch_loss / n_batches
                avg_grad_norm = epoch_grad_norm / n_batches
                training_losses.append(avg_train_loss)

                # Validation phase
                val_metrics, avg_val_loss = self._validate(val_loader)
                validation_losses.append(avg_val_loss)

                logger.info(
                    f"Epoch {epoch}/{epochs} ({n_batches} steps) — "
                    f"train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}, "
                    f"grad norm: {avg_grad_norm:.4f}"
                )
                logger.info(
                    f"  Val Metrics — MSE: {val_metrics['mse']:.6f}, "
                    f"RMSE: {val_metrics['rmse']:.6f}, "
                    f"MAE: {val_metrics['mae']:.6f}, "
                    f"R²: {val_metrics['r2']:.4f}, "
                    f"QLIKE: {val_metrics['qlike']:.6f}, "
                    f"Dir Acc: {val_metrics['directional_accuracy']:.2f}%"
                )

                # Log to MLflow with per-epoch exception handling
                if mlflow_experiment_name:
                    try:
                        mlflow.log_metrics({
                            "epoch": epoch,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "grad_norm": avg_grad_norm,
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                        }, step=epoch)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics for epoch {epoch} to MLflow: {e}")
                        # Continue training - don't crash on MLflow failure

                # Save best adapter and check for early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_metrics = val_metrics
                    patience_counter = 0  # Reset patience counter

                    # Use timestamp to prevent overwrites in concurrent runs
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(output_dir) / f"best_model_{timestamp}"
                    output_path.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(output_path)
                    logger.info(f"  ✓ Saved best LoRA adapter → {output_path}")

                    # Save training info metadata
                    import json
                    metadata = {
                        "timestamp": timestamp,
                        "best_val_loss": float(best_val_loss),
                        "best_epoch": epoch,
                        "best_metrics": {k: float(v) for k, v in best_val_metrics.items()},
                        "hyperparameters": {
                            "model_id": self.model_id,
                            "context_len": self.context_len,
                            "horizon_len": self.horizon_len,
                            "lora_r": self.lora_config.r,
                            "lora_alpha": self.lora_config.lora_alpha,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                        }
                    }
                    metadata_path = output_path / "training_metadata.json"
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    logger.debug(f"  Training metadata saved → {metadata_path}")

                    # Log artifact directory to MLflow (skip model logging to avoid pt2 format issues)
                    if mlflow_experiment_name:
                        mlflow.log_artifacts(output_path, artifact_path="lora_adapter")
                else:
                    patience_counter += 1
                    logger.info(f"  No improvement for {patience_counter} epoch(s) (patience={patience})")

                    # Early stopping check
                    if patience_counter >= patience:
                        logger.info(
                            f"  Early stopping triggered after {epoch} epochs "
                            f"(no improvement for {patience} epochs)"
                        )
                        break

        finally:
            # Ensure MLflow run ends even on exception
            if mlflow_experiment_name:
                try:
                    # Plot learning curves
                    self._plot_learning_curves(
                        training_losses, validation_losses, output_dir
                    )

                    # Log final results
                    mlflow.log_metrics({f"best_val_{k}": v for k, v in best_val_metrics.items()})
                except Exception as e:
                    logger.warning(f"Failed to log training artifacts to MLflow: {e}")
                finally:
                    try:
                        mlflow.end_run()
                    except Exception as e:
                        logger.warning(f"Failed to end MLflow run: {e}")

        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

        # Check we have validation metrics before logging
        if not best_val_metrics:
            logger.warning("No validation metrics collected (training may have failed to complete validation)")
        else:
            logger.info(f"Best val metrics: {best_val_metrics}")

        return best_val_metrics

    def _validate(
        self, val_loader: DataLoader
    ) -> Tuple[Dict[str, float], float]:
        """
        Validate model and compute 6 mandatory metrics.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (metrics_dict, avg_loss)
        """
        self.model.eval()
        val_loss = 0.0
        val_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(self.device)
                target_vals = target_vals.to(self.device)

                # Forward pass
                outputs = self.model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=self._actual_context_len,
                )

                val_loss += outputs.loss.item()
                val_batches += 1

                # Collect predictions and targets
                predictions = outputs.mean_predictions[:, :self.horizon_len]
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_vals.cpu().numpy())

        # Check validation set is not empty
        if not all_predictions or not all_targets:
            raise ValueError(
                f"Validation set is empty or produced no predictions. "
                f"val_batches={val_batches}, predictions={len(all_predictions)}, "
                f"targets={len(all_targets)}"
            )

        # Flatten predictions and targets
        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Compute 6 mandatory metrics
        metrics = evaluate_predictions(all_targets, all_predictions)
        avg_val_loss = val_loss / val_batches  # val_batches >= 1 guaranteed by empty check above

        return metrics, avg_val_loss

    def _plot_learning_curves(
        self,
        train_losses: list,
        val_losses: list,
        output_dir: str,
    ) -> None:
        """
        Plot training and validation learning curves.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            output_dir: Directory to save plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", marker="o")
        plt.plot(val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("TimesFM 2.5 LoRA Fine-Tuning: Learning Curves")
        plt.legend()
        plt.grid(True)

        # Use timestamp to prevent race condition in concurrent runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"learning_curves_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Learning curves saved → {output_path}")

    def evaluate(
        self,
        test_dataset: TimeSeriesLastWindowDataset,
        batch_size: int = 32,
        mlflow_run_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model on test dataset.

        Args:
            test_dataset: Test dataset (last window only)
            batch_size: Evaluation batch size
            mlflow_run_id: MLflow run ID for logging

        Returns:
            Dictionary of 6 mandatory metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Validate dataset type
        if not isinstance(test_dataset, TimeSeriesLastWindowDataset):
            raise TypeError(
                f"test_dataset must be TimeSeriesLastWindowDataset, "
                f"got {type(test_dataset).__name__}"
            )

        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        metrics, _ = self._validate(test_loader)

        logger.info(
            f"Test Results — "
            f"MSE: {metrics['mse']:.6f}, "
            f"RMSE: {metrics['rmse']:.6f}, "
            f"MAE: {metrics['mae']:.6f}, "
            f"R²: {metrics['r2']:.4f}, "
            f"QLIKE: {metrics['qlike']:.6f}, "
            f"Dir Acc: {metrics['directional_accuracy']:.2f}%"
        )

        # Log to MLflow if run ID provided
        if mlflow_run_id:
            try:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            except Exception as e:
                logger.warning(f"Failed to log test metrics to MLflow run {mlflow_run_id}: {e}")

        return metrics


def compare_zero_shot_vs_finetuned(
    test_dataset: TimeSeriesLastWindowDataset,
    adapter_path: str,
    context_len: int = 64,
    horizon_len: int = 5,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compare zero-shot vs LoRA fine-tuned TimesFM 2.5 on test dataset.

    Args:
        test_dataset: Test dataset
        adapter_path: Path to LoRA adapter
        context_len: Context window length
        horizon_len: Forecast horizon length
        device: Device for evaluation

    Returns:
        Tuple of (zero_shot_metrics, finetuned_metrics)

    Raises:
        TypeError: If test_dataset is not TimeSeriesLastWindowDataset
        FileNotFoundError: If adapter_path does not exist
    """
    from peft import PeftModel

    # Validate inputs
    if not isinstance(test_dataset, TimeSeriesLastWindowDataset):
        raise TypeError(
            f"test_dataset must be TimeSeriesLastWindowDataset, "
            f"got {type(test_dataset).__name__}"
        )

    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(
            f"LoRA adapter path does not exist: {adapter_path}"
        )
    if not adapter_path_obj.is_dir():
        raise ValueError(
            f"LoRA adapter path must be a directory, got: {adapter_path}"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load zero-shot base model
    logger.info("Loading zero-shot base model...")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    # Load fine-tuned model
    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    finetuned_model.eval()

    # Evaluate both models
    test_loader = DataLoader(test_dataset, batch_size=32)

    def evaluate_model(model, test_loader):
        """Helper to evaluate a single model."""
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for context, target_vals in test_loader:
                context = context.to(device)
                target_vals = target_vals.to(device)

                outputs = model(past_values=context)
                predictions = outputs.mean_predictions[:, :horizon_len]

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_vals.cpu().numpy())

        all_predictions = np.concatenate(all_predictions).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        return evaluate_predictions(all_targets, all_predictions)

    # Evaluate zero-shot model first, then clean up
    zero_shot_metrics = evaluate_model(base_model, test_loader)

    # Delete base model to free memory before loading fine-tuned model for evaluation
    del base_model
    torch.cuda.empty_cache() if device == "cuda" else None
    logger.debug("Base model deleted from memory after zero-shot evaluation")

    # Evaluate fine-tuned model
    finetuned_metrics = evaluate_model(finetuned_model, test_loader)

    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("Zero-Shot vs LoRA Fine-Tuned Comparison")
    logger.info("="*70)
    logger.info(f"{'Metric':<15} {'Zero-Shot':<15} {'LoRA Fine-Tuned':<20} {'Improvement'}")
    logger.info("-"*70)

    for metric_name in ['mse', 'rmse', 'mae', 'qlike']:
        zs = zero_shot_metrics[metric_name]
        ft = finetuned_metrics[metric_name]
        improvement = (zs - ft) / zs * 100 if zs > 0 else 0
        logger.info(f"{metric_name.upper():<15} {zs:<15.6f} {ft:<20.6f} {improvement:+.2f}%")

    for metric_name in ['r2', 'directional_accuracy']:
        zs = zero_shot_metrics[metric_name]
        ft = finetuned_metrics[metric_name]
        improvement = (ft - zs) / abs(zs) * 100 if zs != 0 else 0
        logger.info(f"{metric_name.upper():<15} {zs:<15.6f} {ft:<20.6f} {improvement:+.2f}%")

    logger.info("="*70 + "\n")

    return zero_shot_metrics, finetuned_metrics
