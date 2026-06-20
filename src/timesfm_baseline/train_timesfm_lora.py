"""
TimesFM 2.5 LoRA Fine-Tuning Main Training Script

This is the main entry point for fine-tuning TimesFM 2.5 with LoRA on
VN30 Parkinson volatility forecasting.

Usage:
    # Single stock training
    python -m src.timesfm_baseline.train_timesfm_lora \
        --data_path data/raw/prices/ACB_ohlcv.csv \
        --output_dir models/timesfm_lora_ACB

    # Multi-stock training
    python -m src.timesfm_baseline.train_timesfm_lora \
        --data_dir data/raw/prices/ \
        --output_dir models/timesfm_lora_VN30

    # Custom hyperparameters
    python -m src.timesfm_baseline.train_timesfm_lora \
        --data_dir data/raw/prices/ \
        --epochs 20 \
        --batch_size 16 \
        --lr 5e-5 \
        --lora_r 8

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.timesfm_baseline.timesfm_lora_finetuning import TimesFMLoRAFineTuner
from src.timesfm_baseline.volatility_dataset import (
    create_volatility_datasets,
    create_multi_stock_volatility_datasets,
)
from src.common.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune TimesFM 2.5 with LoRA for VN30 Parkinson volatility"
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        help="Path to single stock OHLCV CSV file"
    )
    data_group.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory containing multiple OHLCV CSV files"
    )
    data_group.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    data_group.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    data_group.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_id",
        type=str,
        default="google/timesfm-2.5-200m-transformers",
        help="Hugging Face model ID"
    )
    model_group.add_argument(
        "--context_len",
        type=int,
        default=64,
        help="Context window length (default: 64)"
    )
    model_group.add_argument(
        "--horizon_len",
        type=int,
        default=5,
        help="Forecast horizon length (default: 5 for 5-day forecasts)"
    )

    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=4,
        help="LoRA rank (default: 4 for ~0.6%% trainable params)"
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha scaling factor (default: 8)"
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW (default: 1e-4)"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)"
    )
    train_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)"
    )
    train_group.add_argument(
        "--num_train_samples",
        type=int,
        default=5000,
        help="Number of random training windows (default: 5000)"
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save LoRA adapter and results"
    )
    output_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    output_group.add_argument(
        "--mlflow_experiment",
        type=str,
        default="TimesFM_Volatility_FineTuning",
        help="MLflow experiment name"
    )

    return parser.parse_args()


def load_single_stock(data_path: str) -> pd.DataFrame:
    """Load OHLCV data for a single stock."""
    logger.info(f"Loading OHLCV data from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate required columns
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} rows of OHLCV data")
    return df


def load_multi_stock(data_dir: str) -> dict:
    """Load OHLCV data for multiple stocks."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Find all OHLCV CSV files
    csv_files = list(data_dir.glob("*_ohlcv.csv"))
    if not csv_files:
        raise ValueError(f"No OHLCV CSV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} OHLCV files in {data_dir}")

    # Load each stock
    stocks_data = {}
    for csv_file in csv_files:
        stock_id = csv_file.stem.replace("_ohlcv", "")
        logger.info(f"Loading {stock_id}...")

        df = pd.read_csv(csv_file)

        # Validate required columns
        required_cols = ['date', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Skipping {stock_id}: Missing columns {missing_cols}")
            continue

        stocks_data[stock_id] = df

    logger.info(f"Successfully loaded {len(stocks_data)} stocks")
    return stocks_data


def main():
    """Main training function."""
    args = parse_args()

    # Validate data arguments
    if not args.data_path and not args.data_dir:
        raise ValueError("Must provide either --data_path or --data_dir")
    if args.data_path and args.data_dir:
        raise ValueError("Cannot provide both --data_path and --data_dir")

    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Validate hyperparameters
    if args.context_len <= 0:
        raise ValueError(f"context_len must be positive, got {args.context_len}")
    if args.horizon_len <= 0:
        raise ValueError(f"horizon_len must be positive, got {args.horizon_len}")
    if args.lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {args.lora_r}")
    if args.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {args.lora_alpha}")
    if not (0 <= args.lora_dropout <= 1):
        raise ValueError(f"lora_dropout must be in [0, 1], got {args.lora_dropout}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")
    if args.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")
    if args.max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {args.max_grad_norm}")
    if args.num_train_samples <= 0:
        raise ValueError(f"num_train_samples must be positive, got {args.num_train_samples}")
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting TimesFM 2.5 LoRA fine-tuning at {timestamp}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    if args.data_path:
        # Single stock training
        ohlcv_data = load_single_stock(args.data_path)
        train_ds, val_ds, test_ds = create_volatility_datasets(
            ohlcv_data,
            context_len=args.context_len,
            horizon_len=args.horizon_len,
            num_train_samples=args.num_train_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        # Multi-stock training
        stocks_data = load_multi_stock(args.data_dir)
        train_ds, val_ds, test_ds = create_multi_stock_volatility_datasets(
            stocks_data,
            context_len=args.context_len,
            horizon_len=args.horizon_len,
            num_train_samples=args.num_train_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    # Create fine-tuner
    finetuner = TimesFMLoRAFineTuner(
        model_id=args.model_id,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )

    # Train model
    logger.info("Starting LoRA fine-tuning...")
    best_val_metrics = finetuner.train(
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        output_dir=str(output_dir),
        mlflow_experiment_name=args.mlflow_experiment,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = finetuner.evaluate(test_ds, batch_size=args.batch_size)

    # Save results
    results = {
        "timestamp": timestamp,
        "hyperparameters": {
            "model_id": args.model_id,
            "context_len": args.context_len,
            "horizon_len": args.horizon_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "num_train_samples": args.num_train_samples,
            "seed": args.seed,
        },
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    # Calculate val-test difference
    val_test_diff = {}
    for metric_name in test_metrics:
        val_val = best_val_metrics[metric_name]
        test_val = test_metrics[metric_name]
        if metric_name == "directional_accuracy":
            diff = test_val - val_val
        else:
            # For loss metrics (lower is better), positive diff = degradation
            diff = test_val - val_val
        val_test_diff[f"{metric_name}_diff"] = diff

    results["val_test_diff"] = val_test_diff

    # Save results to JSON
    import json
    results_file = output_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Training Complete - Summary")
    logger.info("="*70)
    logger.info(f"{'Metric':<20} {'Validation':<15} {'Test':<15} {'Difference'}")
    logger.info("-"*70)

    for metric_name in ['mse', 'rmse', 'mae', 'qlike']:
        val_val = best_val_metrics[metric_name]
        test_val = test_metrics[metric_name]
        diff = val_test_diff[f"{metric_name}_diff"]
        logger.info(f"{metric_name.upper():<20} {val_val:<15.6f} {test_val:<15.6f} {diff:+.6f}")

    for metric_name in ['r2', 'directional_accuracy']:
        val_val = best_val_metrics[metric_name]
        test_val = test_metrics[metric_name]
        diff = val_test_diff[f"{metric_name}_diff"]
        logger.info(f"{metric_name.upper():<20} {val_val:<15.6f} {test_val:<15.6f} {diff:+.6f}")

    logger.info("="*70)

    # Success criteria check
    success = (
        test_metrics['rmse'] < 0.20 and
        test_metrics['directional_accuracy'] > 55.0
    )

    if success:
        logger.info("✅ Success criteria met: RMSE < 0.20, Dir Acc > 55%")
    else:
        logger.warning(
            f"⚠️ Success criteria not met: "
            f"RMSE={test_metrics['rmse']:.4f} (need < 0.20), "
            f"Dir Acc={test_metrics['directional_accuracy']:.2f}% (need > 55%)"
        )

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
