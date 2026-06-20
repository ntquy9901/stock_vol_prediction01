"""
TimesFM 2.5 Evaluation and Comparison Module

This module provides comprehensive evaluation functions for comparing:
- Zero-shot TimesFM 2.5 (base model)
- LoRA fine-tuned TimesFM 2.5
- Baseline models (HAR-R, LSTM)

All models are evaluated on the same 6 mandatory metrics:
MSE, RMSE, MAE, R², QLIKE, Directional Accuracy

Usage:
    # Compare zero-shot vs fine-tuned
    python -m src.timesfm_baseline.evaluate_timesfm \
        --adapter_path models/timesfm_lora_VN30 \
        --data_dir data/raw/prices/

    # Full comparison with baselines
    python -m src.timesfm_baseline.evaluate_timesfm \
        --adapter_path models/timesfm_lora_VN30 \
        --data_dir data/raw/prices/ \
        --compare_baselines

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.timesfm_baseline.timesfm_lora_finetuning import compare_zero_shot_vs_finetuned
from src.timesfm_baseline.volatility_dataset import create_multi_stock_volatility_datasets
from src.common.evaluation import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TimesFM 2.5 fine-tuning and compare with baselines"
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing OHLCV CSV files"
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=64,
        help="Context window length (default: 64)"
    )
    parser.add_argument(
        "--horizon_len",
        type=int,
        default=5,
        help="Forecast horizon length (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: adapter_path/evaluation)"
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare with HAR-R and LSTM baselines"
    )
    parser.add_argument(
        "--baseline_results_dir",
        type=str,
        default="results",
        help="Directory containing baseline model results"
    )

    return parser.parse_args()


def load_baseline_results(results_dir: str) -> dict:
    """
    Load baseline model results from directory.

    Args:
        results_dir: Directory containing training_results.json files

    Returns:
        Dictionary mapping model names to their test metrics
    """
    results_dir = Path(results_dir)
    baseline_metrics = {}

    # Find all training_results.json files
    json_files = list(results_dir.rglob("training_results.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                results = json.load(f)

            # Extract model name from path
            model_name = json_file.parent.parent.name
            test_metrics = results.get("test_metrics", {})

            if test_metrics:
                baseline_metrics[model_name] = test_metrics
                logger.info(f"Loaded baseline results: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return baseline_metrics


def create_comparison_table(
    zero_shot_metrics: dict,
    finetuned_metrics: dict,
    baseline_metrics: dict = None,
) -> pd.DataFrame:
    """
    Create comparison table for all models and metrics.

    Args:
        zero_shot_metrics: Metrics from zero-shot TimesFM 2.5
        finetuned_metrics: Metrics from LoRA fine-tuned TimesFM 2.5
        baseline_metrics: Optional dict of baseline model metrics

    Returns:
        DataFrame with comparison table
    """
    # Collect all models
    all_models = {
        "TimesFM-ZeroShot": zero_shot_metrics,
        "TimesFM-LoRA": finetuned_metrics,
    }

    if baseline_metrics:
        all_models.update(baseline_metrics)

    # Define all metrics
    metric_names = ["mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]
    metric_labels = ["MSE", "RMSE", "MAE", "R²", "QLIKE", "Dir Acc (%)"]

    # Create comparison table
    comparison_data = []

    for model_name, metrics in all_models.items():
        row = {"Model": model_name}
        for metric_name, label in zip(metric_names, metric_labels):
            row[label] = metrics.get(metric_name, float("nan"))
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Reorder columns
    cols = ["Model"] + metric_labels
    df = df[cols]

    return df


def calculate_improvements(
    finetuned_metrics: dict,
    zero_shot_metrics: dict,
    baseline_metrics: dict = None,
) -> pd.DataFrame:
    """
    Calculate percentage improvements of LoRA over zero-shot and baselines.

    Args:
        finetuned_metrics: Metrics from LoRA fine-tuned TimesFM 2.5
        zero_shot_metrics: Metrics from zero-shot TimesFM 2.5
        baseline_metrics: Optional dict of baseline model metrics

    Returns:
        DataFrame with improvements
    """
    metric_names = ["mse", "rmse", "mae", "qlike"]
    benefit_metrics = ["r2", "directional_accuracy"]

    improvements = []

    # Compare vs Zero-Shot
    vs_zeroshot = {"Comparison": "vs Zero-Shot"}
    for metric_name in metric_names:
        zs = zero_shot_metrics.get(metric_name, 0)
        ft = finetuned_metrics.get(metric_name, 0)
        improvement = (zs - ft) / zs * 100 if zs > 0 else 0
        vs_zeroshot[metric_name.upper()] = improvement

    for metric_name in benefit_metrics:
        zs = zero_shot_metrics.get(metric_name, 0)
        ft = finetuned_metrics.get(metric_name, 0)
        improvement = (ft - zs) / abs(zs) * 100 if zs != 0 else 0
        vs_zeroshot[metric_name.upper()] = improvement

    improvements.append(vs_zeroshot)

    # Compare vs baselines
    if baseline_metrics:
        for baseline_name, baseline_metrics_dict in baseline_metrics.items():
            comparison = {"Comparison": f"vs {baseline_name}"}

            for metric_name in metric_names:
                bl = baseline_metrics_dict.get(metric_name, 0)
                ft = finetuned_metrics.get(metric_name, 0)
                improvement = (bl - ft) / bl * 100 if bl > 0 else 0
                comparison[metric_name.upper()] = improvement

            for metric_name in benefit_metrics:
                bl = baseline_metrics_dict.get(metric_name, 0)
                ft = finetuned_metrics.get(metric_name, 0)
                improvement = (ft - bl) / abs(bl) * 100 if bl != 0 else 0
                comparison[metric_name.upper()] = improvement

            improvements.append(comparison)

    return pd.DataFrame(improvements)


def print_comparison_report(
    comparison_table: pd.DataFrame,
    improvements_table: pd.DataFrame,
    finetuned_metrics: dict,
):
    """
    Print comprehensive comparison report.

    Args:
        comparison_table: DataFrame with all models and metrics
        improvements_table: DataFrame with percentage improvements
        finetuned_metrics: Test metrics from fine-tuned model
    """
    logger.info("\n" + "="*90)
    logger.info("TimesFM 2.5 LoRA Fine-Tuning: Model Comparison Report")
    logger.info("="*90 + "\n")

    # Print comparison table
    logger.info("All Models × All Metrics:")
    logger.info("-"*90)
    logger.info(comparison_table.to_string(index=False))
    logger.info("\n")

    # Print improvements
    logger.info("LoRA Fine-Tuned Improvements:")
    logger.info("-"*90)
    logger.info(improvements_table.to_string(index=False))
    logger.info("\n")

    # Success criteria check
    logger.info("="*90)
    rmse = finetuned_metrics.get("rmse", float("inf"))
    dir_acc = finetuned_metrics.get("directional_accuracy", 0)

    logger.info("Success Criteria Check:")
    logger.info(f"  RMSE < 0.20:           {'✅ PASS' if rmse < 0.20 else '❌ FAIL'} (actual: {rmse:.6f})")
    logger.info(f"  Dir Acc > 55%:         {'✅ PASS' if dir_acc > 55 else '❌ FAIL'} (actual: {dir_acc:.2f}%)")

    if rmse < 0.20 and dir_acc > 55:
        logger.info("\n  🎉 All success criteria met!")
    else:
        logger.info("\n  ⚠️ Some success criteria not met - consider Phase 2 enhancements")

    logger.info("="*90 + "\n")


def save_evaluation_results(
    comparison_table: pd.DataFrame,
    improvements_table: pd.DataFrame,
    zero_shot_metrics: dict,
    finetuned_metrics: dict,
    baseline_metrics: dict = None,
    output_dir: str = None,
):
    """
    Save evaluation results to files.

    Args:
        comparison_table: DataFrame with all models and metrics
        improvements_table: DataFrame with percentage improvements
        zero_shot_metrics: Metrics from zero-shot TimesFM 2.5
        finetuned_metrics: Metrics from LoRA fine-tuned TimesFM 2.5
        baseline_metrics: Optional dict of baseline model metrics
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save comparison table
    comparison_file = output_dir / f"comparison_table_{timestamp}.csv"
    comparison_table.to_csv(comparison_file, index=False)
    logger.info(f"Comparison table saved: {comparison_file}")

    # Save improvements table
    improvements_file = output_dir / f"improvements_{timestamp}.csv"
    improvements_table.to_csv(improvements_file, index=False)
    logger.info(f"Improvements table saved: {improvements_file}")

    # Save detailed JSON results
    results = {
        "timestamp": timestamp,
        "zero_shot_metrics": zero_shot_metrics,
        "finetuned_metrics": finetuned_metrics,
    }

    if baseline_metrics:
        results["baseline_metrics"] = baseline_metrics

    json_file = output_dir / f"evaluation_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved: {json_file}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.adapter_path) / "evaluation"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting TimesFM 2.5 evaluation at {timestamp}")
    logger.info(f"Output directory: {output_dir}\n")

    # Load baseline results if requested
    baseline_metrics = None
    if args.compare_baselines:
        logger.info("Loading baseline model results...")
        baseline_metrics = load_baseline_results(args.baseline_results_dir)
        if not baseline_metrics:
            logger.warning("No baseline results found - proceeding without baseline comparison")
        else:
            logger.info(f"Loaded {len(baseline_metrics)} baseline models\n")

    # Load test data
    logger.info("Loading test data...")
    data_dir = Path(args.data_dir)
    csv_files = list(data_dir.glob("*_ohlcv.csv"))

    if not csv_files:
        raise ValueError(f"No OHLCV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} OHLCV files")

    # Load each stock
    stocks_data = {}
    for csv_file in csv_files:
        stock_id = csv_file.stem.replace("_ohlcv", "")
        df = pd.read_csv(csv_file)

        required_cols = ['date', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Skipping {stock_id}: Missing columns {missing_cols}")
            continue

        stocks_data[stock_id] = df

    logger.info(f"Loaded {len(stocks_data)} stocks\n")

    # Create test dataset
    logger.info("Creating test dataset...")
    _, _, test_dataset = create_multi_stock_volatility_datasets(
        stocks_data,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        num_train_samples=100,  # Minimal, we only need test
        seed=42,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples\n")

    # Compare zero-shot vs fine-tuned
    logger.info("Comparing Zero-Shot vs LoRA Fine-Tuned TimesFM 2.5...")
    zero_shot_metrics, finetuned_metrics = compare_zero_shot_vs_finetuned(
        test_dataset,
        adapter_path=args.adapter_path,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
    )

    # Create comparison tables
    comparison_table = create_comparison_table(
        zero_shot_metrics,
        finetuned_metrics,
        baseline_metrics,
    )

    improvements_table = calculate_improvements(
        finetuned_metrics,
        zero_shot_metrics,
        baseline_metrics,
    )

    # Print report
    print_comparison_report(
        comparison_table,
        improvements_table,
        finetuned_metrics,
    )

    # Save results
    save_evaluation_results(
        comparison_table,
        improvements_table,
        zero_shot_metrics,
        finetuned_metrics,
        baseline_metrics,
        output_dir,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
