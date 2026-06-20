"""
Train models with configuration support
Support multiple data scenarios: VN30, VN100, HNX, Combined

Usage:
    python src/experiment/train_with_config.py --scenario vn30_only --model enhanced_lstm_har
    python src/experiment/train_with_config.py --scenario all_combined --model enhanced_lstm_har
"""

import yaml
import argparse
from pathlib import Path
import sys
import torch
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from common.temporal_split import TemporalSplitter
from common.evaluation import evaluate_predictions


def load_config(config_path: str = None) -> dict:
    """Load training configuration"""
    if config_path is None:
        # Default to relative path from this script location
        script_dir = Path(__file__).parent
        config_path = script_dir / "configs" / "training_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(data_path: str, scenario: dict, config: dict) -> pd.DataFrame:
    """
    Prepare data from specified scenario

    Args:
        data_path: Path to data directory
        scenario: Scenario configuration
        config: Full configuration dict

    Returns:
        Combined DataFrame with all stocks data
    """
    data_path = Path(data_path)

    # Check if we need to combine from multiple sources
    if "combine_from" in scenario:
        print(f"Combining data from {len(scenario['combine_from'])} sources...")
        all_data = []

        for source in scenario['combine_from']:
            source_path = Path(source)
            csv_files = list(source_path.glob("*_ohlcv.csv"))
            print(f"  Found {len(csv_files)} files in {source}")

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                symbol = csv_file.stem.replace("_ohlcv", "")
                df['symbol'] = symbol
                all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined data: {len(combined_df)} rows from {len(all_data)} stocks")
        return combined_df

    else:
        # Single source
        csv_files = list(data_path.glob("*_ohlcv.csv"))
        print(f"Found {len(csv_files)} files in {data_path}")

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            symbol = csv_file.stem.replace("_ohlcv", "")
            df['symbol'] = symbol
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total data: {len(combined_df)} rows from {len(all_data)} stocks")
        return combined_df


def train_model(
    data: pd.DataFrame,
    scenario: dict,
    config: dict,
    model_type: str = "enhanced_lstm_har"
):
    """
    Train model with specified configuration

    Args:
        data: Combined stock data
        scenario: Scenario configuration
        config: Full configuration
        model_type: Type of model to train
    """
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_type.upper()}")
    print(f"Scenario: {scenario['name']}")
    print("=" * 60)

    # Get training parameters
    train_params = config['training']

    # Create temporal split (MANDATORY - prevents data leakage)
    splitter = TemporalSplitter(
        train_ratio=train_params['train_ratio'],
        val_ratio=train_params['val_ratio'],
        test_ratio=train_params['test_ratio']
    )

    # TODO: Implement actual model training based on model_type
    # This is a placeholder for the actual training logic

    print("\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {train_params['num_epochs']}")
    print(f"  Batch size: {train_params['batch_size']}")
    print(f"  Learning rate: {train_params['learning_rate']}")
    print(f"  Device: {train_params['device']}")

    print("\nData split:")
    print(f"  Train: {train_params['train_ratio']*100}%")
    print(f"  Validation: {train_params['val_ratio']*100}%")
    print(f"  Test: {train_params['test_ratio']*100}%")

    print("\nFeatures:")
    features = config['features']
    print(f"  HAR: {features['use_har']}")
    print(f"  Raw volatility: {features['use_raw_volatility']}")

    print("\n[SUCCESS] Configuration validated!")
    print("[INFO] Actual training not yet implemented - use existing train scripts")

    return None


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Train with configuration")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., vn30_only, vn100_only, all_combined)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="enhanced_lstm_har",
        help="Model type (e.g., enhanced_lstm_har, lstm_gat)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: src/experiment/configs/training_config.yaml)"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Get scenario
    if args.scenario not in config['data_scenarios']:
        print(f"[ERROR] Scenario '{args.scenario}' not found in config")
        print(f"Available scenarios: {list(config['data_scenarios'].keys())}")
        return

    scenario = config['data_scenarios'][args.scenario]

    # Prepare data
    print(f"\nPreparing data for scenario: {scenario['name']}")
    data = prepare_data(scenario['data_path'], scenario, config)

    if data is None or data.empty:
        print("[ERROR] No data available for this scenario")
        return

    # Train model
    print("\nTraining model...")
    results = train_model(data, scenario, config, args.model)

    if results:
        print("\n[SUCCESS] Training complete!")
    else:
        print("\n[WARNING] Training configured but not yet implemented")
        print("   Please use existing train scripts:")

        if args.scenario == "vn30_only" and args.model == "enhanced_lstm_har":
            print("   python src/lstm_har_enhanced/train_with_validation.py")


if __name__ == "__main__":
    main()
