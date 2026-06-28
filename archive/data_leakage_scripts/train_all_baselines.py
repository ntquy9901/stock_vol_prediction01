"""
Universal Training Script for All Baseline Models
Supports --data_dir parameter for all models

Usage:
    # Train Enhanced LSTM-HAR with 133 stocks
    python train_all_baselines.py --model enhanced_lstm_har --data_dir data/processed_all

    # Train all baselines with default data (30 stocks)
    python train_all_baselines.py --model all --data_dir data/processed
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def train_har_baseline(data_dir: str):
    """Train HAR-R Linear baseline"""
    print("=" * 80)
    print("TRAINING HAR-R LINEAR BASELINE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    # Import and run
    from src.har_baseline.train import train_har_baseline

    output_dir = f"results/har_baseline_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    train_har_baseline(data_dir, output_dir)

    return output_dir


def train_simple_lstm(data_dir: str):
    """Train Simple LSTM baseline"""
    print("\n" + "=" * 80)
    print("TRAINING SIMPLE LSTM BASELINE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    # Import and run
    from src.lstm_baseline.train_with_validation import train_simple_lstm_with_val

    output_dir = f"results/simple_lstm_val_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    train_simple_lstm_with_val(data_dir, output_dir)

    return output_dir


def train_lstm_har(data_dir: str):
    """Train LSTM-HAR baseline"""
    print("\n" + "=" * 80)
    print("TRAINING LSTM-HAR BASELINE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    # Import and run
    from src.lstm_har_baseline.train_with_validation import train_lstm_har_with_val

    output_dir = f"results/lstm_har_val_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    train_lstm_har_with_val(data_dir, output_dir)

    return output_dir


def train_enhanced_lstm_har(data_dir: str):
    """Train Enhanced LSTM-HAR (best performer)"""
    print("\n" + "=" * 80)
    print("TRAINING ENHANCED LSTM-HAR (BEST PERFORMER)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    # Import and run
    from src.lstm_har_enhanced.train_with_validation import train_enhanced_lstm_har_with_val

    output_dir = f"results/enhanced_lstm_har_val_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    train_enhanced_lstm_har_with_val(data_dir, output_dir)

    return output_dir


def train_all_models(data_dir: str):
    """Train all baseline models sequentially"""
    print("\n" + "=" * 80)
    print("TRAINING ALL BASELINE MODELS")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("Models to train:")
    print("  1. HAR-R Linear")
    print("  2. Simple LSTM")
    print("  3. LSTM-HAR")
    print("  4. Enhanced LSTM-HAR")
    print("=" * 80)

    results = {}

    try:
        print("\n[1/4] Training HAR-R Linear...")
        results['har_linear'] = train_har_baseline(data_dir)

        print("\n[2/4] Training Simple LSTM...")
        results['simple_lstm'] = train_simple_lstm(data_dir)

        print("\n[3/4] Training LSTM-HAR...")
        results['lstm_har'] = train_lstm_har(data_dir)

        print("\n[4/4] Training Enhanced LSTM-HAR...")
        results['enhanced_lstm_har'] = train_enhanced_lstm_har(data_dir)

        print("\n" + "=" * 80)
        print("ALL TRAININGS COMPLETE!")
        print("=" * 80)
        print("\nResults directories:")
        for model, output_dir in results.items():
            print(f"  {model}: {output_dir}")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

    return results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Train baseline models with custom data directory')

    parser.add_argument(
        '--model',
        type=str,
        choices=['har_linear', 'simple_lstm', 'lstm_har', 'enhanced_lstm_har', 'all'],
        default='enhanced_lstm_har',
        help='Model to train (default: enhanced_lstm_har)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Data directory (default: data/processed)'
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data directory: {args.data_dir}")

    # Validate data directory exists
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        print(f"Please ensure processed data exists in: {args.data_dir}")
        print(f"\nAvailable directories:")
        for d in ['data/processed', 'data/processed_all']:
            if os.path.exists(d):
                print(f"  - {d}")
        return

    # Train selected model(s)
    if args.model == 'all':
        train_all_models(args.data_dir)
    elif args.model == 'har_linear':
        train_har_baseline(args.data_dir)
    elif args.model == 'simple_lstm':
        train_simple_lstm(args.data_dir)
    elif args.model == 'lstm_har':
        train_lstm_har(args.data_dir)
    elif args.model == 'enhanced_lstm_har':
        train_enhanced_lstm_har(args.data_dir)


if __name__ == "__main__":
    main()
