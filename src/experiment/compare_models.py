"""
Model Comparison Script - HAR vs LSTM

Compare performance metrics between HAR-R baseline and LSTM models.

Usage:
    python compare_models.py

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import os
import pandas as pd
import glob
from pathlib import Path


def find_latest_results():
    """Find latest HAR and LSTM result directories."""
    project_root = Path(__file__).parent

    # Find latest HAR results
    har_results = glob.glob(str(project_root / "results" / "har_baseline_*"))
    har_results.sort()

    # Find latest LSTM results
    lstm_results = glob.glob(str(project_root / "results" / "simple_lstm_*"))
    lstm_results.sort()

    if not har_results:
        print("[ERROR] No HAR results found!")
        print("   Run: python -m src.har_baseline.train")
        return None, None

    if not lstm_results:
        print("[ERROR] No LSTM results found!")
        print("   Run: python -m src.lstm_baseline.train")
        return None, None

    return har_results[-1], lstm_results[-1]


def load_metrics(result_dir):
    """Load metrics from test_metrics.csv."""
    metrics_file = os.path.join(result_dir, 'test_metrics.csv')

    if not os.path.exists(metrics_file):
        print(f"[ERROR] Metrics file not found: {metrics_file}")
        return None

    df = pd.read_csv(metrics_file)
    return df.iloc[0].to_dict()


def print_comparison(har_metrics, lstm_metrics, har_dir, lstm_dir):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON: HAR-R BASELINE vs LSTM")
    print("=" * 100)

    print("\n[RESULT DIRECTORIES]")
    print(f"  HAR:  {os.path.basename(har_dir)}")
    print(f"  LSTM: {os.path.basename(lstm_dir)}")

    print("\n" + "-" * 100)
    print(f"{'METRIC':<25} {'HAR-R':<20} {'LSTM':<20} {'WINNER':<20} {'IMPROVEMENT'}")
    print("-" * 100)

    # Define metrics (lower is better except Directional_Acc and R2)
    lower_better = ['QLIKE', 'RMSE', 'MAE']
    higher_better = ['Directional_Acc', 'R2']

    for metric in ['QLIKE', 'RMSE', 'MAE', 'R2', 'Directional_Acc']:
        if metric in har_metrics and metric in lstm_metrics:
            har_val = har_metrics[metric]
            lstm_val = lstm_metrics[metric]

            # Determine winner
            if metric in lower_better:
                winner = 'LSTM' if lstm_val < har_val else 'HAR'
                improvement = ((har_val - lstm_val) / har_val * 100) if lstm_val < har_val else ((lstm_val - har_val) / har_val * 100)
            else:
                winner = 'LSTM' if lstm_val > har_val else 'HAR'
                improvement = ((lstm_val - har_val) / har_val * 100) if lstm_val > har_val else ((har_val - lstm_val) / har_val * 100)

            # Format values
            if metric == 'Directional_Acc':
                har_str = f"{har_val:.2f}%"
                lstm_str = f"{lstm_val:.2f}%"
                imp_str = f"{improvement:+.2f}%"
            else:
                har_str = f"{har_val:.6f}"
                lstm_str = f"{lstm_val:.6f}"
                imp_str = f"{improvement:+.2f}%"

            print(f"{metric:<25} {har_str:<20} {lstm_str:<20} {winner:<20} {imp_str}")

    print("-" * 100)

    # Summary
    print("\n[SUMMARY]")

    har_wins = 0
    lstm_wins = 0

    for metric in ['QLIKE', 'RMSE', 'MAE', 'R2', 'Directional_Acc']:
        if metric in har_metrics and metric in lstm_metrics:
            har_val = har_metrics[metric]
            lstm_val = lstm_metrics[metric]

            if metric in lower_better:
                if lstm_val < har_val:
                    lstm_wins += 1
                else:
                    har_wins += 1
            else:
                if lstm_val > har_val:
                    lstm_wins += 1
                else:
                    har_wins += 1

    print(f"  HAR wins:  {har_wins} metrics")
    print(f"  LSTM wins: {lstm_wins} metrics")

    if lstm_wins > har_wins:
        print(f"\n  [WINNER] Overall: LSTM (better on {lstm_wins}/{har_wins + lstm_wins} metrics)")
    elif har_wins > lstm_wins:
        print(f"\n  [WINNER] Overall: HAR (better on {har_wins}/{har_wins + lstm_wins} metrics)")
    else:
        print(f"\n  [TIE] Both models perform equally")

    print("\n" + "=" * 100)


def main():
    """Main comparison function."""
    print("\n[FINDING] Searching for latest results...")

    har_dir, lstm_dir = find_latest_results()

    if har_dir is None or lstm_dir is None:
        print("\n[ERROR] Cannot compare - missing results")
        return

    print(f"[OK] Found HAR results:  {os.path.basename(har_dir)}")
    print(f"[OK] Found LSTM results: {os.path.basename(lstm_dir)}")

    # Load metrics
    print("\n[LOADING] Loading metrics...")
    har_metrics = load_metrics(har_dir)
    lstm_metrics = load_metrics(lstm_dir)

    if har_metrics is None or lstm_metrics is None:
        print("[ERROR] Cannot compare - missing metrics files")
        return

    # Print comparison
    print_comparison(har_metrics, lstm_metrics, har_dir, lstm_dir)

    # Recommendations
    print("\n[RECOMMENDATIONS]")

    # Check if LSTM meets targets
    if lstm_metrics.get('Directional_Acc', 0) > 55 and lstm_metrics.get('QLIKE', 1) < 0.20:
        print("  [SUCCESS] LSTM meets success criteria (QLIKE < 0.20, Dir Acc > 55%)")
        print("     -> Use LSTM for production")
    elif har_metrics.get('Directional_Acc', 0) > 55 and har_metrics.get('QLIKE', 1) < 0.20:
        print("  [SUCCESS] HAR meets success criteria (QLIKE < 0.20, Dir Acc > 55%)")
        print("     -> Use HAR for production (simpler, faster)")
    else:
        print("  [WARNING] Neither model meets success criteria")
        print("     -> Consider: more features, longer training, hyperparameter tuning")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
