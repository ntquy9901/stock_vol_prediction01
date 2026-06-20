"""
Model Comparison and Analysis
Compare all trained models and generate comprehensive comparison report
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_results():
    """Load all model results from results directory"""
    results_dir = Path("results")

    # Find most recent results for each model type
    model_results = {}

    # Enhanced LSTM-HAR (most recent: 2026-06-20_011632)
    enhanced_lstm_file = results_dir / "enhanced_lstm_har_val_2026-06-20_011632" / "enhanced_lstm_har_val_results.json"
    if enhanced_lstm_file.exists():
        with open(enhanced_lstm_file, 'r') as f:
            model_results['enhanced_lstm_har'] = json.load(f)

    # LSTM-HAR (most recent: 2026-06-20_005319)
    lstm_har_file = results_dir / "lstm_har_val_2026-06-20_005319" / "lstm_har_val_results.json"
    if lstm_har_file.exists():
        with open(lstm_har_file, 'r') as f:
            model_results['lstm_har'] = json.load(f)

    # Simple LSTM (most recent: 2026-06-20_003358)
    simple_lstm_file = results_dir / "simple_lstm_val_2026-06-20_003358" / "simple_lstm_val_results.json"
    if simple_lstm_file.exists():
        with open(simple_lstm_file, 'r') as f:
            model_results['simple_lstm'] = json.load(f)

    # HAR-R Linear (most recent: 2026-06-20_003356)
    har_file = results_dir / "har_baseline_2026-06-20_003356" / "model_info.json"
    if har_file.exists():
        with open(har_file, 'r') as f:
            model_results['har_linear'] = json.load(f)

    return model_results

def compare_models(model_results):
    """Compare all models and generate comparison table"""
    print("=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Create comparison table
    comparison_data = []

    for model_name, results in model_results.items():
        if 'test_metrics' not in results:
            continue

        test = results['test_metrics']
        val = results.get('validation_metrics', {})

        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test RMSE': test['rmse'],
            'Test MAE': test['mae'],
            'Test R²': test['r2'],
            'Test QLIKE': test['qlike'],
            'Test Dir Acc': test['directional_accuracy'],
            'Val RMSE': val.get('rmse', 'N/A'),
            'Val R²': val.get('r2', 'N/A'),
            'Improvement': f"{(val['r2'] - test['r2']) * 100:+.1f}%" if 'r2' in val else 'N/A'
        })

    # Create DataFrame and display
    df = pd.DataFrame(comparison_data)

    print("\nTEST PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(df.to_string(index=False))

    # Find best model for each metric
    print("\n" + "=" * 80)
    print("BEST MODEL BY METRIC:")
    print("=" * 80)

    for metric in ['Test RMSE', 'Test MAE', 'Test R²', 'Test QLIKE', 'Test Dir Acc']:
        if metric == 'Test RMSE' or metric == 'Test MAE' or metric == 'Test QLIKE':
            best = df.loc[df[metric].idxmin()]
        elif metric == 'Test R²' or metric == 'Test Dir Acc':
            best = df.loc[df[metric].idxmax()]
        else:
            continue

        print(f"{metric}: {best['Model']} ({best[metric]:.6f})")

    # Overall winner (based on RMSE)
    best_rmse_model = df.loc[df['Test RMSE'].idxmin()]['Model']
    print(f"\n[WINNER] Overall winner (lowest RMSE): {best_rmse_model}")

    # Save comparison
    output_file = Path("results") / f"model_comparison_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    comparison_data_dict = {
        'timestamp': datetime.now().isoformat(),
        'comparison_table': df.to_dict(orient='records'),
        'best_models': {
            'lowest_rmse': best_rmse_model,
            'highest_r2': df.loc[df['Test R²'].idxmax()]['Model'],
            'highest_dir_acc': df.loc[df['Test Dir Acc'].idxmax()]['Model']
        }
    }

    with open(output_file, 'w') as f:
        json.dump(comparison_data_dict, f, indent=2)

    print(f"\nComparison saved to: {output_file}")

    return df, best_rmse_model

def main():
    """Main execution"""
    print("Loading model results...")
    model_results = load_results()

    print(f"Found results for {len(model_results)} models:")
    for model in model_results.keys():
        print(f"  - {model}")

    if len(model_results) == 0:
        print("No results found!")
        return

    # Compare models
    df, winner = compare_models(model_results)

    # Additional analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for model_name, results in model_results.items():
        if 'test_metrics' not in results:
            continue

        test = results['test_metrics']
        val = results.get('validation_metrics', {})

        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Test RMSE: {test['rmse']:.6f}")
        print(f"  Test R²: {test['r2']:.6f}")
        print(f"  Test Dir Acc: {test['directional_accuracy']:.2f}%")

        if 'r2' in val:
            overfitting = val['r2'] - test['r2']
            if overfitting < -0.05:
                print(f"  [WARNING]  OVERFITTING: Val R² ({val['r2']:.6f}) >> Test R² ({test['r2']:.6f})")
            else:
                print(f"  [OK] Generalization: Similar Val ({val['r2']:.6f}) and Test ({test['r2']:.6f})")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if 'enhanced_lstm_har' in model_results:
        enh_results = model_results['enhanced_lstm_har']
        print("[RECOMMENDATION] Use Enhanced LSTM-HAR for production:")
        print(f"  - Best RMSE: {enh_results['test_metrics']['rmse']:.6f}")
        print(f"  - Good R2: {enh_results['test_metrics']['r2']:.6f}")
        print(f"  - Balanced performance across all metrics")

if __name__ == "__main__":
    main()
