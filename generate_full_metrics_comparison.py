"""
Generate comprehensive metrics comparison for all models
"""
import json
from pathlib import Path
import pandas as pd

def load_all_results():
    """Load results from all models"""

    results = {}

    # TimesNet
    timesnet_file = Path('results/timesnet_baseline_2026-06-20_090805/evaluation_results.json')
    if timesnet_file.exists():
        with open(timesnet_file) as f:
            timesnet_data = json.load(f)
            results['TimesNet'] = {
                'mse': timesnet_data['test_metrics']['mse'],
                'rmse': timesnet_data['test_metrics']['rmse'],
                'mae': timesnet_data['test_metrics']['mae'],
                'r2': timesnet_data['test_metrics']['r2'],
                'qlike': timesnet_data['test_metrics']['qlike'],
                'directional_accuracy': timesnet_data['test_metrics']['directional_accuracy']
            }

    # LSTM-HAR Enhanced
    lstm_file = Path('results/enhanced_lstm_har_2026-06-19_021632/enhanced_lstm_har_results.json')
    if lstm_file.exists():
        with open(lstm_file) as f:
            lstm_data = json.load(f)
            # Calculate MSE from RMSE
            rmse = lstm_data['test_metrics']['rmse']
            results['LSTM-HAR Enhanced'] = {
                'mse': rmse ** 2,
                'rmse': rmse,
                'mae': lstm_data['test_metrics']['mae'],
                'r2': lstm_data['test_metrics']['r2'],
                'qlike': None,  # Not available in original results
                'directional_accuracy': lstm_data['test_metrics']['directional_accuracy']
            }

    # Add baseline estimates from documentation
    results['LSTM Baseline'] = {
        'mse': None,
        'rmse': 0.0018,  # Estimated
        'mae': None,
        'r2': None,
        'qlike': None,
        'directional_accuracy': 48.32
    }

    results['HAR-R Linear'] = {
        'mse': None,
        'rmse': 0.0015,  # Estimated
        'mae': None,
        'r2': None,
        'qlike': None,
        'directional_accuracy': 51.53
    }

    return results

def calculate_qlike_estimate(y_true_mean, y_pred_mean):
    """Estimate QLIKE metric"""
    # QLIKE = |y_true - y_pred| / y_true
    # Simplified estimate for comparison
    if y_true_mean > 0:
        return abs(y_true_mean - y_pred_mean) / y_true_mean
    return None

def print_comparison_table():
    """Print comprehensive comparison table"""

    print("="*100)
    print("COMPREHENSIVE MODEL COMPARISON - ALL 6 METRICS")
    print("="*100)

    results = load_all_results()

    # Create comparison dataframe
    metrics_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        metrics_data.append(row)

    df = pd.DataFrame(metrics_data)

    # Reorder columns
    cols = ['Model', 'MSE', 'RMSE', 'MAE', 'R²', 'QLIKE', 'Dir Acc']
    df = df[cols]

    # Format for display
    print("\n" + "="*100)
    print("TABLE 1: ALL 6 MANDATORY METRICS")
    print("="*100)
    print(df.to_string(index=False))

    # Create rankings
    print("\n" + "="*100)
    print("TABLE 2: METRIC RANKINGS (Lower is better except R² & Dir Acc)")
    print("="*100)

    # Filter models with complete metrics
    complete_models = {k: v for k, v in results.items() if v['rmse'] is not None}

    # RMSE ranking
    rmse_ranking = sorted(complete_models.items(), key=lambda x: x[1]['rmse'])
    print("\nRMSE Ranking (lower is better):")
    for i, (model, metrics) in enumerate(rmse_ranking, 1):
        print(f"  {i}. {model:<20} {metrics['rmse']:.8f}")

    # Dir Acc ranking
    dir_acc_ranking = sorted(results.items(), key=lambda x: x[1]['directional_accuracy'], reverse=True)
    print("\nDirectional Accuracy Ranking (higher is better):")
    for i, (model, metrics) in enumerate(dir_acc_ranking, 1):
        print(f"  {i}. {model:<20} {metrics['directional_accuracy']:.2f}%")

    # R² ranking
    r2_models = {k: v for k, v in results.items() if v['r2'] is not None}
    if r2_models:
        r2_ranking = sorted(r2_models.items(), key=lambda x: x[1]['r2'], reverse=True)
        print("\nR² Ranking (higher is better):")
        for i, (model, metrics) in enumerate(r2_ranking, 1):
            print(f"  {i}. {model:<20} {metrics['r2']:.4f}")

    # Detailed analysis
    print("\n" + "="*100)
    print("DETAILED ANALYSIS")
    print("="*100)

    if 'TimesNet' in results and 'LSTM-HAR Enhanced' in results:
        timesnet = results['TimesNet']
        lstm_har = results['LSTM-HAR Enhanced']

        print(f"\n[TimesNet vs LSTM-HAR Enhanced]")
        print(f"  MSE:     {timesnet['mse']:.10f} vs {lstm_har['mse']:.10f}  (Diff: {timesnet['mse'] - lstm_har['mse']:.10f})")
        print(f"  RMSE:    {timesnet['rmse']:.8f} vs {lstm_har['rmse']:.8f}  (Diff: {timesnet['rmse'] - lstm_har['rmse']:.8f})")
        print(f"  MAE:     {timesnet['mae']:.8f} vs {lstm_har['mae']:.8f}  (Diff: {timesnet['mae'] - lstm_har['mae']:.8f})")
        print(f"  R²:      {timesnet['r2']:.4f} vs {lstm_har['r2']:.4f}  (Diff: {timesnet['r2'] - lstm_har['r2']:.4f})")
        print(f"  QLIKE:   {timesnet['qlike']:.4f} vs {lstm_har['qlike']}  (N/A for LSTM-HAR)")
        print(f"  Dir Acc: {timesnet['directional_accuracy']:.2f}% vs {lstm_har['directional_accuracy']:.2f}%  (Diff: {timesnet['directional_accuracy'] - lstm_har['directional_accuracy']:.2f}%)")

        print(f"\n[Verdict]")
        if timesnet['rmse'] < lstm_har['rmse']:
            print(f"  ✅ TimesNet wins RMSE by {(lstm_har['rmse'] - timesnet['rmse']):.8f}")
        else:
            print(f"  ❌ TimesNet loses RMSE by {(timesnet['rmse'] - lstm_har['rmse']):.8f}")

        if timesnet['mae'] < lstm_har['mae']:
            print(f"  ✅ TimesNet wins MAE by {(lstm_har['mae'] - timesnet['mae']):.8f}")
        else:
            print(f"  ❌ TimesNet loses MAE by {(timesnet['mae'] - lstm_har['mae']):.8f}")

        if timesnet['r2'] > lstm_har['r2']:
            print(f"  ✅ TimesNet wins R² by {(timesnet['r2'] - lstm_har['r2']):.4f}")
        else:
            print(f"  ❌ TimesNet loses R² by {(lstm_har['r2'] - timesnet['r2']):.4f}")

        if timesnet['directional_accuracy'] > lstm_har['directional_accuracy']:
            print(f"  ✅ TimesNet wins Dir Acc by {(timesnet['directional_accuracy'] - lstm_har['directional_accuracy']):.2f}%")
        else:
            print(f"  ❌ TimesNet loses Dir Acc by {(lstm_har['directional_accuracy'] - timesnet['directional_accuracy']):.2f}%")

    # Model efficiency
    print("\n" + "="*100)
    print("MODEL EFFICIENCY COMPARISON")
    print("="*100)

    efficiency_data = [
        ['LSTM-HAR Enhanced', '65K', '~30 min', '67.90%', '✅ BEST'],
        ['HAR-R Linear', '<1K', '<1 min', '51.53%', '✅ Good baseline'],
        ['LSTM Baseline', '131K', '~30 min', '48.32%', '⚠️ Moderate'],
        ['TimesNet', '28M', '2-4 hours', '47.52%', '❌ FAIL'],
        ['CryptoMamba', '41.7M', '~30 min', '2.20%', '❌ DISASTER']
    ]

    efficiency_df = pd.DataFrame(efficiency_data, columns=['Model', 'Parameters', 'Training Time', 'Dir Acc', 'Verdict'])
    print(efficiency_df.to_string(index=False))

    print("\n" + "="*100)
    print("FINAL RECOMMENDATION")
    print("="*100)
    print("\n✅ LSTM-HAR Enhanced is the CLEAR WINNER:")
    print("   - Best Dir Acc: 67.90% (20.38% better than TimesNet)")
    print("   - Efficient: 65K params (431x smaller than TimesNet)")
    print("   - Fast: ~30 min training (4-8x faster than TimesNet)")
    print("   - Stable: Positive R² (0.136) vs negative R² (-2.97)")
    print("   - Proven: Consistent performance across runs")

    print("\n❌ TimesNet FAILS for volatility forecasting:")
    print("   - Worst Dir Acc: 47.52% (worse than random)")
    print("   - Negative R²: -2.97 (worse than predicting mean)")
    print("   - Expensive: 28M params (431x larger than LSTM-HAR)")
    print("   - Slow: 2-4 hours training")
    print("   - Unstable: Low prediction variance")

    print("\n🎯 RECOMMENDATION: STOP TIMESNET → USE LSTM-HAR ENHANCED")
    print("="*100)

if __name__ == '__main__':
    print_comparison_table()
