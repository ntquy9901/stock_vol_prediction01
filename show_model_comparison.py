#!/usr/bin/env python3
"""
Model Comparison Summary - Temporal Split (70/15/15)
Generated: 2026-06-19
"""

print("="*80)
print("MODEL COMPARISON - TEMPORAL SPLIT (70/15/15)")
print("="*80)
print()
print("Target: 5-day ahead volatility forecast")
print("Evaluation: 6 mandatory metrics")
print()

print("-"*80)
print("TEST RESULTS (All Models)")
print("-"*80)
print()
print(f"{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<10} {'QLIKE':<12} {'Dir Acc':<10}")
print("-"*80)

# Results from temporal split evaluation
results = [
    {
        "model": "1. HAR-R Linear (Baseline)",
        "mse": "2.63e-07",
        "rmse": 0.000513,
        "mae": 0.000257,
        "r2": 0.105,
        "qlike": 1.298,
        "dir_acc": 51.53
    },
    {
        "model": "2. Simple LSTM",
        "mse": "3.06e-07",
        "rmse": 0.000553,
        "mae": 0.000264,
        "r2": 0.109,
        "qlike": 0.641,
        "dir_acc": 48.05
    },
    {
        "model": "3. LSTM-HAR",
        "mse": "N/A",
        "rmse": 0.000554,
        "mae": 0.000258,
        "r2": 0.110,
        "qlike": float('inf'),  # Missing in results
        "dir_acc": 48.09
    },
    {
        "model": "4. Enhanced LSTM-HAR",
        "mse": "3.08e-07",
        "rmse": 0.000555,
        "mae": 0.000262,
        "r2": 0.107,
        "qlike": 0.638,
        "dir_acc": 48.43
    }
]

# Print results
for r in results:
    model = r['model']
    mse = r['mse']
    rmse = f"{r['rmse']:.6f}" if isinstance(r['rmse'], float) else r['rmse']
    mae = f"{r['mae']:.6f}" if isinstance(r['mae'], float) else r['mae']
    r2 = f"{r['r2']:.6f}" if isinstance(r['r2'], float) else r['r2']
    qlike = f"{r['qlike']:.6f}" if isinstance(r['qlike'], float) and r['qlike'] != float('inf') else r['qlike']
    dir_acc = f"{r['dir_acc']:.2f}%" if isinstance(r['dir_acc'], float) else r['dir_acc']

    print(f"{model:<25} {mse:<12} {rmse:<12} {mae:<12} {r2:<10} {qlike:<12} {dir_acc:<10}")

print()
print("-"*80)
print("BEST MODEL PER METRIC")
print("-"*80)
print()
print("Lower is Better (MSE, RMSE, MAE, QLIKE):")
print("  [1st] MSE:  HAR-R Linear (2.63e-07)")
print("  [1st] RMSE: HAR-R Linear (0.000513)")
print("  [1st] MAE: LSTM-HAR (0.000258)")
print("  [1st] QLIKE: Enhanced LSTM-HAR (0.638)")
print()
print("Higher is Better (R2, Dir Acc):")
print("  [1st] R2: LSTM-HAR (0.110)")
print("  [1st] Dir Acc: HAR-R Linear (51.53%)")
print()

print("-"*80)
print("KEY INSIGHTS")
print("-"*80)
print()
print("[+] HAR-R Linear (Baseline) is SURPRISINGLY STRONG:")
print("   - Best RMSE, MSE, and Dir Acc")
print("   - Simple, interpretable, no overfitting")
print("   - Suggests data has strong linear patterns")
print()
print("[-] LSTM Models UNDERPERFORM:")
print("   - All below 50% Dir Acc (worse than random!)")
print("   - No significant improvement over HAR-R Linear")
print("   - Possible issues:")
print("     o Insufficient training data")
print("     o Over-regularization (dropout=0.2)")
print("     o Wrong architecture")
print("     o Feature scaling problems")
print()

print("-"*80)
print("PROGRESS TOWARDS SUCCESS CRITERIA")
print("-"*80)
print()
print("Target: RMSE < 0.20")
print(f"Status: [+] EXCEEDED (Current: 0.000513)")
print()
print("Target: Dir Acc > 55%")
print(f"Status: [-] NOT MET (Current: 51.53%, Need: +3.5%)")
print()

print("-"*80)
print("RECOMMENDATIONS")
print("-"*80)
print()
print("[HIGH] High Priority:")
print("1. Investigate LSTM underperformance")
print("2. Improve directional accuracy (> 55%)")
print("3. Debug data pipeline & feature scaling")
print()
print("[MEDIUM] Medium Priority:")
print("4. Hyperparameter tuning")
print("5. Feature engineering (add 19 technical indicators)")
print("6. Multi-task learning (volatility + direction)")
print()
print("[ADVANCED] Advanced:")
print("7. LSTM-GAT Hybrid (expected: RMSE 17% percent, Dir Acc 7% percent)")
print()

print("="*80)
print("DETAILED REPORT: MODEL_COMPARISON_TEMPORAL_SPLIT_2026-06-19.md")
print("="*80)