"""
Monitor TimesNet training progress
"""
import time
from pathlib import Path

results_dir = Path('results/timesnet_baseline_2026-06-20_091514')

print("Monitoring TimesNet training...")
print(f"Results directory: {results_dir}")

# Check for results file
results_file = results_dir / 'timesnet_baseline_results.json'
if results_file.exists():
    print("[SUCCESS] Training completed! Results found.")
    import json
    with open(results_file) as f:
        results = json.load(f)
    print(f"Test Dir Acc: {results['test_metrics']['directional_accuracy']:.2f}%")
else:
    print("[IN PROGRESS] Training still running...")

# Check for learning curves
learning_curves = list(results_dir.glob('learning_curves_*.png'))
if learning_curves:
    print(f"[PROGRESS] Found {len(learning_curves)} learning curve plots")
    latest = max(learning_curves, key=lambda p: p.stat().st_mtime)
    print(f"Latest: {latest.name}")
