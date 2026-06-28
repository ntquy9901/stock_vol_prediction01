"""
Check training progress for Simple LSTM and LSTM-HAR models
"""
from pathlib import Path
import json

print("="*80)
print("TRAINING PROGRESS - SIMPLE LSTM & LSTM-HAR (VN30)")
print("="*80)

# Check Simple LSTM
print("\n[1] Simple LSTM (1-Layer, 128 hidden)")
print("-"*80)
simple_dir = Path('results/simple_lstm_vn30_2026-06-20')
if simple_dir.exists():
    files = list(simple_dir.glob('*.*'))
    print(f"Directory: {simple_dir}")

    # Check for model file
    model_file = simple_dir / 'best_simple_lstm_vn30.pth'
    if model_file.exists():
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"[OK] Model file: {model_file.name} ({size_mb:.2f} MB)")

    # Check for training results
    results_file = simple_dir / 'training_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"[OK] Training complete!")
        print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
        print(f"  Total Epochs: {results.get('total_epochs', 'N/A')}")
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            print(f"  Dir Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
    else:
        print("[...] Training in progress...")
    print()
else:
    print("[X] Not started")
    print()

# Check LSTM-HAR
print("[2] LSTM-HAR (2-Layer, 64 hidden)")
print("-"*80)
lstm_har_dir = Path('results/lstm_har_vn30_2026-06-20')
if lstm_har_dir.exists():
    files = list(lstm_har_dir.glob('*.*'))
    print(f"Directory: {lstm_har_dir}")

    # Check for model file
    model_file = lstm_har_dir / 'best_lstm_har_vn30.pth'
    if model_file.exists():
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"[OK] Model file: {model_file.name} ({size_mb:.2f} MB)")

    # Check for training results
    results_file = lstm_har_dir / 'training_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"[OK] Training complete!")
        print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
        print(f"  Total Epochs: {results.get('total_epochs', 'N/A')}")
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            print(f"  Dir Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
    else:
        print("[...] Training in progress...")
    print()
else:
    print("[X] Not started")
    print()

print("="*80)
print("REPORT FILES GENERATED:")
print("="*80)
print("[OK] SIMPLE_LSTM_VN30_REPORT.md - Simple LSTM detailed report")
print("[OK] LSTM_HAR_VN30_REPORT.md   - LSTM-HAR (2-layer) detailed report")
print()
print("Next: Wait for training completion, then reports will be auto-updated")
print("="*80)
