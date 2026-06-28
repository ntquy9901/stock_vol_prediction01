"""
Full Training Script for LSTM-GNN (50 epochs)
With all data leakage fixes applied
"""
import warnings
import sys

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

print("="*80)
print("FULL TRAINING - LSTM-GNN WITH ALL DATA LEAKAGE FIXES")
print("="*80)
print("\nConfiguration:")
print("  - Graph Method: knn")
print("  - Max Epochs: 50")
print("  - Early Stopping: patience=15")
print("  - All 4 critical fixes active:")
print("    ✓ Per-sequence graph construction")
print("    ✓ Training-only normalization")
print("    ✓ Proper temporal split")
print("    ✓ HAR features split-first")
print("\n" + "="*80)

# Run the training script
import subprocess

# Options:
# --graph_method knn: Use k-NN sparse graph (k=8)
# --quick_test: NOT SET (runs full 50 epochs)

result = subprocess.run([
    sys.executable,
    'src/lstm_gat_hybrid/train_parallel_enhanced.py',
    '--graph_method', 'knn'  # Options: 'correlation' or 'knn'
], capture_output=False, text=True)

sys.exit(result.returncode)
