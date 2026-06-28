"""
Quick test script with numpy warnings suppressed
"""
import warnings
import sys

# Suppress ALL numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Also suppress scipy warnings if any
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

print("="*80)
print("QUICK TEST - Numpy warnings suppressed")
print("="*80)

# Run the training script
import subprocess
result = subprocess.run([
    sys.executable,
    'src/lstm_gat_hybrid/train_parallel_enhanced.py',
    '--quick_test',
    '--graph_method', 'knn'
], capture_output=False, text=True)

sys.exit(result.returncode)
