"""
Display generated visualization files for VN30 training results
"""
from pathlib import Path

print("="*80)
print("VN30 TRAINING VISUALIZATION RESULTS")
print("="*80)

result_dir = Path(r'results\enhanced_lstm_har_vn30_2026-06-20')

print("\n[!] Generated Visualization Files:")
print("-"*80)

files = list(result_dir.glob('*.*'))
for f in sorted(files):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  • {f.name}")
    print(f"    Size: {size_mb:.2f} MB")
    print(f"    Path: {f.absolute()}")
    print()

print("="*80)
print("KEY FILES:")
print("="*80)
print(f"1. comprehensive_learning_curves.png")
print("   - 6-plot analysis showing training progress")
print("   - Includes: loss curves, overfitting, convergence, LR, gradients, metrics")
print()
print(f"2. snapshot_mechanism_diagram.txt")
print("   - Detailed ASCII diagrams of sliding window mechanism")
print("   - Shows how 22 days -> 5-day ahead prediction works")
print("   - Includes batching and memory organization details")
print()
print(f"3. training_results.json")
print("   - Final performance metrics")
print("   - Dir Acc: 68.01%, RMSE: 0.000630, R²: 0.119")
print()
print(f"4. best_enhanced_lstm_har_vn30.pth")
print("   - Production model checkpoint (1.3 MB)")
print("   - Contains best weights from epoch 18")
print()
print(f"5. checkpoint.pth")
print("   - Full training state (4.0 MB)")
print("   - Includes optimizer state for resuming training")
print()

print("="*80)
print("VN30 PERFORMANCE REPORT UPDATED")
print("="*80)
print(f"\n[?] Report: VN30_PERFORMANCE_REPORT.md")
print(f"\n[X] Added sections:")
print(f"   • Learning Curves Visualization (6 plots)")
print(f"   • Snapshot Mechanism Documentation")
print(f"   • Detailed architecture diagrams")
print(f"   • Training progress analysis")
print()

print("[X] All visualizations complete!")
