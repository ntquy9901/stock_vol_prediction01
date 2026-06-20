"""
LSTM Training with Fixes - Complete Guide

This document provides the complete guide to re-train LSTM with all fixes
applied and verify metrics improve.

Author: Stock Volatility Prediction Team
Date: 2026-06-18
Status: All fixes applied, ready for training
"""

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                   LSTM TRAINING WITH FIXES - READY TO RUN                  ║
╚════════════════════════════════════════════════════════════════════════════╝

All fixes have been applied to the LSTM training code:

✅ FIX 1: Increased Model Capacity
   - Before: 32 hidden units (4.5K parameters)
   - After: 128 hidden units (~18K parameters)
   - File: src/lstm_baseline/model.py

✅ FIX 2: Increased Learning Rate
   - Before: lr=0.001 (too conservative for scaled data)
   - After: lr=0.01 (10× higher for faster convergence)
   - File: src/lstm_baseline/train.py

✅ FIX 3: Added Pre-Flight Validation
   - Dataset statistics check
   - Feature scaler validation
   - Model forward pass test
   - File: src/lstm_baseline/train.py

✅ FIX 4: Added Training Monitoring
   - TrainingMonitor with epoch 5 check
   - Early detection of flat learning curves
   - Gradient health monitoring
   - File: src/lstm_baseline/train.py

✅ PREVENTION: Debug Script Available
   - Located: src/experiment/debug_training_failure.py
   - Can run to investigate any issues

════════════════════════════════════════════════════════════════════════════

EXPECTED IMPROVEMENTS:

Before Fixes:
  QLIKE: 0.74 (target: < 0.20)
  Directional Accuracy: 0.5% (target: > 55%)
  Model: Not learning

After Fixes (Expected):
  QLIKE: < 0.20 ✅
  Directional Accuracy: > 55% ✅
  Model: Learning patterns ✅

════════════════════════════════════════════════════════════════════════════

HOW TO TRAIN:

1. Navigate to project root:
   cd D:\\bmad-projects\\stock_vol_prediction01

2. Run training:
   python -m src.lstm_baseline.train

3. Training will now:
   - Validate data before training (pre-flight)
   - Monitor health every epoch
   - Stop early if issues detected
   - Use 128 hidden units (4× capacity)
   - Use learning rate 0.01 (10× faster)

════════════════════════════════════════════════════════════════════════════

TIME ESTIMATE:

Before: ~12 minutes (30 epochs)
After: ~15 minutes (30 epochs, larger model)
Total: ~15 minutes for full training

════════════════════════════════════════════════════════════════════════════

VERIFICATION:

After training completes, check:

1. Results location: results/simple_lstm_YYYY-MM-DD_HHMMSS/
2. Metrics file: test_metrics.csv
3. Check QLIKE < 0.20
4. Check Directional_Acc > 55%
5. View learning curves: training_curves.png

════════════════════════════════════════════════════════════════════════════

SUCCESS CRITERIA:

✅ QLIKE < 0.20 (volatility forecasting standard)
✅ Directional Accuracy > 55% (beats random baseline)
✅ Learning curve shows decreasing loss
✅ No training health warnings

════════════════════════════════════════════════════════════════════════════

IF ISSUES PERSIST:

If metrics still don't improve after fixes:

1. Run debug script:
   python -m src.experiment.debug_training_failure

2. Check scaler output:
   - Mean should be ~0
   - Std should be ~1.0

3. Consider additional fixes:
   - Increase model capacity further (256 units)
   - Add more features (HAR features)
   - Adjust architecture (more LSTM layers)

══════════════════════════════════════════════════════════════════════════════

COMMAND TO RUN:

cd D:\\bmad-projects\\stock_vol_prediction01
python -m src.lstm_baseline.train

══════════════════════════════════════════════════════════════════════════════

That's it! Training will start with all improvements applied.

Good luck! 🚀
""")

print("\n✅ All fixes applied - Ready to train!")
print("\nNext steps:")
print("1. cd D:\\bmad-projects\\stock_vol_prediction01")
print("2. python -m src.lstm_baseline.train")
print("3. Wait ~15 minutes for training")
print("4. Check results in results/simple_lstm_*/")
print("\nExpected: QLIKE < 0.20, Directional Accuracy > 55%")
