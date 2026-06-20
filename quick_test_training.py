"""
Quick Test Script for TimesFM 2.5 LoRA Training Pipeline

This script generates synthetic OHLCV data and runs a quick training
to verify the complete pipeline works end-to-end.

Usage:
    python quick_test_training.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate synthetic OHLCV data
logger.info("Generating synthetic OHLCV data...")

np.random.seed(42)
n_days = 2000  # ~8 years of data
dates = pd.date_range(start='2010-01-01', periods=n_days, freq='D')

# Simulate price movements
price = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
high = price + np.abs(np.random.randn(n_days) * 2)
low = price - np.abs(np.random.randn(n_days) * 2)
close = price + np.random.randn(n_days) * 0.5
volume = np.random.randint(1000000, 10000000, n_days)

ohlcv_df = pd.DataFrame({
    'date': dates,
    'open': price,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
})

# Save to temporary file
temp_dir = Path(tempfile.mkdtemp())
sample_file = temp_dir / "sample_stock_ohlcv.csv"
ohlcv_df.to_csv(sample_file, index=False)

logger.info(f"Created synthetic OHLCV data: {sample_file}")
logger.info(f"  Shape: {ohlcv_df.shape}")
logger.info(f"  Date range: {ohlcv_df['date'].min()} to {ohlcv_df['date'].max()}")

# Import training module
from src.timesfm_baseline.train_timesfm_lora import main as train_main
import sys

# Setup command line arguments
sys.argv = [
    'quick_test_training.py',
    '--data_path', str(sample_file),
    '--output_dir', str(temp_dir / 'timesfm_lora_test'),
    '--epochs', '2',  # Quick test: only 2 epochs
    '--batch_size', '16',  # Smaller batch for faster test
    '--num_train_samples', '100',  # Fewer samples for quick test
    '--mlflow_experiment', 'TimesFM_Quick_Test',
]

logger.info("\n" + "="*70)
logger.info("Starting TimesFM 2.5 LoRA Training Test")
logger.info("="*70)
logger.info(f"Configuration:")
logger.info(f"  Epochs: 2")
logger.info(f"  Batch Size: 16")
logger.info(f"  Train Samples: 100")
logger.info(f"  Output: {temp_dir / 'timesfm_lora_test'}")
logger.info("="*70 + "\n")

try:
    # Run training
    train_main()

    logger.info("\n" + "="*70)
    logger.info("✅ Quick Test Complete!")
    logger.info("="*70)
    logger.info(f"\nOutput saved to: {temp_dir}")
    logger.info("  - LoRA adapter: timesfm_lora_test/")
    logger.info("  - Training results: training_results.json")
    logger.info("  - Learning curves: learning_curves.png")

except Exception as e:
    logger.error(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
