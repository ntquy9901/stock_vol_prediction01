"""
Quick test of complete training pipeline with normalization
"""
from src.cryptomamba_baseline.train_enhanced import CryptoMambaEnhancedTrainer
from pathlib import Path

# Create trainer
trainer = CryptoMambaEnhancedTrainer()

# Find data file
data_dir = Path("data/processed")
data_files = list(data_dir.glob("*.csv"))
data_path = data_files[0]

print(f"Testing complete training pipeline with normalization...")

# Load and prepare data (should include normalization now)
train_loader, val_loader, test_loader = trainer.load_and_prepare_data(data_path)

# Create model
trainer.create_model()

# Quick training test (3 epochs)
print("\\n=== QUICK TRAINING TEST (3 epochs) ===")
trainer.training_config['num_epochs'] = 3
trainer.training_config['patience'] = 10
trainer.training_config['min_epochs'] = 1

trainer.train(train_loader, val_loader)

# Test evaluation
test_metrics = trainer.evaluate_test_set(test_loader)

print(f"\\n=== RESULTS ===")
print(f"Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
print(f"Test RMSE: {test_metrics['rmse']:.6f}")

if test_metrics['directional_accuracy'] > 1.0:  # Should be >1% if working
    print(f"\\n[SUCCESS] Model training with normalization works!")
else:
    print(f"\\n[WARNING] Dir Acc suspiciously low")
