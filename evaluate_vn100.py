"""
Evaluate Trained LSTM-GNN Model (VN30) on VN100 Test Set

Model: results/parallel_lstm_gnn_knn_2026-06-22_193440/best_parallel_model.pth
Data: data/processed/vn100_only/

Usage:
    Step 1: python process_vn100_data.py
    Step 2: python evaluate_vn100.py
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')

from src.lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method_fixed
from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.common.evaluation import evaluate_predictions

print("="*80)
print("EVALUATE LSTM-GNN (VN30) ON VN100 TEST SET")
print("="*80)

# Configuration
VN100_DATA_DIR = "data/processed/vn100_only"
MODEL_CHECKPOINT = "results/parallel_lstm_gnn_knn_2026-06-22_193440/best_parallel_model.pth"

print(f"\n[Configuration]")
print(f"  Model: {MODEL_CHECKPOINT}")
print(f"  VN100 Data: {VN100_DATA_DIR}")

# Check if VN100 processed data exists
vn100_path = Path(VN100_DATA_DIR)
if not vn100_path.exists():
    print(f"\n[ERROR] VN100 processed data not found!")
    print(f"\nPlease run: python process_vn100_data.py")
    sys.exit(1)

num_stocks = len(list(vn100_path.glob('*.csv')))
print(f"  Found {num_stocks} stocks")

# Check if model checkpoint exists
checkpoint_path = Path(MODEL_CHECKPOINT)
if not checkpoint_path.exists():
    print(f"\n[ERROR] Model checkpoint not found!")
    print(f"\nAvailable checkpoints:")

    # List recent checkpoints
    results_dirs = sorted(Path('results').glob('parallel_lstm_gnn_knn_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    for dir in results_dirs[:5]:
        ckpt = dir / 'best_parallel_model.pth'
        if ckpt.exists():
            print(f"  - {ckpt}")
    sys.exit(1)

print(f"  [OK] Checkpoint found")

# Check training results to see original performance
training_results = checkpoint_path.parent / 'training_results.json'
if training_results.exists():
    import json
    with open(training_results, 'r') as f:
        results = json.load(f)

    print(f"\n[Model Training Results (VN30)]")
    if 'test_metrics' in results:
        metrics = results['test_metrics']
        print(f"  Test MSE:  {metrics.get('mse', 'N/A')}")
        print(f"  Test RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"  Test Dir Acc:  {metrics.get('dir_acc', 'N/A')}%")
    print(f"  Epochs trained: {results.get('epochs_trained', 'N/A')}")

# Create config
config = LSTMGATConfig()

# Create dataloaders for VN100
print(f"\n[Step 1] Creating dataloaders for VN100...")
try:
    train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders_with_graph_method_fixed(
        data_dir=VN100_DATA_DIR,
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
        graph_method='knn',
        k_neighbors=config.top_k_neighbors,  # Changed from config.k_neighbors
        batch_size=config.batch_size if hasattr(config, 'batch_size') else 32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=0,
        normalize=True,
        remove_outliers=True,
        n_std=3.0,
        data_augmentation=False
    )

    train_dataset, val_dataset, test_dataset = datasets
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(test_dataset)} sequences")

except Exception as e:
    print(f"  [ERROR] Failed to create dataloaders: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load model
print(f"\n[Step 2] Loading trained model from VN30...")
try:
    model = create_parallel_lstm_gat_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is the state_dict itself
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  [OK] Model loaded successfully")

except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate on test set
print(f"\n[Step 3] Evaluating on VN100 test set...")
print(f"  Number of batches: {len(test_loader)}")

all_predictions = []
all_targets = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    for batch_idx, (x, adj_matrix, y, _) in enumerate(test_loader):
        # Move to device
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)

        # Forward pass
        predictions = model(x, adj_matrix)

        # Collect predictions and targets
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y.numpy())

# Concatenate all results
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Flatten for evaluation
all_predictions_flat = all_predictions.flatten()
all_targets_flat = all_targets.flatten()

print(f"  Predictions shape: {all_predictions_flat.shape}")
print(f"  Targets shape: {all_targets_flat.shape}")

# Calculate metrics
print(f"\n[Step 4] Calculating all 6 mandatory metrics...")

try:
    metrics = evaluate_predictions(all_targets_flat, all_predictions_flat)

    print(f"\n{'='*80}")
    print(f"VN100 TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"\nAll 6 Mandatory Metrics:")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  RMSE:      {metrics['rmse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  R²:        {metrics['r2']:.6f}")
    print(f"  QLIKE:     {metrics['qlike']:.6f}")
    print(f"  Dir Acc:   {metrics['directional_accuracy']:.2f}%")

    # Performance assessment
    print(f"\nPerformance Assessment:")
    if metrics['directional_accuracy'] > 60:
        print(f"  [OK] Dir Acc > 60%: EXCELLENT prediction")
    elif metrics['directional_accuracy'] > 55:
        print(f"  [OK] Dir Acc > 55%: GOOD prediction")
    elif metrics['directional_accuracy'] > 50:
        print(f"  [WARN] Dir Acc > 50%: MODERATE prediction")
    else:
        print(f"  [FAIL] Dir Acc < 50%: POOR prediction")

    if metrics['r2'] > 0.7:
        print(f"  [OK] R² > 0.7: EXCELLENT fit")
    elif metrics['r2'] > 0.5:
        print(f"  [OK] R² > 0.5: GOOD fit")
    elif metrics['r2'] > 0.3:
        print(f"  [WARN] R² > 0.3: MODERATE fit")
    else:
        print(f"  [FAIL] R² < 0.3: POOR fit")

    # Transfer learning analysis
    print(f"\nTransfer Learning Analysis (VN30 → VN100):")
    print(f"  Model trained on: 30 VN30 stocks")
    print(f"  Model tested on: {num_stocks} VN100 stocks")
    print(f"  Performance gap indicates domain adaptation capability")

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame([{
        'Timestamp': timestamp,
        'Source_Dataset': 'VN30',
        'Target_Dataset': 'VN100',
        'Model': 'LSTM-GNN',
        'Checkpoint': MODEL_CHECKPOINT,
        'Num_Stocks_Train': 30,
        'Num_Stocks_Test': num_stocks,
        'Test_MSE': metrics['mse'],
        'Test_RMSE': metrics['rmse'],
        'Test_MAE': metrics['mae'],
        'Test_R2': metrics['r2'],
        'Test_QLIKE': metrics['qlike'],
        'Test_Dir_Acc': metrics['directional_accuracy']
    }])

    results_file = Path(f'results/vn100_evaluation_{timestamp}.csv')
    results_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\n  Results saved to: {results_file}")

except Exception as e:
    print(f"  [ERROR] Failed to calculate metrics: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
