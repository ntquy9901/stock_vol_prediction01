"""
Evaluate LSTM-GNN (VN30-trained) on VN100 Training Data Only

Uses simple approach:
- Load VN100 data
- Generate HAR features on ALL data (no splitting)
- Evaluate model directly

Model: results/parallel_lstm_gnn_knn_2026-06-22_193440/best_parallel_model.pth
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')

from src.lstm_gat_hybrid.dataset_with_graph_method import MultiStockDatasetWithGraphMethod
from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.common.evaluation import evaluate_predictions

print("="*80)
print("EVALUATE LSTM-GNN (VN30) ON VN100 TRAINING DATA")
print("="*80)

# Configuration
VN100_DATA_DIR = "data/processed/vn100_only"
MODEL_CHECKPOINT = "results/parallel_lstm_gnn_knn_2026-06-22_193440/best_parallel_model.pth"

print(f"\n[Configuration]")
print(f"  Model: {MODEL_CHECKPOINT}")
print(f"  VN100 Data: {VN100_DATA_DIR}")
print(f"  Approach: Use ALL data (no train/val/test split)")

# Check data exists
vn100_path = Path(VN100_DATA_DIR)
if not vn100_path.exists():
    print(f"\n[ERROR] VN100 processed data not found!")
    print(f"\nPlease run: python process_vn100_data.py")
    sys.exit(1)

num_stocks = len(list(vn100_path.glob('*.csv')))
print(f"  Found {num_stocks} stocks")

# Check model checkpoint exists
checkpoint_path = Path(MODEL_CHECKPOINT)
if not checkpoint_path.exists():
    print(f"\n[ERROR] Model checkpoint not found!")
    sys.exit(1)

print(f"  [OK] Checkpoint found")

# Check training results
training_results_file = checkpoint_path.parent / 'training_results.json'
if training_results_file.exists():
    import json
    with open(training_results_file, 'r') as f:
        train_results = json.load(f)

    print(f"\n[Model Training Results (VN30)]")
    if 'test_metrics' in train_results:
        metrics = train_results['test_metrics']
        print(f"  Test MSE:  {metrics.get('mse', 'N/A')}")
        print(f"  Test RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"  Test Dir Acc: {metrics.get('dir_acc', 'N/A')}%")

# Create config
config = LSTMGATConfig()

# Create dataset directly (no splitting)
print(f"\n[Step 1] Creating dataset from VN100 data...")
try:
    # Create dataset with NO splitting (use all data for evaluation)
    dataset = MultiStockDatasetWithGraphMethod(
        data_dir=str(VN100_DATA_DIR),
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
        graph_method='knn',
        k_neighbors=config.top_k_neighbors,
        normalize=True,
        remove_outliers=True,
        n_std=3.0
    )

    print(f"  [OK] Dataset created: {len(dataset)} sequences")

    if len(dataset) == 0:
        print(f"  [ERROR] Dataset is empty!")
        sys.exit(1)

    # Fit normalizers on the dataset
    print(f"\n  [Step 1.5] Fitting normalizers...")
    # Fit on ALL sequences since we're using entire dataset for evaluation
    for stock_idx, stock_name in enumerate(dataset.stock_names):
        # Collect features and targets from all sequences
        all_features = []
        all_targets = []

        for seq_idx in range(len(dataset)):
            x, adj_matrix, y = dataset.sequences[seq_idx]
            all_features.append(x[:, stock_idx, :])  # All timesteps for this stock
            all_targets.append(y[stock_idx])

        # Concatenate all samples
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.array(all_targets)

        # Fit normalizers
        dataset.feature_normalizers[stock_name].fit(all_features)
        dataset.target_normalizers[stock_name].fit(all_targets.reshape(-1, 1))

    print(f"  [OK] Normalizers fitted on {len(dataset.sequences)} sequences")

except Exception as e:
    print(f"  [ERROR] Failed to create dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create simple dataloader
print(f"\n[Step 2] Creating dataloader...")
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
print(f"  [OK] Dataloader created: {len(dataloader)} batches")

# Load model
print(f"\n[Step 3] Loading trained model...")
try:
    model = create_parallel_lstm_gat_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  [OK] Model loaded successfully")

except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate
print(f"\n[Step 4] Evaluating on VN100...")
all_predictions = []
all_targets = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    for batch_idx, (x, adj_matrix, y, _) in enumerate(dataloader):
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)
        predictions = model(x, adj_matrix)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y.numpy())

# Concatenate
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
all_predictions_flat = all_predictions.flatten()
all_targets_flat = all_targets.flatten()

print(f"  Predictions shape: {all_predictions_flat.shape}")
print(f"  Targets shape: {all_targets_flat.shape}")

# Calculate metrics
print(f"\n[Step 5] Calculating metrics...")
try:
    metrics = evaluate_predictions(all_targets_flat, all_predictions_flat)

    print(f"\n{'='*80}")
    print(f"VN100 EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\nAll 6 Mandatory Metrics:")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  RMSE:      {metrics['rmse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  R²:        {metrics['r2']:.6f}")
    print(f"  QLIKE:     {metrics['qlike']:.6f}")
    print(f"  Dir Acc:   {metrics['directional_accuracy']:.2f}%")

    # Assessment
    print(f"\nPerformance Assessment:")
    if metrics['directional_accuracy'] > 60:
        print(f"  [OK] Dir Acc > 60%: EXCELLENT")
    elif metrics['directional_accuracy'] > 55:
        print(f"  [OK] Dir Acc > 55%: GOOD")
    elif metrics['directional_accuracy'] > 50:
        print(f"  [WARN] Dir Acc > 50%: MODERATE")
    else:
        print(f"  [FAIL] Dir Acc < 50%: POOR")

    # Comparison
    print(f"\nModel Comparison:")
    print(f"  VN30 Training (source):")
    print(f"    Dir Acc: 68.02% (from earlier test)")
    print(f"  VN100 Evaluation (target):")
    print(f"    Dir Acc: {metrics['directional_accuracy']:.2f}%")

    if metrics['directional_accuracy'] > 60:
        print(f"  [OK] Model generalizes well: VN30 -> VN100")
    elif abs(metrics['directional_accuracy'] - 68.02) < 5:
        print(f"  [OK] Similar performance: Good transfer learning")
    else:
        print(f"  [WARN] Performance gap: Domain adaptation needed")

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame([{
        'Timestamp': timestamp,
        'Source_Dataset': 'VN30',
        'Target_Dataset': 'VN100',
        'Evaluation_Type': 'train_only',
        'Model': 'LSTM-GNN',
        'Checkpoint': MODEL_CHECKPOINT,
        'Num_Stocks_Train': 30,
        'Num_Stocks_Test': num_stocks,
        'Test_Sequences': len(dataset),
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
    print(f"  [ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
