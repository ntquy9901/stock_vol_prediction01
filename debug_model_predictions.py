"""
Debug script to investigate what the trained CryptoMamba Enhanced V2 model is predicting
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.cryptomamba_baseline.model_enhanced import create_cryptomamba_model_enhanced
from src.common.har_features import generate_har_features
from src.common.data_normalization import normalize_for_training, denormalize_predictions, VolatilityNormalizer

# Load trained model
model_path = Path('results/cryptomamba_enhanced_2026-06-20_002016/best_cryptomamba_enhanced_model.pth')

# Load data
data_dir = Path('data/processed')
df = pd.read_csv(list(data_dir.glob('*.csv'))[0])
df = generate_har_features(df)

# Create sequences (same as training)
seq_length = 22
forecast_horizon = 5
X_list, y_list = [], []

for i in range(len(df) - seq_length - forecast_horizon):
    X_seq = df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+seq_length].values
    y_target = df['parkinson_volatility'].iloc[i + seq_length + forecast_horizon - 1]
    X_list.append(X_seq)
    y_list.append(y_target)

X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

# Normalize (same as training)
X_norm, y_norm, target_normalizer, feature_stats = normalize_for_training(X, y)

# Split temporally
n = len(X_norm)
test_start = int(0.85 * n)

X_test = torch.FloatTensor(X_norm[test_start:])
y_test = torch.FloatTensor(y_norm[test_start:])

# Create model and load weights
model = create_cryptomamba_model_enhanced()
model.load_state_dict(torch.load(model_path))
model.eval()

print('=== INVESTIGATING TRAINED MODEL PREDICTIONS ===')

# Get predictions
with torch.no_grad():
    predictions_norm = model(X_test)

# Denormalize
predictions = target_normalizer.inverse_transform(predictions_norm.numpy().flatten())
targets = target_normalizer.inverse_transform(y_test.numpy().flatten())

print(f'\nPrediction statistics:')
print(f'  Predictions range: [{predictions.min():.8f}, {predictions.max():.8f}]')
print(f'  Predictions mean: {predictions.mean():.8f}')
print(f'  Predictions std: {predictions.std():.8f}')
print(f'  Unique prediction values: {len(np.unique(predictions))}')

print(f'\nTarget statistics:')
print(f'  Targets range: [{targets.min():.8f}, {targets.max():.8f}]')
print(f'  Targets mean: {targets.mean():.8f}')
print(f'  Targets std: {targets.std():.8f}')
print(f'  Unique target values: {len(np.unique(targets))}')

print(f'\nPrediction variance analysis:')
pred_variance = np.var(predictions)
target_variance = np.var(targets)
print(f'  Prediction variance: {pred_variance:.10f}')
print(f'  Target variance: {target_variance:.10f}')
print(f'  Variance ratio: {pred_variance/target_variance:.4f}')

if pred_variance < 1e-10:
    print(f'  [CRITICAL] Model is predicting constant values!')
elif pred_variance < target_variance * 0.1:
    print(f'  [WARNING] Model predictions have very low variance')
else:
    print(f'  [OK] Model predictions have reasonable variance')

# Check first 20 predictions
print(f'\nFirst 20 predictions vs targets:')
print(f'  Pred      Target    Diff')
print('-' * 35)
for i in range(20):
    print(f'  {predictions[i]:.6f}  {targets[i]:.6f}  {predictions[i]-targets[i]:.6f}')

# Directional accuracy check
actual_changes = np.sign(np.diff(targets))
pred_changes = np.sign(np.diff(predictions))
dir_acc = np.mean(actual_changes == pred_changes) * 100
print(f'\nDirectional accuracy: {dir_acc:.2f}%')
print(f'  Actual changes (first 10): {actual_changes[:10]}')
print(f'  Pred changes (first 10): {pred_changes[:10]}')
print(f'  Agreement: {np.sum(actual_changes[:10] == pred_changes[:10])}/10')
