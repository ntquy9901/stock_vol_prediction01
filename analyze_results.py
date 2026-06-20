"""
Analyze CryptoMamba Enhanced V2 results and compare with baselines
"""
import json

# Load results
with open('results/cryptomamba_enhanced_2026-06-20_002016/cryptomamba_enhanced_results.json', 'r') as f:
    results = json.load(f)

print('=== CRYPTO MAMBA ENHANCED V2 RESULTS ===')
print(f'\nTraining summary:')
print(f'  Epochs trained: {results["config"]["num_epochs_trained"]}')
print(f'  Learning rate: {results["config"]["learning_rate"]}')
print(f'  Hidden dim: {results["config"]["hidden_dim"]}')

print(f'\nFinal metrics:')
print(f'  Validation Dir Acc: {results["validation_metrics"]["directional_accuracy"]:.2f}%')
print(f'  Test Dir Acc: {results["test_metrics"]["directional_accuracy"]:.2f}%')
print(f'  Val-Test Gap: {abs(results["val_test_diff"]["dir_acc_diff"]):.2f}%')

print(f'\nDetailed metrics:')
print(f'  Test RMSE: {results["test_metrics"]["rmse"]:.6f}')
print(f'  Test MAE: {results["test_metrics"]["mae"]:.6f}')
print(f'  Test R2: {results["test_metrics"]["r2"]:.6f}')
print(f'  Test QLIKE: {results["test_metrics"]["qlike"]:.6f}')

print(f'\n=== COMPARISON WITH BASELINES ===')
print(f'Model                    Dir Acc    Status')
print('-' * 50)
print(f'LSTM Baseline           48.32%     baseline')
print(f'HAR-R                   51.53%     target')
print(f'CryptoMamba V2          47.78%     overfitting')
print(f'CryptoMamba Enhanced V2 {results["test_metrics"]["directional_accuracy"]:.2f}%     FAILURE')

print(f'\n=== ANALYSIS ===')
test_acc = results['test_metrics']['directional_accuracy']
if test_acc < 10:
    print('[CRITICAL] Model predicting near-constant values')
    print('Dir Acc of 2.20% means model is NOT learning directional patterns')
    print('This is worse than random guessing (50%)')
elif test_acc < 40:
    print('[POOR] Model performing significantly below baselines')
    print(f'Gap to LSTM: {48.32 - test_acc:.2f}%')
else:
    print('[REASONABLE] Model competitive with baselines')

print(f'\n=== POSSIBLE CAUSES ===')
print('1. Model learning to predict constant values (low variance predictions)')
print('2. Architecture mismatch for volatility forecasting')
print('3. Normalization/denormalization issues')
print('4. Insufficient training signal in data')
print('5. Model capacity still insufficient despite 41.7x increase')
