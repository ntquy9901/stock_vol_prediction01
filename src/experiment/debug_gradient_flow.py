"""
Debug script to check if gradients are flowing properly.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("GRADIENT FLOW DEBUG")
print("=" * 80)

# Create small dataset
dataset = HARVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)
train_size = int(0.8 * len(dataset))
train_dataset, _ = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42)
)

# Small batch for debugging
loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2).to(device)

# Test with different learning rates
for lr in [0.00209, 0.001, 0.0001, 0.00001]:
    print(f"\n{'='*80}")
    print(f"Testing with Learning Rate: {lr}")
    print(f"{'='*80}")

    # Reinitialize model
    model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Get one batch
    X_batch, y_batch = next(iter(loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    # Track weight changes
    initial_lstm_hh = model.lstm.weight_hh_l0[0, 0].item()

    # Forward pass
    model.train()
    optimizer.zero_grad()
    predictions = model(X_batch)
    loss = criterion(predictions, y_batch)

    # Backward pass
    loss.backward()

    # Check gradients
    total_params = 0
    zero_grads = 0
    tiny_grads = 0
    large_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_params += 1

            if grad_norm < 1e-10:
                zero_grads += 1
                print(f"  [X] ZERO gradient: {name} (norm={grad_norm:.2e})")
            elif grad_norm < 1e-6:
                tiny_grads += 1
                print(f"  [!] TINY gradient: {name} (norm={grad_norm:.2e})")
            elif grad_norm > 10:
                large_grads += 1
                print(f"  [!] LARGE gradient: {name} (norm={grad_norm:.2e})")

    print(f"\nGradient Statistics:")
    print(f"  Total parameters: {total_params}")
    print(f"  Zero gradients: {zero_grads} ({zero_grads/total_params*100:.1f}%)")
    print(f"  Tiny gradients: {tiny_grads} ({tiny_grads/total_params*100:.1f}%)")
    print(f"  Large gradients: {large_grads} ({large_grads/total_params*100:.1f}%)")

    # Update weights
    optimizer.step()

    # Check weight change
    final_lstm_hh = model.lstm.weight_hh_l0[0, 0].item()
    weight_change = abs(final_lstm_hh - initial_lstm_hh)

    print(f"\nWeight Change Analysis:")
    print(f"  Initial LSTM weight[0,0]: {initial_lstm_hh:.6f}")
    print(f"  Final LSTM weight[0,0]: {final_lstm_hh:.6f}")
    print(f"  Absolute change: {weight_change:.6e}")

    if weight_change < 1e-8:
        print(f"  [X] ERROR: Weights NOT updating! Change too small.")
    elif weight_change < 1e-6:
        print(f"  [!] WARNING: Weights barely updating.")
    else:
        print(f"  [OK] Weights updating normally.")

print(f"\n{'='*80}")
print("DEBUG COMPLETE")
print(f"{'='*80}")
