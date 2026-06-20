"""
Debug script to check if validation loss is recalculated every epoch.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("VALIDATION LOSS DEBUG")
print("=" * 80)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Create dataset
dataset = HARVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type=='cuda')

# Initialize model
model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2).to(device)
criterion = nn.MSELoss()

# Get one batch and check if it's the same every time
print("\n" + "=" * 80)
print("CHECK 1: Is validation data the same every epoch?")
print("=" * 80)

# Get first batch from test_loader
batch1 = next(iter(test_loader))
X1, y1 = batch1

# Recreate test_loader and get first batch again
test_loader2 = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
batch2 = next(iter(test_loader2))
X2, y2 = batch2

print(f"Batch 1 X mean: {X1.mean().item():.6f}, std: {X1.std().item():.6f}")
print(f"Batch 2 X mean: {X2.mean().item():.6f}, std: {X2.std().item():.6f}")
print(f"X tensors identical: {torch.allclose(X1, X2)}")
print(f"Y tensors identical: {torch.allclose(y1, y2)}")

# Now test validation loss over multiple epochs
print("\n" + "=" * 80)
print("CHECK 2: Does val loss change with random weight initialization?")
print("=" * 80)

val_losses = []
for epoch in range(5):
    # Reinitialize model with different weights each time
    model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2).to(device)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} (random init): Val Loss = {val_loss:.6f}")

print(f"\nVal loss std dev: {np.std(val_losses):.8f}")
print(f"Val loss range: [{min(val_losses):.6f}, {max(val_losses):.6f}]")

if np.std(val_losses) < 1e-6:
    print("[X] ERROR: Val loss is TOO CONSISTENT across random initializations!")
    print("    This suggests val loss calculation might be broken.")
else:
    print("[OK] Val loss varies appropriately across different initializations.")

# Now test with training (weights should update)
print("\n" + "=" * 80)
print("CHECK 3: Does val loss change after training?")
print("=" * 80)

model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Calculate initial val loss
model.eval()
val_loss_initial = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        val_loss_initial += loss.item()
val_loss_initial /= len(test_loader)

print(f"Initial val loss: {val_loss_initial:.6f}")

# Train for 1 epoch
model.train()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    predictions = model(X_batch)
    loss = criterion(predictions, y_batch)
    loss.backward()
    optimizer.step()
    break  # Just one batch for speed

# Calculate val loss after training
model.eval()
val_loss_after = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        val_loss_after += loss.item()
val_loss_after /= len(test_loader)

print(f"Val loss after 1 batch training: {val_loss_after:.6f}")
print(f"Difference: {abs(val_loss_after - val_loss_initial):.8f}")

if abs(val_loss_after - val_loss_initial) < 1e-8:
    print("[X] ERROR: Val loss didn't change after training!")
else:
    print(f"[OK] Val loss changed after training.")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
