"""
Fixed validate() function for train_parallel_enhanced.py

This file contains the corrected validate() function that computes
loss on ORIGINAL scale (consistent with test metrics).

To apply: Replace the validate() function in train_parallel_enhanced.py
(Line 199-281) with this version.
"""

import torch
import numpy as np
from src.common.evaluation import evaluate_predictions


def validate(model, dataloader, criterion, device, dataset=None):
    """
    Validate model with loss computed on ORIGINAL scale (consistent with test)

    FIXED: Inverse transform predictions BEFORE computing loss to ensure
    Val Loss and Test MSE are on the same scale for proper comparison.

    Key changes:
    1. Collect all normalized predictions and targets
    2. Inverse transform to original scale
    3. Compute loss on ORIGINAL scale (not normalized)
    4. Compute metrics on ORIGINAL scale

    This ensures Val Loss and Test MSE are comparable.
    """
    model.eval()
    n_batches = 0

    all_predictions_norm = []
    all_targets_norm = []

    # ========================================================================
    # Step 1: Collect all normalized predictions and targets
    # ========================================================================
    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            x = x.to(device)
            adj_matrix = adj_matrix.to(device)
            y = y.to(device)

            batch_size, num_stocks = y.shape
            y_flat = y.reshape(batch_size * num_stocks)

            predictions = model(x, adj_matrix)
            predictions_flat = predictions.reshape(batch_size * num_stocks)

            # Collect normalized predictions and targets
            all_predictions_norm.extend(predictions_flat.cpu().numpy())
            all_targets_norm.extend(y_flat.cpu().numpy())

            n_batches += 1

    # Convert to numpy arrays
    all_predictions_norm = np.array(all_predictions_norm).flatten()
    all_targets_norm = np.array(all_targets_norm).flatten()

    # ========================================================================
    # DEBUGGING: Check normalization status
    # ========================================================================
    print(f"[DEBUG validate] Before inverse transform:")
    print(f"  predictions_norm mean: {all_predictions_norm.mean():.6f}")
    print(f"  predictions_norm std:  {all_predictions_norm.std():.6f}")
    print(f"  predictions_norm range: [{all_predictions_norm.min():.6f}, {all_predictions_norm.max():.6f}]")
    print(f"  targets_norm mean: {all_targets_norm.mean():.6f}")
    print(f"  targets_norm std:  {all_targets_norm.std():.6f}")
    print(f"  targets_norm range: [{all_targets_norm.min():.6f}, {all_targets_norm.max():.6f}]")

    # ========================================================================
    # Step 2: Extract dataset (handle Subset wrapper)
    # ========================================================================
    actual_dataset = dataset
    if dataset is not None and hasattr(dataset, 'dataset'):
        actual_dataset = dataset.dataset
        print(f"[DEBUG validate] Extracted original dataset from Subset")
        print(f"[DEBUG validate] Original dataset type: {type(actual_dataset).__name__}")
        print(f"[DEBUG validate] Has target_normalizers: {hasattr(actual_dataset, 'target_normalizers')}")

    # ========================================================================
    # Step 3: Inverse transform to original scale
    # ========================================================================
    if actual_dataset is not None and hasattr(actual_dataset, 'target_normalizers'):
        print(f"[DEBUG validate] Applying inverse transform to {len(all_predictions_norm)} predictions...")

        all_predictions_denorm = np.zeros_like(all_predictions_norm)
        all_targets_denorm = np.zeros_like(all_targets_norm)  # ✅ BUG FIX: was zeros_like(all_predictions)

        # Denormalize per-stock predictions
        for i in range(len(all_predictions_norm)):
            stock_idx = i % len(actual_dataset.stock_names)
            stock_name = actual_dataset.stock_names[stock_idx]

            if stock_name in actual_dataset.target_normalizers:
                # Denormalize prediction (reshape to 2D, then flatten)
                all_predictions_denorm[i] = \
                    actual_dataset.target_normalizers[stock_name].inverse_transform(
                        all_predictions_norm[i:i+1].reshape(1, -1)
                    ).flatten()[0]
                # Denormalize target (reshape to 2D, then flatten)
                all_targets_denorm[i] = \
                    actual_dataset.target_normalizers[stock_name].inverse_transform(
                        all_targets_norm[i:i+1].reshape(1, -1)
                    ).flatten()[0]
            else:
                all_predictions_denorm[i] = all_predictions_norm[i]
                all_targets_denorm[i] = all_targets_norm[i]

        print(f"[DEBUG validate] After inverse transform:")
        print(f"  predictions_denorm range: [{all_predictions_denorm.min():.6f}, {all_predictions_denorm.max():.6f}]")
        print(f"  targets_denorm range: [{all_targets_denorm.min():.6f}, {all_targets_denorm.max():.6f}]")

        # ========================================================================
        # Step 4: ✅ FIX: Compute loss on ORIGINAL scale (consistent with test)
        # ========================================================================
        print(f"[DEBUG validate] Computing loss on ORIGINAL scale...")
        loss_tensor = criterion(
            torch.FloatTensor(all_predictions_denorm).to(device),
            torch.FloatTensor(all_targets_denorm).to(device)
        )
        avg_loss = loss_tensor.item()

        # Compute metrics on denormalized data
        metrics = evaluate_predictions(all_targets_denorm, all_predictions_denorm)

        print(f"[DEBUG validate] Loss computed on ORIGINAL scale: {avg_loss:.6f}")
    else:
        # ========================================================================
        # FALLBACK: Compute on normalized scale (WARNING: not consistent with test!)
        # ========================================================================
        print(f"[WARNING validate] No inverse transform applied! Computing loss on NORMALIZED scale!")
        print(f"[WARNING validate] This will cause Val Loss and Test MSE to be on DIFFERENT scales!")

        loss_tensor = criterion(
            torch.FloatTensor(all_predictions_norm).to(device),
            torch.FloatTensor(all_targets_norm).to(device)
        )
        avg_loss = loss_tensor.item()

        # Compute metrics on normalized scale
        metrics = evaluate_predictions(all_targets_norm, all_predictions_norm)

        print(f"[DEBUG validate] Loss computed on NORMALIZED scale: {avg_loss:.6f}")

    return avg_loss, metrics


# ========================================================================
# USAGE INSTRUCTIONS
# ========================================================================
"""
To apply this fix:

1. Open: src/lstm_gat_hybrid/train_parallel_enhanced.py
2. Locate: Line 199-281 (validate() function)
3. Replace: Entire function with this version
4. Save: File

Or use this sed command (from project root):
sed -i '199,281d' src/lstm_gat_hybrid/train_parallel_enhanced.py
sed -i '198r src/lstm_gat_hybrid/validate_fixed.py' src/lstm_gat_hybrid/train_parallel_enhanced.py

Expected results after fix:
- Val Loss: ~1e-4 to ~1e-3 (original scale)
- Test MSE: ~1e-6 to ~1e-5 (original scale)
- Both on same scale → comparable!
"""
