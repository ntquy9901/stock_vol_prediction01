"""
Constant-baseline sanity check for Parallel LSTM-GNN.

Confirms that pooled R² / DirAcc are inflated by cross-sectional spread,
not temporal skill, by running a constant predictor through the same eval.

The constant predictor outputs 0 on the NORMALIZED scale (StandardScaler
centers each stock to mean 0), which is the per-stock training mean.
After inverse-transforming this equals the per-stock training mean volatility.

Expected result:
- Pooled R²    ≈ 0.70  (high — captures cross-sectional level differences)
- Per-stock R² ≈ 0.00  (near zero — no temporal skill whatsoever)
- Per-stock DirAcc ≈ 50%  (random for direction changes)

Any trained model that is genuinely useful must beat per-stock R² > 0
and per-stock DirAcc > 50%.

Usage (from project root):
    python -m src.lstm_gat_hybrid.sanity_constant_baseline
"""

import sys
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lstm_gat_hybrid.dataset_with_graph_method import (
    create_multi_stock_dataloaders_with_graph_method_fixed,
)
from src.common.evaluation import evaluate_predictions


def _per_stock_metrics(preds_denorm: np.ndarray, targets_denorm: np.ndarray, n_stocks: int):
    """
    Compute per-stock R² and directional accuracy along the TIME axis.

    Reshapes flat array (ordered [window, stock] row-major, matching the
    y.reshape used in the training validate() loop) to [n_windows, n_stocks],
    then evaluates each stock column independently.

    Returns:
        mean_r2 (float), mean_dir_acc (float) — averaged across stocks.
        Returns (None, None) on shape mismatch.
    """
    n_total = len(preds_denorm)
    n_windows = n_total // n_stocks
    if n_windows == 0 or n_total != n_windows * n_stocks:
        print(f"[WARN] Shape mismatch: {n_total} elements, {n_stocks} stocks — skipping per-stock metrics")
        return None, None

    preds_2d = preds_denorm.reshape(n_windows, n_stocks)     # [n_windows, n_stocks]
    targets_2d = targets_denorm.reshape(n_windows, n_stocks)

    r2_list = []
    dir_acc_list = []
    for s in range(n_stocks):
        p_s = preds_2d[:, s]
        t_s = targets_2d[:, s]

        ss_res = np.sum((t_s - p_s) ** 2)
        ss_tot = np.sum((t_s - np.mean(t_s)) ** 2)
        r2_s = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_list.append(r2_s)

        actual_ch = np.sign(np.diff(t_s))
        pred_ch = np.sign(np.diff(p_s))
        dir_acc_s = float(np.mean(actual_ch == pred_ch) * 100)
        dir_acc_list.append(dir_acc_s)

    return float(np.mean(r2_list)), float(np.mean(dir_acc_list))


def main():
    print("=" * 70)
    print("CONSTANT BASELINE SANITY CHECK")
    print("Prediction = 0 on normalized scale (= per-stock training mean)")
    print("=" * 70)

    # Load dataloaders with the same arguments as the training script (knn path)
    print("\nLoading dataloaders (graph_method='knn', this may take a minute)...")
    train_loader, val_loader, test_loader, datasets = \
        create_multi_stock_dataloaders_with_graph_method_fixed(
            data_dir='data/processed',
            seq_length=22,
            forecast_horizon=5,
            graph_method='knn',
            k_neighbors=8,
            batch_size=32,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            num_workers=0,
            normalize=True,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False,
            augmentation_prob=0.3,
            augmentation_factor=0.1,
        )

    test_dataset = datasets[2]
    stock_names = test_dataset.stock_names
    n_stocks = len(stock_names)
    print(f"\nTest dataset: {len(test_dataset)} sequences, {n_stocks} stocks")

    # -------------------------------------------------------------------------
    # Collect normalized targets from the test loader
    # -------------------------------------------------------------------------
    all_targets_norm = []
    for x, adj_matrix, y, _ in test_loader:
        batch_size, num_stocks_batch = y.shape
        y_flat = y.reshape(batch_size * num_stocks_batch).numpy()
        all_targets_norm.extend(y_flat)
    all_targets_norm = np.array(all_targets_norm)

    # Constant prediction: 0 on normalized scale
    all_predictions_norm = np.zeros_like(all_targets_norm)

    print(f"\nNormalized targets : mean={all_targets_norm.mean():.4f}, std={all_targets_norm.std():.4f}")
    print(f"Constant prediction: 0.0 (normalized training mean after StandardScaler)")

    # -------------------------------------------------------------------------
    # Denormalize per-stock
    # Flattening order: element i → stock_idx = i % n_stocks  (row-major reshape)
    # -------------------------------------------------------------------------
    all_predictions_denorm = np.zeros_like(all_predictions_norm)
    all_targets_denorm = np.zeros_like(all_targets_norm)

    for i in range(len(all_predictions_norm)):
        stock_idx = i % n_stocks
        stock_name = stock_names[stock_idx]
        norm = test_dataset.target_normalizers.get(stock_name)
        if norm is not None:
            all_predictions_denorm[i] = norm.inverse_transform(
                all_predictions_norm[i:i + 1].reshape(1, -1)
            ).flatten()[0]
            all_targets_denorm[i] = norm.inverse_transform(
                all_targets_norm[i:i + 1].reshape(1, -1)
            ).flatten()[0]
        else:
            all_predictions_denorm[i] = all_predictions_norm[i]
            all_targets_denorm[i] = all_targets_norm[i]

    print(f"\nDenorm predictions : mean={all_predictions_denorm.mean():.6f}, std={all_predictions_denorm.std():.6f}")
    print(f"Denorm targets     : mean={all_targets_denorm.mean():.6f}, std={all_targets_denorm.std():.6f}")

    # -------------------------------------------------------------------------
    # Pooled metrics — same as validate() uses for reporting
    # (evaluate_predictions prints its own DEBUG lines)
    # -------------------------------------------------------------------------
    print("\n--- Pooled Metrics (flat array, same as validate() reports) ---")
    pooled = evaluate_predictions(all_targets_denorm, all_predictions_denorm)
    print(f"  Pooled R²:     {pooled['r2']:.4f}")
    print(f"  Pooled DirAcc: {pooled['directional_accuracy']:.2f}%")
    print(f"  Pooled RMSE:   {pooled['rmse']:.6f}")
    print(f"  Pooled QLIKE:  {pooled['qlike']:.6f}")

    # -------------------------------------------------------------------------
    # Per-stock metrics — temporal skill
    # -------------------------------------------------------------------------
    r2_mean, dir_acc_mean = _per_stock_metrics(all_predictions_denorm, all_targets_denorm, n_stocks)

    print("\n--- Per-Stock Metrics (temporal skill, averaged across stocks) ---")
    if r2_mean is not None:
        print(f"  Per-stock R²:     {r2_mean:.4f}")
        print(f"  Per-stock DirAcc: {dir_acc_mean:.2f}%")
    else:
        print("  Could not compute (shape mismatch).")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print("\n--- Interpretation ---")
    if r2_mean is not None:
        gap = pooled['r2'] - r2_mean
        if gap > 0.3:
            print(f"  [CONFIRMED] Pooled R²={pooled['r2']:.3f}  >>  Per-stock R²={r2_mean:.3f}  (gap={gap:.3f})")
            print(f"  => Pooled R² is inflated by cross-sectional volatility-level spread.")
            print(f"  => A constant predictor scores this high with ZERO temporal skill.")
            print(f"  => Trained model must achieve per-stock R² > 0 and DirAcc > 50% to add value.")
        else:
            print(f"  Pooled R²={pooled['r2']:.3f}, Per-stock R²={r2_mean:.3f} (gap={gap:.3f})")
            print(f"  Per-stock DirAcc={dir_acc_mean:.2f}% (random baseline ~50%)")
    else:
        print(f"  Pooled R²={pooled['r2']:.3f}, Per-stock metrics unavailable.")

    print("\nDone.")


if __name__ == '__main__':
    main()
