"""
Train Parallel LSTM-GNN on S&P 500 (30-ticker subset), multi-horizon.

Reuses src/lstm_gat_hybrid/{model_parallel,dataset,config}.py UNCHANGED
(read-only import, per CLAUDE.md SS3.F.3 hard isolation) -- only the data
directory and forecast horizon differ from the VN30 baseline
(src/lstm_gat_hybrid/train_parallel.py), see design/design.md.

Usage:
    python baselines/2026-08-01_lstm_gnn_sp500_baseline/code/train_sp500_lstm_gnn.py --forecast_horizon 5
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Bootstrap path: baseline folder name has dashes -> not importable as a
# package (CLAUDE.md SS3.F.4). code/ -> baseline_dir -> baselines -> project_root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
for _ in range(3):
    project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.dataset import create_multi_stock_dataloaders
from src.lstm_gat_hybrid.train_parallel import plot_learning_curves
from src.common.evaluation import evaluate_predictions_grouped


def set_seed(seed: int):
    """Fix RNG state for reproducibility (project-context.md Research Best Practices).

    src/lstm_gat_hybrid's model init and dropout draw from the global torch/
    numpy/random RNGs; without this, two runs with identical config produce
    unrelated weight initializations and wildly different metrics after only
    a few epochs (observed: test R^2 ranging from 0.9999 to -777 across 4
    unseeded 2-epoch smoke runs on 2026-08-01).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_new_best(current_dir_acc: float, best_dir_acc: float) -> bool:
    """Checkpoint selection criterion: maximize val directional_accuracy, NOT
    minimize val loss (MSE).

    Found via a real training run: val_loss (normalized MSE) can keep
    decreasing every epoch even as val DirAcc collapses toward 0% from a
    prediction-collapse overfit (train loss -> 0) -- MSE loss is blind to
    directional accuracy, so "best by lowest val_loss" silently picks a
    degenerate checkpoint instead of the actual best epoch.
    """
    return current_dir_acc > best_dir_acc


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, adj_matrix, y, _ in dataloader:
        x, adj_matrix, y = x.to(device), adj_matrix.to(device), y.to(device)
        batch_size, num_stocks = y.shape

        optimizer.zero_grad()
        predictions = model(x, adj_matrix)
        loss = criterion(predictions.reshape(-1), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, criterion, device, stock_names, target_normalizers):
    """Evaluate a dataloader, computing directional_accuracy PER STOCK
    (chronological order within each stock across batches), on the ORIGINAL
    (inverse-transformed) volatility scale.

    y/predictions have shape [batch, num_stocks] per batch -- one row per
    calendar window-start, one column per stock. Flattening this to 1-D
    before computing directional accuracy (as src/lstm_gat_hybrid's own
    train_parallel.py does) interleaves different stocks' values on the
    SAME day; np.diff() on that flattened sequence compares unrelated
    stocks, not a stock against its own next day -- a real bug, found while
    smoke-testing this new baseline (see test_validate_grouped_diracc.py).

    Separately, MultiStockDataset returns z-score NORMALIZED y (mean 0,
    std 1, can be negative). QLIKE is only meaningful on the original
    positive volatility scale -- computing it directly on normalized values
    lets qlike_loss()'s epsilon-clipping blow up the ratio term (observed:
    QLIKE ~19620 in an early smoke run). Each stock's true/pred are
    inverse_transform()-ed with ITS OWN target_normalizers[stock] before
    metrics are computed.

    Args:
        stock_names: list of stock names, index-aligned with y's column
            (stock) dimension -- MultiStockDataset sorts stock_names
            deterministically, same order for train/val/test instances.
        target_normalizers: dict stock_name -> object with
            .inverse_transform(np.ndarray) -> np.ndarray (VolatilityNormalizer).
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    per_stock_true, per_stock_pred = None, None

    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            x, adj_matrix, y = x.to(device), adj_matrix.to(device), y.to(device)
            predictions = model(x, adj_matrix)
            loss = criterion(predictions.reshape(-1), y.reshape(-1))

            total_loss += loss.item()
            n_batches += 1

            batch_size, num_stocks = y.shape
            if per_stock_true is None:
                per_stock_true = [[] for _ in range(num_stocks)]
                per_stock_pred = [[] for _ in range(num_stocks)]

            y_np = y.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            for stock_idx in range(num_stocks):
                per_stock_true[stock_idx].extend(y_np[:, stock_idx])
                per_stock_pred[stock_idx].extend(pred_np[:, stock_idx])

    avg_loss = total_loss / max(n_batches, 1)

    true_groups, pred_groups = [], []
    for stock_idx, stock_name in enumerate(stock_names):
        normalizer = target_normalizers[stock_name]
        true_arr = np.array(per_stock_true[stock_idx])
        pred_arr = np.array(per_stock_pred[stock_idx])
        true_groups.append(normalizer.inverse_transform(true_arr))
        pred_groups.append(normalizer.inverse_transform(pred_arr))

    metrics = evaluate_predictions_grouped(true_groups, pred_groups)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_horizon", type=int, default=5, choices=[1, 5, 10, 22])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", default=os.path.join(project_root, "data", "processed_sp500"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config = LSTMGATConfig()
    config.forecast_horizon = args.forecast_horizon
    config.num_epochs = args.epochs
    config.patience = args.epochs  # no early stop within a capped 10-epoch experimentation run

    device = torch.device(config.device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join(project_root, "results", f"lstm_gnn_sp500_h{args.forecast_horizon}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Forecast horizon: {args.forecast_horizon} day(s) ahead")
    print(f"[INFO] Data dir: {args.data_dir}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Device: {device}")

    train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders(
        data_dir=args.data_dir,
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
        graph_method=config.graph_method,
        batch_size=config.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=config.num_workers,
        normalize=True,
        remove_outliers=True,
        n_std=3.0,
        data_augmentation=False,
    )

    train_dataset, val_dataset, test_dataset = datasets
    val_stock_names = val_dataset.dataset.stock_names
    val_normalizers = val_dataset.dataset.target_normalizers
    test_stock_names = test_dataset.dataset.stock_names
    test_normalizers = test_dataset.dataset.target_normalizers

    model = create_parallel_lstm_gat_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_losses, val_losses = [], []
    best_val_dir_acc = float("-inf")
    best_epoch = 0

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, val_stock_names, val_normalizers)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - Val DirAcc: {val_metrics['directional_accuracy']:.2f}%")

        if is_new_best(val_metrics["directional_accuracy"], best_val_dir_acc):
            best_val_dir_acc = val_metrics["directional_accuracy"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs:
            plot_learning_curves(train_losses, val_losses,
                                  os.path.join(results_dir, f"learning_curve_epoch{epoch+1}.png"))

    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    test_loss, test_metrics = validate(model, test_loader, criterion, device, test_stock_names, test_normalizers)

    print(f"\n[TEST] DirAcc: {test_metrics['directional_accuracy']:.2f}% - "
          f"RMSE: {test_metrics['rmse']:.6f} - QLIKE: {test_metrics['qlike']:.6f}")
    print(f"[INFO] Best checkpoint: epoch {best_epoch} (val DirAcc {best_val_dir_acc:.2f}%)")

    results = {
        "architecture": "ParallelLSTMGNN (src/lstm_gat_hybrid, unmodified)",
        "market": "sp500",
        "num_stocks": len(datasets[0].dataset.stock_names),
        "forecast_horizon": args.forecast_horizon,
        "epochs_trained": config.num_epochs,
        "best_epoch": best_epoch,
        "best_val_dir_acc": float(best_val_dir_acc),
        "seed": args.seed,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {results_dir}")


if __name__ == "__main__":
    main()
