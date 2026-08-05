"""
No-Graph Ablation for the Parallel LSTM-GNN architecture.

Goal
----
Isolate the contribution of the cross-stock GRAPH STRUCTURE (the spatial/GAT
branch's message-passing) from the contribution of the GAT module's parameters
and capacity. To do that, this script runs the EXACT same architecture, data,
hyperparameters and training protocol as the existing HAR-only backbone runs
produced by ``src/lstm_gat_hybrid/train_parallel_enhanced.py`` (graph_method
='knn', 20 epochs, lr=0.001, batch_size=11, 3 seeds 42/123/2026), but replaces
the real adjacency matrix that would be fed to the model on every forward call
with an IDENTITY matrix (self-loops only, no cross-stock edges).

With an identity adjacency the GAT layers can only attend to each node's own
features (off-diagonal entries are masked to -inf before softmax), so the
spatial branch degenerates to a per-node learned transform with the SAME
parameter count as the real model. This is the standard, correct way to ask
"does the cross-stock graph help?" without confounding it with "does the extra
GAT capacity help?".

Isolation
---------
This is a NEW, self-contained script. It does NOT modify model_parallel.py,
train_parallel_enhanced.py, dataset_with_graph_method.py or any shared code —
it only READS/imports them. The adjacency substitution happens locally inside
this script's own training/validation loops, right after unpacking each batch,
before ``model(x, adj_matrix)`` is called.

Run
---
    python scripts/ablation_no_graph/run_no_graph_ablation.py --seeds 42 123 2026
    python scripts/ablation_no_graph/run_no_graph_ablation.py --seeds 42   # single-seed fallback
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Bootstrap sys.path (project root) so `src...` imports resolve regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from src.lstm_gat_hybrid.dataset_with_graph_method import (  # noqa: E402
    create_multi_stock_dataloaders_with_graph_method_fixed,
)
from src.common.evaluation import evaluate_predictions  # noqa: E402
from src.common.provenance import get_provenance  # noqa: E402


def make_identity_adj(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Return a batched identity adjacency with the SAME shape/device/dtype as
    ``adj_matrix`` (i.e. [batch, num_stocks, num_stocks]).

    This is the single point where the graph is removed: identity = self-loops
    only, zero cross-stock edges. The GAT layer masks off-diagonal (==0) entries
    to -inf before softmax, so each node attends only to itself.
    """
    batch_size, num_stocks, _ = adj_matrix.shape
    eye = torch.eye(num_stocks, device=adj_matrix.device, dtype=adj_matrix.dtype)
    return eye.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def verify_graph_removed(model, dataloader, device):
    """Strong, functional verification that the identity substitution genuinely
    disables cross-stock message passing.

    Two checks on the first batch:
      1. Shape/content: the substituted adjacency is exactly a batched identity.
      2. Invariance: with identity adjacency, a node's GNN embedding is
         INVARIANT to other stocks' input features (perturbing every other
         stock leaves stock 0's GNN embedding unchanged). With the REAL
         adjacency the same perturbation DOES change stock 0's embedding —
         proving the substitution is what removes message passing, not a bug
         that silently leaves the real graph in place.
    """
    model.eval()
    x, adj_matrix, y, _ = next(iter(dataloader))
    x = x.to(device)
    adj_matrix = adj_matrix.to(device)

    identity_adj = make_identity_adj(adj_matrix)

    print("\n" + "=" * 78)
    print("SANITY CHECK: identity-adjacency substitution")
    print("=" * 78)
    print(f"  Real adj shape: {tuple(adj_matrix.shape)}  "
          f"mean={adj_matrix.mean():.4f} min={adj_matrix.min():.4f} max={adj_matrix.max():.4f}")
    print(f"  Real adj off-diagonal nonzeros (batch 0): "
          f"{int((adj_matrix[0] != 0).sum() - torch.diagonal(adj_matrix[0]).count_nonzero())}")
    print(f"  Identity adj shape: {tuple(identity_adj.shape)}  "
          f"mean={identity_adj.mean():.4f} min={identity_adj.min():.4f} max={identity_adj.max():.4f}")

    num_stocks = identity_adj.shape[1]
    expected_eye = torch.eye(num_stocks, device=device, dtype=identity_adj.dtype)
    exact_identity = torch.allclose(identity_adj[0], expected_eye)
    print(f"  identity_adj[0] == torch.eye({num_stocks}): {exact_identity}")
    assert exact_identity, "Identity substitution produced a non-identity matrix!"
    print(f"  identity_adj[0] first 4x4 block:\n{identity_adj[0][:4, :4].cpu().numpy()}")

    # Functional invariance check.
    with torch.no_grad():
        _, h_gnn_id = model.get_embeddings(x, identity_adj)
        _, h_gnn_real = model.get_embeddings(x, adj_matrix)

        x_pert = x.clone()
        # Perturb every stock EXCEPT stock 0.
        gen = torch.Generator(device=device) if device.type == "cpu" else torch.Generator()
        noise = torch.randn(x_pert[:, :, 1:, :].shape, device=device)
        x_pert[:, :, 1:, :] = x_pert[:, :, 1:, :] + 5.0 * noise

        _, h_gnn_id_pert = model.get_embeddings(x_pert, identity_adj)
        _, h_gnn_real_pert = model.get_embeddings(x_pert, adj_matrix)

    id_stock0_delta = (h_gnn_id[:, 0, :] - h_gnn_id_pert[:, 0, :]).abs().max().item()
    real_stock0_delta = (h_gnn_real[:, 0, :] - h_gnn_real_pert[:, 0, :]).abs().max().item()

    print(f"\n  [Invariance] Perturbing all OTHER stocks' inputs, watching stock 0's GNN embedding:")
    print(f"    identity adj -> stock 0 max |delta| = {id_stock0_delta:.3e}  (expect ~0: no message passing)")
    print(f"    REAL adj     -> stock 0 max |delta| = {real_stock0_delta:.3e}  (expect >0: message passing on)")

    invariant = id_stock0_delta < 1e-5
    real_leaks = real_stock0_delta > 1e-5
    print(f"    identity-adj embedding invariant to other stocks: {invariant}")
    print(f"    real-adj embedding sensitive to other stocks:     {real_leaks}")
    assert invariant, (
        "identity-adj GNN embedding still depends on other stocks — substitution is NOT taking effect!")
    print("  SANITY CHECK PASSED: cross-stock message passing is genuinely disabled.\n")
    print("=" * 78 + "\n")
    return {
        "identity_exact": bool(exact_identity),
        "identity_stock0_delta": float(id_stock0_delta),
        "real_stock0_delta": float(real_stock0_delta),
        "invariant_under_identity": bool(invariant),
        "sensitive_under_real": bool(real_leaks),
    }


def train_epoch(model, dataloader, criterion, optimizer, device, config, verbose_first=False):
    """One training epoch. Adapted from train_parallel_enhanced.train_epoch, with
    the ONLY functional change being: adj_matrix -> identity right after unpack."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, adj_matrix, y, _ in dataloader:
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)
        adj_matrix = make_identity_adj(adj_matrix)  # <-- NO-GRAPH ABLATION
        y = y.to(device)

        if verbose_first and n_batches == 0:
            print(f"    [DEBUG train first batch] adj (identity) mean={adj_matrix.mean():.6f}, "
                  f"diag_sum={torch.diagonal(adj_matrix[0]).sum():.1f}, "
                  f"offdiag_nonzero={int((adj_matrix[0] != 0).sum() - torch.diagonal(adj_matrix[0]).count_nonzero())}")

        batch_size, num_stocks = y.shape
        y_flat = y.reshape(batch_size * num_stocks)

        optimizer.zero_grad()
        predictions = model(x, adj_matrix)

        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"    WARNING: model output NaN/Inf at batch {n_batches + 1}, skipping")
            continue

        predictions_flat = predictions.reshape(batch_size * num_stocks)
        loss = criterion(predictions_flat, y_flat)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: loss NaN/Inf at batch {n_batches + 1}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    print(f"  Average Train Loss: {avg_loss:.6f}", flush=True)
    return avg_loss


def validate(model, dataloader, criterion, device, dataset=None):
    """Validation. Adapted from train_parallel_enhanced.validate. Same inverse-
    transform / per-ticker DirAcc logic; only change: identity adjacency."""
    model.eval()
    all_predictions_norm = []
    all_targets_norm = []

    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            x = x.to(device)
            adj_matrix = adj_matrix.to(device)
            adj_matrix = make_identity_adj(adj_matrix)  # <-- NO-GRAPH ABLATION
            y = y.to(device)

            batch_size, num_stocks = y.shape
            y_flat = y.reshape(batch_size * num_stocks)

            predictions = model(x, adj_matrix)
            predictions_flat = predictions.reshape(batch_size * num_stocks)

            all_predictions_norm.extend(predictions_flat.cpu().numpy())
            all_targets_norm.extend(y_flat.cpu().numpy())

    all_predictions_norm = np.array(all_predictions_norm).flatten()
    all_targets_norm = np.array(all_targets_norm).flatten()

    actual_dataset = dataset
    if dataset is not None and hasattr(dataset, "dataset"):
        actual_dataset = dataset.dataset

    if actual_dataset is not None and hasattr(actual_dataset, "target_normalizers"):
        all_predictions_denorm = np.zeros_like(all_predictions_norm)
        all_targets_denorm = np.zeros_like(all_targets_norm)

        n_stocks = len(actual_dataset.stock_names)
        for i in range(len(all_predictions_norm)):
            stock_name = actual_dataset.stock_names[i % n_stocks]
            if stock_name in actual_dataset.target_normalizers:
                norm = actual_dataset.target_normalizers[stock_name]
                all_predictions_denorm[i] = norm.inverse_transform(
                    all_predictions_norm[i:i + 1].reshape(1, -1)).flatten()[0]
                all_targets_denorm[i] = norm.inverse_transform(
                    all_targets_norm[i:i + 1].reshape(1, -1)).flatten()[0]
            else:
                all_predictions_denorm[i] = all_predictions_norm[i]
                all_targets_denorm[i] = all_targets_norm[i]

        loss_tensor_norm = criterion(
            torch.FloatTensor(all_predictions_norm).to(device),
            torch.FloatTensor(all_targets_norm).to(device),
        )
        avg_loss = loss_tensor_norm.item()
        metrics = evaluate_predictions(all_targets_denorm, all_predictions_denorm, n_stocks=n_stocks)
    else:
        loss_tensor = criterion(
            torch.FloatTensor(all_predictions_norm).to(device),
            torch.FloatTensor(all_targets_norm).to(device),
        )
        avg_loss = loss_tensor.item()
        # num_stocks from last batch loop
        metrics = evaluate_predictions(all_targets_norm, all_predictions_norm, n_stocks=num_stocks)

    return avg_loss, metrics


def build_config():
    """Replicate the exact non-quick-test / epochs=20 config used by
    train_parallel_enhanced.train_parallel_lstm_gat_enhanced(epochs=20)."""
    config = LSTMGATConfig()
    config.learning_rate = 0.001
    config.batch_size = 11
    # epochs=20 branch: patience=min_epochs=epochs (no early stopping)
    config.num_epochs = 20
    config.patience = 20
    config.min_epochs = 20
    config.weight_decay = 1e-5
    config.gradient_clip = 0.5
    config.lstm_dropout = 0.2
    config.fusion_dropout = 0.15
    return config


def run_one_seed(seed, run_sanity_check=False):
    print("=" * 80)
    print(f"NO-GRAPH ABLATION (identity adjacency) - seed {seed}")
    print("=" * 80)
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = build_config()
    device = torch.device(config.device)

    # Dataloaders: IDENTICAL call to the HAR-only backbone run (graph_method='knn').
    # The real knn adjacency is still built inside the datasets, but we overwrite it
    # with identity in every train/val/test forward call below.
    train_loader, val_loader, test_loader, datasets = \
        create_multi_stock_dataloaders_with_graph_method_fixed(
            data_dir="data/processed",
            seq_length=config.seq_length,
            forecast_horizon=config.forecast_horizon,
            graph_method="knn",
            graph_threshold=None,
            k_neighbors=8,
            batch_size=config.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            num_workers=config.num_workers,
            normalize=True,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=True,
            augmentation_prob=0.15,
            augmentation_factor=0.1,
        )

    model = create_parallel_lstm_gat_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    sanity = None
    if run_sanity_check:
        # Use val_loader for the sanity check (deterministic, no augmentation noise
        # confounding the invariance test).
        sanity = verify_graph_removed(model, val_loader, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = PROJECT_ROOT / f"results/no_graph_ablation_seed{seed}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses, val_losses = [], []
    epoch_times = []

    for epoch in range(config.num_epochs):
        t0 = time.time()
        print(f"\n[Epoch {epoch + 1}/{config.num_epochs}]", flush=True)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config,
                                 verbose_first=(epoch == 0))
        train_losses.append(train_loss)

        val_loss, val_metrics = validate(model, val_loader, criterion, device, datasets[1])
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} "
              f"DirAcc={val_metrics['directional_accuracy']:.2f}% RMSE={val_metrics['rmse']:.6f} "
              f"lr={current_lr:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / "best_no_graph_model.pth")

        epoch_times.append(time.time() - t0)

    print(f"\nTraining done. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
    print(f"Total training time: {sum(epoch_times) / 60:.1f} min")

    model.load_state_dict(torch.load(results_dir / "best_no_graph_model.pth"))
    test_loss, test_metrics = validate(model, test_loader, criterion, device, datasets[2])

    print("\nTest Results (no-graph / identity adjacency):")
    for k in ["mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]:
        print(f"  {k}: {test_metrics[k]:.6f}")

    results = {
        "model": "Parallel LSTM-GNN (NO-GRAPH ablation, identity adjacency)",
        "timestamp": timestamp,
        "seed": seed,
        "architecture": "Parallel (LSTM temporal + GNN spatial) -> Concatenation fusion",
        "graph_method": "identity (no cross-stock edges) — ablation of knn backbone",
        "ablation": {
            "type": "no_graph_identity_adjacency",
            "reference_run": "graph_method='knn', train_parallel_enhanced.py, 20 epochs",
            "description": (
                "Same architecture/data/hyperparameters/protocol as the HAR-only knn "
                "backbone; adjacency replaced with torch.eye at every forward call so the "
                "GAT branch cannot pass messages across stocks (per-node transform only, "
                "same parameter count)."),
            "sanity_check": sanity,
        },
        "config": {
            "num_stocks": config.num_stocks,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs_trained": config.num_epochs,
            "best_epoch": best_epoch,
            "patience": config.patience,
            "weight_decay": config.weight_decay,
        },
        "training_summary": {
            "num_epochs_trained": config.num_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "total_time_minutes": float(sum(epoch_times) / 60),
        },
        "test_metrics": {
            "mse": float(test_metrics["mse"]),
            "rmse": float(test_metrics["rmse"]),
            "mae": float(test_metrics["mae"]),
            "r2": float(test_metrics["r2"]),
            "qlike": float(test_metrics["qlike"]),
            "directional_accuracy": float(test_metrics["directional_accuracy"]),
        },
        "provenance": get_provenance(),
    }

    with open(results_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {results_dir / 'training_results.json'}")
    print(f"Finished at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return results


def main():
    parser = argparse.ArgumentParser(description="No-graph (identity adjacency) ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Seeds to run (default: 42 123 2026, matching the reference protocol)")
    args = parser.parse_args()

    all_results = []
    for i, seed in enumerate(args.seeds):
        # Run the (relatively expensive) sanity check on the first seed only.
        res = run_one_seed(seed, run_sanity_check=(i == 0))
        all_results.append(res)

    print("\n" + "=" * 80)
    print("NO-GRAPH ABLATION SUMMARY")
    print("=" * 80)
    for res in all_results:
        tm = res["test_metrics"]
        print(f"  seed {res['seed']}: QLIKE={tm['qlike']:.4f} RMSE={tm['rmse']:.6f} "
              f"MAE={tm['mae']:.7f} R2={tm['r2']:.4f} DirAcc={tm['directional_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
