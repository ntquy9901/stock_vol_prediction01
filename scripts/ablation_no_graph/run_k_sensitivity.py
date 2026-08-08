"""
k-NN k sensitivity check for the price-only (HAR-only) Parallel LSTM-GNN backbone.

Goal
----
Tonight's no-graph ablation (`scripts/ablation_no_graph/run_no_graph_ablation.py`,
`docs/reports/2026-08-05_graph_ablation_results.md`) found that removing the cross-stock
graph entirely (identity adjacency) produces no statistically significant change vs. the
real k=8 k-NN graph. That was measured at k=8 only. This script asks whether that null
result is specific to k=8: it re-runs the SAME price-only backbone training protocol
(`train_parallel_enhanced.py`, `graph_method='knn'`, 20 epochs, real k-NN graph) at other
k values — k=4 (sparser) and k=16 (denser) — so the k=8 vs no-graph null can be read
against a small k sweep.

How k is varied (isolation)
---------------------------
`config.top_k_neighbors` is what actually controls the k-NN degree. It is read inside
`graph_utils_fixed.DynamicGraphBuilder.build_correlation_graph` as
`self.config.top_k_neighbors`. IMPORTANT: the `k_neighbors=` argument accepted by
`create_multi_stock_dataloaders_with_graph_method_fixed` is stored on the dataset but is
NOT used for graph construction — that wrapper builds its OWN internal `LSTMGATConfig()`
(dataset_with_graph_method.py) and the graph builder reads `config.top_k_neighbors` from
it. So the ONLY lever for k is `LSTMGATConfig.top_k_neighbors`.

To change it without editing `src/lstm_gat_hybrid/config.py` on disk (which would affect
every other concurrent/future run reading the default config), this script patches the
class attribute `LSTMGATConfig.top_k_neighbors` IN THIS PROCESS ONLY, before the wrapper
instantiates its internal config. This is an in-memory, per-process override: other
training runs are separate Python processes that import config.py fresh from disk and
still get the default k=8. No shared file is modified. It is the process-level equivalent
of the "set it on the config instance" override the task asked for (the wrapper does not
expose a config parameter to set it on directly).

Everything else (data, split, normalization, augmentation, model, hyperparameters, 20
epochs, seed) is identical to the reference k=8 run. Strict one-variable-changed
comparison.

Run
---
    python scripts/ablation_no_graph/run_k_sensitivity.py --k 4 --seed 42
    python scripts/ablation_no_graph/run_k_sensitivity.py --k 16 --seed 42
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

from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from src.lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model  # noqa: E402
from src.lstm_gat_hybrid.dataset_with_graph_method import (  # noqa: E402
    create_multi_stock_dataloaders_with_graph_method_fixed,
)
# Reuse the EXACT reference training-loop logic (real adjacency, no substitution).
from src.lstm_gat_hybrid.train_parallel_enhanced import (  # noqa: E402
    train_epoch,
    validate,
)
from src.common.provenance import get_provenance  # noqa: E402


def build_config(k: int) -> LSTMGATConfig:
    """Replicate the exact epochs=20 config used by the reference k=8 run
    (train_parallel_enhanced.train_parallel_lstm_gat_enhanced(epochs=20)), with
    top_k_neighbors set to the requested k on the instance for logging/consistency."""
    config = LSTMGATConfig()
    config.top_k_neighbors = k
    config.learning_rate = 0.001
    config.batch_size = 11
    config.num_epochs = 20
    config.patience = 20
    config.min_epochs = 20
    config.weight_decay = 1e-5
    config.gradient_clip = 0.5
    config.lstm_dropout = 0.2
    config.fusion_dropout = 0.15
    return config


def count_offdiag_nonzeros(adj_batch0: torch.Tensor) -> int:
    """Off-diagonal non-zero entries in a single [num_stocks, num_stocks] adjacency.

    Matches the exact counting convention used in the graph ablation report
    (`docs/reports/2026-08-05_graph_ablation_results.md` §2.1), which reported 402 for
    the real k=8 graph on the same batch."""
    total_nonzero = int((adj_batch0 != 0).sum())
    diag_nonzero = int(torch.diagonal(adj_batch0).count_nonzero())
    return total_nonzero - diag_nonzero


def verify_k_edges(val_loader, k: int) -> dict:
    """Pull the first val batch and report the constructed k-NN edge count so we can
    confirm k actually changed the graph density (sanity check the lever works)."""
    x, adj_matrix, y, _ = next(iter(val_loader))
    num_stocks = adj_matrix.shape[1]
    offdiag = count_offdiag_nonzeros(adj_matrix[0])
    avg_degree = offdiag / num_stocks if num_stocks else float("nan")

    print("\n" + "=" * 78)
    print(f"EDGE-COUNT VERIFICATION (k={k})")
    print("=" * 78)
    print(f"  adj shape: {tuple(adj_matrix.shape)}  num_stocks={num_stocks}")
    print(f"  off-diagonal non-zero edges (batch 0): {offdiag}")
    print(f"  avg off-diagonal degree per node: {avg_degree:.2f}")
    print(f"  (reference k=8 reported 402 off-diagonal edges on the same pipeline)")
    print("=" * 78 + "\n", flush=True)

    return {
        "num_stocks": int(num_stocks),
        "offdiag_nonzero_edges_batch0": int(offdiag),
        "avg_offdiag_degree_batch0": float(avg_degree),
    }


def run_one(k: int, seed: int) -> dict:
    print("=" * 80)
    print(f"k-NN k SENSITIVITY - k={k}, seed={seed}")
    print("=" * 80)
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    # ---- The single lever: patch the class attribute in THIS process only. ----
    # The wrapper builds its own LSTMGATConfig() internally; instances read this class
    # attribute for top_k_neighbors, so patching it here changes the constructed graph.
    original_k = LSTMGATConfig.top_k_neighbors
    LSTMGATConfig.top_k_neighbors = k
    print(f"[k lever] Patched LSTMGATConfig.top_k_neighbors {original_k} -> {k} "
          f"(in-process only; config.py on disk unchanged)")

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = build_config(k)
        device = torch.device(config.device)

        # Dataloaders: IDENTICAL call to the reference knn backbone run.
        train_loader, val_loader, test_loader, datasets = \
            create_multi_stock_dataloaders_with_graph_method_fixed(
                data_dir="data/processed",
                seq_length=config.seq_length,
                forecast_horizon=config.forecast_horizon,
                graph_method="knn",
                graph_threshold=None,
                k_neighbors=k,  # threaded through but unused for graph; kept consistent
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

        edge_verification = verify_k_edges(val_loader, k)

        model = create_parallel_lstm_gat_model(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                         factor=0.5, patience=5)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = PROJECT_ROOT / f"results/k_sensitivity_k{k}_seed{seed}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {results_dir}")

        best_val_loss = float("inf")
        best_epoch = 0
        train_losses, val_losses = [], []
        epoch_times = []

        for epoch in range(config.num_epochs):
            t0 = time.time()
            print(f"\n[Epoch {epoch + 1}/{config.num_epochs}]", flush=True)
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
            train_losses.append(train_loss)

            val_loss, val_metrics = validate(model, val_loader, criterion, device, datasets[1])
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"  Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"DirAcc={val_metrics['directional_accuracy']:.2f}% "
                  f"RMSE={val_metrics['rmse']:.6f} lr={current_lr:.6f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), results_dir / "best_k_sensitivity_model.pth")

            epoch_times.append(time.time() - t0)

        print(f"\nTraining done. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")
        print(f"Total training time: {sum(epoch_times) / 60:.1f} min")

        model.load_state_dict(torch.load(results_dir / "best_k_sensitivity_model.pth"))
        test_loss, test_metrics = validate(model, test_loader, criterion, device, datasets[2])

        print(f"\nTest Results (k={k}, seed={seed}):")
        for m in ["mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]:
            print(f"  {m}: {test_metrics[m]:.6f}")

        results = {
            "model": f"Parallel LSTM-GNN (k-NN graph, k={k})",
            "timestamp": timestamp,
            "seed": seed,
            "k_neighbors": k,
            "architecture": "Parallel (LSTM temporal + GNN spatial) -> Concatenation fusion",
            "graph_method": "knn",
            "sensitivity_check": {
                "type": "knn_k_sweep",
                "reference_run": "graph_method='knn', k=8, train_parallel_enhanced.py, 20 epochs "
                                 "(results/parallel_lstm_gnn_knn_2026-08-03_230722)",
                "k_lever": "LSTMGATConfig.top_k_neighbors patched in-process; config.py unchanged",
                "edge_verification": edge_verification,
            },
            "config": {
                "num_stocks": config.num_stocks,
                "top_k_neighbors": config.top_k_neighbors,
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
    finally:
        LSTMGATConfig.top_k_neighbors = original_k


def main():
    parser = argparse.ArgumentParser(description="k-NN k sensitivity check (price-only backbone)")
    parser.add_argument("--k", type=int, required=True, help="top_k_neighbors for the k-NN graph")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    res = run_one(args.k, args.seed)
    tm = res["test_metrics"]
    print("\n" + "=" * 80)
    print(f"SUMMARY k={args.k} seed={args.seed}: "
          f"QLIKE={tm['qlike']:.4f} RMSE={tm['rmse']:.6f} MAE={tm['mae']:.7f} "
          f"R2={tm['r2']:.4f} DirAcc={tm['directional_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
