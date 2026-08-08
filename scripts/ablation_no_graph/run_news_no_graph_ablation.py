"""News + NO-GRAPH ablation: the genuinely-missing 4th cell of the 2x2 grid.

Goal
----
The paper's ablation grid is (graph on/off) x (news on/off). Three cells were
already trained:
  - (no graph, no news)  -> LSTM-only ablation (identity adjacency on the news-free backbone)
  - (graph,   no news)   -> price-only backbone (train_parallel_enhanced.py)
  - (graph,   news)      -> FULL model (baselines/2026-07-26_per_ticker_news_gate_baseline)

The FOURTH cell — (NO graph, news) — was never actually run. Earlier work
(`docs/reports/2026-08-05_graph_ablation_results.md`) *believed* it had measured
"does the graph help when news is present?", but the script it used
(`run_no_graph_ablation.py`) imports the NEWS-FREE backbone, so it really measured
the graph's effect when news is ABSENT (see the Correction note in the report this
script produces). This script fills the real 4th cell so the graph's marginal
contribution *in the presence of news* can finally be isolated:
    FULL (news + graph)  vs.  news + NO graph (this run).

Method
------
Train the EXACT FULL news model — `PerTickerGatedNewsBaseline`, the same
`create_dual_news_dataloaders` (news panel included), the same hyperparameters and
the same 10+10-epoch resume-chained 20-epoch protocol, the same 3 seeds — that
produced the FULL cell in `train_per_ticker_gate.py`. The ONE and only change: on
every forward call, the real k-NN adjacency is replaced with a batched identity
matrix (`torch.eye(num_stocks)`) before it reaches `model(x_har, adj, x_news)`. With
an identity adjacency each GAT node can only attend to itself (off-diagonal entries
are 0, masked to -inf before the attention softmax), so the spatial branch
degenerates to a per-node transform with the SAME parameter count — the cross-stock
graph is the only thing removed.

Isolation
---------
NEW, self-contained script. It does NOT modify `model_per_ticker_gate.py`,
`train_per_ticker_gate.py`, `dataset_dual_news.py`, `model_parallel.py`,
`train_parallel_enhanced.py`, `evaluation.py`, `run_no_graph_ablation.py`,
`run_lstm_only_ablation.py` or any shared code — it only READS/imports them. The
train/validate loops are copied-and-adapted here so the identity substitution can
happen locally inside the batch loop (the original trainer has no CLI flag for it).
Everything else — MSE loss, the two-param-group optimizer (gate_logits gets its own
`gate_lr`!), the ReduceLROnPlateau scheduler, EarlyStopping, per-ticker
`evaluate_predictions(..., n_stocks=)` metrics — is identical to the FULL trainer, so
the ONLY changed variable is the adjacency content.

The 20-epoch protocol is reproduced as two legs of 10 epochs (mirroring the FULL
run's 10+10 resume chaining, since `train_per_ticker_gate.py` caps invocations at
MAX_EPOCHS=10): each leg re-seeds, builds a fresh optimizer/scheduler/early-stopping,
and leg 2 loads leg 1's best checkpoint with its best-val-loss counter reset — exactly
as a resume invocation of the original does — so test metrics come from the best epoch
among 11-20, matching how the FULL cell's reported numbers were selected.

Run
---
    python scripts/ablation_no_graph/run_news_no_graph_ablation.py --seeds 42 123 2026
    python scripts/ablation_no_graph/run_news_no_graph_ablation.py --seeds 42   # single-seed fallback
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

# Bootstrap sys.path: project root + the FULL news baseline's code dir (so its
# sibling-relative imports — dataset_dual_news, model_dual_news — resolve exactly as
# they do when train_per_ticker_gate.py runs).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GATE_CODE = PROJECT_ROOT / "baselines" / "2026-07-26_per_ticker_news_gate_baseline" / "code"
_DUAL_CODE = PROJECT_ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(PROJECT_ROOT), str(_GATE_CODE), str(_DUAL_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.common.evaluation import evaluate_predictions  # noqa: E402
from src.common.provenance import get_provenance  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from src.lstm_gat_hybrid.train_parallel_enhanced import EarlyStopping  # noqa: E402

from dataset_dual_news import create_dual_news_dataloaders  # noqa: E402 (news baseline, read-only)
from model_per_ticker_gate import PerTickerGatedNewsBaseline  # noqa: E402 (news baseline, read-only)

# FULL-run hyperparameters (train_per_ticker_gate.py argparse defaults, which the FULL
# cell used). Kept as module constants so the "only adjacency changed" claim is auditable.
FULL_LR = 5e-3
FULL_GATE_LR = 0.05
FULL_WEIGHT_DECAY = 1e-5
FULL_D_NEWS = 64
FULL_DROPOUT = 0.5
FULL_BATCH_SIZE = 32
FULL_PATIENCE = 15
FULL_GRAPH_METHOD = "knn"
EPOCHS_PER_LEG = 10          # train_per_ticker_gate.MAX_EPOCHS cap -> 10+10 resume chaining
NUM_LEGS = 2                 # -> 20 epochs total, matching the FULL protocol


def make_identity_adj(adj: torch.Tensor) -> torch.Tensor:
    """Return a batched identity adjacency with the SAME shape/device/dtype as ``adj``
    (i.e. [batch, num_stocks, num_stocks]).

    This is the single point where the graph is removed: identity = self-loops only,
    zero cross-stock edges. The GAT layer masks off-diagonal (==0) entries to -inf
    before softmax, so each node attends only to itself.
    """
    batch_size, num_stocks, _ = adj.shape
    eye = torch.eye(num_stocks, device=adj.device, dtype=adj.dtype)
    return eye.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def verify_graph_removed(model, dataloader, device):
    """Strong, functional verification that the identity substitution genuinely
    disables cross-stock message passing in the NEWS model.

    Same two checks as the sibling scripts, adapted for the news model
    (`forward(x_har, adj, x_news)`; GNN embeddings come from `model.har.get_embeddings`):
      1. Shape/content: the substituted adjacency is exactly a batched identity.
      2. Invariance: with identity adjacency, a node's GNN embedding is INVARIANT to
         other stocks' input features (perturbing every other stock leaves stock 0's
         GNN embedding unchanged). With the REAL adjacency the same perturbation DOES
         change stock 0's embedding — proving the substitution is what removes message
         passing, not a wiring bug that silently leaves the real graph in place.
    """
    model.eval()
    x_har, adj, x_news, y = next(iter(dataloader))
    x_har = x_har.to(device)
    adj = adj.to(device)

    identity_adj = make_identity_adj(adj)

    print("\n" + "=" * 78)
    print("SANITY CHECK: identity-adjacency substitution (news + no-graph)")
    print("=" * 78)
    print(f"  Real adj shape: {tuple(adj.shape)}  "
          f"mean={adj.mean():.4f} min={adj.min():.4f} max={adj.max():.4f}")
    print(f"  Real adj off-diagonal nonzeros (batch 0): "
          f"{int((adj[0] != 0).sum() - torch.diagonal(adj[0]).count_nonzero())}")
    print(f"  Identity adj shape: {tuple(identity_adj.shape)}  "
          f"mean={identity_adj.mean():.4f} min={identity_adj.min():.4f} max={identity_adj.max():.4f}")

    num_stocks = identity_adj.shape[1]
    expected_eye = torch.eye(num_stocks, device=device, dtype=identity_adj.dtype)
    exact_identity = torch.allclose(identity_adj[0], expected_eye)
    print(f"  identity_adj[0] == torch.eye({num_stocks}): {exact_identity}")
    assert exact_identity, "Identity substitution produced a non-identity matrix!"
    print(f"  identity_adj[0] first 4x4 block:\n{identity_adj[0][:4, :4].cpu().numpy()}")

    # Functional invariance check on the news model's GNN branch (model.har).
    with torch.no_grad():
        _, h_gnn_id = model.har.get_embeddings(x_har, identity_adj)
        _, h_gnn_real = model.har.get_embeddings(x_har, adj)

        x_pert = x_har.clone()
        # x_har is [B, seq, num_stocks, feat]; perturb every stock EXCEPT stock 0 (dim 2).
        noise = torch.randn(x_pert[:, :, 1:, :].shape, device=device)
        x_pert[:, :, 1:, :] = x_pert[:, :, 1:, :] + 5.0 * noise

        _, h_gnn_id_pert = model.har.get_embeddings(x_pert, identity_adj)
        _, h_gnn_real_pert = model.har.get_embeddings(x_pert, adj)

    id_stock0_delta = (h_gnn_id[:, 0, :] - h_gnn_id_pert[:, 0, :]).abs().max().item()
    real_stock0_delta = (h_gnn_real[:, 0, :] - h_gnn_real_pert[:, 0, :]).abs().max().item()

    print("\n  [Invariance] Perturbing all OTHER stocks' inputs, watching stock 0's GNN embedding:")
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
        "real_adj_offdiag_nonzeros_batch0": int(
            (adj[0] != 0).sum() - torch.diagonal(adj[0]).count_nonzero()),
        "identity_stock0_delta": float(id_stock0_delta),
        "real_stock0_delta": float(real_stock0_delta),
        "invariant_under_identity": bool(invariant),
        "sensitive_under_real": bool(real_leaks),
    }


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0, verbose_first=False):
    """One training epoch. Copied from train_per_ticker_gate.train_epoch, with the ONLY
    functional change being: adj -> identity right after unpack."""
    model.train()
    total, nb = 0.0, 0
    for x_har, adj, x_news, y in loader:
        x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
        adj = make_identity_adj(adj)  # <-- NEWS + NO-GRAPH ABLATION
        if verbose_first and nb == 0:
            print(f"    [DEBUG train first batch] adj (identity) mean={adj.mean():.6f}, "
                  f"diag_sum={torch.diagonal(adj[0]).sum():.1f}, "
                  f"offdiag_nonzero={int((adj[0] != 0).sum() - torch.diagonal(adj[0]).count_nonzero())}")
        B, S = y.shape
        optimizer.zero_grad()
        pred = model(x_har, adj, x_news)
        loss = criterion(pred.reshape(B * S), y.reshape(B * S))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN/Inf loss, skipping batch {nb + 1}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(1, nb)


def validate(model, loader, criterion, device, dataset):
    """Validation. Copied from train_per_ticker_gate.validate (same inverse-transform +
    per-ticker DirAcc), with the ONLY functional change being: adj -> identity.
    Returns (avg_loss_normalized, metrics_dict_on_denorm_scale)."""
    model.eval()
    preds_n, targs_n = [], []
    with torch.no_grad():
        for x_har, adj, x_news, y in loader:
            x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
            adj = make_identity_adj(adj)  # <-- NEWS + NO-GRAPH ABLATION
            pred = model(x_har, adj, x_news)
            preds_n.append(pred.cpu().numpy().reshape(-1))
            targs_n.append(y.cpu().numpy().reshape(-1))
    preds_n = np.concatenate(preds_n)
    targs_n = np.concatenate(targs_n)
    avg_loss = criterion(torch.tensor(preds_n, dtype=torch.float32),
                         torch.tensor(targs_n, dtype=torch.float32)).item()

    n_stocks = len(dataset.stock_names)
    preds_d = np.zeros_like(preds_n)
    targs_d = np.zeros_like(targs_n)
    for i in range(len(preds_n)):
        sn = dataset.stock_names[i % n_stocks]
        if sn in dataset.target_normalizers:
            preds_d[i] = dataset.target_normalizers[sn].inverse_transform(
                preds_n[i:i + 1].reshape(1, -1)).flatten()[0]
            targs_d[i] = dataset.target_normalizers[sn].inverse_transform(
                targs_n[i:i + 1].reshape(1, -1)).flatten()[0]
        else:
            preds_d[i] = preds_n[i]; targs_d[i] = targs_n[i]

    metrics = evaluate_predictions(targs_d, preds_d, n_stocks=n_stocks)
    if len(preds_d) == (len(preds_d) // n_stocks) * n_stocks and len(preds_d) >= n_stocks * 2:
        nw = len(preds_d) // n_stocks
        p2 = preds_d.reshape(nw, n_stocks); t2 = targs_d.reshape(nw, n_stocks)
        dir_per = []
        for s in range(n_stocks):
            if len(t2[:, s]) >= 2:
                dir_per.append(np.mean(np.sign(np.diff(t2[:, s])) == np.sign(np.diff(p2[:, s]))) * 100)
        if dir_per:
            metrics['directional_accuracy_per_stock'] = float(np.mean(dir_per))
    return avg_loss, metrics


def build_model_and_data(seed, device):
    """Seed, build the FULL news dataloaders + model exactly as train_per_ticker_gate.main()
    does (same config edits, same create_dual_news_dataloaders call, same model args)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only (matches FULL)
    config.gradient_clip = 1.0

    panel_path = str(PROJECT_ROOT / "data" / "features" / "dual_group_news_panel.parquet")
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_dual_news_dataloaders(
            data_dir=str(PROJECT_ROOT / "data" / "processed"),
            news_panel_path=panel_path, graph_method=FULL_GRAPH_METHOD,
            batch_size=FULL_BATCH_SIZE, config=config)

    n_feat = train_ds._n_feat
    stock_names = list(train_ds.stock_names)
    num_stocks = len(stock_names)
    model = PerTickerGatedNewsBaseline(config, n_feat=n_feat, num_stocks=num_stocks,
                                       d_news=FULL_D_NEWS, dropout=FULL_DROPOUT).to(device)
    print(f"[news-no-graph] model n_feat={n_feat}, d_news={FULL_D_NEWS}, num_stocks={num_stocks}")
    return model, (train_loader, val_loader, test_loader), (train_ds, val_ds, test_ds), stock_names


def build_optimizer(model):
    """Same two-param-group optimizer as the FULL trainer: gate_logits gets its own
    (higher) learning rate, everything else gets the base lr."""
    return torch.optim.Adam([
        {"params": [model.gate_logits], "lr": FULL_GATE_LR},
        {"params": [p for n, p in model.named_parameters() if n != "gate_logits"], "lr": FULL_LR},
    ], weight_decay=FULL_WEIGHT_DECAY)


def run_one_seed(seed, run_sanity_check=False):
    """Train one seed of the news + no-graph ablation over the 10+10 (=20) epoch protocol."""
    print("=" * 80)
    print(f"NEWS + NO-GRAPH ABLATION (identity adjacency, news present) - seed {seed}")
    print("=" * 80)
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (train_loader, val_loader, test_loader), (train_ds, val_ds, test_ds), stock_names = \
        build_model_and_data(seed, device)

    sanity = None
    if run_sanity_check:
        sanity = verify_graph_removed(model, val_loader, device)

    criterion = nn.MSELoss()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = PROJECT_ROOT / f"results/news_no_graph_ablation_seed{seed}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = results_dir / "best_news_no_graph_model.pth"
    print(f"Results directory: {results_dir}")

    train_losses, val_losses = [], []
    epoch_times = []
    leg_best_vl = None
    leg_best_vm = None
    leg_best_epoch = None

    for leg in range(NUM_LEGS):
        # Mirror the FULL run's per-invocation behaviour: each 10-epoch leg re-seeds and
        # builds a FRESH optimizer/scheduler/early-stopping. Leg 2 loads leg 1's best
        # checkpoint (the resume path), with its best-val-loss counter reset — so the test
        # model is the best epoch among 11-20, exactly as the FULL cell was selected.
        torch.manual_seed(seed)
        np.random.seed(seed)
        if leg > 0:
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            print(f"\n[leg {leg + 1}] loaded leg-{leg} best checkpoint; "
                  f"continuing epochs {leg * EPOCHS_PER_LEG + 1}-{(leg + 1) * EPOCHS_PER_LEG}")

        optimizer = build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        early = EarlyStopping(patience=FULL_PATIENCE, min_delta=1e-6,
                              min_epochs=min(20, EPOCHS_PER_LEG))
        leg_best_vl = float("inf")
        leg_best_vm = None
        leg_best_epoch = 0

        for ep_local in range(EPOCHS_PER_LEG):
            ep_num = leg * EPOCHS_PER_LEG + ep_local + 1
            t0 = time.time()
            tl = train_epoch(model, train_loader, criterion, optimizer, device,
                             grad_clip=1.0, verbose_first=(leg == 0 and ep_local == 0))
            vl, vm = validate(model, val_loader, criterion, device, val_ds)
            scheduler.step(vl)
            current_lr = optimizer.param_groups[1]["lr"]
            train_losses.append(tl)
            val_losses.append(vl)
            print(f"  Epoch {ep_num}/{NUM_LEGS * EPOCHS_PER_LEG}: train={tl:.6f} val={vl:.6f} "
                  f"DirAcc={vm['directional_accuracy']:.2f}% RMSE={vm['rmse']:.6f} "
                  f"lr={current_lr:.6f}", flush=True)

            if vl < leg_best_vl:
                leg_best_vl = vl
                leg_best_vm = vm
                leg_best_epoch = ep_num
                torch.save(model.state_dict(), best_ckpt_path)
            epoch_times.append(time.time() - t0)

            early(vl, ep_num)
            if early.early_stop:
                print(f"  [early stop] epoch {ep_num}")
                break

    # best_ckpt_path now holds the best epoch of the FINAL leg (11-20), matching FULL selection.
    print(f"\nTraining done. Final-leg best epoch: {leg_best_epoch}, best val loss: {leg_best_vl:.6f}")
    print(f"Total training time: {sum(epoch_times) / 60:.1f} min")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    val_m = leg_best_vm or {}
    test_loss, test_metrics = validate(model, test_loader, criterion, device, test_ds)

    final_gate = model.gate_values().cpu().numpy()

    print("\nTest Results (news + no-graph / identity adjacency):")
    for k in ["mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]:
        print(f"  {k}: {test_metrics[k]:.6f}")

    def _fin(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in d.items()}

    results = {
        "model": "PerTickerGatedNewsBaseline (NEWS + NO-GRAPH ablation, identity adjacency)",
        "timestamp": timestamp,
        "seed": seed,
        "architecture": "HAR(LSTM+GNN) + per-ticker-gated news -> concat fusion",
        "graph_method": "identity (no cross-stock edges) — ablation of the FULL (news+knn) model",
        "ablation": {
            "type": "news_no_graph_identity_adjacency",
            "reference_run": ("FULL per-ticker news gate, "
                              "baselines/2026-07-26_per_ticker_news_gate_baseline/code/"
                              "train_per_ticker_gate.py, graph_method='knn', 10+10=20 epochs"),
            "description": (
                "Same FULL news model / data / hyperparameters / 10+10 resume-chained "
                "20-epoch protocol; the k-NN adjacency is replaced with torch.eye at every "
                "forward call so the GAT branch cannot pass messages across stocks (per-node "
                "transform only, same parameter count). News branch fully present. This is the "
                "genuinely-missing (news, no graph) cell of the 2x2 grid."),
            "sanity_check": sanity,
        },
        "config": {
            "num_stocks": len(stock_names),
            "learning_rate": FULL_LR,
            "gate_lr": FULL_GATE_LR,
            "batch_size": FULL_BATCH_SIZE,
            "weight_decay": FULL_WEIGHT_DECAY,
            "d_news": FULL_D_NEWS,
            "dropout": FULL_DROPOUT,
            "graph_method": FULL_GRAPH_METHOD,
            "num_epochs_trained": NUM_LEGS * EPOCHS_PER_LEG,
            "epochs_per_leg": EPOCHS_PER_LEG,
            "num_legs": NUM_LEGS,
            "final_leg_best_epoch": leg_best_epoch,
        },
        "training_summary": {
            "num_epochs_trained": NUM_LEGS * EPOCHS_PER_LEG,
            "final_leg_best_epoch": leg_best_epoch,
            "final_leg_best_val_loss": float(leg_best_vl),
            "total_time_minutes": float(sum(epoch_times) / 60),
        },
        "final_gate_values": {sn: float(g) for sn, g in zip(stock_names, final_gate)},
        "validation_metrics": _fin(val_m),
        "test_metrics": {
            "mse": float(test_metrics["mse"]),
            "rmse": float(test_metrics["rmse"]),
            "mae": float(test_metrics["mae"]),
            "r2": float(test_metrics["r2"]),
            "qlike": float(test_metrics["qlike"]),
            "directional_accuracy": float(test_metrics["directional_accuracy"]),
            "directional_accuracy_per_stock": float(
                test_metrics.get("directional_accuracy_per_stock", float("nan")))
                if np.isfinite(test_metrics.get("directional_accuracy_per_stock", float("nan")))
                else None,
        },
        "provenance": get_provenance(),
    }

    with open(results_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {results_dir / 'training_results.json'}")
    print(f"Finished at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="News + no-graph (identity adjacency) ablation — the 4th 2x2 cell")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Seeds to run (default: 42 123 2026, matching the FULL protocol)")
    args = parser.parse_args()

    all_results = []
    for i, seed in enumerate(args.seeds):
        # Run the (relatively expensive) sanity check on the first seed only.
        res = run_one_seed(seed, run_sanity_check=(i == 0))
        all_results.append(res)

    print("\n" + "=" * 80)
    print("NEWS + NO-GRAPH ABLATION SUMMARY")
    print("=" * 80)
    for res in all_results:
        tm = res["test_metrics"]
        print(f"  seed {res['seed']}: QLIKE={tm['qlike']:.4f} RMSE={tm['rmse']:.6f} "
              f"MAE={tm['mae']:.7f} R2={tm['r2']:.4f} DirAcc={tm['directional_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
