"""Train the directed-spillover-graph + QLIKE-augmented-loss baseline.

Hard-isolated copy of
`2026-07-25_dual_group_news_embedding_baseline/code/train_dual_news.py` (CLAUDE.md §3.F rule 3).
Two changes vs. the sibling, per design.md:
  1. `create_spillover_news_dataloaders` (graph_method='spillover' by default) instead of the
     sibling's symmetric correlation/k-NN graph.
  2. Loss = `losses.combined_loss` (MSE + qlike_weight * QLIKE-on-denormalized) instead of plain
     `nn.MSELoss`.
Model architecture (`DualGroupNewsBaseline`) is UNCHANGED (read-only import) — the GAT layer
already supports arbitrary (including asymmetric) adjacency matrices, so no model edit is needed.

Training policy (CLAUDE.md): default/max 10 epochs for this experimental run; report every 5;
enforced below (raises if --epochs > 10, matching the 2026-07-25_macro_news_baseline precedent
for unattended runs).

Run (smoke, no panel needed):
  python train_spillover_qlike.py --epochs 2 --smoke

Real run (after data/features/dual_group_news_panel.parquet exists):
  python train_spillover_qlike.py --epochs 10
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
from src.common.evaluation import evaluate_predictions
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.train_parallel_enhanced import (
    EarlyStopping, plot_learning_curves_with_analysis,
)

from model_dual_news import DualGroupNewsBaseline  # noqa: E402 (sibling, read-only)
from dataset_spillover_news import create_spillover_news_dataloaders
from losses import build_denorm_tensors, combined_loss

MAX_EPOCHS = 10  # CLAUDE.md Training policy: >10 epochs needs explicit user approval based on
                  # 5/10-epoch results, which isn't possible during an unattended run.


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0):
    model.train()
    total, nb = 0.0, 0
    for x_har, adj, x_news, y in loader:
        x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
        optimizer.zero_grad()
        pred = model(x_har, adj, x_news)                 # [B, S]
        loss = loss_fn(pred, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN loss, skipping batch {nb+1}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(1, nb)


def validate(model, loader, loss_fn, device, dataset):
    """Returns (avg_loss_normalized, metrics_dict_on_denorm_scale)."""
    model.eval()
    preds_n, targs_n = [], []
    losses = []
    with torch.no_grad():
        for x_har, adj, x_news, y in loader:
            x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
            pred = model(x_har, adj, x_news)
            losses.append(loss_fn(pred, y).item())
            preds_n.append(pred.cpu().numpy().reshape(-1))
            targs_n.append(y.cpu().numpy().reshape(-1))
    preds_n = np.concatenate(preds_n)
    targs_n = np.concatenate(targs_n)
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    n_stocks = len(dataset.stock_names)
    preds_d = np.zeros_like(preds_n)
    targs_d = np.zeros_like(targs_n)
    for i in range(len(preds_n)):
        sn = dataset.stock_names[i % n_stocks]
        if sn in dataset.target_normalizers:
            preds_d[i] = dataset.target_normalizers[sn].inverse_transform(
                preds_n[i:i+1].reshape(1, -1)).flatten()[0]
            targs_d[i] = dataset.target_normalizers[sn].inverse_transform(
                targs_n[i:i+1].reshape(1, -1)).flatten()[0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--news_panel_path",
                    default=str(_ROOT / "data" / "features" / "dual_group_news_panel.parquet"))
    ap.add_argument("--epochs", type=int, default=10,
                    help="CLAUDE.md Training policy: default 5-10 epochs; capped at 10 (see MAX_EPOCHS)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--plot_every", type=int, default=5)
    ap.add_argument("--qlike_weight", type=float, default=0.1,
                    help="weight of QLIKE term in combined loss (design.md §2.2, not tuned)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick run (no real panel needed; verifies shapes/forward)")
    ap.add_argument("--graph_method", default="spillover",
                    choices=["spillover", "correlation", "knn"])
    ap.add_argument("--k_neighbors", type=int, default=8,
                    help="top-k incoming spillover edges per node (graph_method='spillover')")
    args = ap.parse_args()

    if args.epochs > MAX_EPOCHS:
        raise ValueError(
            f"--epochs={args.epochs} exceeds CLAUDE.md Training policy cap ({MAX_EPOCHS}) for "
            "experimental runs without explicit user approval based on 5/10-epoch results.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[train] device={device}, smoke={args.smoke}, graph_method={args.graph_method}, "
          f"qlike_weight={args.qlike_weight}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    config.gradient_clip = 1.0

    panel_path = args.news_panel_path if not args.smoke else None
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_spillover_news_dataloaders(
            data_dir=args.data_dir, news_panel_path=panel_path, graph_method=args.graph_method,
            k_neighbors=args.k_neighbors, batch_size=args.batch_size, config=config)

    n_feat = train_ds._n_feat
    model = DualGroupNewsBaseline(config, n_feat=n_feat, d_news=args.d_news,
                                   dropout=args.dropout).to(device)
    print(f"[train] model n_feat={n_feat}, d_news={args.d_news}")

    mean_t, std_t = build_denorm_tensors(train_ds, device)
    loss_fn = lambda pred, y: combined_loss(pred, y, mean_t, std_t, qlike_weight=args.qlike_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"spillover_qlike_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"spillover_qlike_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=min(20, args.epochs))
    best_path = ckpt_dir / "best.pt"
    best_vl = float('inf')
    best_vm = None
    train_losses, val_losses = [], []

    print(f"\n{'ep':>4} | {'train_loss':>11} | {'val_loss':>9} | {'DirAcc':>7} | {'RMSE':>10}")
    print("-" * 55)
    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, loss_fn, optimizer, device)
        vl, vm = validate(model, val_loader, loss_fn, device, val_ds)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"{ep+1:>4} | {tl:>11.6f} | {vl:>9.6f} | "
              f"{vm['directional_accuracy']:>6.2f}% | {vm['rmse']:>10.6f}", flush=True)
        if (ep + 1) % 5 == 0:
            print(f"      [5-epoch report] MSE={vm['mse']:.6f} RMSE={vm['rmse']:.6f} "
                  f"MAE={vm['mae']:.6f} R2={vm['r2']:.6f} QLIKE={vm['qlike']:.6f} "
                  f"DirAcc={vm['directional_accuracy']:.2f}%", flush=True)
        if vl < best_vl:
            best_vl = vl
            best_vm = vm
            torch.save(model.state_dict(), best_path)
        if (ep + 1) % args.plot_every == 0 and len(train_losses) >= 2:
            try:
                plot_learning_curves_with_analysis(
                    train_losses, val_losses, out_dir, ep, gap_threshold=0.05)
            except Exception as e:
                print(f"[warn] learning-curve plot failed at epoch {ep+1}: {e}")
        early(vl, ep + 1)
        if early.early_stop:
            print(f"[early stop] epoch {ep+1}")
            break

    if len(train_losses) >= 2:
        try:
            plot_learning_curves_with_analysis(
                train_losses, val_losses, out_dir, len(train_losses) - 1, gap_threshold=0.05)
        except Exception as e:
            print(f"[warn] final learning-curve plot failed: {e}")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_path)

    val_m = best_vm or {}
    test_loss, test_m = validate(model, test_loader, loss_fn, device, test_ds)

    def _fin(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in d.items()}

    print("\n=== Val/Test Comparison (best ckpt) ===")
    print(f"{'Metric':<12}{'Validation':>14}{'Test':>14}{'Diff':>14}")
    print("-" * 54)
    metric_keys = ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']
    val_test_diff = {}
    for k in metric_keys:
        v = val_m.get(k, float('nan'))
        t = test_m.get(k, float('nan'))
        diff = t - v
        val_test_diff[f"{k}_diff"] = float(diff) if np.isfinite(diff) else None
        print(f"{k:<12}{v:>14.6f}{t:>14.6f}{diff:>+14.6f}")

    results = {
        "model": "DualGroupNewsBaseline",
        "graph_method": args.graph_method, "qlike_weight": args.qlike_weight,
        "n_feat": int(n_feat), "d_news": args.d_news, "smoke": bool(args.smoke),
        "validation_metrics": _fin(val_m),
        "test_metrics": _fin(test_m),
        "val_test_diff": val_test_diff,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(f"\n[done] results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
