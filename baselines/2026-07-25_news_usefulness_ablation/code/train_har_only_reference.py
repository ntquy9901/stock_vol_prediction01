"""Train a fresh HAR-only reference model (ParallelLSTMGNN, own fusion, NOT frozen) on the EXACT
SAME data pipeline as `2026-07-25_dual_group_news_embedding_baseline` (same 32 common stocks,
same train/val/test window split) — needed for a valid per-ticker ablation against the all-ON
dual-group model (see requirements.md).

Reuses `create_dual_news_dataloaders` (sibling baseline, read-only) purely for its x_har/adj/y
construction — `news_panel_path=None` means x_news is a dummy all-zero tensor that this script
never even looks at (only x_har, adj, y are used). x_har/adj/y depend only on
`_load_raw_stock_data`/`_split_raw_data_by_date`, NOT on the news panel, so this gives IDENTICAL
windows/splits to the real dual-group run for a fair per-ticker comparison.

Not comparable to the June/July "HAR-only 69.98%/69.61%" reference elsewhere in the project —
those used a different pipeline (batch_size=11, data augmentation on). This one intentionally
matches TODAY's dual-group pipeline (batch_size=32, no augmentation) for apples-to-apples
per-ticker deltas, not for an absolute benchmark number.

Run: python train_har_only_reference.py --epochs 10
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_DUAL_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_DUAL_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
from src.common.evaluation import evaluate_predictions
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN
from src.lstm_gat_hybrid.train_parallel_enhanced import EarlyStopping

from dataset_dual_news import create_dual_news_dataloaders  # sibling, read-only


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total, nb = 0.0, 0
    for x_har, adj, _x_news, y in loader:
        x_har = x_har.to(device); adj = adj.to(device); y = y.to(device)
        B, S = y.shape
        optimizer.zero_grad()
        pred = model(x_har, adj)
        loss = criterion(pred.reshape(B * S), y.reshape(B * S))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN loss, skipping batch {nb+1}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(1, nb)


def per_stock_metrics(preds_d, targs_d, stock_names):
    """Per-ticker MSE, QLIKE, DirAcc (delta computation prefers MSE/QLIKE — continuous, less
    noisy than DirAcc's binary sign-agreement over ~163 points per ticker)."""
    n_stocks = len(stock_names)
    if len(preds_d) < n_stocks * 2 or len(preds_d) % n_stocks != 0:
        return {}
    nw = len(preds_d) // n_stocks
    p2 = preds_d.reshape(nw, n_stocks)
    t2 = targs_d.reshape(nw, n_stocks)
    out = {}
    for s, name in enumerate(stock_names):
        p_s, t_s = p2[:, s], t2[:, s]
        m = evaluate_predictions(t_s, p_s)
        dir_acc = None
        if len(t_s) >= 2:
            dir_acc = float(np.mean(np.sign(np.diff(t_s)) == np.sign(np.diff(p_s))) * 100)
        out[name] = {"mse": float(m["mse"]), "qlike": float(m["qlike"]), "dir_acc": dir_acc}
    return out


def validate(model, loader, criterion, device, dataset):
    model.eval()
    preds_n, targs_n = [], []
    with torch.no_grad():
        for x_har, adj, _x_news, y in loader:
            x_har = x_har.to(device); adj = adj.to(device); y = y.to(device)
            pred = model(x_har, adj)
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
                preds_n[i:i+1].reshape(1, -1)).flatten()[0]
            targs_d[i] = dataset.target_normalizers[sn].inverse_transform(
                targs_n[i:i+1].reshape(1, -1)).flatten()[0]
        else:
            preds_d[i] = preds_n[i]; targs_d[i] = targs_n[i]

    metrics = evaluate_predictions(targs_d, preds_d, n_stocks=n_stocks)
    metrics['_per_ticker'] = per_stock_metrics(preds_d, targs_d, dataset.stock_names)
    return avg_loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--graph_method", default="knn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[train] device={device}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    config.gradient_clip = 1.0
    config.fusion_dropout = args.dropout

    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_dual_news_dataloaders(
            data_dir=args.data_dir, news_panel_path=None, graph_method=args.graph_method,
            batch_size=args.batch_size, config=config)

    model = ParallelLSTMGNN(config).to(device)
    print(f"[train] HAR-only reference, {len(train_ds.stock_names)} stocks, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"har_only_ablation_ref_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"har_only_ablation_ref_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=min(20, args.epochs))
    best_path = ckpt_dir / "best.pt"
    best_vl = float('inf')
    best_vm = None

    print(f"\n{'ep':>4} | {'train_loss':>11} | {'val_loss':>9} | {'DirAcc':>7} | {'RMSE':>10}")
    print("-" * 55)
    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, vm = validate(model, val_loader, criterion, device, val_ds)
        scheduler.step(vl)
        print(f"{ep+1:>4} | {tl:>11.6f} | {vl:>9.6f} | "
              f"{vm['directional_accuracy']:>6.2f}% | {vm['rmse']:>10.6f}", flush=True)
        if vl < best_vl:
            best_vl = vl
            best_vm = vm
            torch.save(model.state_dict(), best_path)
        early(vl, ep + 1)
        if early.early_stop:
            print(f"[early stop] epoch {ep+1}")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_path)

    val_m = best_vm or {}
    test_loss, test_m = validate(model, test_loader, criterion, device, test_ds)

    def _fin(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in d.items() if not k.startswith('_')}

    print("\n=== Val/Test Comparison ===")
    for k in ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']:
        print(f"{k:<24}{val_m.get(k, float('nan')):>14.6f}{test_m.get(k, float('nan')):>14.6f}")

    results = {
        "model": "ParallelLSTMGNN_HAR_only_ablation_reference",
        "stock_names": list(train_ds.stock_names),
        "validation_metrics": _fin(val_m),
        "test_metrics": _fin(test_m),
        "per_ticker_test_metrics": test_m.get('_per_ticker', {}),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\n[done] results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
