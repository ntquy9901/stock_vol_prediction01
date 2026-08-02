"""Train the Selective News Gate baseline (HAR + news branch, masked to 22 EDA-selected tickers).

Reuses `create_dual_news_dataloaders` from the sibling baseline
`2026-07-25_dual_group_news_embedding_baseline` (read-only import, no data rebuild — same
dual_group_news_panel.parquet). Adds a per-stock DirAcc breakdown split by NEWS_ON/NEWS_OFF group
(requirements.md §4 success criterion), on top of the standard 6 mandatory metrics.

Training policy (CLAUDE.md): default 10 epochs for this first experimental run.

Run (smoke, no panel needed):
  python baselines/2026-07-25_selective_news_gate_baseline/code/train_selective_gate.py --epochs 2 --smoke

Real run:
  python baselines/2026-07-25_selective_news_gate_baseline/code/train_selective_gate.py --epochs 10
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
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
from src.common.evaluation import evaluate_predictions
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.train_parallel_enhanced import (
    EarlyStopping, plot_learning_curves_with_analysis,
)

from dataset_dual_news import create_dual_news_dataloaders  # sibling, read-only
from model_selective_gate import NEWS_OFF_TICKERS, NEWS_ON_TICKERS, SelectiveGateNewsBaseline


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total, nb = 0.0, 0
    for x_har, adj, x_news, y in loader:
        x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
        B, S = y.shape
        optimizer.zero_grad()
        pred = model(x_har, adj, x_news)
        loss = criterion(pred.reshape(B * S), y.reshape(B * S))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN loss, skipping batch {nb+1}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(1, nb)


def _per_stock_dir_acc(preds_d, targs_d, stock_names):
    """Per-ticker directional accuracy + NEWS_ON/NEWS_OFF group averages."""
    n_stocks = len(stock_names)
    if len(preds_d) < n_stocks * 2 or len(preds_d) % n_stocks != 0:
        return {}, None, None
    nw = len(preds_d) // n_stocks
    p2 = preds_d.reshape(nw, n_stocks)
    t2 = targs_d.reshape(nw, n_stocks)
    per_ticker = {}
    for s, name in enumerate(stock_names):
        if len(t2[:, s]) >= 2:
            per_ticker[name] = float(np.mean(
                np.sign(np.diff(t2[:, s])) == np.sign(np.diff(p2[:, s]))) * 100)
    on_vals = [v for k, v in per_ticker.items() if k in NEWS_ON_TICKERS]
    off_vals = [v for k, v in per_ticker.items() if k in NEWS_OFF_TICKERS]
    on_avg = float(np.mean(on_vals)) if on_vals else None
    off_avg = float(np.mean(off_vals)) if off_vals else None
    return per_ticker, on_avg, off_avg


def validate(model, loader, criterion, device, dataset):
    model.eval()
    preds_n, targs_n = [], []
    with torch.no_grad():
        for x_har, adj, x_news, y in loader:
            x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
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
                preds_n[i:i+1].reshape(1, -1)).flatten()[0]
            targs_d[i] = dataset.target_normalizers[sn].inverse_transform(
                targs_n[i:i+1].reshape(1, -1)).flatten()[0]
        else:
            preds_d[i] = preds_n[i]; targs_d[i] = targs_n[i]

    metrics = evaluate_predictions(targs_d, preds_d, n_stocks=n_stocks)
    per_ticker, on_avg, off_avg = _per_stock_dir_acc(preds_d, targs_d, dataset.stock_names)
    if on_avg is not None:
        metrics['dir_acc_news_on_avg'] = on_avg
    if off_avg is not None:
        metrics['dir_acc_news_off_avg'] = off_avg
    metrics['_per_ticker_dir_acc'] = per_ticker
    return avg_loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--news_panel_path",
                    default=str(_ROOT / "data" / "features" / "dual_group_news_panel.parquet"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--plot_every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--graph_method", default="knn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[train] device={device}, smoke={args.smoke}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    config.gradient_clip = 1.0

    panel_path = args.news_panel_path if not args.smoke else None
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_dual_news_dataloaders(
            data_dir=args.data_dir, news_panel_path=panel_path, graph_method=args.graph_method,
            batch_size=args.batch_size, config=config)

    n_feat = train_ds._n_feat
    model = SelectiveGateNewsBaseline(config, n_feat=n_feat, stock_names=train_ds.stock_names,
                                      d_news=args.d_news, dropout=args.dropout).to(device)
    n_on = int(model.stock_mask.sum().item())
    print(f"[train] model n_feat={n_feat}, d_news={args.d_news}, "
          f"NEWS_ON stocks={n_on}/{len(train_ds.stock_names)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"selective_gate_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"selective_gate_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=min(20, args.epochs))
    best_path = ckpt_dir / "best.pt"
    best_vl = float('inf')
    best_vm = None
    train_losses, val_losses = [], []

    print(f"\n{'ep':>4} | {'train_loss':>11} | {'val_loss':>9} | {'DirAcc':>7} | "
          f"{'ON':>6} | {'OFF':>6} | {'RMSE':>10}")
    print("-" * 75)
    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, vm = validate(model, val_loader, criterion, device, val_ds)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        on_s = f"{vm['dir_acc_news_on_avg']:.2f}%" if vm.get('dir_acc_news_on_avg') is not None else "n/a"
        off_s = f"{vm['dir_acc_news_off_avg']:.2f}%" if vm.get('dir_acc_news_off_avg') is not None else "n/a"
        print(f"{ep+1:>4} | {tl:>11.6f} | {vl:>9.6f} | {vm['directional_accuracy']:>6.2f}% | "
              f"{on_s:>6} | {off_s:>6} | {vm['rmse']:>10.6f}", flush=True)
        if (ep + 1) % 5 == 0:
            print(f"      [5-epoch report] MSE={vm['mse']:.6f} RMSE={vm['rmse']:.6f} "
                  f"MAE={vm['mae']:.6f} R2={vm['r2']:.6f} QLIKE={vm['qlike']:.6f} "
                  f"DirAcc={vm['directional_accuracy']:.2f}% NEWS_ON={on_s} NEWS_OFF={off_s}",
                  flush=True)
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
    test_loss, test_m = validate(model, test_loader, criterion, device, test_ds)

    def _fin(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in d.items() if not k.startswith('_')}

    print("\n=== Val/Test Comparison (best ckpt) ===")
    print(f"{'Metric':<24}{'Validation':>14}{'Test':>14}{'Diff':>14}")
    print("-" * 66)
    metric_keys = ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy',
                   'dir_acc_news_on_avg', 'dir_acc_news_off_avg']
    val_test_diff = {}
    for k in metric_keys:
        v = val_m.get(k, float('nan'))
        t = test_m.get(k, float('nan'))
        diff = t - v
        val_test_diff[f"{k}_diff"] = float(diff) if np.isfinite(diff) else None
        print(f"{k:<24}{v:>14.6f}{t:>14.6f}{diff:>+14.6f}")

    print("\n=== Per-ticker Test DirAcc ===")
    for name, acc in sorted(test_m.get('_per_ticker_dir_acc', {}).items(), key=lambda kv: -kv[1]):
        group = "ON " if name in NEWS_ON_TICKERS else "OFF"
        print(f"  [{group}] {name:<6} {acc:>6.2f}%")

    results = {
        "model": "SelectiveGateNewsBaseline",
        "n_feat": int(n_feat), "d_news": args.d_news, "smoke": bool(args.smoke),
        "news_on_count": n_on, "news_off_count": len(train_ds.stock_names) - n_on,
        "validation_metrics": _fin(val_m),
        "test_metrics": _fin(test_m),
        "val_test_diff": val_test_diff,
        "per_ticker_test_dir_acc": test_m.get('_per_ticker_dir_acc', {}),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(f"\n[done] results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
