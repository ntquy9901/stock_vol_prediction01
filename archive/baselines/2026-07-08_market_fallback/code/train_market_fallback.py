"""Train the market-fallback baseline.

Custom train_epoch/validate handle the 7-tuple batch and the model's
forward(x_har, adj, x_emb, mask, x_market, market_mask). Reuses EarlyStopping,
evaluate_predictions, plot_learning_curves_with_analysis. §3.B Val/Test table.
§3.E weight_decay. §3.C learning curves every --plot_every epochs.

Run (smoke — no real caches needed):
  python baselines/2026-07-08_market_fallback/code/train_market_fallback.py --epochs 2 --smoke

Real run (after extract_market_embeddings.py + per-ticker extract):
  python baselines/2026-07-08_market_fallback/code/train_market_fallback.py --epochs 40
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
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

from market_fallback_model_embedding import MarketFallbackBaseline
from market_fallback_dataset_embedding import create_market_dataloaders


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total, nb = 0.0, 0
    for x_har, adj, x_emb, mask, x_market, market_mask, y in loader:
        x_har = x_har.to(device); adj = adj.to(device); x_emb = x_emb.to(device)
        mask = mask.to(device); x_market = x_market.to(device)
        market_mask = market_mask.to(device); y = y.to(device)
        B, S = y.shape
        optimizer.zero_grad()
        pred = model(x_har, adj, x_emb, mask, x_market, market_mask)
        loss = criterion(pred.reshape(B * S), y.reshape(B * S))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN loss, skipping batch {nb+1}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(1, nb)


def validate(model, loader, criterion, device, dataset):
    model.eval()
    preds_n, targs_n = [], []
    with torch.no_grad():
        for x_har, adj, x_emb, mask, x_market, market_mask, y in loader:
            x_har = x_har.to(device); adj = adj.to(device); x_emb = x_emb.to(device)
            mask = mask.to(device); x_market = x_market.to(device)
            market_mask = market_mask.to(device); y = y.to(device)
            pred = model(x_har, adj, x_emb, mask, x_market, market_mask)
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
    ap.add_argument("--emb_dir", default=str(_ROOT / "data" / "sentiment_embedding"))
    ap.add_argument("--market_cache", default=str(_ROOT / "data" / "sentiment_embedding" / "market_emb.npz"))
    ap.add_argument("--epochs", type=int, default=70,
                    help="CLAUDE.md standardized 70; pass explicit --epochs for matched-epoch comparison")  # [MED-6]
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="CLAUDE.md §3.E")
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--plot_every", type=int, default=5, help="learning curve cadence (§3.C)")
    ap.add_argument("--smoke", action="store_true",
                    help="dummy embeddings (no real caches needed; verifies shapes/forward)")
    ap.add_argument("--graph_method", default="knn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[train] device={device}, smoke={args.smoke}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    config.gradient_clip = 1.0

    if args.smoke:
        emb_dir = None
        market_cache = None
    else:
        emb_dir = args.emb_dir
        market_cache = args.market_cache

    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_market_dataloaders(
            data_dir=args.data_dir, emb_dir=emb_dir, market_cache_path=market_cache,
            graph_method=args.graph_method, batch_size=args.batch_size, config=config)

    emb_dim = train_ds._emb_dim or args.d_news
    model = MarketFallbackBaseline(config, emb_dim=emb_dim, d_news=args.d_news,
                                   dropout=args.dropout).to(device)
    print(f"[train] model emb_dim={emb_dim}, d_news={args.d_news}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"market_fallback_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"market_fallback_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=20)
    best_path = ckpt_dir / "best.pt"
    best_vl = float('inf')
    best_vm = None
    train_losses, val_losses = [], []

    print(f"\n{'ep':>4} | {'train_loss':>11} | {'val_loss':>9} | {'DirAcc':>7} | {'RMSE':>10}")
    print("-" * 55)
    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, vm = validate(model, val_loader, criterion, device, val_ds)
        scheduler.step(vl)
        train_losses.append(tl); val_losses.append(vl)
        print(f"{ep+1:>4} | {tl:>11.6f} | {vl:>9.6f} | "
              f"{vm['directional_accuracy']:>6.2f}% | {vm['rmse']:>10.6f}", flush=True)
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
        "model": "MarketFallbackBaseline",
        "emb_dim": int(emb_dim), "d_news": args.d_news, "smoke": bool(args.smoke),
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
