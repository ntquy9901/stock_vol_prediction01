"""Train the Latent Noise Injection baseline (Tier A).

Reuses Embedding Baseline's dataset, train_epoch, validate (read-only imports).
Only difference vs embedding baseline: model = LatentNoiseBaseline (adds Gaussian
noise on news_rep during training) and default --epochs 5 (experiment cap, CLAUDE.md
Training Policy).

Run (real PhoBERT cache at data/sentiment_embedding/):
  python baselines/2026-07-11_latent_noise/code/train_latent_noise.py --epochs 5
Smoke (no cache needed):
  python baselines/2026-07-11_latent_noise/code/train_latent_noise.py --epochs 2 --smoke
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_EMB_CODE = _ROOT / "baselines" / "2026-07-07_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_EMB_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

from src.common.evaluation import evaluate_predictions  # noqa: F401 (used by imported validate)
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.train_parallel_enhanced import (
    EarlyStopping, plot_learning_curves_with_analysis,
)
# read-only reuse from sibling embedding baseline
from dataset_embedding import create_embedding_dataloaders
from train_embedding_baseline import train_epoch, validate

from model_latent_noise import LatentNoiseBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--emb_dir", default=str(_ROOT / "data" / "sentiment_embedding"))
    ap.add_argument("--epochs", type=int, default=5,
                    help="default 5 (experiment cap, CLAUDE.md Training Policy)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5,
                    help="L2 weight decay (CLAUDE.md §3.E mandates 1e-5)")
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--noise_std", type=float, default=0.1,
                    help="Gaussian noise std on news_rep (Tier A); 0 disables")
    ap.add_argument("--plot_every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="quick run (no real embedding cache; verifies shapes/forward)")
    ap.add_argument("--graph_method", default="knn")
    ap.add_argument("--resume_from", default=None,
                    help="path to a .pt state_dict to resume weights from (continue training)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[train] device={device}, noise_std={args.noise_std}, smoke={args.smoke}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    config.gradient_clip = 1.0

    emb_dir = args.emb_dir if not args.smoke else None  # smoke -> dummy embeddings
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_embedding_dataloaders(
            data_dir=args.data_dir, emb_dir=emb_dir, graph_method=args.graph_method,
            batch_size=args.batch_size, config=config)

    emb_dim = train_ds._emb_dim or args.d_news
    model = LatentNoiseBaseline(config, emb_dim=emb_dim, d_news=args.d_news,
                                dropout=args.dropout, noise_std=args.noise_std).to(device)
    print(f"[train] model emb_dim={emb_dim}, d_news={args.d_news}, noise_std={args.noise_std}")

    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(sd)
        print(f"[resume] loaded weights from {args.resume_from}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"latent_noise_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"latent_noise_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # min_epochs small for short experiment runs (Training Policy: 5-epoch test)
    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=min(5, args.epochs))
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
        train_losses.append(tl)
        val_losses.append(vl)
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

    # eval() ensures noise is OFF for fair test eval
    model.eval()
    val_m = best_vm or {}
    test_loss, test_m = validate(model, test_loader, criterion, device, test_ds)

    def _fin(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in d.items()}

    print("\n=== Val/Test Comparison (best ckpt, noise OFF at eval) ===")
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
        "model": "LatentNoiseBaseline",
        "tier": "A (additive Gaussian noise, no loss change)",
        "noise_std": float(args.noise_std),
        "emb_dim": int(emb_dim), "d_news": args.d_news, "smoke": bool(args.smoke),
        "epochs_run": len(train_losses),
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
