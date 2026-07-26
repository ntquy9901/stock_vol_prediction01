"""Train the per-ticker isolated-gradient news gate baseline.

Hard-isolated copy-modification of
`2026-07-25_dual_group_news_embedding_baseline/code/train_dual_news.py` (CLAUDE.md §3.F rule 3).
Same HAR+news dual-group panel, same MSE loss, same dataloaders/eval/early-stopping utilities
(read-only import) — the ONLY architectural change is `PerTickerGatedNewsBaseline` (per-ticker
gate) instead of `DualGroupNewsBaseline` (no gate). This isolates exactly one experimental
variable (design.md §4 Simplicity Gate).

Debug output (user requirement, 2026-07-26 — "in debug ra màn hình... lưu ra file... xem quá
trình học tin tức ảnh hưởng tốt/xấu đến cổ phiếu nào"):
  - every epoch: console table of all tickers sorted by gate value (desc), with delta vs. the
    previous epoch;
  - every epoch: append to `gate_history.json` (epoch -> {ticker: gate_value});
  - every 5 epochs (CLAUDE.md §3.C cadence): gate-evolution PNG (one line per ticker) alongside
    the mandatory train/val loss learning curve (unchanged, reused from the sibling).

Run (smoke, no panel needed):
  python train_per_ticker_gate.py --epochs 2 --smoke

Real run:
  python train_per_ticker_gate.py --epochs 10

Continue training an existing run for N more epochs (loads weights + gate/loss history so the
console table, gate_history.json and gate-evolution plot are continuous across the resume
boundary — epoch numbering picks up where the previous run left off, not reset to 1):
  python train_per_ticker_gate.py --epochs 10 \
      --resume_checkpoint models/per_ticker_gate_<ts>/best.pt \
      --resume_results_dir results/per_ticker_gate_<ts>
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
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.evaluation import evaluate_predictions
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.train_parallel_enhanced import (
    EarlyStopping, plot_learning_curves_with_analysis,
)

from dataset_dual_news import create_dual_news_dataloaders  # noqa: E402 (sibling, read-only)
from model_per_ticker_gate import PerTickerGatedNewsBaseline

MAX_EPOCHS = 10  # CLAUDE.md Training policy: >10 epochs needs explicit user approval based on
                  # 5/10-epoch results.


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


def validate(model, loader, criterion, device, dataset):
    """Returns (avg_loss_normalized, metrics_dict_on_denorm_scale)."""
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

    metrics = evaluate_predictions(targs_d, preds_d)
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


def print_gate_table(stock_names, gate_values, prev_gate_values, epoch):
    """Console debug output (user requirement): all tickers sorted by gate value desc, with
    delta vs. the previous epoch, so the "which ticker does news help/hurt" process is visible
    epoch by epoch, not just at the end."""
    order = np.argsort(-gate_values)
    print(f"\n  [gate debug, epoch {epoch}] ticker gate values (sorted desc; delta vs. prev epoch):")
    print(f"  {'ticker':<8}{'gate':>8}{'delta':>10}")
    for idx in order:
        sn = stock_names[idx]
        g = gate_values[idx]
        d = g - prev_gate_values[idx] if prev_gate_values is not None else 0.0
        arrow = "+" if d > 1e-4 else ("-" if d < -1e-4 else "=")
        print(f"  {sn:<8}{g:>8.4f}{d:>+10.4f} {arrow}")


def load_resume_state(resume_results_dir: Path | None) -> tuple[dict, list, list, int]:
    """Load a previous run's `gate_history.json` (always saved) and `loss_history.json` (saved
    from this fix onward — absent for runs started before it existed, handled gracefully) so a
    resumed run's epoch numbering, gate table, and loss curve continue where the previous run
    left off instead of resetting to epoch 1.

    Returns (gate_history, train_losses, val_losses, start_epoch). `start_epoch` is the highest
    epoch number already recorded in `gate_history` (0 if none/not resuming).
    """
    if resume_results_dir is None:
        return {}, [], [], 0
    resume_results_dir = Path(resume_results_dir)
    gate_history_path = resume_results_dir / "gate_history.json"
    if not gate_history_path.exists():
        raise FileNotFoundError(f"--resume_results_dir given but no gate_history.json at {gate_history_path}")
    gate_history = json.loads(gate_history_path.read_text(encoding="utf-8"))
    start_epoch = max((int(e) for e in gate_history.keys()), default=0)

    loss_history_path = resume_results_dir / "loss_history.json"
    if loss_history_path.exists():
        lh = json.loads(loss_history_path.read_text(encoding="utf-8"))
        train_losses, val_losses = lh["train_losses"], lh["val_losses"]
    else:
        print(f"[resume] no loss_history.json found at {loss_history_path} "
              "(run predates this being saved) -- loss curve will only cover the new epochs")
        train_losses, val_losses = [], []
    return gate_history, train_losses, val_losses, start_epoch


def plot_gate_evolution(gate_history: dict, stock_names: list, out_path: Path):
    """1 line per ticker (colormap, not a 30-entry legend — unreadable at that count),
    x=epoch, y=gate value. Companion to the mandatory loss learning curve (CLAUDE.md §3.C),
    same 5-epoch cadence, but for the gate instead of the loss."""
    epochs = sorted(int(e) for e in gate_history.keys())
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(stock_names))
    for i, sn in enumerate(stock_names):
        ys = [gate_history[str(ep)].get(sn, np.nan) for ep in epochs]
        ax.plot(epochs, ys, color=cmap(i), linewidth=1.2, alpha=0.85, label=sn)
    ax.set_xlabel("epoch")
    ax.set_ylabel("gate value (sigmoid)")
    ax.set_title("Per-ticker news gate evolution during training")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(ncol=3, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gate evolution plot saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--news_panel_path",
                    default=str(_ROOT / "data" / "features" / "dual_group_news_panel.parquet"))
    ap.add_argument("--epochs", type=int, default=10,
                    help="CLAUDE.md Training policy: default 5-10 epochs; capped at 10 (MAX_EPOCHS)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3, help="LR for all params except gate_logits")
    ap.add_argument("--gate_lr", type=float, default=0.05,
                    help="LR for gate_logits (higher than --lr on purpose: 30 scalars need to "
                         "move fast enough to be visible within the 10-epoch cap; design.md §2)")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--plot_every", type=int, default=5)
    ap.add_argument("--graph_method", default="knn")
    ap.add_argument("--smoke", action="store_true",
                    help="quick run (no real panel needed; verifies shapes/forward)")
    ap.add_argument("--resume_checkpoint", default=None,
                    help="path to a previous best.pt to load weights from before training")
    ap.add_argument("--resume_results_dir", default=None,
                    help="path to a previous run's results/ dir (gate_history.json + "
                         "loss_history.json) to continue epoch numbering/history from")
    args = ap.parse_args()

    if args.epochs > MAX_EPOCHS:
        raise ValueError(
            f"--epochs={args.epochs} exceeds CLAUDE.md Training policy cap ({MAX_EPOCHS}) for "
            "experimental runs without explicit user approval based on 5/10-epoch results. "
            "(This cap applies per invocation; resuming for another <=10-epoch run after "
            "reviewing this run's results is the explicit-approval path CLAUDE.md describes.)")
    if bool(args.resume_checkpoint) != bool(args.resume_results_dir):
        raise ValueError("--resume_checkpoint and --resume_results_dir must be given together")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}, smoke={args.smoke}, gate_lr={args.gate_lr}, lr={args.lr}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    config.gradient_clip = 1.0

    panel_path = args.news_panel_path if not args.smoke else None
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_dual_news_dataloaders(
            data_dir=args.data_dir, news_panel_path=panel_path, graph_method=args.graph_method,
            batch_size=args.batch_size, config=config)

    n_feat = train_ds._n_feat
    stock_names = list(train_ds.stock_names)
    num_stocks = len(stock_names)
    model = PerTickerGatedNewsBaseline(config, n_feat=n_feat, num_stocks=num_stocks,
                                       d_news=args.d_news, dropout=args.dropout).to(device)
    print(f"[train] model n_feat={n_feat}, d_news={args.d_news}, num_stocks={num_stocks}")

    gate_history, train_losses, val_losses, start_epoch = load_resume_state(args.resume_results_dir)
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
        print(f"[resume] loaded weights from {args.resume_checkpoint}, "
              f"continuing from epoch {start_epoch} for {args.epochs} more epoch(s) "
              f"(-> epoch {start_epoch + args.epochs})")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {"params": [model.gate_logits], "lr": args.gate_lr},
        {"params": [p for n, p in model.named_parameters() if n != "gate_logits"], "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"per_ticker_gate_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _ROOT / "models" / f"per_ticker_gate_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early = EarlyStopping(patience=args.patience, min_delta=1e-6, min_epochs=min(20, args.epochs))
    best_path = ckpt_dir / "best.pt"
    best_vl = float('inf')
    best_vm = None

    gate_history_path = out_dir / "gate_history.json"
    loss_history_path = out_dir / "loss_history.json"
    prev_gate = None
    if gate_history and start_epoch > 0:
        last_key = str(start_epoch)
        prev_gate = np.array([gate_history[last_key][sn] for sn in stock_names])

    print(f"\n{'ep':>4} | {'train_loss':>11} | {'val_loss':>9} | {'DirAcc':>7} | {'RMSE':>10}")
    print("-" * 55)
    for ep_local in range(args.epochs):
        ep_num = start_epoch + ep_local + 1   # continuous numbering across resumes
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, vm = validate(model, val_loader, criterion, device, val_ds)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"{ep_num:>4} | {tl:>11.6f} | {vl:>9.6f} | "
              f"{vm['directional_accuracy']:>6.2f}% | {vm['rmse']:>10.6f}", flush=True)
        if ep_num % 5 == 0:
            print(f"      [5-epoch report] MSE={vm['mse']:.6f} RMSE={vm['rmse']:.6f} "
                  f"MAE={vm['mae']:.6f} R2={vm['r2']:.6f} QLIKE={vm['qlike']:.6f} "
                  f"DirAcc={vm['directional_accuracy']:.2f}%", flush=True)

        # --- debug output (user requirement): per-ticker gate table + gate_history.json ---
        gate_np = model.gate_values().cpu().numpy()
        print_gate_table(stock_names, gate_np, prev_gate, ep_num)
        gate_history[str(ep_num)] = {sn: float(g) for sn, g in zip(stock_names, gate_np)}
        gate_history_path.write_text(json.dumps(gate_history, indent=2), encoding="utf-8")
        loss_history_path.write_text(
            json.dumps({"train_losses": train_losses, "val_losses": val_losses}, indent=2),
            encoding="utf-8")
        prev_gate = gate_np

        if vl < best_vl:
            best_vl = vl
            best_vm = vm
            torch.save(model.state_dict(), best_path)
        if ep_num % args.plot_every == 0 and len(train_losses) >= 2:
            try:
                plot_learning_curves_with_analysis(
                    train_losses, val_losses, out_dir, ep_num - 1, gap_threshold=0.05)
            except Exception as e:
                print(f"[warn] learning-curve plot failed at epoch {ep_num}: {e}")
            plot_gate_evolution(gate_history, stock_names,
                                out_dir / f"gate_evolution_epoch_{ep_num}_{timestamp}.png")
        early(vl, ep_num)
        if early.early_stop:
            print(f"[early stop] epoch {ep_num}")
            break

    if len(train_losses) >= 2:
        try:
            plot_learning_curves_with_analysis(
                train_losses, val_losses, out_dir, len(train_losses) - 1, gap_threshold=0.05)
        except Exception as e:
            print(f"[warn] final learning-curve plot failed: {e}")
        plot_gate_evolution(gate_history, stock_names,
                            out_dir / f"gate_evolution_final_{timestamp}.png")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_path)

    val_m = best_vm or {}
    test_loss, test_m = validate(model, test_loader, criterion, device, test_ds)

    final_gate = model.gate_values().cpu().numpy()
    print("\n=== Final per-ticker gate values (from best checkpoint) ===")
    print_gate_table(stock_names, final_gate, None, "best-ckpt")

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
        "model": "PerTickerGatedNewsBaseline",
        "n_feat": int(n_feat), "d_news": args.d_news, "smoke": bool(args.smoke),
        "gate_lr": args.gate_lr, "lr": args.lr,
        "final_gate_values": {sn: float(g) for sn, g in zip(stock_names, final_gate)},
        "validation_metrics": _fin(val_m),
        "test_metrics": _fin(test_m),
        "val_test_diff": val_test_diff,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(f"\n[done] results -> {out_dir / 'results.json'}")
    print(f"[done] gate history -> {gate_history_path}")


if __name__ == "__main__":
    main()
