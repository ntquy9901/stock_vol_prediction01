"""Extract + visualize the learned per-ticker news gate from a trained
GatedCrossAttnBaseline checkpoint.

Question this answers: after training, did the model learn to rely on news more for some
VN30 tickers than others on its own (via the gate's gradient), rather than via an externally
pre-selected ON/OFF list? Contrast with the 3 externally-derived selective-gate attempts
(`baselines/2026-07-25_*selective*`, `*_news_usefulness_ablation`, `*_ablation_derived_gate*`)
that used a fixed mask decided BEFORE training — none of those transferred.

Run: python .../analyze_gate_per_ticker.py --checkpoint models/gated_crossattn_<ts>/best.pt
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_gated_crossattn import GatedCrossAttnBaseline
from dataset_embedding import create_embedding_dataloaders  # noqa: E402 (sibling, read-only)


@torch.no_grad()
def extract_gate_per_ticker(model, loader, stock_names, device):
    """Run `model` over `loader`, capture gate_mlp's sigmoid output via a forward hook.

    Returns {ticker: np.ndarray of gate values, one per (test window, ticker)}.
    """
    if not stock_names:
        raise ValueError("stock_names is empty — nothing to extract")

    gates_by_ticker = {s: [] for s in stock_names}
    captured = {}

    def _hook(_module, _inp, out):
        captured["gate"] = out.detach()

    handle = model.gate_mlp.register_forward_hook(_hook)
    model.eval()
    try:
        for x_har, adj, x_emb, mask, _y in loader:
            x_har = x_har.to(device); adj = adj.to(device)
            x_emb = x_emb.to(device); mask = mask.to(device)
            captured.clear()
            model(x_har, adj, x_emb, mask)
            if "gate" not in captured:
                raise RuntimeError("gate_mlp forward hook did not fire — model has no gate_mlp "
                                    "output for this batch")
            gate = captured["gate"]
            if gate.shape[-1] != 1:
                raise RuntimeError(f"gate_mlp output width is {gate.shape[-1]}, expected 1 "
                                    "(squeeze assumption violated)")
            gate = gate.squeeze(-1).cpu().numpy()  # [B, S]
            if gate.shape[1] != len(stock_names):
                raise ValueError(f"gate has {gate.shape[1]} stocks but stock_names has "
                                  f"{len(stock_names)} — ordering/count mismatch")
            for s_idx, sname in enumerate(stock_names):
                gates_by_ticker[sname].append(gate[:, s_idx])
    finally:
        handle.remove()

    if not any(gates_by_ticker.values()):
        raise ValueError("loader produced zero batches — no gate values to extract")
    return {s: np.concatenate(v) for s, v in gates_by_ticker.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--emb_dir", default=str(_ROOT / "data" / "sentiment_embedding"))
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--graph_method", default="knn",
                    help="must match the graph_method the checkpoint was TRAINED with")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = LSTMGATConfig()
    config.num_features_per_stock = 3

    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
        create_embedding_dataloaders(
            data_dir=args.data_dir, emb_dir=args.emb_dir, graph_method=args.graph_method,
            batch_size=32, config=config)

    emb_dim = train_ds._emb_dim
    model = GatedCrossAttnBaseline(config, emb_dim=emb_dim, d_news=args.d_news,
                                   num_heads=args.num_heads, dropout=0.0).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        raise RuntimeError(
            "checkpoint state_dict does not match model built from --d_news/--num_heads/"
            f"--graph_method (got d_news={args.d_news}, num_heads={args.num_heads}, "
            f"graph_method={args.graph_method}) — check these match the training run. "
            f"Original error: {e}") from e

    loader, ds = (test_loader, test_ds) if args.split == "test" else (val_loader, val_ds)
    gates = extract_gate_per_ticker(model, loader, ds.stock_names, device)

    stats = {s: {"mean": float(np.mean(v)), "std": float(np.std(v))} for s, v in gates.items()}
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    n_windows = len(next(iter(gates.values())))

    print(f"\n=== Learned news-gate per ticker ({args.split} split, n_windows={n_windows}) ===")
    print(f"{'ticker':<8}{'mean_gate':>12}{'std_gate':>12}")
    for s, st in ranked:
        print(f"{s:<8}{st['mean']:>12.4f}{st['std']:>12.4f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"gate_per_ticker_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_stats.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "split": args.split, "n_windows": n_windows,
                    "stats": stats}, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")

    tickers = [s for s, _ in ranked]
    means = [st["mean"] for _, st in ranked]
    stds = [st["std"] for _, st in ranked]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(tickers)), means, yerr=stds, capsize=3, color="#4C72B0")
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=90)
    ax.set_ylabel("Learned news gate (mean over test windows)")
    ax.set_title(f"Per-ticker learned reliance on news (GatedCrossAttn, {args.split} split)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="gate=0.5 (half-open)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "gate_per_ticker.png", dpi=150)
    plt.close(fig)

    print(f"\n[done] stats -> {out_dir / 'gate_stats.json'}")
    print(f"[done] chart -> {out_dir / 'gate_per_ticker.png'}")


if __name__ == "__main__":
    main()
