"""Per-ticker MSE/QLIKE/DirAcc breakdown for an ALREADY-TRAINED dual-group news checkpoint
(no training here — pure inference + metrics).

Default checkpoint: the 40-epoch, fully-converged, all-32-stocks-ON dual-group model
(`models/dual_group_news_2026-07-25_071825/best.pt`) — the most reliable "all-ON" reference
available (early-stopped at epoch 36, see docs/reports/2026-07-25_0131_summaryOfUpdate_report.md).

Run: python eval_checkpoint_per_ticker.py
"""
import sys
import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_DUAL_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_DUAL_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from src.common.evaluation import evaluate_predictions
from src.lstm_gat_hybrid.config import LSTMGATConfig

from dataset_dual_news import create_dual_news_dataloaders  # sibling, read-only
from model_dual_news import DualGroupNewsBaseline  # sibling, read-only


def per_stock_metrics(preds_d, targs_d, stock_names):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(_ROOT / "data" / "processed"))
    ap.add_argument("--news_panel_path",
                    default=str(_ROOT / "data" / "features" / "dual_group_news_panel.parquet"))
    ap.add_argument("--checkpoint",
                    default=str(_ROOT / "models" / "dual_group_news_2026-07-25_071825" / "best.pt"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_news", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--graph_method", default="knn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}, checkpoint={args.checkpoint}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3

    _, _, test_loader, (train_ds, val_ds, test_ds) = create_dual_news_dataloaders(
        data_dir=args.data_dir, news_panel_path=args.news_panel_path,
        graph_method=args.graph_method, batch_size=args.batch_size, config=config)

    n_feat = train_ds._n_feat
    model = DualGroupNewsBaseline(config, n_feat=n_feat, d_news=args.d_news, dropout=args.dropout).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[eval] loaded checkpoint, n_feat={n_feat}, stocks={len(test_ds.stock_names)}")

    preds_n, targs_n = [], []
    with torch.no_grad():
        for x_har, adj, x_news, y in test_loader:
            x_har = x_har.to(device); adj = adj.to(device); x_news = x_news.to(device); y = y.to(device)
            pred = model(x_har, adj, x_news)
            preds_n.append(pred.cpu().numpy().reshape(-1))
            targs_n.append(y.cpu().numpy().reshape(-1))
    preds_n = np.concatenate(preds_n)
    targs_n = np.concatenate(targs_n)

    n_stocks = len(test_ds.stock_names)
    preds_d = np.zeros_like(preds_n)
    targs_d = np.zeros_like(targs_n)
    for i in range(len(preds_n)):
        sn = test_ds.stock_names[i % n_stocks]
        if sn in test_ds.target_normalizers:
            preds_d[i] = test_ds.target_normalizers[sn].inverse_transform(
                preds_n[i:i+1].reshape(1, -1)).flatten()[0]
            targs_d[i] = test_ds.target_normalizers[sn].inverse_transform(
                targs_n[i:i+1].reshape(1, -1)).flatten()[0]
        else:
            preds_d[i] = preds_n[i]; targs_d[i] = targs_n[i]

    overall = evaluate_predictions(targs_d, preds_d, n_stocks=n_stocks)
    per_ticker = per_stock_metrics(preds_d, targs_d, test_ds.stock_names)

    print(f"\n[overall test] DirAcc={overall['directional_accuracy']:.2f}% "
          f"R2={overall['r2']:.4f} QLIKE={overall['qlike']:.4f}")

    out_path = _ROOT / "results" / "all_on_dual_group_per_ticker_eval.json"
    out_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "stock_names": list(test_ds.stock_names),
        "overall_test_metrics": {k: float(v) for k, v in overall.items()},
        "per_ticker_test_metrics": per_ticker,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
