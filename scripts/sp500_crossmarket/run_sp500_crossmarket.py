"""Cross-market check on S&P500: does the VN pattern (price-only LSTM beats HAR at short
horizons, HAR wins long) replicate on the US market?

Self-contained runner (does NOT touch the VN news/graph harness). Two models only:
  HAR       : pooled OLS on HAR features [daily=Parkinson(t), weekly=roll5, monthly=roll22].
  lstm_only : per-ticker StandardScaler(features + target, TRAIN-fit), 2-layer LSTM(hidden=64,
              dropout=0.2) + Linear head, linear output on normalized scale, inverse-transform
              for eval. NO news/graph/gate.

Target: Parkinson VARIANCE at t+h (point forecast). seq_length=22. horizons {1,5,10,22}.
Split: per-ticker chronological 70/15/15 (early-stop on val), test read once. No leakage:
windows within-split, scalers fit on TRAIN only.

Metrics: QLIKE (shared floor 1e-8, identical to VN dm_report._qlike) + RMSE. DM (HLN, HAC lag
h-1) of lstm_only vs HAR on per-observation QLIKE (negative => lstm beats HAR), reusing the VN
diebold_mariano() implementation.

Run:
  .venv_gpu_encode/Scripts/python.exe scripts/sp500_crossmarket/run_sp500_crossmarket.py \
      --n-tickers 120 --seed 42 --epochs 10 --patience 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"))
from diebold_mariano import diebold_mariano  # noqa: E402

SEQ = 22
HORIZONS = (1, 5, 10, 22)
QLIKE_FLOOR = 1e-8  # identical to VN dm_report._qlike
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def qlike(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    t = np.maximum(target, QLIKE_FLOOR)
    p = np.maximum(prediction, QLIKE_FLOOR)
    r = t / p
    return r - np.log(r) - 1.0


def build_ticker_features(path: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Return (ticker, feats[N,3] raw HAR, pk[N]) with monthly-valid rows only."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    pk = df["parkinson_variance"].to_numpy(dtype=np.float64)
    if len(pk) < 200:
        return None
    s = pd.Series(pk)
    daily = pk
    weekly = s.rolling(5, min_periods=5).mean().to_numpy()
    monthly = s.rolling(22, min_periods=22).mean().to_numpy()
    feats = np.stack([daily, weekly, monthly], axis=1)  # [N,3]
    ticker = Path(path).name.replace("_processed.csv", "")
    return ticker, feats, pk


def make_windows(feats: np.ndarray, pk: np.ndarray, horizon: int) -> list[int]:
    """Valid anchor indices t: window [t-21..t] all monthly-valid, target pk[t+h] exists."""
    n = len(pk)
    first_valid = 21  # monthly rolling valid from index 21
    anchors = []
    for t in range(first_valid + (SEQ - 1), n - horizon):
        # every timestep in [t-21..t] must be monthly-valid; earliest = t-21 >= first_valid
        if (t - (SEQ - 1)) >= first_valid:
            anchors.append(t)
    return anchors


class LstmOnly(nn.Module):
    def __init__(self, price_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def run_horizon(tickers: list[tuple[str, np.ndarray, np.ndarray]], horizon: int, args) -> dict:
    """Build pooled windows, fit HAR (OLS) + lstm_only, evaluate on identical test set."""
    # per-ticker window arrays (float32) + per-ticker train-fit scalers
    Xtr, ytr_norm, Xva, yva_norm, va_raw, va_tk = [], [], [], [], [], []
    Xte, yte_norm, te_raw, te_tk, te_date = [], [], [], [], []
    har_train_feat, har_train_y = [], []
    har_test_feat = []
    tk_scalers = {}  # ticker -> (target_mean, target_std)

    for tk_id, (ticker, feats, pk) in enumerate(tickers):
        anchors = make_windows(feats, pk, horizon)
        if len(anchors) < 60:
            continue
        anchors = np.asarray(anchors)
        n = len(anchors)
        i_tr, i_va = int(n * 0.70), int(n * 0.85)
        a_tr, a_va, a_te = anchors[:i_tr], anchors[i_tr:i_va], anchors[i_va:]
        if len(a_tr) < 30 or len(a_va) < 5 or len(a_te) < 5:
            continue

        # per-ticker feature scaler (fit on TRAIN anchor feature rows) + target scaler
        feat_tr_rows = feats[a_tr]  # [ntr,3]
        f_mean = feat_tr_rows.mean(0)
        f_std = feat_tr_rows.std(0) + 1e-8
        y_tr_raw = pk[a_tr + horizon]
        t_mean = float(y_tr_raw.mean())
        t_std = float(y_tr_raw.std() + 1e-8)
        tk_scalers[ticker] = (t_mean, t_std)

        def windows_for(anchor_arr):
            # [M, SEQ, 3] normalized features
            idx = anchor_arr[:, None] - (SEQ - 1) + np.arange(SEQ)[None, :]  # [M,SEQ]
            w = feats[idx]  # [M,SEQ,3]
            return ((w - f_mean) / f_std).astype(np.float32)

        # LSTM windows
        Xtr.append(windows_for(a_tr))
        ytr_norm.append(((pk[a_tr + horizon] - t_mean) / t_std).astype(np.float32))
        Xva.append(windows_for(a_va))
        yva_norm.append(((pk[a_va + horizon] - t_mean) / t_std).astype(np.float32))
        va_raw.append(pk[a_va + horizon].astype(np.float64))
        va_tk.append(np.full(len(a_va), tk_id))
        Xte.append(windows_for(a_te))
        yte_norm.append(((pk[a_te + horizon] - t_mean) / t_std).astype(np.float32))
        te_raw.append(pk[a_te + horizon].astype(np.float64))
        te_tk.append(np.full(len(a_te), tk_id))
        te_date.append(a_te + horizon)

        # HAR: raw features at anchor -> raw target (pooled)
        har_train_feat.append(feats[a_tr])
        har_train_y.append(pk[a_tr + horizon])
        har_test_feat.append(feats[a_te])

    Xtr = np.concatenate(Xtr); ytr_norm = np.concatenate(ytr_norm)
    Xva = np.concatenate(Xva); yva_norm = np.concatenate(yva_norm)
    va_raw = np.concatenate(va_raw); va_tk = np.concatenate(va_tk)
    Xte = np.concatenate(Xte); yte_norm = np.concatenate(yte_norm)
    te_raw = np.concatenate(te_raw); te_tk = np.concatenate(te_tk)
    te_date = np.concatenate(te_date)

    tk_mean = np.array([tk_scalers[t][0] for t, _, _ in tickers if t in tk_scalers])
    # map tk_id -> (mean,std); rebuild by ticker order among used
    used_ids = sorted(set(te_tk.tolist()))
    id2ms = {}
    # tk_id corresponds to enumerate index; recover from tickers list
    for tk_id, (ticker, _, _) in enumerate(tickers):
        if ticker in tk_scalers:
            id2ms[tk_id] = tk_scalers[ticker]
    means_te = np.array([id2ms[i][0] for i in te_tk])
    stds_te = np.array([id2ms[i][1] for i in te_tk])
    means_va = np.array([id2ms[i][0] for i in va_tk])
    stds_va = np.array([id2ms[i][1] for i in va_tk])

    # ---- HAR (pooled OLS) ----
    Htr = np.concatenate(har_train_feat); ytr_har = np.concatenate(har_train_y)
    Hte = np.concatenate(har_test_feat)
    A = np.column_stack([np.ones(len(Htr)), Htr])
    coef, *_ = np.linalg.lstsq(A, ytr_har, rcond=None)
    har_pred_test = np.column_stack([np.ones(len(Hte)), Hte]) @ coef
    har_pred_test = np.maximum(har_pred_test, QLIKE_FLOOR)

    # ---- lstm_only (torch, GPU) ----
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = LstmOnly(price_dim=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr_norm)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    bs = args.batch_size
    ntr = len(Xtr_t)

    best_val = np.inf; best_state = None; patience_ctr = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx].to(DEVICE, non_blocking=True)
            yb = ytr_t[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # val QLIKE (raw scale)
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva_t), 4096):
                vp.append(model(Xva_t[i:i + 4096]).cpu().numpy())
            vp = np.concatenate(vp)
        vp_raw = np.maximum(vp * stds_va + means_va, QLIKE_FLOOR)
        val_q = float(qlike(va_raw, vp_raw).mean())
        if val_q < best_val - 1e-6:
            best_val = val_q; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = []
        for i in range(0, len(Xte_t), 4096):
            tp.append(model(Xte_t[i:i + 4096]).cpu().numpy())
        tp = np.concatenate(tp)
    lstm_pred_test = np.maximum(tp * stds_te + means_te, QLIKE_FLOOR)

    # ---- metrics on identical test set ----
    q_har = qlike(te_raw, har_pred_test)
    q_lstm = qlike(te_raw, lstm_pred_test)
    rmse_har = float(np.sqrt(np.mean((te_raw - har_pred_test) ** 2)))
    rmse_lstm = float(np.sqrt(np.mean((te_raw - lstm_pred_test) ** 2)))
    dm = diebold_mariano(q_lstm, q_har, h=horizon)  # A=lstm, B=har; neg => lstm beats HAR

    return {
        "horizon": horizon,
        "n_test": int(len(te_raw)),
        "qlike_har": float(q_har.mean()),
        "qlike_lstm": float(q_lstm.mean()),
        "rmse_har": rmse_har,
        "rmse_lstm": rmse_lstm,
        "dm_hln": float(dm.dm_hln),
        "dm_p": float(dm.p_value),
        "dm_mean_diff": float(dm.mean_diff),
        "epochs_ran": epoch + 1,
        "n_train_windows": int(ntr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    all_files = sorted(glob.glob(str(ROOT / "data" / "processed" / "sp500" / "*_processed.csv")))
    rng = np.random.default_rng(args.seed)
    if args.n_tickers < len(all_files):
        sel = sorted(rng.choice(len(all_files), size=args.n_tickers, replace=False).tolist())
        files = [all_files[i] for i in sel]
    else:
        files = all_files
    print(f"[info] device={DEVICE} tickers_selected={len(files)}/{len(all_files)} seed={args.seed}")

    t0 = time.time()
    tickers = []
    for f in files:
        r = build_ticker_features(f)
        if r is not None:
            tickers.append(r)
    dropped = len(files) - len(tickers)
    print(f"[info] loaded {len(tickers)} tickers, dropped {dropped} (<200 rows)")

    results = []
    for h in HORIZONS:
        th = time.time()
        res = run_horizon(tickers, h, args)
        res["seconds"] = round(time.time() - th, 1)
        results.append(res)
        print(f"[h{h}] QLIKE HAR={res['qlike_har']:.4f} lstm={res['qlike_lstm']:.4f} "
              f"DM={res['dm_hln']:+.2f}(p={res['dm_p']:.3f}) n={res['n_test']} "
              f"ep={res['epochs_ran']} {res['seconds']}s")

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = {
        "meta": {
            "stamp": stamp, "device": str(DEVICE), "seed": args.seed,
            "n_tickers_selected": len(files), "n_tickers_used": len(tickers),
            "n_tickers_total_available": len(all_files),
            "selected_tickers": [Path(f).name.replace("_processed.csv", "") for f in files],
            "seq_length": SEQ, "horizons": list(HORIZONS),
            "epochs_max": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "qlike_floor": QLIKE_FLOOR,
            "split": "per-ticker 70/15/15 chronological, early-stop on val QLIKE, test read once",
            "total_seconds": round(time.time() - t0, 1),
        },
        "results": results,
    }
    outdir = ROOT / "results" / f"sp500_crossmarket_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] {time.time() - t0:.1f}s -> {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
