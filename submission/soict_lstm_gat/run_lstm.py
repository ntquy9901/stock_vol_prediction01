"""Main model: per-OBSERVATION pooled LSTM (no graph) vs HAR + GARCH baselines.

Per the updated requirements, the headline model is a price-only **LSTM** trained on per-observation
pooled windows (every ticker x window), per-stock chronological 80/10/10, MSE loss, early-stop on
val MSE. This is the data design under which the LSTM is competitive with / beats HAR at short horizons
(unlike the common-date snapshot design used for the graph-check in run_all.py).

CLI:  python run_lstm.py <dataset> <lookback> <horizon> [--smoke] [--data-root DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import baselines as B  # noqa: E402
import data_utils as du  # noqa: E402
import metrics as M  # noqa: E402
from config import Config, SMOKE  # noqa: E402

HERE = Path(__file__).resolve().parent


class LSTM(nn.Module):
    """Price-only per-observation LSTM: [B, seq, 3] -> [B]."""

    def __init__(self, in_dim=3, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def build_perobs(files, lookback, horizon, cfg):
    """Per-observation pooled data + aligned raw-HAR (for HAR) and per-ticker train series (for GARCH)."""
    Xtr, ytr_n, Xva, yva_n, yva_raw, va_id = [], [], [], [], [], []
    Xte, yte_raw, te_id, te_date = [], [], [], []
    har_tr_f, har_tr_y, har_te_f = [], [], []
    garch_train = {}          # ticker_id -> train target series
    scalers = {}
    tickers = []
    for f in sorted(files):
        df = du.pd.read_csv(f, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        pk = df["parkinson_volatility"].to_numpy(float)
        if len(pk) < 200 or not np.isfinite(pk).all():   # skip tickers with too little / non-finite data
            continue
        feats = du.har_features(pk)
        a = du.make_windows(pk, lookback, horizon)
        if len(a) < 60:
            continue
        a_tr, a_va, a_te = du.per_stock_split(a, cfg.train_frac, cfg.val_frac)
        if len(a_tr) < 30 or len(a_va) < 5 or len(a_te) < 5:
            continue
        tid = len(tickers); tickers.append(Path(f).name.replace("_processed.csv", ""))
        s = du.TickerScaler(); s.fit_features(feats[a_tr]); s.fit_target(pk[np.asarray(a_tr) + horizon])
        scalers[tid] = (s.t_mean, s.t_std)
        # GARCH: fit on all pre-test targets (train + val) so the forecast starts at the test boundary
        # (the earlier train-only fit left a val-length gap that mis-aligned the forecast to test dates).
        garch_train[tid] = pk[np.concatenate([np.asarray(a_tr), np.asarray(a_va)]) + horizon]

        def win(anchors):
            idx = np.asarray(anchors)[:, None] - (lookback - 1) + np.arange(lookback)[None, :]
            return s.transform_windows(feats[idx])
        Xtr.append(win(a_tr)); ytr_n.append(((pk[np.asarray(a_tr) + horizon] - s.t_mean) / s.t_std).astype(np.float32))
        Xva.append(win(a_va)); yva_n.append(((pk[np.asarray(a_va) + horizon] - s.t_mean) / s.t_std).astype(np.float32))
        yva_raw.append(pk[np.asarray(a_va) + horizon]); va_id.append(np.full(len(a_va), tid))
        Xte.append(win(a_te)); yte_raw.append(pk[np.asarray(a_te) + horizon]); te_id.append(np.full(len(a_te), tid))
        te_date.append([du.pd.Timestamp(df.iloc[t + horizon]["date"]).strftime("%Y-%m-%d") for t in a_te])
        har_tr_f.append(feats[a_tr]); har_tr_y.append(pk[np.asarray(a_tr) + horizon]); har_te_f.append(feats[a_te])
    cat = lambda xs: np.concatenate(xs)
    return {
        "Xtr": cat(Xtr).astype(np.float32), "ytr_n": cat(ytr_n),
        "Xva": cat(Xva).astype(np.float32), "yva_n": cat(yva_n),
        "yva_raw": cat(yva_raw), "va_id": cat(va_id),
        "Xte": cat(Xte).astype(np.float32), "yte_raw": cat(yte_raw), "te_id": cat(te_id),
        "te_date": sum(te_date, []), "scalers": scalers, "tickers": tickers,
        "har_tr_f": cat(har_tr_f), "har_tr_y": cat(har_tr_y), "har_te_f": cat(har_te_f),
        "garch_train": garch_train,
    }


def train_lstm(data, cfg, seed, out_dir, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(data["Xtr"]); ytr = torch.from_numpy(data["ytr_n"])
    Xva = torch.from_numpy(data["Xva"]).to(device); yva = torch.from_numpy(data["yva_n"]).to(device)
    model = LSTM(3, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lossf = nn.MSELoss(); n = len(Xtr); bs = cfg.batch_size
    hist = {"train_mse": [], "val_mse": []}
    best = np.inf; best_state = None; wait = 0
    log.write(f"[hyperparams] seed={seed} model=LSTM(per-obs) hidden={cfg.hidden} lr={cfg.lr} "
              f"epochs={cfg.epochs} train_obs={n} device={device}\n"); log.flush()
    for ep in range(cfg.epochs):
        model.train(); perm = torch.randperm(n); tl = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(device, non_blocking=True); yb = ytr[idx].to(device, non_blocking=True)
            opt.zero_grad(); out = model(xb); loss = lossf(out, yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva), 4096):
                vp.append(model(Xva[i:i + 4096]))
            val_mse = lossf(torch.cat(vp), yva).item()      # early-stop on val MSE (per requirements)
        hist["train_mse"].append(tl / max(nb, 1)); hist["val_mse"].append(val_mse); sched.step(val_mse)
        log.write(f"[epoch {ep+1}/{cfg.epochs}] train_mse={tl/max(nb,1):.6f} val_mse={val_mse:.6f}\n"); log.flush()
        if val_mse < best - 1e-7:
            best = val_mse; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if (ep + 1) % 5 == 0:
            plt.figure(figsize=(5, 3)); plt.plot(hist["train_mse"], label="train"); plt.plot(hist["val_mse"], label="val")
            plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.tight_layout()
            plt.savefig(out_dir / f"lc_seed{seed}_ep{ep+1}.png", dpi=110); plt.close()
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    # test predictions -> {(tid,date): (y_raw, pred_raw)}
    model.eval(); Xte = torch.from_numpy(data["Xte"]).to(device)
    with torch.no_grad():
        tp = []
        for i in range(0, len(Xte), 4096):
            tp.append(model(Xte[i:i + 4096]).cpu().numpy())
    tp = np.concatenate(tp)
    t_mean = np.array([data["scalers"][int(j)][0] for j in data["te_id"]])
    t_std = np.array([data["scalers"][int(j)][1] for j in data["te_id"]])
    pred_raw = np.maximum(tp * t_std + t_mean, 0.0)
    return {(int(data["te_id"][k]), data["te_date"][k]): (float(data["yte_raw"][k]), float(pred_raw[k]))
            for k in range(len(tp))}


def _ens(dicts):
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    keys = sorted(keys)
    return {k: (dicts[0][k][0], float(np.mean([d[k][1] for d in dicts]))) for k in keys}


def _metrics(pred, floor):
    ks = sorted(pred); y = np.array([pred[k][0] for k in ks]); p = np.array([pred[k][1] for k in ks])
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor=floor), "r2": M.r2(y, p), "n": len(ks)}


def _dm(a, b, h, loss, floor):
    ks = sorted(set(a) & set(b)); y = np.array([a[k][0] for k in ks])
    pa = np.array([a[k][1] for k in ks]); pb = np.array([b[k][1] for k in ks])
    la = M.per_obs_qlike(y, pa, floor) if loss == "qlike" else M.per_obs_se(y, pa)
    lb = M.per_obs_qlike(y, pb, floor) if loss == "qlike" else M.per_obs_se(y, pb)
    r = M.diebold_mariano(la, lb, h=h)
    return {"dm_hln": r.dm_hln, "p_value": r.p_value, "favors": "LSTM" if r.mean_diff < 0 else "other", "n": r.n}


def run(dataset, files, lookback, horizon, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train.log", "w", encoding="utf-8"); t0 = time.time()
    data = build_perobs(files, lookback, horizon, cfg)
    print(f"[perobs] {dataset} lb{lookback} h{horizon} tickers={len(data['tickers'])} "
          f"train_obs={len(data['Xtr'])} test_obs={len(data['Xte'])}", flush=True)
    lstm = _ens([train_lstm(data, cfg, s, out_dir, log) for s in cfg.seeds]) if len(cfg.seeds) > 1 \
        else train_lstm(data, cfg, cfg.seeds[0], out_dir, log)
    # HAR
    coef = B.har_fit(data["har_tr_f"], data["har_tr_y"]); hp = B.har_predict(data["har_te_f"], coef, floor=cfg.qlike_floor)
    HAR = {(int(data["te_id"][k]), data["te_date"][k]): (float(data["yte_raw"][k]), float(hp[k])) for k in range(len(hp))}
    # GARCH per ticker
    GAR = {}
    te_by = {}
    for k in range(len(data["te_id"])):
        te_by.setdefault(int(data["te_id"][k]), []).append(k)
    for tid, ks in te_by.items():
        try:
            fc = B.garch_forecast(data["garch_train"][tid], n_test=len(ks), horizon=horizon, floor=cfg.qlike_floor)
        except Exception:
            fc = np.full(len(ks), max(float(np.var(data["garch_train"][tid])), cfg.qlike_floor))
        for i, k in enumerate(ks):
            GAR[(tid, data["te_date"][k])] = (float(data["yte_raw"][k]), float(fc[i]))
    preds = {"HAR": HAR, "GARCH": GAR, "LSTM": lstm}
    metrics = {n: _metrics(p, cfg.qlike_floor) for n, p in preds.items()}
    dm = {f"LSTM_vs_{o}": {"qlike": _dm(lstm, preds[o], horizon, "qlike", cfg.qlike_floor),
                           "se": _dm(lstm, preds[o], horizon, "se", cfg.qlike_floor)} for o in ("HAR", "GARCH")}
    log.close()
    res = {"dataset": dataset, "lookback": lookback, "horizon": horizon, "design": "per-observation",
           "tickers": len(data["tickers"]), "n_test": metrics["HAR"]["n"], "seeds": list(cfg.seeds),
           "metrics": metrics, "dm": dm, "row_order": ["HAR", "GARCH", "LSTM"],
           "seconds": round(time.time() - t0, 1)}
    (out_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("lookback", type=int); ap.add_argument("horizon", type=int)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--data-root", default=str(HERE / "data"))
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else Config()
    root = Path(a.data_root)
    mp = {"vn30": [str(root / "vn30" / "*_processed.csv"), str(root / "*_processed.csv")],
          "vn100": [str(root / "vn100" / "*_processed.csv"), str(root / "vn100_vnstock" / "*_processed.csv")],
          "sp500": [str(root / "sp500" / "*_processed.csv")]}
    files = next((glob.glob(p) for p in mp[a.dataset] if glob.glob(p)), [])
    if not files:
        raise FileNotFoundError(f"no files for {a.dataset} under {a.data_root}")
    out = HERE.parents[1] / "results" / "soict_perobs" / f"{a.dataset}_lb{a.lookback}_h{a.horizon}"
    r = run(a.dataset, files, a.lookback, a.horizon, cfg, out)
    ql = {k: round(v["qlike"], 4) for k, v in r["metrics"].items()}
    print(f"[done] {a.dataset} lb{a.lookback} h{a.horizon} n_test={r['n_test']} QLIKE={ql} "
          f"LSTM_vs_HAR={r['dm']['LSTM_vs_HAR']['qlike']}", flush=True)


if __name__ == "__main__":
    main()
