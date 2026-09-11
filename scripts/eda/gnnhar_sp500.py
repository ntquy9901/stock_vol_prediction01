"""Faithful re-implementation of GNNHAR (Zhang, Pu, Cucuringu & Dong, Int. J. Forecasting 2024,
arXiv:2308.01419; official code https://github.com/chaozhang-ox/GNNHAR) evaluated under OUR SP500
protocol so a real learned multi-hop message-passing GNN sits in the SAME table as full_matrix.py.

Why: our paper argues cross-firm graph structure adds no reliable incremental QLIKE beyond own-history
+ a market factor on SP500. Our graph models feed ONE hand-built neighbour-mean scalar into a gamma-GBM.
A reviewer objects a real GNN (learned multi-hop message passing) could exploit graph structure a scalar
cannot. GNNHAR is the flagship HAR-baselined, QLIKE-evaluated graph-vol model, so running it under our
protocol directly answers that objection.

Faithful to the official repo:
  * GraphConvLayer: output = adj @ (X @ W) + b  (xavier_uniform W, bias init ones) -- GNNHAR.py:129-147
  * HAR / GNNHAR1L / GNNHAR2L: res = relu(H1 + H_gcn), H1 = Linear(F,1) own-feature linear term,
    H_gcn = (multi-layer) GCN -> mlp(nhid,1).  GNNHAR.py:150-277  (generalised from F=3 to F in {3,9})
  * QLIKE loss: tf = y/f ; L = mean(tf - log tf)  -- GNNHAR.py:317-328 (Loss.forward, QLike branch).
    Equivalent (up to +const) to submission/soict_lstm_gat metrics.per_obs_qlike used for evaluation.
  * Adam(lr=1e-3, weight_decay=1e-5), best-validation-checkpoint, multi-seed ensemble -- GNNHAR.py:357,332-433
  * n_hid = 9 (paper default).

Adaptations (documented, protocol-matched, causal):
  * Target = daily Parkinson variance pk.shift(-h) (OUR target, IDENTICAL to full_matrix) for h in
    {1,5,10,22} -- NOT the paper's horizon-averaged target, so GNNHAR's QLIKE goes in the same table.
  * Graph: correlation top-k graph built on TRAIN log-vol only (S1.build_graph) -- this is EXACTLY the
    graph our GBM+corr uses, so "learned GNN message passing vs scalar neighbour-mean on the same graph"
    is a clean comparison. The paper's glasso precision graph is ill-conditioned at p=497 (it used 27/80
    node universes); correlation is explicitly an accepted graph for this study.
  * No-graph control = A=I (per-node nonlinear MLP, GNNHAR's own key ablation isolating spillover from
    nonlinearity). Placebo = random edges of matched density (S1 Wp). Both train-only.
  * Node features: GNNHAR-HAR3 = 3 HAR lags (the paper's inputs); GNNHAR-OWN9 = full_matrix.OWN (3 HAR +
    rq + 5 log-vol momentum) for a matched-input comparison to GBM(own)/GBM+corr.
  * Features z-scored (train stats), target scaled by 1/train-mean (QLIKE is scale-equivariant up to an
    additive constant, so scaling is loss-equivalent) purely for numerical conditioning -- the paper's
    LOBSTER variance is already O(1); our pk ~ 1e-4.
  * Cross-section has missing tickers (IPO/delist) -> masked loss + masked forward (absent node features
    set to train-mean/0 after z-score), excluded from QLIKE/DM exactly like full_matrix's dropna rows.

Walk-forward folds, embargo, >=30000-train-row gate, FL floor, pooled QLIKE and date-clustered DM are
imported from full_matrix / S1 / the har_anchored stats module so the numbers are directly comparable.

Run: python gnnhar_sp500.py [--smoke]    (writes results/gnnhar/gnnhar_sp500.json)
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
import full_matrix as FM  # noqa: E402  (reuse load/panel/_har_ols/gbm/nb + OWN/HAR/FL/SEEDS)
import vn_gbm_graph_stage1 as S1  # noqa: E402  (FOLDS/TRAIN_START/build_graph/TOPK/RNG_SEED)
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
HAR3 = FM.HAR
OWN9 = FM.OWN
SEEDS = (0, 1, 2)
N_HID = 9
VALID_LEN = 22         # paper default trailing-validation length (GNNHAR.py --valid_len)
LR, WD = 1e-3, 1e-5
BS = 256               # batch of dates (paper used 128 over O(1) variance; larger for step-count/speed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- faithful model
class GraphConvLayer(nn.Module):
    """adj @ (X @ W) + b. xavier_uniform(W), bias init ones -- GNNHAR.py:129-147."""

    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_f, out_f))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        self.bias = nn.Parameter(torch.ones(1, out_f)) if bias else None

    def forward(self, x, adj):                       # x:(B,N,in_f) adj:(N,N)
        h = torch.matmul(x, self.weight)             # (B,N,out_f)
        out = torch.matmul(adj, h)                   # (N,N)@(B,N,out_f) broadcasts over batch
        return out + self.bias if self.bias is not None else out


class GNNHAR(nn.Module):
    """HAR linear term H1=Linear(F,1) + n_gcn-layer GCN -> mlp(nhid,1); res=relu(H1+Hg).
    n_gcn=1 -> GNNHAR1L, n_gcn=2 -> GNNHAR2L (GNNHAR.py:190-244)."""

    def __init__(self, in_f, n_hid, n_gcn):
        super().__init__()
        self.linear1 = nn.Linear(in_f, 1, bias=True)
        nn.init.constant_(self.linear1.bias, 1.0)    # positive start (target scaled to mean~1) avoids
        #                                              dead-ReLU collapse under QLIKE; mirrors the repo's
        #                                              bias=ones init + its restart-on-collapse loop
        gcns = [GraphConvLayer(in_f, n_hid, bias=False)]
        gcns += [GraphConvLayer(n_hid, n_hid, bias=False) for _ in range(n_gcn - 1)]
        self.gcns = nn.ModuleList(gcns)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h1 = self.linear1(x)                         # (B,N,1)
        hg = x
        for gcn in self.gcns:
            hg = self.relu(gcn(hg, adj))
        hg = self.mlp1(hg)                           # (B,N,1)
        return self.relu(h1 + hg).squeeze(-1)        # (B,N)


def _qlike_loss(pred, y, mask):
    """Paper QLIKE: tf = y/(f+1e-4); mean(tf - log tf) over masked cells (GNNHAR.py:322-324)."""
    f = pred + 1e-4
    tf = torch.clamp(y, min=FL) / f
    loss = tf - torch.log(tf)
    return (loss * mask).sum() / mask.sum()


def _train_once(X, Ys, Mt, adj, tr, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    torch.manual_seed(seed)
    model = GNNHAR(in_f, N_HID, n_gcn).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state, bad = float("inf"), None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g, device=DEVICE)]
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            opt.zero_grad()
            loss = _qlike_loss(model(X[b], adj), Ys[b], Mt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = _qlike_loss(model(X[va_idx], adj), Ys[va_idx], Mt[va_idx]).item()
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience and epoch >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def train_predict(X, Ys, Mt, adj, tr_idx, va_idx, te_idx, in_f, n_gcn, seed, max_epochs, patience):
    """Train GNNHAR on scaled target, early-stop on validation QLIKE; restart on collapse (faithful to
    GNNHAR.py:428-433). Scaled-QLIKE optimum ~1; >10 => dead-ReLU/non-convergence -> retry w/ new seed."""
    tr = torch.as_tensor(tr_idx, device=DEVICE)
    model, best_val = _train_once(X, Ys, Mt, adj, tr, va_idx, in_f, n_gcn, seed, max_epochs, patience)
    tries = 0
    while (not np.isfinite(best_val) or best_val > 1.6) and tries < 4:   # scaled QLIKE+1; >1.6 => poor fit
        tries += 1
        model, best_val = _train_once(X, Ys, Mt, adj, tr, va_idx, in_f, n_gcn,
                                      seed + 1000 * tries, max_epochs, patience)
    model.eval()
    with torch.no_grad():
        return model(X[te_idx], adj).cpu().numpy(), best_val   # (len(te_idx), N) scaled, val loss


def build_fold_tensors(fold, tickers, feats, tr_dates_mask_date):
    """Pivot fold -> (D,N,F) z-scored features (train stats), (D,N) target, (D,N) mask, date/ticker maps."""
    dts = np.sort(fold["date"].unique())
    dpos = {d: i for i, d in enumerate(dts)}
    cpos = {t: j for j, t in enumerate(tickers)}
    D, N = len(dts), len(tickers)
    r = fold["date"].map(dpos).to_numpy()
    c = fold["ticker"].map(cpos).to_numpy()
    mask = np.zeros((D, N), np.float32)
    mask[r, c] = 1.0
    Y = np.zeros((D, N), np.float32)
    Y[r, c] = fold["y"].to_numpy(np.float32)
    is_train_date = np.array([tr_dates_mask_date(d) for d in dts])           # bool over rows
    train_cells = mask.astype(bool) & is_train_date[:, None]
    X = np.zeros((D, N, len(feats)), np.float32)
    for fi, col in enumerate(feats):
        mm = np.zeros((D, N), np.float32)
        mm[r, c] = fold[col].to_numpy(np.float32)
        vals = mm[train_cells]                                               # train-only stats
        mu, sd = float(vals.mean()), float(vals.std())
        sd = sd if sd > 1e-12 else 1.0
        z = (mm - mu) / sd
        z[~mask.astype(bool)] = 0.0                                          # absent nodes -> train mean
        X[:, :, fi] = z
    sc = 1.0 / max(float(Y[train_cells].mean()), FL)                        # causal target scale
    return X, Y, Y * sc, mask, sc, dpos, cpos, dts


CMP = [("GNNHAR2L-corr-HAR3", "HAR"), ("GNNHAR2L-corr-OWN9", "GBM"),
       ("GNNHAR2L-corr-OWN9", "GBM+corr"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR2L-none-HAR3"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR2L-plac-HAR3"),
       ("GNNHAR2L-corr-OWN9", "GNNHAR2L-none-OWN9"),
       ("GNNHAR2L-corr-OWN9", "GNNHAR2L-plac-OWN9"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR1L-corr-HAR3")]


def pool_write(out, h, seeds, base, gnn_names, preds, yy, dts_all, gnn_seed_q, outpath, verbose):
    """Pool per-obs QLIKE over folds completed so far, run date-clustered DM, write JSON. Called after
    every fold (crash-resilient checkpoint) and at horizon end (verbose table)."""
    y = np.concatenate(yy)
    dates = np.concatenate(dts_all)
    e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in preds}
    q = {m: float(np.mean(e[m])) for m in e}
    if verbose:
        print(f"\n===== SP500 h{h} (n={len(y):,}, {len(seeds)} seeds, {len(yy)} folds) =====", flush=True)
        for m in base + gnn_names:
            spread = ""
            if gnn_seed_q.get(m):
                arr = np.array(gnn_seed_q[m])
                spread = f"  (per-seed QLIKE {arr.mean():.4f}+-{arr.std():.4f})"
            print(f"  {m:20s} QLIKE {q[m]:.4f}{spread}", flush=True)
        print("  --- DM (pos pct = first/row model better, lower QLIKE) ---", flush=True)
    dmr = {}
    for x, b in CMP:
        p = ST.date_clustered_dm(e[x], e[b], dates, h)["p_value"]
        dmr[f"{x}_vs_{b}"] = p
        if verbose:
            print(f"  DM {x:20s} vs {b:20s}: {(q[b]-q[x])/q[b]*100:+.2f}% (p={p:.3f})", flush=True)
    out[f"h{h}"] = {"n": int(len(y)), "n_folds": len(yy), "qlike": q, "dm": dmr,
                    "per_seed_qlike": {m: [round(v, 6) for v in gnn_seed_q[m]] for m in gnn_seed_q}}
    outpath.write_text(json.dumps(out, indent=2))


def main():  # pragma: no cover - entry driver: argparse + full training loop over folds/seeds (helpers tested)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 2 folds, few epochs, 1 seed")
    args = ap.parse_args()
    horizons = (1,) if args.smoke else (1, 5, 10, 22)
    seeds = (0,) if args.smoke else SEEDS
    max_epochs, patience = (250, 30) if args.smoke else (150, 20)
    fold_cap = 1 if args.smoke else None

    # GNN configs: (name, feature_set, n_gcn, adj_type)
    configs = [
        ("GNNHAR2L-corr-HAR3", HAR3, 2, "corr"),
        ("GNNHAR2L-none-HAR3", HAR3, 2, "none"),
        ("GNNHAR2L-plac-HAR3", HAR3, 2, "plac"),
        ("GNNHAR1L-corr-HAR3", HAR3, 1, "corr"),
        ("GNNHAR2L-corr-OWN9", OWN9, 2, "corr"),
        ("GNNHAR2L-none-OWN9", OWN9, 2, "none"),
        ("GNNHAR2L-plac-OWN9", OWN9, 2, "plac"),
    ]
    base = ["HAR", "GBM", "GBM+corr"]
    gnn_names = [c[0] for c in configs]

    frames, sect, _ = FM.load("sp500")
    print(f"loaded {len(frames)} sp500 tickers; device={DEVICE}", flush=True)
    outdir = REPO / "results" / "gnnhar"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    outpath = outdir / f"gnnhar_sp500{tag}.json"
    out = {}
    for h in horizons:
        t0 = time.time()
        a = FM.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        N = len(tickers)
        preds = {m: [] for m in base + gnn_names}
        gnn_seed_q = {c[0]: [] for c in configs}      # per-seed QLIKE spread (per fold concat later)
        yy, dts_all = [], []
        n_done = 0
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < 30000:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            Wc, Wp = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            keep = list(dict.fromkeys(OWN9 + ["date", "ticker", "y", "parkinson_variance"]))
            fold = a.loc[(a.date >= S1.TRAIN_START) & (a.date < tend), keep].copy()
            fold["g_corr"] = FM.nb(fold, tickers, Wc)
            fold["g_corr"] = fold["g_corr"].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            y = tef["y"].to_numpy(float)

            # ---- baselines (same folds/rows as full_matrix) ----
            preds["HAR"].append(FM._har_ols(trf, tef))
            preds["GBM"].append(np.mean([FM.gbm(trf, tef, OWN9, s) for s in SEEDS], 0))
            preds["GBM+corr"].append(np.mean([FM.gbm(trf, tef, OWN9 + ["g_corr"], s) for s in SEEDS], 0))

            # ---- GNNHAR ----
            tcut = ts - embargo
            tr_date_fn = (lambda d, tcut=tcut: d < np.datetime64(tcut))
            adj_cache = {"corr": torch.as_tensor(Wc, dtype=torch.float32, device=DEVICE),
                         "plac": torch.as_tensor(Wp, dtype=torch.float32, device=DEVICE),
                         "none": torch.eye(N, dtype=torch.float32, device=DEVICE)}
            x_cache = {}
            for name, feats, n_gcn, adj_type in configs:
                key = tuple(feats)
                if key not in x_cache:
                    Xn, Yraw, Ys, mask, sc, dpos, cpos, dts = build_fold_tensors(fold, tickers, feats, tr_date_fn)
                    Xt = torch.as_tensor(Xn, device=DEVICE)
                    Yst = torch.as_tensor(Ys, device=DEVICE)
                    Mt = torch.as_tensor(mask, device=DEVICE)
                    dmask = np.array([tr_date_fn(d) for d in dts])
                    tr_idx = np.where(dmask)[0]
                    va_idx = tr_idx[-VALID_LEN:]
                    tr_idx = tr_idx[:-VALID_LEN]
                    te_idx = np.where((dts >= np.datetime64(ts)) & (dts < np.datetime64(tend)))[0]
                    cidx = tef["ticker"].map(cpos).to_numpy()
                    te_pos = {d: i for i, d in enumerate(te_idx)}
                    row_in_te = np.array([te_pos[dpos[d]] for d in tef["date"]])
                    x_cache[key] = (Xt, Yst, Mt, sc, tr_idx, va_idx, te_idx, cidx, row_in_te)
                Xt, Yst, Mt, sc, tr_idx, va_idx, te_idx, cidx, row_in_te = x_cache[key]
                adj = adj_cache[adj_type]
                seed_preds, vlosses = [], []
                for sd in seeds:
                    pm, bv = train_predict(Xt, Yst, Mt, adj, tr_idx, va_idx, te_idx,
                                           len(feats), n_gcn, sd, max_epochs, patience)
                    pr = np.maximum(pm[row_in_te, cidx] / sc, FL)           # raw-scale, floored
                    seed_preds.append(pr)
                    vlosses.append(bv)
                    gnn_seed_q[name].append(float(np.mean(M.per_obs_qlike(y, pr, floor=FL))))
                preds[name].append(np.mean(seed_preds, 0))                  # seed ensemble
                print(f"    {name:20s} val(QLIKE+1)~{np.mean(vlosses):.3f} "
                      f"testQLIKE~{np.mean(gnn_seed_q[name][-len(seeds):]):.3f}", flush=True)
            yy.append(y)
            dts_all.append(tef["date"].to_numpy())
            del fold, trf, tef, x_cache, adj_cache
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            # crash-resilient: checkpoint pooled result over folds completed so far (quiet)
            pool_write(out, h, seeds, base, gnn_names, preds, yy, dts_all, gnn_seed_q, outpath, verbose=False)
            print(f"  h{h} fold {k} ({ts.date()}) done, {time.time()-t0:.0f}s (checkpointed)", flush=True)

        pool_write(out, h, seeds, base, gnn_names, preds, yy, dts_all, gnn_seed_q, outpath, verbose=True)
        print(f"  saved {outpath.name} (through h{h})", flush=True)
        del a
        gc.collect()
    print(f"\nsaved {outpath}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
