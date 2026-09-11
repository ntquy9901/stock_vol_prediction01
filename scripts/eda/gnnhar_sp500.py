"""Faithful re-implementation of GNNHAR (Zhang, Pu, Cucuringu & Dong, Int. J. Forecasting 2024,
arXiv:2308.01419; official code https://github.com/chaozhang-ox/GNNHAR) evaluated under OUR full-matrix
protocol so a real learned multi-hop message-passing GNN sits in the SAME table as full_matrix.py, on
both S&P 500 and (thin-market) HOSE.

Why: our paper argues cross-firm graph structure adds no reliable incremental QLIKE beyond own-history
+ a market factor. Our graph models feed ONE hand-built neighbour-mean scalar into a gamma-GBM. A
reviewer objects a real GNN (learned multi-hop message passing) could exploit graph structure a scalar
cannot. GNNHAR is the flagship HAR-baselined, QLIKE-evaluated graph-vol model, so running it under our
protocol directly answers that objection.

Reported models, in the ORIGINAL-PAPER-FIRST order:
  * HAR, GHAR  -- the GNNHAR paper's own baselines (HAR + a LINEAR graph-HAR: OLS on the 3 HAR lags plus
    the corr-graph neighbour-average of those 3 lags, floored like HAR).
  * GBM, GBM+corr  -- our own-history gamma-GBM (the delivered SP500/VN champion) and its scalar-graph
    variant.
  * GNNHAR2L/1L * {corr, none(=A=I no-graph control), plac(=random edges)} * {HAR3, OWN9}  -- our learned
    GNN variants (the reviewer's objection made concrete).

Faithful to the official repo:
  * GraphConvLayer: output = adj @ (X @ W) + b  (xavier_uniform W, bias init ones) -- GNNHAR.py:129-147
  * HAR / GNNHAR1L / GNNHAR2L: res = relu(H1 + H_gcn), H1 = Linear(F,1) own-feature linear term,
    H_gcn = (multi-layer) GCN -> mlp(nhid,1).  GNNHAR.py:150-277  (generalised from F=3 to F in {3,10})
  * QLIKE loss: tf = y/f ; L = mean(tf - log tf)  -- GNNHAR.py:317-328 (Loss.forward, QLike branch).
  * Adam(lr=1e-3, weight_decay=1e-5), best-validation-checkpoint, multi-seed ensemble -- GNNHAR.py:357,332-433
  * n_hid = 9 (paper default).

Adaptations (documented, protocol-matched, causal):
  * Target = daily Parkinson variance pk.shift(-h) (OUR target, IDENTICAL to full_matrix) for h in
    {1,5,10,22} -- NOT the paper's horizon-averaged target, so GNNHAR's QLIKE goes in the same table.
  * Graph = correlation top-k on TRAIN log-vol only (S1.build_graph) -- EXACTLY the graph our GBM+corr and
    GHAR use, so "learned GNN message passing vs scalar neighbour-mean on the same graph" is a clean test.
  * No-graph control = A=I (per-node nonlinear MLP, GNNHAR's own key ablation). Placebo = random edges of
    matched density. Both train-only.
  * Node features: HAR3 = 3 HAR lags (paper's inputs); OWN9 = full_matrix.OWN (own-history) for a
    matched-input comparison to GBM.
  * Features z-scored (train stats), target scaled by 1/train-mean for numerical conditioning (QLIKE is
    scale-equivariant up to an additive constant).
  * Cross-section has missing tickers (IPO/delist) -> masked loss + masked forward, excluded from
    QLIKE/DM exactly like full_matrix's dropna rows.

Over/under-fit evidence (gate-compliant): ONE JSON PER HORIZON -- results/gnnhar/gnnhar_<market>_h<h>.json --
carrying test ``metrics`` + ``train_metrics`` + ``val_metrics`` (all 5: mse/rmse/mae/r2/qlike, all models) and
``fit_diagnostics`` (overfit_check.classify_fit verdict for each learned GNN). HOSE additionally carries
``per_fold_qlike`` because HOSE QLIKE is dominated by regime-spike folds (COVID 2020 / 2022 / Apr-2025) that
must be visible, not hidden in the pooled mean.

Run: python gnnhar_sp500.py [--market {sp500,hose}] [--smoke]
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
sys.path.insert(0, str(REPO / "scripts" / "quality_gate"))
import overfit_check as OF  # noqa: E402  (classify_fit -> per-model over/under-fit verdict)

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


def fit_gnn(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    """Train GNNHAR on scaled target, early-stop on validation QLIKE; restart on collapse (faithful to
    GNNHAR.py:428-433). Scaled-QLIKE optimum ~1; >1.6 => dead-ReLU/non-convergence -> retry w/ new seed.
    Returns the best-validation model + its val loss (predictions produced separately by ``row_preds`` so
    the SAME trained model scores the train / val / test windows for the over/under-fit evidence)."""
    tr = torch.as_tensor(tr_idx, device=DEVICE)
    model, best_val = _train_once(X, Ys, Mt, adj, tr, va_idx, in_f, n_gcn, seed, max_epochs, patience)
    tries = 0
    while (not np.isfinite(best_val) or best_val > 1.6) and tries < 4:   # scaled QLIKE+1; >1.6 => poor fit
        tries += 1
        model, best_val = _train_once(X, Ys, Mt, adj, tr, va_idx, in_f, n_gcn,
                                      seed + 1000 * tries, max_epochs, patience)
    return model, best_val


def row_preds(model, X, adj, date_idx, row_in, cidx, sc):
    """Raw-scale, floored GNN predictions for the (date,ticker) rows a split covers: score the fold tensor
    at ``date_idx`` dates, then gather cell (row_in[i], cidx[i]) per observation and undo the target scale."""
    model.eval()
    with torch.no_grad():
        pm = model(X[date_idx], adj).cpu().numpy()             # (len(date_idx), N) scaled
    return np.maximum(pm[row_in, cidx] / sc, FL)


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


def _row_in_te(tef_dates, dpos, te_idx):
    """Index of each row's date within ``te_idx`` (the fold-pivot positions of that split's dates).

    Uses pandas ``.map`` for the date->fold-position lookup so it is robust to a datetime64-vs-Timestamp
    key mismatch across pandas versions: a direct ``dpos[d]`` dict lookup raised ``KeyError`` on pandas
    builds where ``fold['date'].unique()`` yields ``np.datetime64`` keys while ``tef['date']`` iterates
    ``Timestamp`` values.
    """
    te_pos = {int(p): i for i, p in enumerate(te_idx)}
    return np.array([te_pos[int(p)] for p in tef_dates.map(dpos)])


def _nb_col(fold, tickers, W, col):
    """Weighted neighbour average of ``col`` for every (date,ticker) row, via graph weight matrix W
    (row-normalised, W[i,j] = weight of neighbour j for node i). Same pivot/aggregate shape as FM.nb but
    for an arbitrary column (FM.nb is hard-wired to parkinson_variance)."""
    piv = fold.pivot_table(index="date", columns="ticker", values=col).reindex(columns=tickers).sort_index()
    V = piv.to_numpy(float)
    Vf = np.nan_to_num(np.where(np.isnan(V), np.nanmean(V, axis=1, keepdims=True), V))
    NB = Vf @ W.T
    dpos = {d: i for i, d in enumerate(piv.index)}
    cpos = {c: j for j, c in enumerate(tickers)}
    return NB[fold["date"].map(dpos).to_numpy(), fold["ticker"].map(cpos).to_numpy()]


def ghar_ols(tr, te, cols):
    """GHAR = the GNNHAR paper's LINEAR graph-HAR baseline: OLS on the 3 HAR lags PLUS the corr-graph
    neighbour-average of those 3 lags. Floored identically to FM._har_ols (0.01 * mean floored train y)."""
    x = lambda df: np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])   # noqa: E731
    c = np.linalg.lstsq(x(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    return np.maximum(x(te) @ c, 0.01 * np.maximum(tr["y"], FL).mean())


def _metrics5(y, p):
    """The 5 mandatory metrics for one model on one split."""
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": M.qlike(y, p, floor=FL)}


# DM comparisons (first model better = lower QLIKE). Paper baselines first, then our GBM, then the GNNs.
CMP = [("GHAR", "HAR"),
       ("GNNHAR2L-corr-HAR3", "HAR"),
       ("GNNHAR2L-corr-HAR3", "GHAR"),
       ("GNNHAR2L-corr-OWN9", "GBM"),
       ("GNNHAR2L-corr-OWN9", "GBM+corr"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR2L-none-HAR3"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR2L-plac-HAR3"),
       ("GNNHAR2L-corr-OWN9", "GNNHAR2L-none-OWN9"),
       ("GNNHAR2L-corr-OWN9", "GNNHAR2L-plac-OWN9"),
       ("GNNHAR2L-corr-HAR3", "GNNHAR1L-corr-HAR3")]


def pool_write(h, market, seeds, base, gnn_names, learned, preds_te, yy_te, dts_te,
               preds_tr, yy_tr, preds_va, yy_va, gnn_seed_q, outpath, verbose):
    """Pool per-obs QLIKE over folds completed so far; compute the 5 metrics on the pooled train / val /
    test predictions of every model; stamp a classify_fit verdict for each learned GNN; run date-clustered
    DM on the test losses; write ONE per-horizon JSON carrying the full over/under-fit evidence. Called
    after every fold (crash-resilient checkpoint, quiet) and at horizon end (verbose table)."""
    order = base + gnn_names
    y = np.concatenate(yy_te)
    dates = np.concatenate(dts_te)
    pooled = {m: np.concatenate(preds_te[m]) for m in preds_te}
    e = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in preds_te}
    metrics = {m: _metrics5(y, pooled[m]) for m in order}
    y_tr = np.concatenate(yy_tr)
    y_va = np.concatenate(yy_va)
    train_metrics = {m: _metrics5(y_tr, np.concatenate(preds_tr[m])) for m in order}
    val_metrics = {m: _metrics5(y_va, np.concatenate(preds_va[m])) for m in order}
    fit_diagnostics = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in learned}
    if verbose:
        print(f"\n===== {market} h{h} (n={len(y):,}, {len(seeds)} seeds, {len(yy_te)} folds) =====", flush=True)
        for m in order:
            spread = ""
            if gnn_seed_q.get(m):
                arr = np.array(gnn_seed_q[m])
                spread = f"  (per-seed QLIKE {arr.mean():.4f}+-{arr.std():.4f})"
            v = fit_diagnostics.get(m)
            fit = f"  [fit={v['status']}]" if v else ""
            print(f"  {m:20s} QLIKE {metrics[m]['qlike']:.4f}{spread}{fit}", flush=True)
        print("  --- DM (pos pct = first/row model better, lower QLIKE) ---", flush=True)
    dm = {}
    for x, b in CMP:
        p = ST.date_clustered_dm(e[x], e[b], dates, h)["p_value"]
        dm[f"{x}_vs_{b}"] = p
        if verbose:
            qb, qx = metrics[b]["qlike"], metrics[x]["qlike"]
            print(f"  DM {x:20s} vs {b:20s}: {(qb - qx) / qb * 100:+.2f}% (p={p:.3f})", flush=True)
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy_te),
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit_diagnostics, "dm": dm,
           "per_seed_qlike": {m: [round(v, 6) for v in gnn_seed_q[m]] for m in gnn_seed_q}}
    if market == "hose":
        # HOSE QLIKE is dominated by regime-spike folds (COVID 2020 / 2022 / Apr-2025) -> expose per-fold.
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy_te[i], preds_te[m][i], floor=FL)))
                                     for i in range(len(yy_te))] for m in preds_te}
    outpath.write_text(json.dumps(doc, indent=2))


def main():  # pragma: no cover - entry driver: argparse + full training loop over folds/seeds (helpers tested)
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=("sp500", "hose"), default="sp500")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, few epochs, 1 seed")
    args = ap.parse_args()
    market = args.market
    horizons = (1,) if args.smoke else (1, 5, 10, 22)
    seeds = (0,) if args.smoke else SEEDS
    max_epochs, patience = (60, 15) if args.smoke else (150, 20)
    fold_cap = 1 if args.smoke else None
    min_train = 30000 if market == "sp500" else 3000

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
    base = ["HAR", "GHAR", "GBM", "GBM+corr"]     # paper baselines first (HAR, GHAR), then our GBM
    gnn_names = [c[0] for c in configs]
    learned = list(gnn_names)                     # only the GNNs carry a fit verdict

    frames, sect, edates = FM.load(market)
    print(f"loaded {len(frames)} {market} tickers; device={DEVICE}", flush=True)
    outdir = REPO / "results" / "gnnhar"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    ghar_cols = HAR3 + [f"g_{c}" for c in HAR3]
    for h in horizons:
        t0 = time.time()
        outpath = outdir / f"gnnhar_{market}{tag}_h{h}.json"
        a = FM.panel(frames, {}, h)               # no earnings features (edates gated out; none for either)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        N = len(tickers)
        preds_te = {m: [] for m in base + gnn_names}
        preds_tr = {m: [] for m in base + gnn_names}
        preds_va = {m: [] for m in base + gnn_names}
        gnn_seed_q = {c[0]: [] for c in configs}
        yy_te, yy_tr, yy_va, dts_te = [], [], [], []
        n_done = 0
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_train:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            Wc, Wp = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            keep = list(dict.fromkeys(OWN9 + ["date", "ticker", "y", "parkinson_variance"]))
            fold = a.loc[(a.date >= S1.TRAIN_START) & (a.date < tend), keep].copy()
            fold["g_corr"] = FM.nb(fold, tickers, Wc)
            fold["g_corr"] = fold["g_corr"].fillna(0.0)
            for c in HAR3:                          # neighbour-average of each HAR lag for the GHAR baseline
                fold[f"g_{c}"] = np.nan_to_num(_nb_col(fold, tickers, Wc, c), nan=0.0)
            # train/val/test dataframe row sets (val = last VALID_LEN train dates -> matches the GNN split)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            val_dates = np.sort(trf["date"].unique())[-VALID_LEN:]
            is_val = trf["date"].isin(val_dates)
            trf_e = trf[~is_val]
            vaf = trf[is_val]
            y_te = tef["y"].to_numpy(float)

            # ---- paper/own baselines (same folds/rows as full_matrix), scored on train/val/test ----
            for split, sub in (("te", tef), ("tr", trf_e), ("va", vaf)):
                dst = {"te": preds_te, "tr": preds_tr, "va": preds_va}[split]
                dst["HAR"].append(FM._har_ols(trf, sub))
                dst["GHAR"].append(ghar_ols(trf, sub, ghar_cols))
                dst["GBM"].append(np.mean([FM.gbm(trf, sub, OWN9, s) for s in SEEDS], 0))
                dst["GBM+corr"].append(np.mean([FM.gbm(trf, sub, OWN9 + ["g_corr"], s) for s in SEEDS], 0))

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
                    all_tr = np.where(np.array([tr_date_fn(d) for d in dts]))[0]
                    va_idx, tr_idx = all_tr[-VALID_LEN:], all_tr[:-VALID_LEN]
                    te_idx = np.where((dts >= np.datetime64(ts)) & (dts < np.datetime64(tend)))[0]
                    map_te = (te_idx, _row_in_te(tef["date"], dpos, te_idx), tef["ticker"].map(cpos).to_numpy())
                    map_tr = (tr_idx, _row_in_te(trf_e["date"], dpos, tr_idx), trf_e["ticker"].map(cpos).to_numpy())
                    map_va = (va_idx, _row_in_te(vaf["date"], dpos, va_idx), vaf["ticker"].map(cpos).to_numpy())
                    x_cache[key] = (Xt, Yst, Mt, sc, tr_idx, va_idx, map_te, map_tr, map_va)
                Xt, Yst, Mt, sc, tr_idx, va_idx, map_te, map_tr, map_va = x_cache[key]
                adj = adj_cache[adj_type]
                seed_te, seed_tr, seed_va, vlosses = [], [], [], []
                for sd in seeds:
                    model, bv = fit_gnn(Xt, Yst, Mt, adj, tr_idx, va_idx, len(feats), n_gcn, sd, max_epochs, patience)
                    seed_te.append(row_preds(model, Xt, adj, *map_te, sc))
                    seed_tr.append(row_preds(model, Xt, adj, *map_tr, sc))
                    seed_va.append(row_preds(model, Xt, adj, *map_va, sc))
                    vlosses.append(bv)
                    gnn_seed_q[name].append(float(np.mean(M.per_obs_qlike(y_te, seed_te[-1], floor=FL))))
                preds_te[name].append(np.mean(seed_te, 0))          # seed ensemble
                preds_tr[name].append(np.mean(seed_tr, 0))
                preds_va[name].append(np.mean(seed_va, 0))
                print(f"    {name:20s} val(QLIKE+1)~{np.mean(vlosses):.3f} "
                      f"testQLIKE~{np.mean(gnn_seed_q[name][-len(seeds):]):.3f}", flush=True)
            yy_te.append(y_te)
            yy_tr.append(trf_e["y"].to_numpy(float))
            yy_va.append(vaf["y"].to_numpy(float))
            dts_te.append(tef["date"].to_numpy())
            del fold, trf, tef, trf_e, vaf, x_cache, adj_cache
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            # crash-resilient: checkpoint pooled result over folds completed so far (quiet)
            pool_write(h, market, seeds, base, gnn_names, learned, preds_te, yy_te, dts_te,
                       preds_tr, yy_tr, preds_va, yy_va, gnn_seed_q, outpath, verbose=False)
            print(f"  h{h} fold {k} ({ts.date()}) done, {time.time()-t0:.0f}s (checkpointed)", flush=True)

        pool_write(h, market, seeds, base, gnn_names, learned, preds_te, yy_te, dts_te,
                   preds_tr, yy_tr, preds_va, yy_va, gnn_seed_q, outpath, verbose=True)
        print(f"  saved {outpath.name}", flush=True)
        del a
        gc.collect()
    print(f"\nsaved results/gnnhar/gnnhar_{market}{tag}_h*.json", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
