"""Learned GNN node embedding for the GBME+GNN-embed baseline, with leakage-safe out-of-fold cross-fitting.

The embedding is the penultimate hidden of the faithful GNNHAR (`gnnhar_sp500.GNNHAR`) — the R^{n_hid}
representation AFTER the two GCN layers and BEFORE the final `mlp1` head. `GNNEmbedder` is a thin subclass
that adds an `embed()` method and NO parameters, so it shares GNNHAR's weights exactly.

Out-of-fold cross-fitting (design section 3): within a walk-forward fold's train window the train-row
embeddings are produced by an inner temporal K-fold — every train row is embedded by a GNN that did NOT see
it. The test embeddings come from a GNN trained on the FULL train window. Graph, feature scaler, GNN and
(downstream) GBM all fit on train / inner-train rows only. This avoids the stacking-leakage trap where the
GBM overfits to in-sample GNN features that do not generalise.

Reuses `gnnhar_sp500` (model, tensor builder, QLIKE loss, hyper-parameters) and `vn_gbm_graph_stage1`
(train-only correlation graph). No shared module is edited.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402,F401  (import first so `metrics`/`stats` resolve via its path setup)
import gnnhar_sp500 as G  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import gnn_embed_config as C  # noqa: E402

DEVICE = G.DEVICE


class GNNEmbedder(G.GNNHAR):
    """GNNHAR exposing the post-GCN / pre-mlp1 hidden as an R^{n_hid} node embedding.

    Adds no parameters, so a fitted `GNNHAR.state_dict()` loads into it unchanged: train with the proven
    `G.fit_gnn` loop (or `train_embedder` here for learning curves), then wrap for embedding.
    """

    def embed(self, x, adj):
        """Return the penultimate hidden `hg` (B, N, n_hid) after the GCN stack, before the mlp1 head."""
        hg = x
        for gcn in self.gcns:
            hg = self.relu(gcn(hg, adj))
        return hg


def train_embedder(X, Ys, Mt, adj, tr_idx, va_idx, in_f, seed, max_epochs, patience, record=False):
    """Train a `GNNEmbedder` on `tr_idx` dates (batched over dates on GPU), early-stopping on `va_idx`
    QLIKE. Reuses GNNHAR's loss + LR/WD/batch (single source). Returns (model, best_val, curves) where
    `curves` is [] unless `record` (then per-epoch train/val QLIKE for the learning-curve evidence).

    Batching: `G.BS` dates per step, whole cross-section per date, masked QLIKE — no batch=1.
    """
    if C.ARCH != "gcn":
        raise NotImplementedError(f"ARCH={C.ARCH!r} not implemented; only 'gcn' is supported")
    torch.manual_seed(seed)
    model = GNNEmbedder(in_f, G.N_HID, C.N_GCN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=G.LR, weight_decay=G.WD)
    tr = torch.as_tensor(tr_idx, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best_val, best_state, bad, curves = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen, device=DEVICE)]
        for s in range(0, len(perm), G.BS):
            b = perm[s:s + G.BS]
            opt.zero_grad()
            loss = G._qlike_loss(model(X[b], adj), Ys[b], Mt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = G._qlike_loss(model(X[va_idx], adj), Ys[va_idx], Mt[va_idx]).item()
            if record:
                tloss = G._qlike_loss(model(X[tr], adj), Ys[tr], Mt[tr]).item()
                curves.append({"epoch": epoch, "train_qlike": tloss, "val_qlike": vloss})
        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and epoch >= C.MIN_EPOCH:
                break
    if best_state is not None:  # pragma: no cover - always set (epoch 0 improves from inf); defensive guard
        model.load_state_dict(best_state)
    return model, best_val, curves


def seed_embed(X, Ys, Mt, adj, tr_idx, va_idx, in_f, seeds, max_epochs, patience, emb_idx, row_in, cidx,
               record=False):
    """Seed-ensembled per-row embedding: train the GNN once per seed, embed the (emb_idx dates, cell
    (row_in, cidx)) rows, average `z` over seeds. Returns (z [n_rows, n_hid], curves-of-first-seed)."""
    zs, curves = [], []
    for i, sd in enumerate(seeds):
        model, _, cv = train_embedder(X, Ys, Mt, adj, tr_idx, va_idx, in_f, sd, max_epochs, patience,
                                      record=record and i == 0)
        if record and i == 0:
            curves = cv
        model.eval()
        with torch.no_grad():
            Z = model.embed(X[emb_idx], adj).cpu().numpy()      # (len(emb_idx), N, n_hid)
        zs.append(Z[row_in, cidx].astype(np.float32))
    return np.mean(zs, 0), curves


def _val_len(n_train_dates):
    """Trailing-validation length for the GNN early stop: `C.VALID_LEN` but never >= the train-date count."""
    return max(1, min(C.VALID_LEN, n_train_dates - 1))


def _group_z(df, tickers, feats, train_dates, emb_dates, graph_seed, seeds, max_epochs, patience, trainer,
             record=False):
    """Train an embedding GNN on `train_dates` (stats + graph from those rows only) and embed the rows of
    `df` whose date is in `emb_dates`. Returns (z [n_emb_rows, n_hid], emb_row_index, curves).

    `df` must contain exactly the union of train_dates and emb_dates. The graph is built on the train-date
    rows only (leakage-safe); the feature z-scoring uses train-date cells only (via G.build_fold_tensors)."""
    train_set = {np.datetime64(d) for d in train_dates}
    is_train_date = lambda d: np.datetime64(d) in train_set   # noqa: E731
    X, _Y, Ys, mask, sc, dpos, cpos, dts = G.build_fold_tensors(df, tickers, feats, is_train_date)
    Xt = torch.as_tensor(X, device=DEVICE)
    Yst = torch.as_tensor(Ys, device=DEVICE)
    Mt = torch.as_tensor(mask, device=DEVICE)
    W, _Wp = S1.build_graph(df[df["date"].isin(train_dates)], tickers, np.random.default_rng(graph_seed))
    adj = torch.as_tensor(W, dtype=torch.float32, device=DEVICE)
    all_tr = np.where(np.array([is_train_date(d) for d in dts]))[0]
    if len(all_tr) < 2:                                        # fail loud: cannot hold out a val date and
        raise ValueError(f"inner-train has {len(all_tr)} date(s); need >=2 to train + validate")  # still train
    vl = _val_len(len(all_tr))
    tr_idx, va_idx = all_tr[:-vl], all_tr[-vl:]
    emb_rows = df[df["date"].isin(emb_dates)]
    emb_idx = np.where(np.isin(dts, np.array([np.datetime64(d) for d in emb_dates])))[0]
    row_in = G._row_in_te(emb_rows["date"], dpos, emb_idx)
    cidx = emb_rows["ticker"].map(cpos).to_numpy()
    z, curves = trainer(Xt, Yst, Mt, adj, tr_idx, va_idx, len(feats), seeds, max_epochs, patience,
                        emb_idx, row_in, cidx, record=record)
    return z, emb_rows.index.to_numpy(), curves


def inner_blocks(train_dates, k):
    """Split sorted unique `train_dates` into up to `k` contiguous temporal blocks (drop empties)."""
    return [b for b in np.array_split(np.sort(np.unique(train_dates)), k) if len(b)]


def oof_train_z(trf, tickers, feats, seeds, max_epochs, patience, base_seed, k=None, trainer=None):
    """Out-of-fold train embeddings: inner temporal K-fold over `trf` — every train row is embedded by a
    GNN whose training dates EXCLUDE that row's date. Returns z aligned to `trf`'s row order (n_trf, n_hid).

    Raises if any train row is left unfilled (fail-loud coverage guarantee)."""
    trainer = trainer or seed_embed
    k = C.INNER_K if k is None else k
    dates = np.sort(trf["date"].unique())
    blocks = inner_blocks(dates, k)
    z = np.full((len(trf), G.N_HID), np.nan, np.float32)
    posmap = {ix: i for i, ix in enumerate(trf.index.to_numpy())}
    for j, held in enumerate(blocks):
        held_set = {np.datetime64(d) for d in held}
        inner_train = np.array([d for d in dates if np.datetime64(d) not in held_set])
        zj, idx_j, _ = _group_z(trf, tickers, feats, inner_train, held, base_seed + j, seeds,
                                max_epochs, patience, trainer, record=False)
        for zi, ix in zip(zj, idx_j):
            z[posmap[ix]] = zi
    if np.isnan(z).any():
        raise RuntimeError("OOF coverage gap: some train rows were not embedded")
    return z


def test_z(fold, trf, tef, tickers, feats, seeds, max_epochs, patience, base_seed, trainer=None):
    """Test embeddings: one GNN trained on the FULL train window (graph + stats from train only), embed the
    test rows. Returns (z aligned to `tef` row order (n_tef, n_hid), learning_curves of the first seed)."""
    trainer = trainer or seed_embed
    train_dates = np.sort(trf["date"].unique())
    test_dates = np.sort(tef["date"].unique())
    z, idx, curves = _group_z(fold, tickers, feats, train_dates, test_dates, base_seed, seeds,
                              max_epochs, patience, trainer, record=True)
    order = {ix: i for i, ix in enumerate(idx)}
    z_aligned = z[[order[ix] for ix in tef.index.to_numpy()]]
    return z_aligned, curves


def frozen_z(df, tickers, feats, burnin_dates, seeds, max_epochs, patience, base_seed, trainer=None):
    """Frozen-basis embeddings (design section 9 follow-up, the fixed-basis variant).

    Train ONE GNN on `burnin_dates` only -- graph, feature scaler and weights all come from those rows -- then
    embed EVERY row of `df` with that single frozen model. All folds therefore share ONE latent basis: there is
    no per-fold refit and hence none of the inner-vs-test embedding basis drift that made the OOF variant's
    long-horizon QLIKE detonate (a GNN hidden is only defined up to rotation/permutation, so stitching `z` from
    separately-fitted GNNs is ill-posed; a single frozen GNN removes that ambiguity). Causal iff every
    downstream test date is strictly AFTER `burnin_dates` (the GNN never sees any test row). This is the
    standard frozen-feature-extractor setup: test embeddings are out-of-sample to the GNN, train embeddings on
    later folds are too; only the burn-in fold's own train rows are in-sample to the extractor, which is
    inherent to transfer learning and carries no label leakage.

    Returns ({int row_index -> z (n_hid,)}, learning_curves)."""
    trainer = trainer or seed_embed
    all_dates = np.sort(df["date"].unique())
    z, idx, curves = _group_z(df, tickers, feats, burnin_dates, all_dates, base_seed, seeds,
                              max_epochs, patience, trainer, record=True)
    return {int(ix): row for ix, row in zip(idx, z)}, curves
