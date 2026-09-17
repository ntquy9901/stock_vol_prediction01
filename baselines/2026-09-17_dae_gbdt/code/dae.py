"""Denoising autoencoder (DAE) representation for the GBME+DAE baseline (Jahrer/Kaggle-Grandmaster paradigm).

An UNSUPERVISED autoencoder over the champion OWN-8 + EARN tabular features, trained to reconstruct the clean
(standardised) input from a SWAP-NOISE-corrupted copy (Vincent et al. 2008 denoising AE; Jahrer 2018 Porto
Seguro 1st-place solution -- swap noise = replace a fraction of each column's cells with values drawn from
other rows of the SAME column). The bottleneck hidden `Z in R^K` is extracted and concatenated to the raw
features for the downstream gamma-GBM.

FROZEN-BASIS (design section 4): ONE DAE is trained on the burn-in window (the first eligible walk-forward
fold's causal train rows), FROZEN, and used to embed EVERY panel row -- a single shared latent basis. This
deliberately avoids the per-fold neural-basis drift that detonated the sibling GNN-embed baseline's
long-horizon QLIKE (a neural hidden is only defined up to rotation/permutation, so stitching Z from
separately-fitted networks is ill-posed). The standardiser (feature mean/std) and DAE weights fit on burn-in
rows only, so embedding later folds' rows is out-of-sample to the extractor; the DAE never sees any label, so
there is no label leakage even for the burn-in rows that are in-sample to the extractor (inherent to a frozen
feature extractor).

No shared module is edited; the downstream GBM + floor are reused from `full_matrix`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import dae_config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DAE(nn.Module):
    """Symmetric MLP autoencoder: encoder in_f -> HIDDEN... -> K bottleneck; decoder K -> ...HIDDEN -> in_f.

    `embed(x)` returns the bottleneck Z; `forward(x)` returns the reconstruction. ReLU between hidden layers.
    """

    def __init__(self, in_f, hidden, k):
        super().__init__()
        enc, prev = [], in_f
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc += [nn.Linear(prev, k)]
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], k
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec += [nn.Linear(prev, in_f)]
        self.decoder = nn.Sequential(*dec)

    def embed(self, x):
        """Return the bottleneck representation Z (n, K) for standardised inputs x (n, in_f)."""
        return self.encoder(x)

    def forward(self, x):
        """Return the reconstruction (n, in_f) of standardised inputs x (n, in_f)."""
        return self.decoder(self.encoder(x))


def swap_noise(x, rate, generator):
    """Return a swap-noise-corrupted copy of `x` (rows = observations, cols = features).

    Each cell is, INDEPENDENTLY with probability `rate`, replaced by the value of the SAME column drawn from
    another random row of `x` (Jahrer swap noise). Works on CPU or GPU tensors (batched, no per-cell loop).
    """
    n, f = x.shape
    mask = torch.rand(n, f, generator=generator, device=x.device) < rate
    src = torch.randint(0, n, (n, f), generator=generator, device=x.device)
    gathered = torch.gather(x, 0, src)          # gathered[i, j] = x[src[i, j], j]  (same-column value)
    return torch.where(mask, gathered, x)


def standardize(Xburn, Xall):
    """Z-score `Xburn` and `Xall` by the BURN-IN column mean/std (causal: stats from burn-in rows only).

    A zero-variance (constant) column keeps std=1 so it maps to 0, not NaN. Returns (Xb_std, Xa_std)."""
    mu = Xburn.mean(0)
    sd = Xburn.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((Xburn - mu) / sd).astype(np.float32), ((Xall - mu) / sd).astype(np.float32)


def val_len(n_dates):
    """Trailing-validation length: `C.VALID_LEN` but never >= the date count (>=1 train date remains)."""
    return max(1, min(C.VALID_LEN, n_dates - 1))


def val_split(dates):
    """Row indices (tr_idx, va_idx) for a TEMPORAL burn-in split: the last `val_len` unique dates form the
    clean validation slice, the rest are train."""
    uniq = np.sort(np.unique(dates))
    val_dates = set(uniq[-val_len(len(uniq)):])
    is_val = np.array([d in val_dates for d in dates])
    return np.where(~is_val)[0], np.where(is_val)[0]


def train_dae(Xtr, Xva, in_f, hidden, k, epochs, patience, seed, record=False):
    """Train a `DAE` to reconstruct clean standardised rows from swap-noise-corrupted inputs, batched on GPU,
    early-stopping on the CLEAN validation reconstruction MSE. Reuses the config optimiser constants (single
    source). Returns (model, best_val, curves); `curves` is [] unless `record` (then per-epoch train/val
    reconstruction MSE for the learning-curve evidence).

    Batching: `C.BATCH` rows per step over the GPU-resident tensor, fully vectorised swap noise -- no batch=1.
    """
    torch.manual_seed(seed)
    model = DAE(in_f, hidden, k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lossfn = nn.MSELoss()
    Xt = torch.as_tensor(Xtr, device=DEVICE)
    Xv = torch.as_tensor(Xva, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best_val, best_state, bad, curves = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen, device=DEVICE)
        for s in range(0, len(perm), C.BATCH):
            b = perm[s:s + C.BATCH]
            clean = Xt[b]
            opt.zero_grad()
            loss = lossfn(model(swap_noise(clean, C.SWAP_RATE, gen)), clean)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossfn(model(Xv), Xv).item()
            if record:
                tloss = lossfn(model(Xt), Xt).item()
                curves.append({"epoch": epoch, "train_recon_mse": tloss, "val_recon_mse": vloss})
        if vloss < best_val - 1e-9:
            best_val, best_state, bad = vloss, {kk: v.detach().clone() for kk, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and epoch >= C.MIN_EPOCH:
                break
    if best_state is not None:  # pragma: no cover - always set (epoch 0 improves from inf); defensive guard
        model.load_state_dict(best_state)
    return model, best_val, curves


def default_trainer(Xburn, burn_dates, Xall, k, epochs, patience, seed, record=False):
    """Standardise on burn-in, temporally split it for early stopping, train ONE DAE, and embed EVERY row of
    `Xall`. Returns (z_all (n_all, k), curves). This is the injectable trainer the driver calls; tests inject
    a cheap fake with the same signature so the walk-forward driver runs without real torch training."""
    Xb, Xa = standardize(Xburn, Xall)
    tr_idx, va_idx = val_split(burn_dates)
    model, _bv, curves = train_dae(Xb[tr_idx], Xb[va_idx], Xburn.shape[1], C.HIDDEN, k, epochs, patience,
                                   seed, record=record)
    model.eval()
    with torch.no_grad():
        z = model.embed(torch.as_tensor(Xa, device=DEVICE)).cpu().numpy().astype(np.float32)
    return z, curves


def frozen_z(df, feats, burnin_dates, k, epochs, patience, seed, trainer=None):
    """Frozen-basis embeddings: train ONE DAE on the burn-in rows of `df` (dates in `burnin_dates`) and embed
    EVERY row of `df` with that single frozen DAE (one shared latent basis; no per-fold refit).

    `df` must contain `feats` + a `date` column. Causal iff every downstream test date is >= the burn-in
    dates (the DAE, being unsupervised and frozen, never sees any label). Returns
    ({int row_index -> z (k,)}, learning_curves)."""
    trainer = trainer or default_trainer
    burn = df[df["date"].isin(burnin_dates)]
    if burn["date"].nunique() < 2:
        raise ValueError(f"burn-in has {burn['date'].nunique()} date(s); need >=2 to train + validate the DAE")
    Xburn = burn[feats].to_numpy(np.float32)
    Xall = df[feats].to_numpy(np.float32)
    z, curves = trainer(Xburn, burn["date"].to_numpy(), Xall, k, epochs, patience, seed, record=True)
    return {int(ix): row for ix, row in zip(df.index.to_numpy(), z)}, curves
