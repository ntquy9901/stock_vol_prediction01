"""Small sequential LSTM feature-extractor for the LSTM-feature / GBME-blend baseline, with leakage-safe
out-of-fold cross-fitting.

The LSTM reads a fixed ``SEQ_LEN``-day window of the OWN-8 (+earnings) own-history feature vectors ending at
day ``t`` and regresses the h-step-ahead log-variance ``log(pk_{t+h})``. Its (variance-space) prediction is
then handed to the champion gamma-GBM as ONE extra causal feature (``lstmfeat``). This is the transferable germ
of arXiv:2505.23084 (deep-sequential (+) tree-ensemble) adapted to the variance-QLIKE thesis.

Out-of-fold cross-fitting (mirrors baselines/2026-09-14_gbm_gnn_embed/embed.py): within a walk-forward fold's
train window the train-row features are produced by an inner temporal K-fold -- every train row is embedded by
an LSTM that did NOT see its date as a training target; the test features come from an LSTM trained on the FULL
train window. Feature/target standardisation and the early-stop split all use train / inner-train rows only.
This avoids the stacking-leakage trap where the GBM overfits an in-sample deep feature that does not generalise.

Sequences are built ONCE over the full panel (each sequence uses only PAST feature rows of its own ticker, so
it is causal), then selected positionally per fold; the LSTM only ever trains on train-window target rows.

Batching: whole standardised sequence tensors go to the GPU and are trained in ``BATCH``-sized steps (no
batch=1). Reuses ``full_matrix`` (FL / gamma-GBM) via the driver's path bootstrap; edits no shared module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402,F401  (import first so `metrics`/`stats` resolve via its path setup)
import lstm_blend_config as C  # noqa: E402

FL = FM.FL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- model
class SeqLSTM(nn.Module):
    """A small LSTM over a [B, SEQ_LEN, F] window -> scalar (standardised log-variance at the last step)."""

    def __init__(self, in_f, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(in_f, hidden, layers, batch_first=True,
                            dropout=(dropout if layers > 1 else 0.0))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                 # [B, SEQ_LEN, hidden]
        return self.head(out[:, -1, :]).squeeze(-1)   # [B]


def _forward_batched(model, X, batch):
    """Forward ``X`` through ``model`` in ``batch``-sized chunks (no whole-set forward -> bounded VRAM even for
    the hundreds-of-thousands of sequences in a HOSE train window). Returns a single concatenated tensor."""
    outs = [model(X[s:s + batch]) for s in range(0, len(X), batch)]
    return torch.cat(outs) if len(outs) > 1 else outs[0]


# ----------------------------------------------------------------------------- sequence construction
def build_sequences(a, feats, seq_len):
    """Build one causal sequence per panel row: the ``seq_len`` own-history feature rows of that row's ticker
    ending at (and including) its own date, left-padded by repeating the ticker's earliest row when short.

    ``a`` is the walk-forward panel (reset index, so row position == ``a.index``). Returns
    ``(X_all [N, seq_len, F] float32, logvar_all [N] float64)`` aligned to ``a``'s row order, where
    ``logvar_all = log(max(y, FL))`` is the h-step-ahead log-variance target."""
    fvals = a[feats].to_numpy(np.float32)
    n, fdim = len(a), len(feats)
    X_all = np.zeros((n, seq_len, fdim), np.float32)
    for _tk, idx in a.groupby("ticker", sort=False).indices.items():
        idx = np.asarray(idx)                 # positional rows of this ticker, in date order
        block = fvals[idx]                    # [T_tk, F]
        for j, pos in enumerate(idx):
            lo = max(0, j - seq_len + 1)
            win = block[lo:j + 1]             # up to seq_len rows ending at j (causal)
            if len(win) < seq_len:            # left-pad by repeating the earliest available row
                win = np.vstack([np.repeat(win[:1], seq_len - len(win), axis=0), win])
            X_all[pos] = win
    logvar_all = np.log(np.maximum(a["y"].to_numpy(float), FL))
    return X_all, logvar_all


def inner_blocks(train_dates, k):
    """Split sorted unique ``train_dates`` into up to ``k`` contiguous temporal blocks (drop empties)."""
    return [b for b in np.array_split(np.sort(np.unique(train_dates)), k) if len(b)]


def _val_split(dates, pos_train, valid_len):
    """Temporal early-stop split of ``pos_train``: the trailing ``valid_len`` train dates form validation, the
    rest train. ``dates`` is the whole-panel date array. Returns (inner_train_pos, inner_val_pos)."""
    d = dates[pos_train]
    uniq = np.sort(np.unique(d))
    vl = max(1, min(valid_len, len(uniq) - 1))
    val_dates = set(uniq[-vl:])
    is_val = np.array([x in val_dates for x in d])
    return pos_train[~is_val], pos_train[is_val]


# ----------------------------------------------------------------------------- torch training
def train_lstm(Xtr, ytr, Xva, yva, seed, max_epochs, patience, record=False):
    """Train a ``SeqLSTM`` on standardised (Xtr, ytr), early-stopping on the validation MSE (Xva, yva).

    Batches ``C.BATCH`` sequences per step on ``DEVICE`` (no batch=1). Returns ``(model, curves)`` where
    ``curves`` is [] unless ``record`` (then per-epoch train/val MSE for the learning-curve evidence)."""
    torch.manual_seed(seed)
    model = SeqLSTM(Xtr.shape[2], C.HIDDEN, C.LAYERS, C.DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WD)
    lossf = nn.MSELoss()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    n = len(Xtr)
    best_val, best_state, bad, curves = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=DEVICE)
        for s in range(0, n, C.BATCH):
            b = perm[s:s + C.BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(_forward_batched(model, Xva, C.BATCH), yva).item()
            if record:
                tloss = lossf(_forward_batched(model, Xtr, C.BATCH), ytr).item()
                curves.append({"epoch": epoch, "train_mse": tloss, "val_mse": vloss})
        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and epoch >= C.MIN_EPOCH:
                break
    if best_state is not None:  # pragma: no cover - always set (epoch 0 improves from inf); defensive guard
        model.load_state_dict(best_state)
    return model, curves


def torch_trainer(Xtr, ytr, Xva, yva, Xpred, seed, max_epochs, patience, record=False):
    """Default trainer: train an LSTM then predict the ``Xpred`` rows. Returns (pred_std [n_pred], curves).

    ``pred_std`` is in the STANDARDISED log-variance space (the caller de-standardises). Tests inject a cheap
    fake with this same signature to exercise the walk-forward driver without real torch training."""
    model, curves = train_lstm(Xtr, ytr, Xva, yva, seed, max_epochs, patience, record=record)
    model.eval()
    with torch.no_grad():
        pred = _forward_batched(model, Xpred, C.BATCH).cpu().numpy().astype(np.float64)
    return pred, curves


# ----------------------------------------------------------------------------- feature generation
def _group_pred(X_all, logvar_all, dates, pos_train, pos_pred, seeds, max_epochs, patience, trainer,
                record=False):
    """Train LSTM(s) on ``pos_train`` (standardise + early-stop split from those rows only) and predict the
    ``pos_pred`` rows' h-ahead log-variance, seed-averaged. Returns (pred_logvar [len(pos_pred)], curves).

    Feature standardisation uses per-feature mean/std over the ``pos_train`` cells; the target is standardised
    by the ``pos_train`` log-variance mean/std. Fails loud if the early-stop split leaves no validation row."""
    flat = X_all[pos_train].reshape(-1, X_all.shape[2])
    mu = flat.mean(0).astype(np.float32)                       # float32 stats: keep standardisation in float32
    sd = (flat.std(0) + FL).astype(np.float32)                 # (a float64 broadcast would double a ~1GB array)
    ymu = float(logvar_all[pos_train].mean())
    ysd = float(logvar_all[pos_train].std()) + FL
    tr_pos, va_pos = _val_split(dates, pos_train, C.VALID_LEN)
    if len(va_pos) == 0 or len(tr_pos) == 0:
        raise ValueError(f"early-stop split degenerate: {len(tr_pos)} train / {len(va_pos)} val rows")

    def _to_dev(pos):
        return torch.as_tensor((X_all[pos] - mu) / sd, device=DEVICE)   # float32 in, float32 out

    Xtr, Xva, Xpred = _to_dev(tr_pos), _to_dev(va_pos), _to_dev(pos_pred)
    ytr = torch.as_tensor(((logvar_all[tr_pos] - ymu) / ysd).astype(np.float32), device=DEVICE)
    yva = torch.as_tensor(((logvar_all[va_pos] - ymu) / ysd).astype(np.float32), device=DEVICE)
    preds, curves = [], []
    for i, sd_seed in enumerate(seeds):
        p_std, cv = trainer(Xtr, ytr, Xva, yva, Xpred, sd_seed, max_epochs, patience, record=(record and i == 0))
        preds.append(np.asarray(p_std, float) * ysd + ymu)     # de-standardise back to log-variance
        if record and i == 0:
            curves = cv
    return np.mean(preds, 0), curves


def _to_variance(logvar):
    """Map a log-variance prediction to a clipped variance-space feature in ``[FL, PRED_CAP]``."""
    return np.clip(np.exp(np.clip(logvar, np.log(FL), np.log(C.PRED_CAP))), FL, C.PRED_CAP)


def oof_train_feat(X_all, logvar_all, dates, pos_train, seeds, max_epochs, patience, base_seed, k=None,
                   trainer=None):
    """Out-of-fold train-row lstmfeat: inner temporal K-fold over ``pos_train`` -- every train row is predicted
    by an LSTM whose training dates EXCLUDE that row's date. Returns a variance-space feature aligned to
    ``pos_train`` order. Raises if any train row is left unfilled (fail-loud coverage guarantee)."""
    trainer = trainer or torch_trainer
    k = C.INNER_K if k is None else k
    d_train = dates[pos_train]
    blocks = inner_blocks(d_train, k)
    feat = np.full(len(pos_train), np.nan)
    for j, held in enumerate(blocks):
        held_set = {np.datetime64(x) for x in held}
        in_held = np.array([np.datetime64(x) in held_set for x in d_train])
        inner_train_pos = pos_train[~in_held]
        held_pos = pos_train[in_held]
        pred_logvar, _ = _group_pred(X_all, logvar_all, dates, inner_train_pos, held_pos, seeds,
                                     max_epochs, patience, trainer, record=False)
        feat[in_held] = _to_variance(pred_logvar)
    if np.isnan(feat).any():
        raise RuntimeError("OOF coverage gap: some train rows were not embedded")
    return feat


def test_feat(X_all, logvar_all, dates, pos_train, pos_test, seeds, max_epochs, patience, base_seed,
              trainer=None):
    """Test-row lstmfeat: ONE LSTM trained on the FULL train window (``pos_train``), predicting ``pos_test``.
    Returns (variance-space feature aligned to ``pos_test`` order, learning_curves of the first seed)."""
    trainer = trainer or torch_trainer
    pred_logvar, curves = _group_pred(X_all, logvar_all, dates, pos_train, pos_test, seeds,
                                      max_epochs, patience, trainer, record=True)
    return _to_variance(pred_logvar), curves
