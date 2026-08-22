"""E9 static gated residual + E10 dynamic regime-aware soft gate (plan section 12).

The residual expert is frozen; its per-node correction ``c`` (normalized) and scale are supplied. The gate
scales the correction: additive ``har + lam*scale*c`` or multiplicative ``(har+eps)*exp(lam*scale*c)``.
  E9: one static ``lam`` per horizon, fit on validation QLIKE (grid). lam -> 0 = collapse to HAR.
  E10: ``lam(z) = sigmoid(g(z))`` from a small MLP over OBSERVABLE state z at t; HAR-biased bias init so
       lam starts small (~0.1); trained on train QLIKE, early-stopped on validation QLIKE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import metrics as M  # noqa: E402


def reconstruct(framing, harp, c, scale, eps, lam, pred_floor=0.0):
    """Raw prediction from a gated correction, floored at pred_floor (train-derived positive mapping)."""
    if framing == "additive":
        return np.maximum(harp + lam * scale * c, pred_floor)
    return np.maximum((harp + eps) * np.exp(lam * scale * c), pred_floor)


def fit_lambda_static(y_va, harp_va, c_va, scale, framing, eps, floor,
                      lam_max: float = 2.0, grid: int = 201, pred_floor=0.0) -> float:
    """E9: grid-search the static correction strength that minimizes validation QLIKE."""
    best_l, best_q = 0.0, np.inf
    for L in np.linspace(0.0, lam_max, grid):
        p = reconstruct(framing, harp_va, c_va, scale, eps, L, pred_floor)
        q = M.qlike(y_va.reshape(-1), p.reshape(-1), floor=floor)
        if q < best_q:
            best_l, best_q = float(L), q
    return best_l


def _state_features(harp, c, scale):
    """Observable-at-t state z per (date,node): [market HAR level, cross-sectional dispersion,
    node HAR level, |correction magnitude|]. All known at the prediction cutoff."""
    market = harp.mean(1, keepdims=True) * np.ones_like(harp)      # [n,N]
    disp = harp.std(1, keepdims=True) * np.ones_like(harp)
    mag = np.abs(scale * c)
    z = np.stack([market, disp, harp, mag], axis=-1)              # [n,N,4]
    return z.reshape(-1, 4).astype(np.float32)


class _Gate(nn.Module):
    def __init__(self, in_dim=4, hidden=8, lam_max=2.0, bias0=-2.2):
        super().__init__()
        self.lam_max = lam_max
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias0)               # sigmoid(-2.2)~0.10 -> HAR-biased start

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(-1) * self.lam_max


def fit_gate_dynamic(D, corr, cfg, seed, eps, lam_max: float = 2.0):
    """E10: train a small soft gate over observable state; early-stop on validation QLIKE.

    ``corr`` is the frozen residual expert's bundle (c_tr/c_va/c_te, framing). Returns (test pred dict,
    val pred dict, mean_test_lambda)."""
    framing = corr["framing"]
    scale = D["add_scale"] if framing == "additive" else D["mul_scale"]
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prep(c_arr, harp, y):
        z = torch.from_numpy(_state_features(harp, c_arr, scale)).to(device)
        return z, harp, y

    z_tr, _, _ = prep(corr["c_tr"], D["harp_tr"], D["y_tr"])
    gate = _Gate(lam_max=lam_max).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=cfg.weight_decay)

    harp_tr_t = torch.from_numpy(D["harp_tr"].astype(np.float32)).to(device).reshape(-1)
    c_tr_t = torch.from_numpy(corr["c_tr"].astype(np.float32)).to(device).reshape(-1)
    y_tr_t = torch.from_numpy(D["y_tr"].astype(np.float32)).to(device).reshape(-1)
    scale_t = torch.from_numpy(np.broadcast_to(scale, D["harp_tr"].shape).astype(np.float32)).to(device).reshape(-1)
    floor = cfg.qlike_floor
    pf = D.get("pred_floor", floor)
    pf_arr = np.broadcast_to(pf, D["harp_tr"].shape).astype(np.float32)
    pf_t = torch.from_numpy(pf_arr).to(device).reshape(-1)

    def recon_torch(lam, harp, c, sc):
        if framing == "additive":
            p = harp + lam * sc * c
        else:
            p = (harp + eps) * torch.exp(lam * sc * c)
        return torch.maximum(p, pf_t)

    best = np.inf; best_state = None; wait = 0
    for ep in range(max(cfg.epochs, 15)):
        gate.train(); opt.zero_grad()
        lam = gate(z_tr)
        p = recon_torch(lam, harp_tr_t, c_tr_t, scale_t)
        qlike = (y_tr_t / p + torch.log(p)).mean()               # QLIKE up to constant
        qlike.backward(); opt.step()
        # val QLIKE
        gate.eval()
        with torch.no_grad():
            zv = torch.from_numpy(_state_features(corr["c_va"], D["harp_va"], scale)).to(device)
            lamv = gate(zv).cpu().numpy().reshape(D["harp_va"].shape)
        pv = reconstruct(framing, D["harp_va"], corr["c_va"], scale, eps, lamv, pf)
        vq = M.qlike(D["y_va"].reshape(-1), pv.reshape(-1), floor=floor)
        if vq < best - 1e-9:
            best = vq; best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}; wait = 0
        else:
            wait += 1
        if ep + 1 >= 5 and wait >= max(cfg.patience, 3):
            break
    if best_state:
        gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        zt = torch.from_numpy(_state_features(corr["c_te"], D["harp_te"], scale)).to(device)
        lamt = gate(zt).cpu().numpy().reshape(D["harp_te"].shape)
        zv = torch.from_numpy(_state_features(corr["c_va"], D["harp_va"], scale)).to(device)
        lamv = gate(zv).cpu().numpy().reshape(D["harp_va"].shape)
    pte = reconstruct(framing, D["harp_te"], corr["c_te"], scale, eps, lamt, pf)
    pva = reconstruct(framing, D["harp_va"], corr["c_va"], scale, eps, lamv, pf)
    te = {(j, D["d_te"][i]): (float(D["y_te"][i, j]), float(pte[i, j]))
          for i in range(len(D["d_te"])) for j in range(D["N"])}
    va = {(j, D["d_va"][i]): (float(D["y_va"][i, j]), float(pva[i, j]))
          for i in range(len(D["d_va"])) for j in range(D["N"])}
    return te, va, float(np.mean(lamt))
