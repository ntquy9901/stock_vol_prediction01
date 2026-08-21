"""E3/E4 static convex forecast combination with frozen experts (plan sections 8, 9).

Both experts are frozen; the weight alpha is fit on VALIDATION predictions only (never the test set).
For squared error the validation optimum is closed-form; for QLIKE it is found on a dense [0,1] grid.
Running this per horizon yields E4 (horizon-specific alpha). alpha -> 1 means the neural expert adds no
reliable incremental value under a convex blend (report it; not an implementation failure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import metrics as M  # noqa: E402


def _align(da: dict, db: dict):
    keys = sorted(set(da) & set(db))
    y = np.array([da[k][0] for k in keys])
    a = np.array([da[k][1] for k in keys]); b = np.array([db[k][1] for k in keys])
    return keys, y, a, b


def fit_alpha_mse(y, har, nn) -> float:
    """Closed-form validation optimum for squared error, clipped to [0, 1]."""
    den = float(np.sum((har - nn) ** 2))
    if den <= 0:
        return 1.0
    return float(np.clip(np.sum((y - nn) * (har - nn)) / den, 0.0, 1.0))


def fit_alpha_qlike(y, har, nn, floor: float, grid: int = 201) -> float:
    """Dense-grid validation minimizer of QLIKE over alpha in [0, 1]."""
    best_a, best_q = 1.0, np.inf
    for a in np.linspace(0.0, 1.0, grid):
        p = np.maximum(a * har + (1.0 - a) * nn, floor)
        q = M.qlike(y, p, floor=floor)
        if q < best_q:
            best_a, best_q = float(a), q
    return best_a


def blend(har_test: dict, nn_test: dict, har_val: dict, nn_val: dict,
          loss: str, floor: float) -> tuple[float, dict]:
    """Fit alpha on validation, apply to test. Returns (alpha, blended test pred dict)."""
    _, yv, hv, nv = _align(har_val, nn_val)
    alpha = fit_alpha_mse(yv, hv, nv) if loss == "mse" else fit_alpha_qlike(yv, hv, nv, floor)
    keys, yt, ht, nt = _align(har_test, nn_test)
    out = {k: (float(yt[i]), float(alpha * ht[i] + (1.0 - alpha) * nt[i])) for i, k in enumerate(keys)}
    return alpha, out
