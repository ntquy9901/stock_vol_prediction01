"""GARCH(1,1) / GJR-GARCH(1,1,1) estimation + causal recursion + multi-step forecast.

Pure, unit-tested core. Estimation delegates to the ``arch`` package (Kevin Sheppard); the conditional-
variance filter and the h-step-ahead closed form are implemented here so they can be verified against
the published recursion (CLAUDE.md: named estimators use the published formula + a test-vs-formula).

Units: returns are scaled by ``config.SCALE`` (=100) for ``arch``'s numerical conditioning; every forecast
is divided by ``SCALE**2`` to land on the same variance scale as ``parkinson_variance`` (variance of the
daily LOG-return). See design/design.md section 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import arch  # module-level: a missing/broken arch install fails LOUD at import, never silently -> all-fallback
import numpy as np

import config


@dataclass(frozen=True)
class Params:
    """Fitted GARCH parameters (on the SCALE'd return series) plus a fallback marker.

    ``ok=False`` means estimation failed / was degenerate / had too few observations; the caller then
    forecasts the unconditional sample variance ``fallback_var`` (ORIGINAL return scale)."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    ok: bool
    fallback_var: float


def reversion_persistence(p: Params, variant: str) -> float:
    """Mean-reversion persistence φ governing the multi-step forecast.

    GARCH: φ = α + β. GJR: φ = α + β + γ/2 (expected leverage under a symmetric zero-mean
    innovation, ``P(ε<0)=0.5`` for Normal / Student-t)."""
    phi = p.alpha + p.beta
    if variant == "gjr":
        phi += p.gamma / 2.0
    return phi


def multistep(s_next_scaled: np.ndarray, p: Params, variant: str, h: int) -> np.ndarray:
    """h-step-ahead conditional variance (SCALE'd) from the one-step forecast σ²_{t+1|t}.

    ``σ²_{t+h|t} = σ̄² + φ^{h-1}·(σ²_{t+1|t} − σ̄²)`` with σ̄² = ω/(1−φ). For h=1 returns σ²_{t+1|t};
    as h→∞ reverts to σ̄². Vectorised over ``s_next_scaled``."""
    phi = reversion_persistence(p, variant)
    uncond = p.omega / (1.0 - phi)
    s_next_scaled = np.asarray(s_next_scaled, dtype=float)
    return uncond + phi ** (h - 1) * (s_next_scaled - uncond)


def one_step_next(returns_scaled: np.ndarray, p: Params, variant: str) -> np.ndarray:
    """Causal one-step-ahead conditional variance σ²_{t+1|t} (SCALE'd) for every index t.

    Filters σ²_i = ω + (α + γ·1(ε_{i-1}<0))·ε²_{i-1} + β·σ²_{i-1} over the full return history
    (initialised at the unconditional variance), then returns
    ``s_next[t] = ω + (α + γ·1(ε_t<0))·ε²_t + β·σ²_t`` = σ²_{t+1|t}. ``s_next[t]`` depends only on
    returns with index ≤ t (causal)."""
    eps = np.asarray(returns_scaled, dtype=float) - p.mu
    n = eps.shape[0]
    neg = (eps < 0.0).astype(float)
    e2 = eps ** 2
    sig = np.empty(n, dtype=float)
    sig[0] = p.omega / (1.0 - reversion_persistence(p, variant))     # unconditional variance init
    lev = p.gamma if variant == "gjr" else 0.0
    for i in range(1, n):
        sig[i] = p.omega + (p.alpha + lev * neg[i - 1]) * e2[i - 1] + p.beta * sig[i - 1]
    return p.omega + (p.alpha + lev * neg) * e2 + p.beta * sig


def fit_params(train_returns: np.ndarray, variant: str) -> Params:
    """ML-fit GARCH/GJR params on ``train_returns * SCALE``; degrade to a fallback marker on failure.

    Falls back (``ok=False``) when there are fewer than ``config.MIN_TRAIN_OBS`` returns, the ``arch``
    fit raises, or the fitted params are degenerate (ω ≤ 0 or reversion persistence φ ∉ (LO, HI)).
    ``fallback_var`` is the unconditional sample variance of the train returns in ORIGINAL scale."""
    train_returns = np.asarray(train_returns, dtype=float)
    fallback_var = float(np.var(train_returns, ddof=1)) if train_returns.size >= 2 else 0.0
    bad = Params(0.0, 0.0, 0.0, 0.0, 0.0, ok=False, fallback_var=fallback_var)
    if train_returns.size < config.MIN_TRAIN_OBS:
        return bad
    y = train_returns * config.SCALE
    kw = dict(mean=config.MEAN, vol="GARCH", p=1, q=1, dist=config.DIST)
    if variant == "gjr":
        kw["o"] = 1
    try:
        res = arch.arch_model(y, **kw).fit(disp="off", show_warning=False)
        pr = res.params
        p = Params(mu=float(pr.get("mu", 0.0)), omega=float(pr["omega"]), alpha=float(pr["alpha[1]"]),
                   gamma=float(pr.get("gamma[1]", 0.0)) if variant == "gjr" else 0.0,
                   beta=float(pr["beta[1]"]), ok=True, fallback_var=fallback_var)
    except Exception:   # arch/scipy convergence or numeric failure -> unconditional-variance fallback
        return bad
    phi = reversion_persistence(p, variant)
    if not (p.omega > 0.0 and config.PERSIST_LO < phi < config.PERSIST_HI):
        return bad          # degenerate / non-stationary fit -> fallback (never emit garbage)
    uncond_orig = (p.omega / (1.0 - phi)) / (config.SCALE ** 2)   # model-implied unconditional variance
    if not (fallback_var > 0.0
            and fallback_var / config.VAR_RATIO_CAP <= uncond_orig <= fallback_var * config.VAR_RATIO_CAP):
        return bad          # near-IGARCH collapse (omega~0 -> uncond~0) or explosion (phi~1 -> uncond huge):
    return p                # implied unconditional variance is orders of magnitude off the sample variance


def forecast(returns: np.ndarray, p: Params, variant: str, test_idx: np.ndarray, h: int) -> np.ndarray:
    """h-step forecasts (ORIGINAL variance scale) at ``test_idx`` given fitted ``p``.

    On fallback returns the unconditional sample variance for every requested index; otherwise runs the
    causal recursion on the full return history and rescales by ``÷ SCALE**2``."""
    test_idx = np.asarray(test_idx, dtype=int)
    if not p.ok:
        return np.full(test_idx.shape[0], p.fallback_var, dtype=float)
    s_next = one_step_next(np.asarray(returns, dtype=float) * config.SCALE, p, variant)
    f = multistep(s_next[test_idx], p, variant, h) / (config.SCALE ** 2)
    # safety net: a variance forecast cannot credibly sit VAR_RATIO_CAP x off the ticker's own sample
    # variance -- clip transient multi-step explosions / collapses (p.ok fits have fallback_var > 0).
    return np.clip(f, p.fallback_var / config.VAR_RATIO_CAP, p.fallback_var * config.VAR_RATIO_CAP)


def ticker_forecast(returns: np.ndarray, n_train: int, test_idx: np.ndarray, h: int,
                    variant: str) -> np.ndarray:
    """Convenience: fit on the first ``n_train`` returns then forecast ``test_idx`` at horizon ``h``."""
    returns = np.asarray(returns, dtype=float)
    p = fit_params(returns[:n_train], variant)
    return forecast(returns, p, variant, test_idx, h)
