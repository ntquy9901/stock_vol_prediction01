"""Self-contained classical volatility baselines for the SOICT LSTM-GAT submission.

Two families are exposed:

* **HAR** (Heterogeneous AutoRegressive) via plain OLS with intercept on the three HAR
  components (daily / weekly / monthly). ``har_fit`` returns ``[intercept, b_d, b_w, b_m]``
  through ``np.linalg.lstsq`` on the design matrix ``[1, X]``; ``har_predict`` applies the
  coefficients and floors the output so a volatility forecast is never negative.

* **GARCH(1,1)** via the ``arch`` package. The forecast target here is the Parkinson
  **VARIANCE** series (the processed ``parkinson_volatility`` column is a variance, not a
  standard deviation). GARCH models the conditional variance of a *return-like* series, so we
  build pseudo returns ``r_t = sign_t * sqrt(var_t)`` (random +/-1 sign, zero mean, variance
  ``var_t``) and scale them x100 for numerical conditioning exactly as the project's classical
  ladder does (``baselines/classical_baselines/code/classical_baselines.py``). The GARCH
  conditional-variance forecast in that scaled space is divided by ``100**2 = 1e4`` to recover a
  forecast on the raw Parkinson-variance scale, directly comparable to the target.

Robustness: if ``arch`` is unavailable, or the fit / forecast raises or returns a non-finite or
non-positive value for a series, ``garch_forecast`` falls back to the mean of the training
variance series (a valid, strictly positive constant forecast) and logs the reason. The floor
guarantees every returned forecast is ``>= floor > 0``.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# `arch` availability is probed once at import time so the fallback path is explicit.
try:
    from arch import arch_model

    _ARCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only when arch is missing
    arch_model = None
    _ARCH_AVAILABLE = False
    logger.warning("arch package unavailable (%s); garch_forecast will use variance-mean fallback", exc)

FLOOR = 1e-8
_RETURN_SCALE = 100.0  # scale pseudo returns to ~percent range for arch's optimiser conditioning
_MAX_PERSISTENCE = 0.999  # cap alpha+beta so near-unit-root (IGARCH) fits do not diverge multi-step


# --------------------------------------------------------------------------------------------------
# HAR (OLS)
# --------------------------------------------------------------------------------------------------
def har_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS with intercept: y ~ [1, X]. Returns coef = [intercept, b1, b2, b3] (shape (4,))."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must have shape (n, 3), got {X.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def har_predict(X: np.ndarray, coef: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """Predict intercept + X @ coef[1:], clamped to be >= floor (volatility is non-negative)."""
    X = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must have shape (n, 3), got {X.shape}")
    if coef.shape != (4,):
        raise ValueError(f"coef must have shape (4,), got {coef.shape}")
    pred = coef[0] + X @ coef[1:]
    return np.maximum(pred, floor)


# --------------------------------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------------------------------
def _fallback_forecast(train_series: np.ndarray, n_test: int, floor: float, reason: str) -> np.ndarray:
    """Constant variance forecast = mean of the (positive-floored) train variance series.

    Uses only the FINITE entries of ``train_series`` (this path is reached precisely when the
    series is short/degenerate, possibly with NaN/inf), so the docstring guarantee of a finite,
    positive, ``>= floor`` output holds; if nothing finite remains, returns ``floor``.
    """
    if not (np.isfinite(floor) and floor > 0.0):
        raise ValueError(f"floor must be finite and positive, got {floor}")
    logger.warning("garch_forecast falling back to train-variance mean: %s", reason)
    finite = np.asarray(train_series, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    mean_var = float(np.mean(np.maximum(finite, floor))) if finite.size else floor
    # R-02: extreme finite values (e.g. 1e308) can overflow the mean to +inf -> fall back to the floor
    # rather than emit a non-finite forecast (the docstring guarantees finite/positive/>= floor).
    if not np.isfinite(mean_var):
        mean_var = floor
    mean_var = max(mean_var, floor)
    return np.full(n_test, mean_var, dtype=float)


def _cap_params(omega, alpha, beta, uncond_target, cap=_MAX_PERSISTENCE):
    """Cap GARCH(1,1) persistence at ``cap`` via variance targeting.

    If ``alpha + beta <= cap`` the fitted parameters are returned unchanged. Otherwise the
    ARCH/GARCH mass is scaled down to ``cap`` (preserving the ``alpha:beta`` ratio) and ``omega``
    is re-set so the long-run (unconditional) variance equals ``uncond_target``:
    ``omega = uncond_target * (1 - cap)``. This gives a finite unconditional variance, so the
    multi-step forecast converges instead of diverging (a near-unit-root IGARCH fit would grow
    roughly linearly over a long frozen forecast path). Returns ``(omega, alpha, beta)``.
    """
    persist = alpha + beta
    if persist <= cap:
        return omega, alpha, beta
    factor = cap / persist
    return uncond_target * (1.0 - cap), alpha * factor, beta * factor


def _capped_forecast_path(omega, alpha, beta, last_h, last_eps2, uncond_target, total_steps,
                          cap=_MAX_PERSISTENCE):
    """Analytic multi-step GARCH(1,1) conditional-variance path with persistence capped at ``cap``.

    ``last_h`` is the last in-sample conditional variance and ``last_eps2`` the last squared
    residual, both in the scaled (pseudo-return) space. For a stationary fit this equals arch's
    analytic ``res.forecast`` path exactly; for a capped (IGARCH) fit it converges to the finite
    capped unconditional variance. Returns an array of length ``total_steps`` (1-, 2-, ...,
    ``total_steps``-step-ahead variances).
    """
    omega_c, alpha_c, beta_c = _cap_params(omega, alpha, beta, uncond_target, cap)
    persist = alpha_c + beta_c
    sig2_1 = omega_c + alpha_c * last_eps2 + beta_c * last_h   # one-step-ahead variance
    uncond = omega_c / (1.0 - persist)                         # finite: persist <= cap < 1
    k = np.arange(total_steps)
    return uncond + (persist ** k) * (sig2_1 - uncond)


def garch_forecast(
    train_series: np.ndarray,
    n_test: int,
    horizon: int = 1,
    floor: float = FLOOR,
    seed: int = 42,
    *,
    return_status: bool = False,
):
    """GARCH(1,1) variance forecasts for ``n_test`` observations from the end of ``train_series``.

    ``train_series`` is a Parkinson-VARIANCE series. Pseudo returns ``sign * sqrt(var) * 100`` are
    fit with ``arch_model(vol="GARCH", p=1, q=1)``; the frozen-parameter multi-step conditional
    variance path from the last training point is read off and rescaled by ``1/1e4`` back to the
    raw-variance scale. Element ``i`` is the ``(horizon + i)``-step-ahead variance forecast, so for
    the default ``horizon=1`` the returned array is the 1-, 2-, ..., ``n_test``-step-ahead path.

    Always returns shape ``(n_test,)``, every entry finite, positive, and ``>= floor``.

    ``return_status=True`` additionally returns a status dict ``{"fallback": bool, "reason": str,
    "arch_available": bool}`` so a caller can record GARCH degradation (external review M-08) instead
    of a silent mean-variance fallback being written as an ordinary benchmark metric.
    """
    def _pack(arr, fallback, reason):
        if return_status:
            return arr, {"fallback": bool(fallback), "reason": reason, "arch_available": _ARCH_AVAILABLE}
        return arr

    train_series = np.asarray(train_series, dtype=float).ravel()
    if n_test < 1:
        raise ValueError(f"n_test must be >= 1, got {n_test}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if train_series.size < 2 or not np.all(np.isfinite(train_series)):
        reason = "series too short or non-finite"
        return _pack(_fallback_forecast(train_series, n_test, floor, reason), True, reason)

    if not _ARCH_AVAILABLE:
        reason = "arch package unavailable"
        return _pack(_fallback_forecast(train_series, n_test, floor, reason), True, reason)

    try:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(np.maximum(train_series, floor))
        signs = rng.choice([-1.0, 1.0], size=scale.size)
        pseudo_returns = signs * scale * _RETURN_SCALE

        am = arch_model(pseudo_returns, mean="Constant", vol="GARCH", p=1, q=1,
                        dist="normal", rescale=False)
        res = am.fit(disp="off")

        # Read the fitted GARCH(1,1) parameters and last in-sample state, then build the analytic
        # multi-step variance path with persistence capped (variance targeting) so a near-unit-root
        # (IGARCH) fit converges to a finite unconditional variance instead of diverging over the
        # long frozen forecast path. For a stationary fit this reproduces arch's analytic forecast.
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        last_h = float(res.conditional_volatility[-1]) ** 2
        last_eps2 = float(res.resid[-1]) ** 2
        uncond_target = float(np.var(pseudo_returns))    # sample long-run variance (scaled space)
        total_steps = n_test + horizon - 1
        variance_path = _capped_forecast_path(omega, alpha, beta, last_h, last_eps2,
                                              uncond_target, total_steps)
        if variance_path.size < total_steps:
            reason = "forecast path shorter than requested"
            return _pack(_fallback_forecast(train_series, n_test, floor, reason), True, reason)

        out = variance_path[horizon - 1: horizon - 1 + n_test] / (_RETURN_SCALE ** 2)
        out = np.asarray(out, dtype=float)
        if out.shape != (n_test,) or not np.all(np.isfinite(out)) or not np.all(out > 0):
            reason = "non-finite / non-positive forecast"
            return _pack(_fallback_forecast(train_series, n_test, floor, reason), True, reason)
        return _pack(np.maximum(out, floor), False, "")
    except Exception as exc:  # arch may fail to converge / raise on pathological series
        reason = f"arch fit/forecast error: {exc}"
        return _pack(_fallback_forecast(train_series, n_test, floor, reason), True, reason)
