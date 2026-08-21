"""Pre-gating diagnostics for the HAR-anchored residual study (plan section 14).

Before spending compute on residual heads or gates (E5-E10), these cheap checks answer whether a
neural expert can plausibly correct HAR: are HAR and NN errors complementary (low correlation leaves
room to combine), does NN win where the two forecasts disagree, does a residual model explain any
out-of-sample variance, and does the full model beat HAR in relative R^2. All operate on the terminal
Parkinson-variance target; ``har_pred``/``nn_pred``/``model_pred``/``resid_pred`` are aligned arrays.

Pure numpy; scipy.stats only for the correlation coefficients.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps


def error_complementarity(
    y: np.ndarray, har_pred: np.ndarray, nn_pred: np.ndarray
) -> dict[str, float]:
    """Pearson and Spearman correlation between HAR errors and NN errors.

    Errors are ``y - har_pred`` and ``y - nn_pred``. Correlation near 1 means the two experts miss in
    the same direction (little to gain from combining); low/negative correlation signals
    complementary errors worth blending.
    """
    y = np.asarray(y, dtype=float)
    har_err = y - np.asarray(har_pred, dtype=float)
    nn_err = y - np.asarray(nn_pred, dtype=float)
    pearson = float(_sps.pearsonr(har_err, nn_err).statistic)
    spearman = float(_sps.spearmanr(har_err, nn_err).statistic)
    return {"pearson": pearson, "spearman": spearman}


def disagreement_winrate(
    y: np.ndarray, har_pred: np.ndarray, nn_pred: np.ndarray, n_bins: int = 5
) -> list[dict[str, float]]:
    """NN win-rate as a function of how far NN and HAR disagree.

    Disagreement ``d = nn_pred - har_pred``; rows are bucketed into ``n_bins`` quantile bins of
    ``|d|``. For each bin, ``p_nn_better`` is the fraction of rows where the NN forecast is closer to
    the target than HAR (``|y - nn| < |y - har|``). A rising win-rate with disagreement is evidence a
    gate could route to NN where it disagrees most. Returns one dict per bin (low to high ``|d|``) with
    ``bin_lo``, ``bin_hi``, ``count``, ``p_nn_better`` (``nan`` for empty bins). Bin counts partition
    all rows.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y = np.asarray(y, dtype=float)
    har_pred = np.asarray(har_pred, dtype=float)
    nn_pred = np.asarray(nn_pred, dtype=float)
    absd = np.abs(nn_pred - har_pred)
    nn_better = np.abs(y - nn_pred) < np.abs(y - har_pred)

    edges = np.quantile(absd, np.linspace(0.0, 1.0, n_bins + 1))
    # Assign by interior edges so every row lands in exactly one of n_bins buckets (counts partition).
    bin_idx = np.clip(np.digitize(absd, edges[1:-1], right=False), 0, n_bins - 1)

    out: list[dict[str, float]] = []
    for k in range(n_bins):
        mask = bin_idx == k
        count = int(mask.sum())
        p_nn = float(nn_better[mask].mean()) if count > 0 else float("nan")
        out.append(
            {
                "bin_lo": float(edges[k]),
                "bin_hi": float(edges[k + 1]),
                "count": count,
                "p_nn_better": p_nn,
            }
        )
    return out


def residual_r2_oos(
    y: np.ndarray, har_pred: np.ndarray, resid_pred: np.ndarray
) -> float:
    """Out-of-sample R^2 of the residual predictor against a predict-zero benchmark.

    Residual ``r = y - har_pred``. The benchmark forecast for the residual is 0 (HAR already carries
    the level), so ``R^2 = 1 - sum((r - resid_pred)^2) / sum(r^2)``. This is the deployment-honest OOS
    form (no test-set residual mean); ``<= 0`` means the residual model adds nothing over leaving HAR
    untouched.
    """
    y = np.asarray(y, dtype=float)
    r = y - np.asarray(har_pred, dtype=float)
    resid_pred = np.asarray(resid_pred, dtype=float)
    ss_res = float(np.sum((r - resid_pred) ** 2))
    ss_tot = float(np.sum(r ** 2))
    if ss_tot == 0.0:
        raise ValueError("HAR residuals are all zero; residual R^2 undefined")
    return 1.0 - ss_res / ss_tot


def relative_r2_vs_har(
    y: np.ndarray, model_pred: np.ndarray, har_pred: np.ndarray
) -> float:
    """Relative out-of-sample R^2 of a model against the HAR benchmark (plan section 17).

    ``R^2 = 1 - sum((y - model_pred)^2) / sum((y - har_pred)^2)``. Positive means the model beats HAR
    on squared error; 0 means it matches HAR; negative means it is worse.
    """
    y = np.asarray(y, dtype=float)
    model_pred = np.asarray(model_pred, dtype=float)
    har_pred = np.asarray(har_pred, dtype=float)
    ss_model = float(np.sum((y - model_pred) ** 2))
    ss_har = float(np.sum((y - har_pred) ** 2))
    if ss_har == 0.0:
        raise ValueError("HAR squared error is zero; relative R^2 undefined")
    return 1.0 - ss_model / ss_har
