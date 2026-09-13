"""Causal principled-HAR feature panel.

The principled feature set uses columns ALREADY present in the enriched frames — HAR-RV core at fixed Corsi
windows (``har_daily``/``har_weekly``/``har_monthly``) and the published range estimators
(``garman_klass_variance``/``rogers_satchell_variance``/``yang_zhang_n20``) — plus the only new feature,
realized semivariance (``semi_neg``/``semi_pos``) computed per-ticker causally from ``daily_return``.
"""
import config
import estimators as E

FEATURES = ["har_daily", "har_weekly", "har_monthly",
            "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20",
            "semi_neg", "semi_pos"]


def add_features(frame):
    """Add ``semi_neg``/``semi_pos`` to one ticker frame (causal, sorted by date). Other FEATURES are already
    columns in the enriched frame."""
    d = frame.sort_values("date").reset_index(drop=True).copy()
    rm, rp = E.semivariance(d["daily_return"].to_numpy(float), config.SEMI_WINDOW, config.SEMI_MIN_PERIODS)
    d["semi_neg"], d["semi_pos"] = rm, rp
    return d


def feature_frames(frames):
    """Map :func:`add_features` over every ticker frame."""
    return {tk: add_features(fr) for tk, fr in frames.items()}
