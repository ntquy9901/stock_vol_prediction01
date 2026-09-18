"""Causal zero-shot foundation-model forecaster + per-(ticker,date) cache (Hướng B).

For every trading date ``t`` of a ticker, a frozen pretrained time-series model (Amazon Chronos-Bolt) forecasts
the next ``PRED_LEN`` steps of the own parkinson-variance series from the TRAILING context ``<= t`` only. The
forecast at step ``h`` predicts the variance ``h`` trading days ahead, i.e. exactly the panel target
``parkinson_variance.shift(-h)`` for a row dated ``t`` — so the model output merges into the panel by
``(ticker, t)`` with no leakage (the frozen model never trains on this data; the context stops at ``t``).

The heavy part is the ~1.4M sliding-window forecasts; it is cached to a parquet so re-runs are cheap. Inference
is BATCHED (``FORECAST_BATCH`` contexts per model call) on GPU when available — never batch=1.

The forecaster is injected as a ``predict_fn(list_of_1d_arrays, pred_len) -> (median[B,L], spread[B,L])`` so the
driver test can pass a fast fake and only a separate opt-in test calls the real (slow) Chronos model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import vn_gbm_graph_stage1 as S1  # noqa: E402
import foundation_config as C  # noqa: E402

FL = S1.FL


def chronos_predict_fn(model_name=None, device=None):
    """Build the real Chronos-Bolt ``predict_fn``. Imports torch/chronos lazily so tests that inject a fake do
    not need the model. Falls back to CPU if CUDA is unavailable. Returns ``fn(contexts, pred_len)`` giving
    ``(median[B,pred_len], spread[B,pred_len])`` from the configured quantile levels."""
    import torch  # local import: only needed for the real model
    from chronos import BaseChronosPipeline

    dev = device or C.DEVICE
    if dev == "cuda" and not torch.cuda.is_available():  # pragma: no cover - host-dependent hardware branch
        dev = "cpu"
    pipe = BaseChronosPipeline.from_pretrained(model_name or C.MODEL_NAME, device_map=dev,
                                               torch_dtype=torch.float32)
    q_levels = list(C.QUANTILE_LEVELS)

    def _fn(contexts, pred_len):
        inputs = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
        q, _mean = pipe.predict_quantiles(inputs=inputs, prediction_length=pred_len, quantile_levels=q_levels)
        q = q.detach().cpu().numpy()                       # [B, pred_len, n_quantiles]
        median = q[:, :, 1]
        spread = q[:, :, 2] - q[:, :, 0]
        return median, spread

    return _fn


def _batched_forecast(predict_fn, contexts, pred_len, batch):
    """Run ``predict_fn`` over ``contexts`` in chunks of ``batch`` (batched GPU inference, never batch=1);
    return ``(median[N,pred_len], spread[N,pred_len])``."""
    meds, sprs = [], []
    for i in range(0, len(contexts), batch):
        m, s = predict_fn(contexts[i:i + batch], pred_len)
        meds.append(np.asarray(m, dtype=float))
        sprs.append(np.asarray(s, dtype=float))
    return np.vstack(meds), np.vstack(sprs)


def forecast_series(predict_fn, dates, values, cfg=C):
    """Causal per-date zero-shot forecast for ONE ticker.

    ``dates`` (sorted) and ``values`` (parkinson variance) align by index. For each index ``i`` with at least
    ``MIN_CONTEXT`` trailing observations, the context is ``values[max(0, i-CTX_LEN+1) : i+1]`` (<= t) and the
    step-``h`` output is the median forecast of the variance ``h`` days ahead. Returns a DataFrame indexed by
    date with ``fnd_h{h}`` (+ ``fspread_h{h}`` when ``USE_SPREAD``) columns for every ``h`` in ``HORIZONS``.
    """
    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates)
    n = len(values)
    idxs, contexts = [], []
    for i in range(n):
        if i + 1 < cfg.MIN_CONTEXT:
            continue
        lo = max(0, i - cfg.CTX_LEN + 1)
        idxs.append(i)
        contexts.append(values[lo:i + 1])
    if not contexts:                                       # pragma: no cover - ticker shorter than MIN_CONTEXT
        return pd.DataFrame(columns=["date"])
    median, spread = _batched_forecast(predict_fn, contexts, cfg.PRED_LEN, cfg.FORECAST_BATCH)
    out = {"date": dates[idxs]}
    for h in cfg.HORIZONS:
        step = h - 1                                       # step index of the h-days-ahead forecast
        out[f"fnd_h{h}"] = np.clip(median[:, step], FL, cfg.PRED_CAP)
        if cfg.USE_SPREAD:
            out[f"fspread_h{h}"] = np.clip(spread[:, step], 0.0, cfg.PRED_CAP)
    return pd.DataFrame(out)


def build_cache(frames, predict_fn, cfg=C):
    """Forecast every ticker's series and stack into one ``(ticker, date, fnd_h*, fspread_h*)`` DataFrame."""
    parts = []
    for tk, d in frames.items():
        d = d.sort_values("date")
        df = forecast_series(predict_fn, d["date"].to_numpy(), d["parkinson_variance"].to_numpy(float), cfg)
        if len(df):
            parts.append(df.assign(ticker=tk))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ticker", "date"])


def load_or_build_cache(frames, cache_path, predict_fn=None, cfg=C, rebuild=False):
    """Return the foundation cache, building it (with the real Chronos ``predict_fn`` by default) and writing the
    parquet on first run; subsequent runs read the parquet so the ~1.4M forecasts are computed once."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)
    fn = predict_fn or chronos_predict_fn(cfg.MODEL_NAME, cfg.DEVICE)
    cache = build_cache(frames, fn, cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(cache_path, index=False)
    return cache
