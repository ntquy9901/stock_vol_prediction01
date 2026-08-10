"""Classical econometric volatility baselines on the consistent Track-B ladder basis.

Every baseline is scored on EXACTLY the pooled val/test observations (14418 / 14464 at h5),
same (ticker_id, target_date) keys and same raw Parkinson-volatility targets used to score the
deep-model rungs P0 -> P1 -> P2 -> P3 -> G1, via the identical ``train.evaluate_records`` scorer.

Baselines: persistence (random walk), EWMA/RiskMetrics (lambda=0.94), HAR, HARQ (daily range-based
RQ proxy), log-HAR, and per-ticker GARCH(1,1) / GJR-GARCH / EGARCH (``arch`` package).

Run:  python baselines/classical_baselines/code/run_classical_baselines.py <TS> [horizon]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parents[2]
_TRACKB = _ROOT / "submission_track_b" / "trackb_code"
for _p in (str(_CODE), str(_TRACKB), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

FLOOR = 1e-8
EWMA_LAMBDA = 0.94
_RATIOS = (0.7, 0.15, 0.15)


# --------------------------------------------------------------------------------------------------
# Basis construction (reuses the ladder manifest + per-ticker scalers)
# --------------------------------------------------------------------------------------------------
def build_manifest(horizon: int = 5):
    """Return (pooled_manifest, store) on the SAME basis the ladder scored P0-G1 on."""

    from run_pilot import build_screening_inputs

    inputs = build_screening_inputs(False, None, "P0", horizon=horizon)
    return inputs.manifest, inputs.store


def data_root() -> Path:
    return _ROOT / "data"


@dataclass
class TickerSeries:
    """Continuous per-ticker series with trailing (causal) HAR + EWMA components."""

    ticker: str
    dates: list[str]
    vol: np.ndarray  # NOTE: the processed "parkinson_volatility" column is the Parkinson VARIANCE
    sigma_d: np.ndarray
    sigma_w: np.ndarray
    sigma_m: np.ndarray
    ewma_pred: np.ndarray  # RiskMetrics EWMA of the realized-variance series
    train_end_date: str
    pos: dict[str, int]
    logret: pd.Series  # index = date str


def load_ticker_series(ticker: str) -> TickerSeries:
    """Load the full Parkinson-vol series (+ log returns) and precompute causal components."""

    proc = pd.read_csv(data_root() / "processed" / f"{ticker}_processed.csv")
    proc["date"] = proc["date"].astype(str)
    proc = proc.sort_values("date").reset_index(drop=True)
    vol = proc["parkinson_volatility"].to_numpy(dtype=float)
    dates = proc["date"].tolist()
    series = pd.Series(vol)
    sigma_w = series.rolling(5).mean().to_numpy()
    sigma_m = series.rolling(22).mean().to_numpy()
    ewma_pred = _ewma_variance(vol, EWMA_LAMBDA)
    train_end_idx = int(len(proc) * _RATIOS[0])
    train_end_date = dates[train_end_idx - 1]
    logret = _load_log_returns(ticker)
    return TickerSeries(
        ticker=ticker, dates=dates, vol=vol, sigma_d=vol.copy(),
        sigma_w=sigma_w, sigma_m=sigma_m, ewma_pred=ewma_pred,
        train_end_date=train_end_date, pos={d: i for i, d in enumerate(dates)}, logret=logret,
    )


def _ewma_variance(rv: np.ndarray, lam: float) -> np.ndarray:
    """RiskMetrics EWMA smoother of the realized-variance series (v_t = lam v_{t-1} + (1-lam) RV_t).

    The target series is the Parkinson VARIANCE, so the EWMA is applied to it directly and its level
    is the (flat) multi-step variance forecast.
    """

    out = np.empty_like(rv, dtype=float)
    out[0] = rv[0]
    for i in range(1, len(rv)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * rv[i]
    return out


def _load_log_returns(ticker: str) -> pd.Series:
    path = data_root() / "raw" / "prices" / f"{ticker}_ohlcv.csv"
    if not path.exists():
        # No raw OHLCV -> return-based GARCH family cannot cover this ticker (e.g. LPB).
        return pd.Series(dtype=float)
    ohlcv = pd.read_csv(path)
    # Some tickers (VPB, VRE) store tz-aware timestamps ("2017-08-18 00:00:00+07:00") instead of
    # plain YYYY-MM-DD; normalize to date-only strings so returns align with the obs date keys.
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.strftime("%Y-%m-%d")
    ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
    close = ohlcv["close"].to_numpy(dtype=float)
    logret = np.diff(np.log(close))
    return pd.Series(logret, index=ohlcv["date"].tolist()[1:])


# --------------------------------------------------------------------------------------------------
# Baseline predictors -> {(ticker_id, target_date): pred_raw}
# --------------------------------------------------------------------------------------------------
def _obs(samples):
    """Yield (ticker_id, ticker, target_date, obs_date) for each pooled sample."""

    for s in samples:
        yield int(s.key.ticker_id), s.key.ticker, s.key.target_date, s.input_dates[-1]


def predict_persistence(samples, series_by_ticker) -> dict:
    out = {}
    for tid, ticker, target_date, obs_date in _obs(samples):
        s = series_by_ticker[ticker]
        out[(tid, target_date)] = float(s.vol[s.pos[obs_date]])
    return out


def predict_ewma(samples, series_by_ticker) -> dict:
    out = {}
    for tid, ticker, target_date, obs_date in _obs(samples):
        s = series_by_ticker[ticker]
        out[(tid, target_date)] = float(s.ewma_pred[s.pos[obs_date]])
    return out


def _har_design(series: TickerSeries, obs_date: str, kind: str) -> np.ndarray:
    i = series.pos[obs_date]
    # Floor components: some days have zero Parkinson vol (H==L), which would make log-HAR ln(0).
    d = max(float(series.sigma_d[i]), FLOOR)
    w = max(float(series.sigma_w[i]), FLOOR)
    m = max(float(series.sigma_m[i]), FLOOR)
    if kind == "har":
        return np.array([d, w, m], dtype=float)
    if kind == "harq":
        return np.array([d, w, m, d * np.sqrt(d * d)], dtype=float)  # RQ_d proxy = d^2
    if kind == "log_har":
        return np.log(np.array([d, w, m], dtype=float))
    raise ValueError(f"unknown kind {kind}")


def _fit_regression_baseline(train_samples, eval_samples, series_by_ticker, kind: str) -> dict:
    """Per-ticker OLS on train samples; predict eval samples. log_har fits/pred in log space."""

    train_by_ticker: dict[str, list] = {}
    for s in train_samples:
        train_by_ticker.setdefault(s.key.ticker, []).append(s)
    models: dict[str, LinearRegression] = {}
    for ticker, rows in train_by_ticker.items():
        series = series_by_ticker[ticker]
        x, y = [], []
        for s in rows:
            design = _har_design(series, s.input_dates[-1], kind)
            target = float(s.y_eval_raw)
            if not np.isfinite(design).all() or target <= 0:
                continue
            x.append(design)
            y.append(np.log(target) if kind == "log_har" else target)
        models[ticker] = LinearRegression().fit(np.asarray(x), np.asarray(y))
    out = {}
    for tid, ticker, target_date, obs_date in _obs(eval_samples):
        design = _har_design(series_by_ticker[ticker], obs_date, kind).reshape(1, -1)
        pred = float(models[ticker].predict(design)[0])
        out[(tid, target_date)] = float(np.exp(pred)) if kind == "log_har" else pred
    return out


def predict_har(train_samples, eval_samples, series_by_ticker) -> dict:
    return _fit_regression_baseline(train_samples, eval_samples, series_by_ticker, "har")


def predict_harq(train_samples, eval_samples, series_by_ticker) -> dict:
    return _fit_regression_baseline(train_samples, eval_samples, series_by_ticker, "harq")


def predict_log_har(train_samples, eval_samples, series_by_ticker) -> dict:
    return _fit_regression_baseline(train_samples, eval_samples, series_by_ticker, "log_har")


_GARCH_SPECS = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "method": "analytic"},
    "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "method": "analytic"},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "method": "simulation"},
}


def garch_ticker_forecast(series: TickerSeries, origin_dates, horizon: int, spec_name: str,
                          seed: int = 42) -> dict[str, float]:
    """Frozen-param h-step marginal-VARIANCE forecast at each origin (train-only estimation).

    Returns are scaled x100 for estimation, so the conditional variance is in percent^2; dividing by
    100^2 = 1e4 recovers the raw-return variance, directly comparable to the Parkinson variance
    target (both estimate the daily return variance).
    """

    from arch import arch_model

    spec = _GARCH_SPECS[spec_name]
    r = (series.logret * 100.0).dropna()
    dates = list(r.index)
    r_values = r.to_numpy(dtype=float)
    train_count = int(np.searchsorted(np.asarray(dates), series.train_end_date, side="right"))
    if train_count < 100:
        raise ValueError(f"{series.ticker}: too few train returns ({train_count}) for {spec_name}")
    am = arch_model(r_values, mean="Constant", vol=spec["vol"], p=spec["p"], o=spec["o"],
                    q=spec["q"], dist="normal", rescale=False)
    res = am.fit(last_obs=train_count, disp="off")
    pos = {d: i for i, d in enumerate(dates)}
    requested = sorted(set(origin_dates))
    aligned = [d for d in requested if d in pos]
    if not aligned:
        raise ValueError(f"{series.ticker}: no requested origin is a trading day for {spec_name}")
    start = min(pos[d] for d in aligned)
    kwargs = {"horizon": horizon, "start": start, "reindex": True}
    if spec["method"] == "simulation":
        kwargs.update(method="simulation", simulations=1000,
                      rng=np.random.default_rng(seed).standard_normal)
    fc = res.forecast(**kwargs)
    variance = fc.variance.to_numpy()  # rows aligned to r index, cols h.1..h.h

    def _forecast_row(obs_date: str) -> int:
        """Forecast-origin row for obs_date, carrying forward across non-trading-day gaps.

        A few observation dates fall on holidays absent from this ticker's trading calendar
        (e.g. LPB's SSI feed misses Tet / national holidays present in the processed series). For
        those the freshest available origin <= obs_date is used, since the conditional variance
        forecast is persistent across a short gap.
        """

        if obs_date in pos and pos[obs_date] >= start:
            return pos[obs_date]
        prior = [i for d, i in pos.items() if d <= obs_date and i >= start]
        return max(prior) if prior else start

    out: dict[str, float] = {}
    for d in requested:
        v = variance[_forecast_row(d), horizon - 1]
        out[d] = float(v / 1e4) if np.isfinite(v) and v > 0 else FLOOR
    return out


def predict_garch(eval_samples, series_by_ticker, horizon: int, spec_name: str) -> dict:
    origins_by_ticker: dict[str, set] = {}
    keys_by_ticker: dict[str, list] = {}
    for tid, ticker, target_date, obs_date in _obs(eval_samples):
        origins_by_ticker.setdefault(ticker, set()).add(obs_date)
        keys_by_ticker.setdefault(ticker, []).append((tid, target_date, obs_date))
    out = {}
    for ticker, origins in origins_by_ticker.items():
        fc = garch_ticker_forecast(series_by_ticker[ticker], origins, horizon, spec_name)
        for tid, target_date, obs_date in keys_by_ticker[ticker]:
            out[(tid, target_date)] = fc[obs_date]
    return out


# --------------------------------------------------------------------------------------------------
# Evaluation (identical ladder scorer via a per-ticker scaler round-trip)
# --------------------------------------------------------------------------------------------------
def evaluate_baseline(preds: dict, samples, store) -> dict:
    """Score raw predictions with the ladder's ``evaluate_records`` (round-trip through store)."""

    from train import evaluate_records

    records = []
    for s in samples:
        key = (int(s.key.ticker_id), s.key.target_date)
        pred_raw = max(float(preds[key]), FLOOR)
        scaler = store.get(int(s.key.ticker_id)).target_scaler
        pred_norm = float(scaler.transform(np.array([pred_raw]))[0])
        records.append({
            "ticker_id": int(s.key.ticker_id),
            "target_date": s.key.target_date,
            "prediction_norm": pred_norm,
            "target_raw": float(s.y_eval_raw if s.y_eval_raw is not None else s.y_raw),
        })
    return evaluate_records(records, store)["metrics"]


def assert_full_coverage(preds: dict, samples) -> None:
    keys = {(int(s.key.ticker_id), s.key.target_date) for s in samples}
    missing = keys - set(preds)
    if missing:
        raise ValueError(f"predictions miss {len(missing)} observation(s), e.g. {sorted(missing)[:3]}")
    if len(preds) != len(keys):
        raise ValueError(f"prediction count {len(preds)} != observation count {len(keys)}")


# --------------------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------------------
_REGRESSION = {"HAR": "har", "HARQ": "harq", "logHAR": "log_har"}
_GARCH = {"GARCH": "garch", "GJR-GARCH": "gjr", "EGARCH": "egarch"}


def run_all(horizon: int = 5, tickers: list[str] | None = None) -> dict:
    """Compute every classical baseline's 6-metric val+test table on the ladder observation set."""

    manifest, store = build_manifest(horizon)
    train_s, val_s, test_s = (manifest.samples[k] for k in ("train", "val", "test"))
    names = tickers if tickers is not None else sorted({s.key.ticker for s in val_s})
    series = {t: load_ticker_series(t) for t in names}
    # GARCH family needs close-to-close returns; tickers without raw OHLCV are excluded from it.
    garch_tickers = {t for t in names if not series[t].logret.empty}
    excluded = sorted(set(names) - garch_tickers)
    garch_val = [s for s in val_s if s.key.ticker in garch_tickers]
    garch_test = [s for s in test_s if s.key.ticker in garch_tickers]

    results: dict[str, dict] = {}

    def _record(name: str, val_preds: dict, test_preds: dict, vs, ts) -> None:
        assert_full_coverage(val_preds, vs)
        assert_full_coverage(test_preds, ts)
        results[name] = {
            "val": evaluate_baseline(val_preds, vs, store),
            "test": evaluate_baseline(test_preds, ts, store),
            "n_val_obs": len(vs),
            "n_test_obs": len(ts),
        }

    _record("Persistence", predict_persistence(val_s, series),
            predict_persistence(test_s, series), val_s, test_s)
    _record("EWMA", predict_ewma(val_s, series), predict_ewma(test_s, series), val_s, test_s)
    for name, kind in _REGRESSION.items():
        vp = _fit_regression_baseline(train_s, val_s, series, kind)
        tp = _fit_regression_baseline(train_s, test_s, series, kind)
        _record(name, vp, tp, val_s, test_s)
    for name, spec in _GARCH.items():
        vp = predict_garch(garch_val, series, horizon, spec)
        tp = predict_garch(garch_test, series, horizon, spec)
        _record(name, vp, tp, garch_val, garch_test)

    return {
        "horizon": horizon,
        "n_val_obs": len(val_s),
        "n_test_obs": len(test_s),
        "n_tickers": len(names),
        "garch_excluded_tickers": excluded,
        "garch_n_val_obs": len(garch_val),
        "garch_n_test_obs": len(garch_test),
        "results": results,
    }
