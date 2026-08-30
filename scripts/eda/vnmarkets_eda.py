"""Market-parameterized EDA + data-mining over the VN price panels (vn100, vn30, hose) mirroring the
parallel HNX EDA. CPU/pandas only (NO GPU): the single GPU is committed to overnight training. Markets are
processed SEQUENTIALLY (one fully, discard its frames, then the next) to keep RAM bounded.

Data (READ-ONLY):
  * raw OHLCV  -> ``volatility_estimators.PRICE[panel]`` (``*_ohlcv.csv``: date,open,high,low,close,volume).
  * processed  -> the Parkinson-VARIANCE target the delivered runners read (``parkinson_volatility`` column;
    NOTE it is sigma^2, a VARIANCE, not sigma). HOSE/HNX processed values are ETL upper-clipped at 0.1;
    vn30/vn100 processed are not clipped.
  * screened universe -> ``floor_sensitivity.screen_files`` (>=250 rows, zero-Parkinson frac <=0.5,
    NaN frac <=0.5) for HOSE/HNX; vn30/vn100 keep all tickers.

Parkinson = ln(H/L)^2 / (4 ln2) = sigma^2 (daily variance). market_pk = cross-sectional per-day mean of the
Parkinson variance (common market factor); volume_zscore_20 = 20-day rolling z-score of volume; HAR
weekly/monthly = 5-/22-day rolling means of the Parkinson variance (mirrors the reference runner).

Outputs one self-contained HTML per market (charts embedded as base64 PNG, no external CDN) plus a
cross-market comparison HTML + short markdown.

Usage:
    python scripts/eda/vnmarkets_eda.py                 # full run, all markets + comparison
    python scripts/eda/vnmarkets_eda.py --limit 8       # quick smoke on <=8 tickers/panel
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless, no display, CPU only
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))

import volatility_estimators as VE  # noqa: E402
import floor_sensitivity as FS      # noqa: E402

# processed dirs the DELIVERED runners read (raw dirs come from VE.PRICE)
PROCESSED = {
    "vn30": REPO / "submission" / "soict_lstm_gat" / "data" / "vn30",
    "vn100": REPO / "submission" / "soict_lstm_gat" / "data" / "vn100",
    "hose": REPO / "data" / "processed" / "hose",
    "hnx": REPO / "data" / "processed" / "hnx",
}
SCREEN_PANELS = {"hose", "hnx"}          # liquidity/history screen applies to the two full exchanges
_LN2 = float(np.log(2.0))
_OHLC_RTOL = 1e-5                        # float32 storage noise absorber (matches the raw-quality gate)
SPLIT_THRESH = 0.50                      # |1-day simple return| > 50% -> candidate unadjusted split/dividend
OUT_DIR = REPO / "docs" / "reports"


# --------------------------------------------------------------------------------------------------
# Pure detectors / statistics (unit-tested)
# --------------------------------------------------------------------------------------------------
def log_returns(close: np.ndarray) -> np.ndarray:
    """Daily log returns ln(C_t / C_{t-1}); first element NaN. Non-positive closes -> NaN."""
    c = np.asarray(close, dtype=np.float64)
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (c > 0) & (prev > 0)
        r = np.where(ok, np.log(c / prev), np.nan)
    return r


def simple_returns(close: np.ndarray) -> np.ndarray:
    """Daily simple returns C_t/C_{t-1} - 1; first element NaN. Non-positive prior close -> NaN."""
    c = np.asarray(close, dtype=np.float64)
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev > 0, c / prev - 1.0, np.nan)
    return r


def skewness(x: np.ndarray) -> float:
    """Population skew = mean(((x-mu)/sigma)^3); NaNs dropped. NaN if <3 finite points or sigma==0."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    mu = v.mean()
    sigma = v.std()
    if sigma == 0:
        return float("nan")
    return float(np.mean(((v - mu) / sigma) ** 3))


def excess_kurtosis(x: np.ndarray) -> float:
    """Population excess kurtosis = mean(((x-mu)/sigma)^4) - 3; NaN if <4 finite points or sigma==0."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float("nan")
    mu = v.mean()
    sigma = v.std()
    if sigma == 0:
        return float("nan")
    return float(np.mean(((v - mu) / sigma) ** 4) - 3.0)


def summary_stats(x: np.ndarray) -> dict:
    """Descriptive stats over finite values: n, mean, std, min, p1/p25/median/p75/p99, max, skew, kurtosis."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "p1": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"),
                "p99": float("nan"), "max": float("nan"), "skew": float("nan"), "kurtosis": float("nan")}
    p1, p25, med, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "p1": float(p1), "p25": float(p25), "median": float(med), "p75": float(p75), "p99": float(p99),
            "max": float(v.max()), "skew": skewness(v), "kurtosis": excess_kurtosis(v)}


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score (x-median)/(1.4826*MAD). All-NaN or zero-MAD -> zeros (no outliers flagged)."""
    v = np.asarray(x, dtype=np.float64)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return np.zeros_like(v)
    med = np.median(fin)
    mad = np.median(np.abs(fin - med))
    scale = 1.4826 * mad
    if scale == 0:
        return np.zeros_like(v)
    return (v - med) / scale


def count_robust_outliers(x: np.ndarray, thresh: float = 3.5) -> int:
    """Number of finite points with |robust z| > thresh."""
    z = robust_z(x)
    return int(np.sum(np.isfinite(z) & (np.abs(z) > thresh)))


def iqr_outlier_count(x: np.ndarray, k: float = 1.5) -> int:
    """Number of finite points outside [Q1 - k*IQR, Q3 + k*IQR]."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return int(np.sum((v < lo) | (v > hi)))


def detect_ohlc_violations(df: pd.DataFrame) -> dict:
    """Count OHLC geometry violations on a raw OHLCV frame. Uses the same relative tolerance as the
    raw-data quality gate so float32 storage noise is not flagged."""
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    finite = np.isfinite([o, h, lo, c]).all(0)
    nonpos = finite & ((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0))
    hi_lt_lo = finite & (h < lo)
    hi_oc = np.maximum(o, c)
    lo_oc = np.minimum(o, c)
    open_close_out = finite & ((h < hi_oc * (1 - _OHLC_RTOL)) | (lo > lo_oc * (1 + _OHLC_RTOL)))
    zero_range = finite & (h == lo)
    nan_rows = int(np.sum(~finite))
    return {"n": int(len(df)), "nonpositive": int(nonpos.sum()), "high_lt_low": int(hi_lt_lo.sum()),
            "open_close_outside": int(open_close_out.sum()), "zero_range": int(zero_range.sum()),
            "nan_rows": nan_rows}


def zero_parkinson_fraction(pk: np.ndarray) -> float:
    """Fraction of exactly-zero Parkinson-variance days (H==L / illiquid). NaN if no finite values."""
    v = np.asarray(pk, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v == 0.0))


def detect_split_jumps(dates: np.ndarray, close: np.ndarray, thresh: float = SPLIT_THRESH) -> list:
    """Days whose |simple return| exceeds ``thresh`` (default 50%) -- candidate UNADJUSTED split/dividend
    (VN prices are not split-adjusted). Returns [(date_str, ret), ...] sorted by |ret| descending."""
    r = simple_returns(close)
    idx = np.where(np.isfinite(r) & (np.abs(r) > thresh))[0]
    out = [(str(dates[i]), float(r[i])) for i in idx]
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out


def detect_stale_runs(dates: np.ndarray, close: np.ndarray, min_run: int = 5) -> list:
    """Maximal runs of >= ``min_run`` consecutive identical (finite, positive) closes -- stale/repeated
    prices. Returns [(start_date, end_date, length), ...]."""
    c = np.asarray(close, dtype=np.float64)
    runs = []
    i = 0
    n = len(c)
    while i < n:
        j = i
        while j + 1 < n and np.isfinite(c[j]) and c[j] > 0 and c[j + 1] == c[i]:
            j += 1
        length = j - i + 1
        if length >= min_run and np.isfinite(c[i]) and c[i] > 0:
            runs.append((str(dates[i]), str(dates[j]), int(length)))
        i = j + 1
    return runs


def zero_volume_fraction(volume: np.ndarray) -> float:
    """Fraction of finite volume rows that are exactly zero. NaN if no finite values."""
    v = np.asarray(volume, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v == 0.0))


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """Autocorrelation function of a series for lags 1..``lags`` (finite values, mean-subtracted).
    Returns array length ``lags``; NaN entries where variance is zero / too few points."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    out = np.full(lags, np.nan)
    if n < 3:
        return out
    v = v - v.mean()
    denom = np.sum(v * v)
    if denom == 0:
        return out
    for k in range(1, lags + 1):
        if k >= n:
            break
        out[k - 1] = float(np.sum(v[k:] * v[:-k]) / denom)
    return out


def zero_parkinson_by_year(dates: np.ndarray, pk: np.ndarray) -> dict:
    """Map year -> fraction of exactly-zero Parkinson days that year (illiquidity over time)."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    v = pd.to_numeric(pd.Series(pk), errors="coerce")
    m = d.notna() & v.notna()
    if not m.any():
        return {}
    years = d[m].dt.year.to_numpy()
    zero = (v[m].to_numpy() == 0.0)
    out = {}
    for y in np.unique(years):
        sel = years == y
        out[int(y)] = float(np.mean(zero[sel]))
    return out


def correlation_stats(ret_wide: pd.DataFrame, min_overlap: int = 250) -> dict:
    """Pairwise Pearson return-correlation stats on a wide [date x ticker] return frame. Returns median /
    mean |rho|, quantiles, fraction of pairs with |rho|>0.3, the # tickers, and the correlation matrix."""
    corr = ret_wide.corr(min_periods=min_overlap)
    m = corr.to_numpy()
    iu = np.triu_indices_from(m, k=1)
    off = m[iu]
    off = off[np.isfinite(off)]
    if off.size == 0:
        return {"n_tickers": corr.shape[0], "n_pairs": 0, "median_abs_rho": float("nan"),
                "mean_abs_rho": float("nan"), "p90_abs_rho": float("nan"),
                "frac_gt_0.3": float("nan"), "mean_rho": float("nan"), "corr": corr}
    a = np.abs(off)
    return {"n_tickers": corr.shape[0], "n_pairs": int(off.size),
            "median_abs_rho": float(np.median(a)), "mean_abs_rho": float(np.mean(a)),
            "p90_abs_rho": float(np.percentile(a, 90)), "frac_gt_0.3": float(np.mean(a > 0.3)),
            "mean_rho": float(np.mean(off)), "corr": corr}


def build_cross_market_table(summaries: dict) -> list:
    """Assemble the cross-market comparison rows from per-market summary dicts (ordered vn100, vn30, hose,
    hnx when present). Each row: market, raw/screened universe, rows, median |rho|, zero-Parkinson rate,
    dirty-data totals, liquidity (zero-volume rate), earliest/latest date."""
    canon = [p for p in ("vn100", "vn30", "hose", "hnx") if p in summaries]
    order = canon + [p for p in summaries if p not in canon]
    rows = []
    for p in order:
        s = summaries[p]
        rows.append({
            "market": p,
            "raw_tickers": s["raw_tickers"],
            "screened_tickers": s["screened_tickers"],
            "total_rows": s["total_rows"],
            "date_min": s["date_min"],
            "date_max": s["date_max"],
            "median_abs_rho": s["corr"]["median_abs_rho"],
            "mean_abs_rho": s["corr"]["mean_abs_rho"],
            "zero_parkinson_rate": s["zero_parkinson_rate"],
            "zero_volume_rate": s["zero_volume_rate"],
            "ohlc_high_lt_low": s["dirty"]["high_lt_low"],
            "ohlc_open_close_outside": s["dirty"]["open_close_outside"],
            "ohlc_nonpositive": s["dirty"]["nonpositive"],
            "zero_range_days": s["dirty"]["zero_range"],
            "split_jump_days": s["dirty"]["split_jumps"],
            "stale_run_tickers": s["dirty"]["stale_run_tickers"],
        })
    return rows


# --------------------------------------------------------------------------------------------------
# Data loading (READ-ONLY)
# --------------------------------------------------------------------------------------------------
def _raw_files(panel: str) -> list:
    return sorted(glob.glob(str(VE.PRICE[panel] / "*_ohlcv.csv")))


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def screened_set(panel: str) -> set | None:
    """Screened ticker set for HOSE/HNX (via floor_sensitivity); None (keep all) for vn30/vn100."""
    if panel not in SCREEN_PANELS:
        return None
    kept = FS.screen_files(sorted(glob.glob(str(PROCESSED[panel] / "*_processed.csv"))))
    return {Path(f).name.replace("_processed.csv", "") for f in kept}


# --------------------------------------------------------------------------------------------------
# Charts (base64 PNG, embedded)
# --------------------------------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:  # pragma: no cover - thin matplotlib IO wrapper
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str, alt: str = "") -> str:  # pragma: no cover - trivial html
    return f"<img alt='{alt}' style='max-width:100%;height:auto' src='data:image/png;base64,{b64}'>"


def hist_chart(x: np.ndarray, title: str, log_x: bool = False, bins: int = 60) -> str:  # pragma: no cover
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if v.size:
        if log_x:
            v = v[v > 0]
            ax.hist(np.log10(v), bins=bins, color="#3b6ea5")
            ax.set_xlabel("log10(value)")
        else:
            ax.hist(v, bins=bins, color="#3b6ea5")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    return _img(_fig_to_b64(fig), title)


def heatmap_chart(corr: pd.DataFrame, title: str) -> str:  # pragma: no cover
    m = corr.to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _img(_fig_to_b64(fig), title)


def abs_rho_hist(corr: pd.DataFrame, title: str) -> str:  # pragma: no cover
    m = corr.to_numpy()
    iu = np.triu_indices_from(m, k=1)
    a = np.abs(m[iu])
    a = a[np.isfinite(a)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if a.size:
        ax.hist(a, bins=50, color="#a5533b")
        ax.axvline(float(np.median(a)), color="k", ls="--", lw=1, label=f"median={np.median(a):.3f}")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("|rho|")
    ax.tick_params(labelsize=8)
    return _img(_fig_to_b64(fig), title)


def acf_chart(vals: np.ndarray, title: str) -> str:  # pragma: no cover
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(np.arange(1, len(vals) + 1), vals, color="#3b8a5a")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.tick_params(labelsize=8)
    return _img(_fig_to_b64(fig), title)


def line_by_year_chart(year_map: dict, title: str, ylabel: str) -> str:  # pragma: no cover
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if year_map:
        ys = sorted(year_map)
        ax.plot(ys, [year_map[y] for y in ys], marker="o", color="#7a3b8a")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8)
    return _img(_fig_to_b64(fig), title)


def tickers_per_year_chart(counts: dict, title: str) -> str:  # pragma: no cover
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if counts:
        ys = sorted(counts)
        ax.bar(ys, [counts[y] for y in ys], color="#3b6ea5")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("# active tickers")
    ax.tick_params(labelsize=8)
    return _img(_fig_to_b64(fig), title)


# --------------------------------------------------------------------------------------------------
# Per-market analysis (sequential; discards frames after each market)
# --------------------------------------------------------------------------------------------------
def analyze_panel(panel: str, limit: int | None = None) -> dict:
    """Load ONE market's raw + processed data, run all detectors, and return a summary dict (compact
    numbers + a handful of charts as base64). Big frames are released before returning so the caller can
    process the next market without holding all three in RAM."""
    raw_files = _raw_files(panel)
    if limit is not None:
        raw_files = raw_files[:limit]
    screen = screened_set(panel)
    proc_dir = PROCESSED[panel]

    all_returns = {}          # ticker -> pd.Series(return indexed by date) for correlation
    all_ret_vals = []         # pooled returns for distribution
    all_pk_vals = []          # pooled processed Parkinson variance
    all_vol_vals = []         # pooled volume
    all_har_w, all_har_m = [], []
    per_ticker = []           # coverage rows
    dirty = {"nonpositive": 0, "high_lt_low": 0, "open_close_outside": 0, "zero_range": 0,
             "nan_rows": 0, "split_jumps": 0, "stale_run_tickers": 0, "robust_outliers": 0}
    split_examples = []       # (ticker, date, ret)
    stale_examples = []       # (ticker, start, end, length)
    zero_vol_num, zero_vol_den = 0, 0
    zero_pk_num, zero_pk_den = 0, 0
    year_active = {}          # year -> set of tickers active
    pk_frames = {}            # ticker -> (dates, pk) for market_pk + by-year (released after use)
    date_min, date_max = None, None

    for f in raw_files:
        ticker = Path(f).stem.replace("_ohlcv", "")
        raw = _load_raw(f)
        if not {"open", "high", "low", "close"} <= set(raw.columns) or len(raw) < 2:
            continue
        dv = detect_ohlc_violations(raw)
        for k in ("nonpositive", "high_lt_low", "open_close_outside", "zero_range", "nan_rows"):
            dirty[k] += dv[k]
        dates = raw["date"].dt.strftime("%Y-%m-%d").to_numpy() if "date" in raw else np.arange(len(raw)).astype(str)
        close = raw["close"].to_numpy(float)
        vol = raw["volume"].to_numpy(float) if "volume" in raw else np.full(len(raw), np.nan)

        r = log_returns(close)
        all_ret_vals.append(r[np.isfinite(r)])
        dirty["robust_outliers"] += count_robust_outliers(r)

        jumps = detect_split_jumps(dates, close)
        dirty["split_jumps"] += len(jumps)
        split_examples.extend((ticker, d, rr) for d, rr in jumps[:3])
        stales = detect_stale_runs(dates, close)
        if stales:
            dirty["stale_run_tickers"] += 1
            stale_examples.extend((ticker, s, e, ln) for s, e, ln in stales[:2])

        zv = np.asarray(vol, float)
        zv = zv[np.isfinite(zv)]
        if zv.size:
            zero_vol_num += int(np.sum(zv == 0))
            zero_vol_den += int(zv.size)
            all_vol_vals.append(zv)

        # correlation series (indexed by date) on screened universe only
        if screen is None or ticker in screen:
            if "date" in raw:
                s = pd.Series(r, index=raw["date"].to_numpy())
                all_returns[ticker] = s[np.isfinite(s.to_numpy())]

        # processed Parkinson target
        pf = proc_dir / f"{ticker}_processed.csv"
        if pf.exists():
            proc = pd.read_csv(pf)
            if "parkinson_volatility" in proc.columns:
                pk = pd.to_numeric(proc["parkinson_volatility"], errors="coerce").to_numpy(float)
                pdates = pd.to_datetime(proc["date"], errors="coerce") if "date" in proc.columns else None
                keep_ticker = (screen is None or ticker in screen)
                fin = pk[np.isfinite(pk)]
                if keep_ticker and fin.size:
                    all_pk_vals.append(fin)
                    zero_pk_num += int(np.sum(fin == 0.0))
                    zero_pk_den += int(fin.size)
                    sp = pd.Series(pk)
                    all_har_w.append(sp.rolling(5, min_periods=5).mean().dropna().to_numpy())
                    all_har_m.append(sp.rolling(22, min_periods=22).mean().dropna().to_numpy())
                    if pdates is not None:
                        pk_frames[ticker] = (pdates.dt.strftime("%Y-%m-%d").to_numpy(), pk)

        # coverage
        d0 = raw["date"].min() if "date" in raw else None
        d1 = raw["date"].max() if "date" in raw else None
        if d0 is not None and pd.notna(d0):
            date_min = d0 if date_min is None else min(date_min, d0)
            date_max = d1 if date_max is None else max(date_max, d1)
            for y in range(d0.year, d1.year + 1):
                year_active.setdefault(y, set()).add(ticker)
        per_ticker.append({"ticker": ticker, "rows": len(raw),
                           "first": str(d0.date()) if d0 is not None and pd.notna(d0) else "",
                           "last": str(d1.date()) if d1 is not None and pd.notna(d1) else "",
                           "screened_in": (screen is None or ticker in screen)})

    # ---- aggregate cross-sectional structure ----
    ret_wide = pd.DataFrame(all_returns) if all_returns else pd.DataFrame()
    min_overlap = min(250, max(20, len(ret_wide) // 4)) if len(ret_wide) else 250
    corr = correlation_stats(ret_wide, min_overlap=min_overlap) if not ret_wide.empty else \
        {"n_tickers": 0, "n_pairs": 0, "median_abs_rho": float("nan"), "mean_abs_rho": float("nan"),
         "p90_abs_rho": float("nan"), "frac_gt_0.3": float("nan"), "mean_rho": float("nan"),
         "corr": pd.DataFrame()}

    # ---- market_pk (cross-sectional per-day mean of Parkinson) + illiquidity-by-year ----
    market_pk_vals = np.array([])
    zero_pk_year = {}
    if pk_frames:
        mk = pd.DataFrame({t: pd.Series(pk, index=pd.to_datetime(dts)) for t, (dts, pk) in pk_frames.items()})
        market_pk_vals = mk.mean(axis=1).dropna().to_numpy()
        # pooled zero-Parkinson by year
        by_year = {}
        for t, (dts, pk) in pk_frames.items():
            ym = zero_parkinson_by_year(dts, pk)
            for y, _f in ym.items():
                z = by_year.setdefault(y, [0, 0])
                v = np.asarray(pk, float)
                dd = pd.to_datetime(pd.Series(dts), errors="coerce")
                sel = (dd.dt.year == y).to_numpy() & np.isfinite(v)
                z[0] += int(np.sum(v[sel] == 0.0))
                z[1] += int(np.sum(sel))
        zero_pk_year = {y: (c[0] / c[1] if c[1] else float("nan")) for y, c in sorted(by_year.items())}
        del mk

    pooled_ret = np.concatenate(all_ret_vals) if all_ret_vals else np.array([])
    pooled_pk = np.concatenate(all_pk_vals) if all_pk_vals else np.array([])
    pooled_vol = np.concatenate(all_vol_vals) if all_vol_vals else np.array([])
    pooled_har_w = np.concatenate(all_har_w) if all_har_w else np.array([])
    pooled_har_m = np.concatenate(all_har_m) if all_har_m else np.array([])

    # volume_zscore_20 pooled (per-ticker rolling z, then pool)
    vz_vals = []
    for f in raw_files:
        ticker = Path(f).stem.replace("_ohlcv", "")
        if not (screen is None or ticker in screen):
            continue
        raw = _load_raw(f)
        if "volume" not in raw.columns:
            continue
        s = pd.to_numeric(raw["volume"], errors="coerce")
        mu = s.rolling(20, min_periods=20).mean()
        sd = s.rolling(20, min_periods=20).std()
        z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if z.size:
            vz_vals.append(z)
    pooled_vz = np.concatenate(vz_vals) if vz_vals else np.array([])

    # ACF of |returns| (clustering) on the pooled-per-ticker average
    abs_ret_acf = acf(np.abs(pooled_ret), 20) if pooled_ret.size else np.full(20, np.nan)
    pk_acf = acf(market_pk_vals, 20) if market_pk_vals.size else np.full(20, np.nan)

    tickers_per_year = {y: len(s) for y, s in sorted(year_active.items())}

    raw_n = len(raw_files) if limit is None else len(raw_files)
    screened_n = (len(screen) if screen is not None else raw_n) if limit is None else \
        sum(1 for pt in per_ticker if pt["screened_in"])

    summary = {
        "panel": panel,
        "raw_tickers": len([1 for _ in raw_files]),
        "screened_tickers": screened_n,
        "total_rows": int(sum(pt["rows"] for pt in per_ticker)),
        "date_min": str(date_min.date()) if date_min is not None else "",
        "date_max": str(date_max.date()) if date_max is not None else "",
        "per_ticker": per_ticker,
        "tickers_per_year": tickers_per_year,
        "corr": corr,
        "zero_parkinson_rate": float(zero_pk_num / zero_pk_den) if zero_pk_den else float("nan"),
        "zero_volume_rate": float(zero_vol_num / zero_vol_den) if zero_vol_den else float("nan"),
        "zero_parkinson_by_year": zero_pk_year,
        "dirty": dirty,
        "split_examples": split_examples,
        "stale_examples": stale_examples,
        "stats": {
            "returns": summary_stats(pooled_ret),
            "parkinson": summary_stats(pooled_pk),
            "volume": summary_stats(pooled_vol),
            "har_weekly": summary_stats(pooled_har_w),
            "har_monthly": summary_stats(pooled_har_m),
            "market_pk": summary_stats(market_pk_vals),
            "volume_zscore_20": summary_stats(pooled_vz),
        },
        "_charts": {
            "ret": pooled_ret, "pk": pooled_pk, "vol": pooled_vol,
            "har_w": pooled_har_w, "har_m": pooled_har_m, "market_pk": market_pk_vals,
            "vz": pooled_vz, "abs_ret_acf": abs_ret_acf, "pk_acf": pk_acf,
        },
    }
    # release big frames
    del all_returns, ret_wide, all_ret_vals, all_pk_vals, all_vol_vals, pk_frames
    return summary


# --------------------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------------------
_CSS = ("body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1080px;color:#222}"
        "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;"
        "padding:4px 8px;text-align:center}th{background:#f2f2f2}h2{border-bottom:2px solid #ddd;margin-top:30px}"
        ".note{color:#555;font-size:13px}.grid{display:flex;flex-wrap:wrap;gap:10px}"
        ".card{flex:1 1 340px}.warn{color:#b00;font-weight:bold}code{background:#f4f4f4;padding:1px 4px}")


def _stats_table(stats: dict) -> str:  # pragma: no cover - html assembly, covered via smoke
    cols = ["n", "mean", "std", "min", "p1", "p25", "median", "p75", "p99", "max", "skew", "kurtosis"]
    head = "".join(f"<th>{c}</th>" for c in ["feature"] + cols)
    body = []
    for name, s in stats.items():
        cells = "".join(f"<td>{s[c]:.4g}</td>" if isinstance(s[c], float) else f"<td>{s[c]}</td>" for c in cols)
        body.append(f"<tr><td><b>{name}</b></td>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def render_market_html(summary: dict) -> str:  # pragma: no cover - presentation, exercised by smoke test
    p = summary["panel"]
    ch = summary["_charts"]
    c = summary["corr"]
    dirty = summary["dirty"]
    parts = [f"<html><head><meta charset='utf-8'><title>{p.upper()} EDA</title><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>{p.upper()} — exploratory data analysis &amp; data mining</h1>")
    parts.append(f"<p class='note'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | raw OHLCV "
                 f"<code>{VE.PRICE[p].name}</code> | processed target <code>{PROCESSED[p].name}</code>. "
                 "Parkinson column is sigma^2 (daily VARIANCE), not sigma. CPU/pandas only.</p>")

    # Executive summary
    parts.append("<h2>Executive summary</h2><ul>")
    parts.append(f"<li>Universe: <b>{summary['raw_tickers']}</b> raw tickers, "
                 f"<b>{summary['screened_tickers']}</b> after liquidity/history screen; "
                 f"{summary['total_rows']:,} ticker-days, {summary['date_min']} to {summary['date_max']}.</li>")
    parts.append(f"<li>Cross-sectional structure: median |rho| = "
                 f"<b>{c['median_abs_rho']:.3f}</b> (mean {c['mean_abs_rho']:.3f}, "
                 f"{c['frac_gt_0.3']*100:.1f}% of pairs |rho|&gt;0.3) over {c['n_tickers']} tickers.</li>")
    parts.append(f"<li>Zero-Parkinson (H==L) rate: <b>{summary['zero_parkinson_rate']*100:.2f}%</b>; "
                 f"zero-volume rate: {summary['zero_volume_rate']*100:.2f}%.</li>")
    parts.append(f"<li class='warn'>Dirty-data: high&lt;low {dirty['high_lt_low']}, "
                 f"open/close-outside {dirty['open_close_outside']}, nonpositive {dirty['nonpositive']}, "
                 f"zero-range {dirty['zero_range']}, NaN-rows {dirty['nan_rows']}, "
                 f"split-jump days {dirty['split_jumps']}, tickers with stale runs {dirty['stale_run_tickers']}.</li>")
    parts.append("</ul>")

    # 1. Coverage
    parts.append("<h2>1. Coverage &amp; structure</h2>")
    parts.append(tickers_per_year_chart(summary["tickers_per_year"], f"{p}: active tickers per year"))
    rows = sorted(summary["per_ticker"], key=lambda r: r["rows"])
    parts.append("<p class='note'>Shortest / longest histories (rows):</p><table>"
                 "<tr><th>ticker</th><th>rows</th><th>first</th><th>last</th><th>screened-in</th></tr>")
    for r in rows[:5] + rows[-5:]:
        parts.append(f"<tr><td>{r['ticker']}</td><td>{r['rows']}</td><td>{r['first']}</td>"
                     f"<td>{r['last']}</td><td>{'yes' if r['screened_in'] else 'NO'}</td></tr>")
    parts.append("</table>")

    # 2. Distributions
    parts.append("<h2>2. Distributions</h2>")
    parts.append(_stats_table(summary["stats"]))
    parts.append("<div class='grid'>")
    parts.append("<div class='card'>" + hist_chart(ch["ret"], "daily log-returns") + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["pk"], "Parkinson sigma^2 (log10)", log_x=True) + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["vol"], "volume (log10)", log_x=True) + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["har_w"], "HAR weekly (log10)", log_x=True) + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["har_m"], "HAR monthly (log10)", log_x=True) + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["market_pk"], "market_pk (log10)", log_x=True) + "</div>")
    parts.append("<div class='card'>" + hist_chart(ch["vz"], "volume_zscore_20") + "</div>")
    parts.append("</div>")

    # 3. Dirty-data / anomalies
    parts.append("<h2>3. Dirty-data / anomaly detection</h2>")
    parts.append("<p class='note'>Split-jump candidates (|1-day return|&gt;50%; VN not split-adjusted). "
                 "Top examples:</p><table><tr><th>ticker</th><th>date</th><th>return</th></tr>")
    for t, d, rr in sorted(summary["split_examples"], key=lambda x: abs(x[2]), reverse=True)[:20]:
        parts.append(f"<tr><td>{t}</td><td>{d}</td><td>{rr*100:.1f}%</td></tr>")
    parts.append("</table>")
    parts.append("<p class='note'>Stale price runs (&ge;5 identical consecutive closes). Examples:</p>"
                 "<table><tr><th>ticker</th><th>start</th><th>end</th><th>length</th></tr>")
    for t, s, e, ln in sorted(summary["stale_examples"], key=lambda x: x[3], reverse=True)[:20]:
        parts.append(f"<tr><td>{t}</td><td>{s}</td><td>{e}</td><td>{ln}</td></tr>")
    parts.append("</table>")

    # 4. Cross-sectional
    parts.append("<h2>4. Cross-sectional structure</h2>")
    if not c["corr"].empty:
        parts.append("<div class='grid'><div class='card'>"
                     + heatmap_chart(c["corr"], f"{p}: return-correlation heatmap") + "</div>")
        parts.append("<div class='card'>" + abs_rho_hist(c["corr"], f"{p}: |rho| distribution") + "</div></div>")
    parts.append(f"<p class='note'>median |rho| = <b>{c['median_abs_rho']:.3f}</b>, mean |rho| = "
                 f"{c['mean_abs_rho']:.3f}, 90th pct = {c['p90_abs_rho']:.3f}. Weak cross-sectional "
                 "structure implies limited headroom for a correlation-graph model.</p>")

    # 5. Temporal
    parts.append("<h2>5. Temporal structure</h2><div class='grid'>")
    parts.append("<div class='card'>" + acf_chart(ch["abs_ret_acf"], "ACF of |returns| (clustering)") + "</div>")
    parts.append("<div class='card'>" + acf_chart(ch["pk_acf"], "ACF of market_pk") + "</div>")
    parts.append("<div class='card'>" + line_by_year_chart(summary["zero_parkinson_by_year"],
                 "zero-Parkinson rate by year", "frac H==L") + "</div></div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _fmt(x, d=3, pct=False):  # pragma: no cover - formatting helper, covered via smoke
    if not isinstance(x, (int, float)) or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x*100:.2f}%" if pct else (f"{x:,}" if isinstance(x, int) else f"{x:.{d}f}")


def render_comparison_html(table_rows: list) -> str:  # pragma: no cover - presentation, covered via smoke
    parts = [f"<html><head><meta charset='utf-8'><title>VN markets EDA comparison</title>"
             f"<style>{_CSS}</style></head><body>"]
    parts.append("<h1>Cross-market EDA comparison — VN100 / VN30 / HOSE / HNX</h1>")
    parts.append(f"<p class='note'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
                 "Parkinson column is sigma^2 (variance). HOSE/HNX processed values are ETL upper-clipped "
                 "at 0.1; vn30/vn100 are not.</p>")
    cols = [("market", "market"), ("raw_tickers", "raw"), ("screened_tickers", "screened"),
            ("total_rows", "ticker-days"), ("median_abs_rho", "median |rho|"),
            ("mean_abs_rho", "mean |rho|"), ("zero_parkinson_rate", "zero-Parkinson"),
            ("zero_volume_rate", "zero-volume"), ("ohlc_high_lt_low", "high<low"),
            ("ohlc_open_close_outside", "O/C-outside"), ("ohlc_nonpositive", "nonpos"),
            ("zero_range_days", "zero-range"), ("split_jump_days", "split-jumps"),
            ("stale_run_tickers", "stale tickers")]
    parts.append("<table><tr>" + "".join(f"<th>{lbl}</th>" for _k, lbl in cols) + "</tr>")
    for row in table_rows:
        cells = []
        for k, _lbl in cols:
            v = row[k]
            if k == "market":
                cells.append(f"<td><b>{v}</b></td>")
            elif k in ("median_abs_rho", "mean_abs_rho"):
                cells.append(f"<td>{_fmt(v, 3)}</td>")
            elif k in ("zero_parkinson_rate", "zero_volume_rate"):
                cells.append(f"<td>{_fmt(v, pct=True)}</td>")
            else:
                cells.append(f"<td>{_fmt(v)}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    parts.append("<p class='note'>Prioritize dirty-data issues that affect the volatility TARGET "
                 "(zero-range / zero-Parkinson days floor the variance and inflate QLIKE) over cosmetic "
                 "ones. Weak median |rho| (as on HNX) caps the value of a correlation-graph model.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def run(panels=("vn30", "vn100", "hose"), comparison_extra=("hnx",), limit=None, out_dir=OUT_DIR):
    """Process each panel SEQUENTIALLY: analyze -> write per-market HTML -> discard. Then build the
    cross-market comparison (adding ``comparison_extra`` panels, e.g. hnx, for the table only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    written = []
    for panel in panels:
        s = analyze_panel(panel, limit=limit)
        html = render_market_html(s)
        path = out_dir / f"2026-08-30_{panel}_eda.html"
        path.write_text(html, encoding="utf-8")
        written.append(str(path))
        # keep only the compact fields needed for the comparison; drop charts/per-ticker to bound RAM
        summaries[panel] = _compact(s)
        del s
    for panel in comparison_extra:
        if panel in summaries:
            continue
        s = analyze_panel(panel, limit=limit)
        summaries[panel] = _compact(s)
        del s
    table_rows = build_cross_market_table(summaries)
    comp_html = render_comparison_html(table_rows)
    comp_path = out_dir / "2026-08-30_vn_markets_eda_comparison.html"
    comp_path.write_text(comp_html, encoding="utf-8")
    written.append(str(comp_path))
    md_path = out_dir / "2026-08-30_vn_markets_eda_comparison.md"
    md_path.write_text(_comparison_md(table_rows), encoding="utf-8")
    written.append(str(md_path))
    return {"written": written, "table_rows": table_rows}


def _compact(summary: dict) -> dict:
    """Strip the heavy chart arrays / per-ticker list, keep only comparison-table inputs."""
    return {k: summary[k] for k in ("panel", "raw_tickers", "screened_tickers", "total_rows",
                                    "date_min", "date_max", "corr", "zero_parkinson_rate",
                                    "zero_volume_rate", "dirty")}


def _comparison_md(table_rows: list) -> str:
    lines = ["# Cross-market EDA comparison — VN100 / VN30 / HOSE / HNX", "",
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. Parkinson = sigma^2 (variance).", "",
             "| market | raw | screened | ticker-days | median rho | zero-Parkinson | zero-volume "
             "| high<low | O/C-outside | nonpos | zero-range | split-jumps | stale tickers |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table_rows:
        lines.append("| {market} | {raw_tickers} | {screened_tickers} | {total_rows:,} | "
                     "{mr:.3f} | {zp:.2%} | {zv:.2%} | {hl} | {oc} | {np_} | {zr} | {sj} | {st} |".format(
                         mr=r["median_abs_rho"], zp=r["zero_parkinson_rate"], zv=r["zero_volume_rate"],
                         hl=r["ohlc_high_lt_low"], oc=r["ohlc_open_close_outside"], np_=r["ohlc_nonpositive"],
                         zr=r["zero_range_days"], sj=r["split_jump_days"], st=r["stale_run_tickers"], **r))
    return "\n".join(lines) + "\n"


def main():  # pragma: no cover - CLI entry
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", default=["vn30", "vn100", "hose"])
    ap.add_argument("--extra", nargs="+", default=["hnx"], help="panels for the comparison table only")
    ap.add_argument("--limit", type=int, default=None, help="cap tickers/panel (smoke)")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    res = run(panels=a.panels, comparison_extra=a.extra, limit=a.limit, out_dir=a.out)
    for w in res["written"]:
        print("wrote", w)


if __name__ == "__main__":  # pragma: no cover
    main()
