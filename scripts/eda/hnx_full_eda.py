"""Broad exploratory + dirty-data EDA over the ENTIRE HNX panel (raw OHLCV + processed Parkinson).

Complements the per-ticker diagnostic table: this is the CROSS-SECTIONAL / TEMPORAL / ANOMALY view
(distributions, outliers, correlation structure, illiquidity over time, data-source seams) rather than a
per-stock issue list.

Target column semantics (units trap, per CLAUDE.md): ``parkinson_volatility`` is Parkinson VARIANCE
sigma^2 = ln(H/L)^2 / (4 ln 2), NOT sigma. Every "Parkinson" number below is a variance.

Derived node features replicate the delivered runner (``masked_rich``): HAR weekly/monthly = trailing
5/22-day rolling means of Parkinson; ``market_pk`` = cross-sectional MEDIAN of sqrt(pk) over valid nodes
at day t; ``volume_zscore_20`` = trailing 20-day z-score of log1p(volume) per ticker.

READ-ONLY: never writes into data/. Usage:
    python scripts/eda/hnx_full_eda.py            # -> docs/reports/2026-08-30_hnx_full_eda.{html,md}
"""
from __future__ import annotations

import base64
import glob
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HNX_RAW = REPO / "data" / "raw" / "prices" / "hnx_vnstock"
HNX_PROC = REPO / "data" / "processed" / "hnx"

_LN2 = np.log(2.0)
OHLC_RTOL = 1e-5           # same tolerance as tests/test_raw_prices_quality.py (absorbs float32 noise)
WEEKLY_WIN = 5
MONTHLY_WIN = 22
VOL_WIN = 20
STALE_MIN_RUN = 5         # >= this many identical consecutive closes == a "stale" run
JUMP_THR = 0.50           # |log-return| above this flags a candidate unadjusted split/dividend
SCREEN_MAX_ZERO_FRAC = 0.5   # floor_sensitivity.screen_files default (illiquid drop)
SCREEN_MIN_ROWS = 250

OHLC = ["open", "high", "low", "close"]


# --------------------------------------------------------------------------------------------------
# Pure detectors (unit-tested)
# --------------------------------------------------------------------------------------------------
def ohlc_violation_mask(df: pd.DataFrame, rtol: float = OHLC_RTOL) -> dict:
    """Per-row OHLC geometry violation masks for one ticker frame.

    Returns a dict of boolean numpy arrays (aligned to df rows):
      ``nonpositive`` : any of O/H/L/C <= 0
      ``high_lt_low`` : high < low
      ``oc_out``      : open or close outside [low, high] (beyond rtol)
      ``zero_range``  : high == low (limit / no intraday movement)
      ``any``         : union of nonpositive | high_lt_low | oc_out (the CORRUPT rows; zero_range is benign)
    ``rtol`` matches the raw-data quality gate so float32 storage noise is not flagged as corruption.
    """
    o, h, lo, c = (pd.to_numeric(df[k], errors="coerce").to_numpy(float) for k in OHLC)
    hi_oc, lo_oc = np.maximum(o, c), np.minimum(o, c)
    nonpositive = ~((o > 0) & (h > 0) & (lo > 0) & (c > 0))
    high_lt_low = h < lo
    oc_out = (h < hi_oc * (1 - rtol)) | (lo > lo_oc * (1 + rtol))
    zero_range = np.isclose(h, lo)
    return {
        "nonpositive": nonpositive,
        "high_lt_low": high_lt_low,
        "oc_out": oc_out & ~high_lt_low,          # count high<low once, in its own bucket
        "zero_range": zero_range & ~nonpositive,
        "any": nonpositive | high_lt_low | oc_out,
    }


def parkinson_variance(df: pd.DataFrame) -> np.ndarray:
    """Parkinson VARIANCE sigma^2 = ln(H/L)^2/(4 ln2); NaN where OHLC geometry is invalid."""
    o, h, lo, c = (pd.to_numeric(df[k], errors="coerce").to_numpy(float) for k in OHLC)
    hi_oc, lo_oc = np.maximum(o, c), np.minimum(o, c)
    ok = ((o > 0) & (h > 0) & (lo > 0) & (c > 0) & (h >= lo)
          & (h >= hi_oc * (1 - OHLC_RTOL)) & (lo <= lo_oc * (1 + OHLC_RTOL)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.log(h / lo) ** 2 / (4 * _LN2)
    pk[~ok] = np.nan
    return pk


def log_returns(close: np.ndarray) -> np.ndarray:
    """Close-to-close log-returns; first element NaN. Non-positive closes -> NaN return."""
    c = np.asarray(close, dtype=float)
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c / prev)
    r[(prev <= 0) | (c <= 0)] = np.nan
    return r


def stale_close_runs(close: np.ndarray, min_run: int = STALE_MIN_RUN) -> dict:
    """Detect runs of consecutive IDENTICAL closes (stale / non-trading prices).

    Returns ``{"max_run": int, "n_runs": int, "n_days_in_runs": int}`` where a run counts only when its
    length >= ``min_run``. A single unchanged close (run length 1) is normal; long flat runs signal a
    suspended / illiquid ticker.
    """
    c = np.asarray(close, dtype=float)
    if c.size == 0:
        return {"max_run": 0, "n_runs": 0, "n_days_in_runs": 0}
    change = np.ones(c.size, dtype=bool)
    change[1:] = c[1:] != c[:-1]
    grp = np.cumsum(change)
    _, lengths = np.unique(grp, return_counts=True)
    long = lengths[lengths >= min_run]
    return {"max_run": int(lengths.max()), "n_runs": int(long.size),
            "n_days_in_runs": int(long.sum())}


def split_jump_days(close: np.ndarray, thr: float = JUMP_THR) -> np.ndarray:
    """Indices where |close-to-close log-return| > ``thr`` (candidate unadjusted split/dividend).

    VN prices are not split-adjusted, so a >50% one-day move is more likely a corporate action than a
    true return."""
    r = log_returns(close)
    return np.flatnonzero(np.abs(r) > thr)


def robust_z_outliers(x: np.ndarray, thr: float = 5.0) -> np.ndarray:
    """Boolean mask of robust-z (MAD) outliers: |x-median| > thr * 1.4826 * MAD. All-equal -> no outliers."""
    v = np.asarray(x, dtype=float)
    finite = np.isfinite(v)
    out = np.zeros(v.shape, dtype=bool)
    if finite.sum() == 0:
        return out
    med = np.median(v[finite])
    mad = np.median(np.abs(v[finite] - med))
    if mad == 0:
        return out
    out[finite] = np.abs(v[finite] - med) > thr * 1.4826 * mad
    return out


def leading_zero_volume(volume: np.ndarray) -> int:
    """Number of leading rows with zero/NaN volume (backfilled-before-listing candidate prefix)."""
    v = np.asarray(volume, dtype=float)
    nz = np.flatnonzero((v > 0) & np.isfinite(v))
    return int(nz[0]) if nz.size else int(v.size)


def _skew_kurt(x: np.ndarray) -> tuple:
    """Sample skewness and excess kurtosis of the finite entries (0 when < 3 points or zero variance)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0, 0.0
    m, s = v.mean(), v.std()
    if s == 0:
        return 0.0, 0.0
    z = (v - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def acf(x: np.ndarray, nlags: int = 30) -> np.ndarray:
    """Sample autocorrelation of the finite series for lags 1..nlags (lag 0 omitted)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < nlags + 2:
        return np.full(nlags, np.nan)
    v = v - v.mean()
    denom = np.dot(v, v)
    if denom == 0:
        return np.zeros(nlags)
    return np.array([np.dot(v[:-k], v[k:]) / denom for k in range(1, nlags + 1)])


# --------------------------------------------------------------------------------------------------
# Loading + per-ticker aggregation
# --------------------------------------------------------------------------------------------------
@dataclass
class Panel:
    tickers: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)          # ticker -> raw OHLCV frame (date-sorted, unique)
    pk_wide: pd.DataFrame = None                       # date x ticker Parkinson variance (processed target)
    ret_wide: pd.DataFrame = None                      # date x ticker log-returns


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", kind="stable")
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def load_panel(raw_dir: Path = HNX_RAW, proc_dir: Path = HNX_PROC) -> Panel:
    """Load every HNX ticker with BOTH a raw OHLCV file and a processed Parkinson file."""
    raw_files = {Path(f).stem.replace("_ohlcv", ""): Path(f)
                 for f in glob.glob(str(raw_dir / "*_ohlcv.csv"))}
    proc_files = {Path(f).stem.replace("_processed", ""): Path(f)
                  for f in glob.glob(str(proc_dir / "*_processed.csv"))}
    tickers = sorted(set(raw_files) & set(proc_files))
    raw, pk_series, ret_series = {}, {}, {}
    for tk in tickers:
        df = _read_raw(raw_files[tk])
        raw[tk] = df
        proc = pd.read_csv(proc_files[tk], parse_dates=["date"]).sort_values("date")
        pk_series[tk] = proc.set_index("date")["parkinson_volatility"]
        ret_series[tk] = pd.Series(log_returns(df["close"].to_numpy()), index=df["date"])
    pk_wide = pd.DataFrame(pk_series).sort_index()
    ret_wide = pd.DataFrame(ret_series).sort_index()
    return Panel(tickers=tickers, raw=raw, pk_wide=pk_wide, ret_wide=ret_wide)


def per_ticker_stats(panel: Panel) -> pd.DataFrame:
    """One row per ticker: coverage, dirty-data counts, illiquidity, stale/jump flags."""
    rows = []
    for tk in panel.tickers:
        df = panel.raw[tk]
        v = ohlc_violation_mask(df)
        pk = panel.pk_wide[tk].dropna()
        close = df["close"].to_numpy()
        vol = df["volume"].to_numpy() if "volume" in df else np.zeros(len(df))
        stale = stale_close_runs(close)
        rows.append({
            "ticker": tk,
            "n_raw": len(df),
            "n_proc": int(pk.size),
            "start": df["date"].min(),
            "end": df["date"].max(),
            "nonpositive": int(v["nonpositive"].sum()),
            "high_lt_low": int(v["high_lt_low"].sum()),
            "oc_out": int(v["oc_out"].sum()),
            "corrupt_rows": int(v["any"].sum()),
            "zero_range": int(v["zero_range"].sum()),
            "zero_range_frac": float(v["zero_range"].mean()),
            "zero_pk": int((pk == 0).sum()),
            "zero_pk_frac": float((pk == 0).mean()) if pk.size else np.nan,
            "zero_vol_days": int((vol == 0).sum()),
            "leading_zero_vol": leading_zero_volume(vol),
            "stale_max_run": stale["max_run"],
            "stale_days": stale["n_days_in_runs"],
            "jump_days": int(split_jump_days(close).size),
        })
    return pd.DataFrame(rows)


def screened_universe(pt: pd.DataFrame) -> pd.Series:
    """Boolean per-ticker: passes the delivered liquidity/history screen (>=250 rows AND zero-pk frac<=0.5)."""
    return (pt["n_proc"] >= SCREEN_MIN_ROWS) & (pt["zero_pk_frac"] <= SCREEN_MAX_ZERO_FRAC)


# --------------------------------------------------------------------------------------------------
# Charts (matplotlib -> base64 PNG, no external assets)
# --------------------------------------------------------------------------------------------------
def _fig_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str, alt: str) -> str:
    return f"<img alt='{alt}' src='data:image/png;base64,{b64}'/>"


def chart_tickers_per_year(panel: Panel) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    active = {}
    for tk in panel.tickers:
        yrs = panel.raw[tk]["date"].dt.year.unique()
        for y in yrs:
            active[int(y)] = active.get(int(y), 0) + 1
    years = sorted(active)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(years, [active[y] for y in years], color="#3b6")
    ax.set_title("Active HNX tickers per year (>=1 trading day)")
    ax.set_xlabel("year"); ax.set_ylabel("# tickers")
    return _fig_b64(fig)


def chart_hist(values: np.ndarray, title: str, xlabel: str, log_x: bool = False,
               bins: int = 60, color: str = "#37c") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6, 3))
    if log_x:
        v = v[v > 0]
        if v.size:
            ax.hist(np.log10(v), bins=bins, color=color)
        ax.set_xlabel(f"log10({xlabel})")
    else:
        ax.hist(v, bins=bins, color=color)
        ax.set_xlabel(xlabel)
    ax.set_title(title); ax.set_ylabel("count")
    return _fig_b64(fig)


def chart_zero_pk_by_year(panel: Panel) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stacked = panel.pk_wide.stack()
    by_year = stacked.groupby(stacked.index.get_level_values(0).year)
    frac = by_year.apply(lambda s: float((s == 0).mean()))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frac.index, frac.values, "-o", color="#c33")
    ax.set_ylim(0, 1); ax.set_title("Zero-Parkinson (illiquid) day rate by year, pooled")
    ax.set_xlabel("year"); ax.set_ylabel("fraction of ticker-days == 0")
    return _fig_b64(fig)


def chart_corr_heatmap(corr: np.ndarray, labels: list) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Pairwise return correlation (screened tickers)")
    step = max(1, len(labels) // 20)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=5)
    ax.set_yticks(range(0, len(labels), step))
    ax.set_yticklabels(labels[::step], fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _fig_b64(fig)


def chart_line(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str,
               color: str = "#639") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, "-", color=color)
    ax.axhline(0, color="#999", lw=0.7)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return _fig_b64(fig)


# --------------------------------------------------------------------------------------------------
# Cross-sectional correlation
# --------------------------------------------------------------------------------------------------
def return_correlations(ret_wide: pd.DataFrame, min_overlap: int = 100) -> tuple:
    """Pairwise Pearson correlation of daily log-returns over overlapping dates.

    Returns ``(corr_matrix, labels, offdiag_values)``. Pairs with < ``min_overlap`` overlapping dates
    are NaN in the matrix and excluded from ``offdiag_values``.
    """
    c = ret_wide.corr(min_periods=min_overlap)
    labels = list(c.columns)
    m = c.to_numpy()
    iu = np.triu_indices(m.shape[0], k=1)
    off = m[iu]
    off = off[np.isfinite(off)]
    return m, labels, off


# --------------------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------------------
def _tbl(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"<td>{floatfmt.format(v)}</td>" if np.isfinite(v) else "<td>-</td>")
            else:
                cells.append(f"<td>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def build_report(panel: Panel) -> dict:
    """Compute all statistics + charts. Returns ``{"html": str, "md": str, "stats": dict}``."""
    pt = per_ticker_stats(panel)
    kept = screened_universe(pt)
    pt = pt.assign(screened=kept)

    # ---- pooled dirty-data headline numbers ----
    total_ticker_days = int(pt["n_proc"].sum())
    pooled_pk = panel.pk_wide.stack()
    zero_pk_frac = float((pooled_pk == 0).mean())
    corrupt_total = int(pt["corrupt_rows"].sum())
    zero_range_total = int(pt["zero_range"].sum())
    jump_total = int(pt["jump_days"].sum())

    # ---- corrupt-row date clustering (data-source seam detection) ----
    corrupt_dates = {}
    for tk in panel.tickers:
        df = panel.raw[tk]
        m = ohlc_violation_mask(df)["any"]
        for d in df["date"][m]:
            corrupt_dates[d] = corrupt_dates.get(d, 0) + 1
    seam = (pd.Series(corrupt_dates).sort_values(ascending=False).head(12)
            if corrupt_dates else pd.Series(dtype=int))

    # ---- distributions ----
    ret_all = panel.ret_wide.to_numpy().ravel()
    pk_all = pooled_pk.to_numpy()
    vol_all = np.concatenate([panel.raw[tk]["volume"].to_numpy(dtype=float) for tk in panel.tickers])
    ret_sk, ret_ku = _skew_kurt(ret_all)
    pk_sk, pk_ku = _skew_kurt(pk_all)

    # HAR features + volume z-score (pooled) computed from processed pk / raw volume
    har_w, har_m = [], []
    for tk in panel.tickers:
        s = panel.pk_wide[tk]
        har_w.append(s.rolling(WEEKLY_WIN, min_periods=WEEKLY_WIN).mean().to_numpy())
        har_m.append(s.rolling(MONTHLY_WIN, min_periods=MONTHLY_WIN).mean().to_numpy())
    har_w = np.concatenate(har_w); har_m = np.concatenate(har_m)
    market_pk = np.nanmedian(np.sqrt(panel.pk_wide.to_numpy(float)), axis=1)

    # ---- correlation structure ----
    ret_screened = panel.ret_wide[[tk for tk in panel.tickers if bool(kept[pt.index[pt["ticker"] == tk][0]])]]
    corr_m, corr_labels, corr_off = return_correlations(ret_screened)
    med_abs_corr = float(np.median(np.abs(corr_off))) if corr_off.size else np.nan
    frac_high = float(np.mean(corr_off > 0.7)) if corr_off.size else np.nan
    n_high = int(np.sum(corr_off > 0.7))
    n_pairs = int(corr_off.size)

    # ---- volatility clustering (ACF of |returns|, market-average) ----
    mkt_abs_ret = panel.ret_wide.abs().mean(axis=1).to_numpy()
    acf_abs = acf(mkt_abs_ret, nlags=30)

    stats = {
        "n_tickers": len(panel.tickers),
        "n_screened": int(kept.sum()),
        "n_dropped": int((~kept).sum()),
        "total_ticker_days": total_ticker_days,
        "zero_pk_frac": zero_pk_frac,
        "corrupt_total": corrupt_total,
        "zero_range_total": zero_range_total,
        "jump_total": jump_total,
        "med_abs_corr": med_abs_corr,
        "n_high_corr_pairs": n_high,
        "n_pairs": n_pairs,
        "ret_skew": ret_sk, "ret_kurt": ret_ku,
        "pk_skew": pk_sk, "pk_kurt": pk_ku,
        "date_min": str(panel.pk_wide.index.min().date()),
        "date_max": str(panel.pk_wide.index.max().date()),
    }

    # ---- charts ----
    charts = {
        "tickers_year": chart_tickers_per_year(panel),
        "ret_hist": chart_hist(ret_all, "Daily log-return distribution", "log-return", bins=80),
        "pk_hist": chart_hist(pk_all, "Parkinson variance sigma^2 (log10 x)", "parkinson_var", log_x=True),
        "vol_hist": chart_hist(vol_all, "Daily volume (log10 x)", "volume", log_x=True, color="#a63"),
        "harw_hist": chart_hist(har_w, "HAR weekly (5d mean pk, log10 x)", "har_weekly", log_x=True, color="#585"),
        "harm_hist": chart_hist(har_m, "HAR monthly (22d mean pk, log10 x)", "har_monthly", log_x=True, color="#575"),
        "mktpk_hist": chart_hist(market_pk, "market_pk (cross-sectional median sqrt(pk))", "market_pk", color="#b58"),
        "zeropk_year": chart_zero_pk_by_year(panel),
        "zeropk_ticker": chart_hist(pt["zero_pk_frac"].to_numpy(), "Per-ticker zero-Parkinson fraction",
                                    "zero_pk_frac", bins=40, color="#c33"),
        "corr_hist": chart_hist(corr_off, "Pairwise return-correlation distribution (screened)",
                                "corr", bins=60, color="#369"),
        "corr_heatmap": chart_corr_heatmap(corr_m, corr_labels),
        "acf_abs": chart_line(np.arange(1, 31), acf_abs, "ACF of |market return| (volatility clustering)",
                              "lag (days)", "autocorrelation"),
    }

    html = _render_html(stats, pt, seam, charts, med_abs_corr, frac_high)
    md = _render_md(stats, pt, seam)
    return {"html": html, "md": md, "stats": stats, "per_ticker": pt}


def _render_html(stats, pt, seam, charts, med_abs_corr, frac_high) -> str:
    s = stats
    dirty_pct = 100 * s["zero_pk_frac"]
    top_zero = pt.sort_values("zero_pk_frac", ascending=False).head(12)[
        ["ticker", "n_proc", "zero_pk_frac", "zero_range", "stale_max_run", "screened"]]
    top_corrupt = pt[pt["corrupt_rows"] > 0].sort_values("corrupt_rows", ascending=False).head(15)[
        ["ticker", "corrupt_rows", "nonpositive", "high_lt_low", "oc_out", "n_raw"]]
    top_jump = pt[pt["jump_days"] > 0].sort_values("jump_days", ascending=False).head(15)[
        ["ticker", "jump_days", "n_raw", "start"]]
    seam_tbl = ("<table><tr><th>date</th><th># tickers with corrupt bar</th></tr>"
                + "".join(f"<tr><td>{d.date()}</td><td>{n}</td></tr>" for d, n in seam.items())
                + "</table>") if len(seam) else "<p>None.</p>"

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:22px;max-width:1080px;color:#222}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;"
           "padding:3px 8px;text-align:center}th{background:#f2f2f2}h2{border-bottom:2px solid #ddd;margin-top:34px}"
           "img{max-width:100%;height:auto;margin:6px 0}.kpi{display:inline-block;border:1px solid #ccc;"
           "border-radius:8px;padding:8px 14px;margin:5px;background:#fafafa}.kpi b{font-size:20px;display:block}"
           ".warn{color:#b00}.note{color:#555;font-size:13px}")

    def kpi(label, val, warn=False):
        cls = "kpi warn" if warn else "kpi"
        return f"<span class='{cls}'>{label}<b>{val}</b></span>"

    p = [f"<html><head><meta charset='utf-8'><title>HNX full EDA</title><style>{css}</style></head><body>",
         "<h1>HNX panel — exploratory data analysis &amp; dirty-data audit</h1>",
         f"<p class='note'>Universe: {s['n_tickers']} HNX tickers with both raw OHLCV and processed "
         f"Parkinson-variance files, {s['date_min']} .. {s['date_max']}. "
         "The <code>parkinson_volatility</code> column is VARIANCE sigma^2, not sigma.</p>",
         "<h2>Executive summary</h2>",
         kpi("Tickers (raw&cap;proc)", s["n_tickers"]),
         kpi("Pass liquidity screen", s["n_screened"]),
         kpi("Dropped (illiquid/short)", s["n_dropped"], warn=True),
         kpi("Ticker-days", f"{s['total_ticker_days']:,}"),
         kpi("Zero-Parkinson days", f"{dirty_pct:.1f}%", warn=True),
         kpi("Corrupt OHLC rows", s["corrupt_total"], warn=s["corrupt_total"] > 0),
         kpi("Zero-range (H=L) rows", f"{s['zero_range_total']:,}"),
         kpi("Split-candidate jumps", s["jump_total"], warn=True),
         kpi("Median |corr|", f"{med_abs_corr:.3f}"),
         kpi(">0.7 corr pairs", f"{s['n_high_corr_pairs']}/{s['n_pairs']:,}"),
         "<h3>Most important findings</h3><ol>",
         f"<li><b>Extreme illiquidity dominates the target.</b> {dirty_pct:.1f}% of all ticker-days have "
         f"Parkinson variance EXACTLY zero (H=L limit/no-trade days). These are floored in QLIKE and make "
         "the volatility target degenerate for many names — the single biggest data caveat.</li>",
         f"<li><b>The liquidity screen removes {s['n_dropped']}/{s['n_tickers']} tickers.</b> Only "
         f"{s['n_screened']} survive (>= {SCREEN_MIN_ROWS} rows and <= {int(SCREEN_MAX_ZERO_FRAC*100)}% "
         "zero-Parkinson days); the raw HNX universe is far larger than the usable one.</li>",
         f"<li><b>Cross-sectional structure is weak.</b> Median pairwise |return correlation| = "
         f"{med_abs_corr:.3f}; only {s['n_high_corr_pairs']}/{s['n_pairs']:,} pairs exceed 0.7. A spatial "
         "graph over HNX returns has little signal to exploit.</li>",
         f"<li><b>{s['corrupt_total']} OHLC-geometry violations cluster on specific dates</b> (see seam table) "
         "— open/close outside [low,high] recurring across many tickers on the same day indicates a "
         "data-source artifact, not independent per-ticker errors.</li>",
         f"<li><b>Fat tails + volatility clustering are present.</b> Return excess kurtosis = "
         f"{s['ret_kurt']:.1f}; |return| ACF stays positive over weeks (see chart) — standard for a real "
         "equity market and consistent with HAR being a strong baseline.</li></ol>"]

    p += ["<h2>1. Coverage &amp; structure</h2>", _img(charts["tickers_year"], "tickers per year"),
          "<p class='note'>Each ticker enters on its listing year; the panel is an unbalanced (staggered) "
          "panel with heavy left-censoring.</p>"]

    p += ["<h2>2. Distributions</h2>",
          f"<p class='note'>Return skew={s['ret_skew']:.2f}, excess-kurt={s['ret_kurt']:.1f}; "
          f"Parkinson-variance skew={s['pk_skew']:.2f}, excess-kurt={s['pk_kurt']:.1f}. "
          "Parkinson/volume/HAR are shown on log10 x (heavy right tails; zeros dropped in log view).</p>",
          _img(charts["ret_hist"], "returns"), _img(charts["pk_hist"], "parkinson"),
          _img(charts["vol_hist"], "volume"), _img(charts["harw_hist"], "har weekly"),
          _img(charts["harm_hist"], "har monthly"), _img(charts["mktpk_hist"], "market pk")]

    p += ["<h2>3. Dirty-data / anomaly detection</h2>",
          "<h3>3a. Zero-Parkinson (illiquid) days over time &amp; per ticker</h3>",
          _img(charts["zeropk_year"], "zero pk by year"), _img(charts["zeropk_ticker"], "zero pk per ticker"),
          "<p class='note'>Zero-Parkinson = H==L days (limit-price / no intraday movement). "
          "Right-skewed per-ticker distribution: a subset of names is almost entirely illiquid.</p>",
          "<h4>Top illiquid tickers</h4>", _tbl(top_zero),
          "<h3>3b. OHLC-geometry violations</h3>",
          ("<p class='note'>These rows are genuinely corrupt (nonpositive price, high&lt;low, or open/close "
           "outside [low,high] beyond float tolerance) — they must be excluded from any estimator.</p>"
           + _tbl(top_corrupt.astype({c: int for c in ["corrupt_rows", "nonpositive", "high_lt_low", "oc_out", "n_raw"]}))
           ) if len(top_corrupt) else "<p>No corrupt OHLC rows detected.</p>",
          "<h4>Corrupt-bar date clustering (data-source seam)</h4>", seam_tbl,
          "<h3>3c. Candidate unadjusted splits / large jumps (|return|&gt;50%)</h3>",
          (_tbl(top_jump) if len(top_jump) else "<p>None.</p>"),
          "<p class='note'>VN prices are NOT split-adjusted; a &gt;50% one-day move is more likely a "
          "corporate action than a real return. Flagged for review, not auto-corrected.</p>"]

    p += ["<h2>4. Cross-sectional structure</h2>",
          _img(charts["corr_hist"], "corr dist"), _img(charts["corr_heatmap"], "corr heatmap"),
          f"<p class='note'>Median |corr| = {med_abs_corr:.3f}; fraction of pairs &gt;0.7 = {frac_high:.4f}. "
          "Weak co-movement implies a return-correlation graph adds little cross-stock information — "
          "consistent with the project's negative graph-ablation result.</p>"]

    p += ["<h2>5. Temporal patterns</h2>", _img(charts["acf_abs"], "acf abs returns"),
          "<p class='note'>Slowly-decaying positive autocorrelation of |returns| = volatility clustering; "
          "the zero-Parkinson-by-year chart (section 3a) also shows the illiquidity regime shrinking over "
          "time as the exchange matured.</p>"]

    p += ["<h2>Appendix: full per-ticker table</h2>",
          _tbl(pt[["ticker", "n_raw", "n_proc", "zero_pk_frac", "zero_range", "corrupt_rows",
                   "jump_days", "stale_max_run", "leading_zero_vol", "screened"]]),
          "</body></html>"]
    return "\n".join(p)


def _render_md(stats, pt, seam) -> str:
    s = stats
    lines = [
        "# HNX full EDA — key findings & dirty-data triage",
        "",
        f"Universe: **{s['n_tickers']}** HNX tickers (raw OHLCV ∩ processed Parkinson), "
        f"{s['date_min']} .. {s['date_max']}, {s['total_ticker_days']:,} ticker-days. "
        "`parkinson_volatility` is VARIANCE σ², not σ.",
        "",
        "## Headline dirty-data figures",
        f"- **Zero-Parkinson (H=L illiquid) days: {100*s['zero_pk_frac']:.1f}%** of all ticker-days "
        "(target is exactly zero → floored in QLIKE).",
        f"- **Liquidity screen keeps {s['n_screened']}/{s['n_tickers']}** tickers "
        f"(≥{SCREEN_MIN_ROWS} rows and ≤{int(SCREEN_MAX_ZERO_FRAC*100)}% zero-Parkinson); "
        f"**{s['n_dropped']} dropped**.",
        f"- **{s['corrupt_total']} corrupt OHLC rows** (nonpositive / high<low / open-close outside "
        f"[low,high]); **{s['zero_range_total']:,} zero-range (H=L)** rows.",
        f"- **{s['jump_total']} candidate unadjusted-split jumps** (|return|>50%).",
        "",
        "## New insights",
        f"1. Extreme illiquidity is the dominant data property, not an edge case ({100*s['zero_pk_frac']:.1f}% "
        "zero-variance days). The usable HNX universe is a minority of the listed universe.",
        f"2. Return co-movement is weak: median |corr| = {s['med_abs_corr']:.3f}, only "
        f"{s['n_high_corr_pairs']}/{s['n_pairs']:,} pairs >0.7 → little for a spatial graph to exploit.",
        "3. OHLC-geometry violations cluster on shared calendar dates across many tickers "
        "(data-source seam), not random per-ticker noise.",
        f"4. Heavy tails (return excess kurtosis {s['ret_kurt']:.1f}) + persistent |return| autocorrelation "
        "= genuine volatility clustering; supports HAR as a strong baseline.",
        "",
        "## Prioritized dirty-data issues",
        "| issue | severity | affects the volatility TARGET/results? |",
        "|---|---|---|",
        "| Zero-Parkinson illiquid days (H=L) | HIGH | YES — floors the target, inflates QLIKE, "
        "makes point metrics uninformative for illiquid names; already mitigated by the liquidity screen. |",
        "| Corrupt OHLC rows (open/close outside [low,high], nonpositive) | MEDIUM | YES if used — "
        "the estimator already NaNs them; must stay excluded. |",
        "| Unadjusted split/dividend jumps (|ret|>50%) | MEDIUM | YES for close-to-close / overnight "
        "estimators; Parkinson (intraday range) is immune. |",
        "| Stale/flat close runs & zero-volume days | LOW-MEDIUM | INDIRECT — a symptom of the same "
        "illiquidity captured by the zero-Parkinson screen. |",
        "| Leading backfilled prefix before listing | LOW | COSMETIC once the screen + rolling warm-up "
        "drop the early rows. |",
        "",
        "## Recommendation for the paper's data-limitations section",
        f"Report HNX as a **thin, illiquid market**: {100*s['zero_pk_frac']:.1f}% zero-variance ticker-days, "
        f"only {s['n_screened']}/{s['n_tickers']} names surviving a ≤{int(SCREEN_MAX_ZERO_FRAC*100)}% "
        "zero-Parkinson liquidity screen. State that (a) prices are NOT split-adjusted (jumps flagged, "
        "Parkinson intraday-range target is robust to overnight gaps), (b) the target is a VARIANCE, "
        "(c) the zero-Parkinson floor makes QLIKE floor-sensitive on HNX, and (d) weak cross-sectional "
        "correlation limits the headroom for spatial-graph models on this panel.",
        "",
    ]
    if len(seam):
        lines += ["## Corrupt-bar date clusters (seam)", ""]
        lines += [f"- {d.date()}: {n} tickers" for d, n in seam.items()]
        lines += [""]
    return "\n".join(lines)


DEFAULT_HTML = REPO / "docs" / "reports" / "2026-08-30_hnx_full_eda.html"
DEFAULT_MD = REPO / "docs" / "reports" / "2026-08-30_hnx_full_eda.md"


def run_eda(raw_dir: Path = HNX_RAW, proc_dir: Path = HNX_PROC,
            out_html: Path = DEFAULT_HTML, out_md: Path = DEFAULT_MD) -> dict:
    """Load the panel, build the report, write the HTML + MD, and return the stats dict."""
    panel = load_panel(raw_dir, proc_dir)
    rep = build_report(panel)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(rep["html"], encoding="utf-8")
    Path(out_md).write_text(rep["md"], encoding="utf-8")
    return rep["stats"]


def main():  # pragma: no cover
    stats = run_eda()
    print("[hnx-eda] wrote docs/reports/2026-08-30_hnx_full_eda.{html,md}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()
