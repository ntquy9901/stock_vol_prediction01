"""ETL-clean + ENRICH per-ticker processed data (baseline A3).

Reuses the already-tested building blocks (does NOT re-derive any formula):
  * ``scripts/etl_audit/etl_cleaning.py``          -- ETL cleaners (each unit-tested).
  * ``scripts/eda/volatility_estimators.py``       -- parkinson/garman_klass/rogers_satchell/yang_zhang.
  * ``submission/soict_lstm_gat/pipeline_config``  -- canonical windows (never hardcoded here).

Output = ``data/processed_enriched/<market>/<ticker>.csv`` with the causal columns listed in
``ENRICHED_COLUMNS``. Every per-row column depends solely on dates <= t (schema-spec leakage rule);
no train/val/test-boundary statistic, scaler, adjacency, or future/centered window is baked in.

The ``open``/``high``/``low``/``close``/``volume`` columns are the POST-ETL CLEANED OHLC used to
compute the estimators (== raw except on flagged dirty bars; the raw pre-clean values live in
``data/raw/prices/...``), so each row is self-verifiable next to its derived estimators.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in ("scripts/etl_audit", "scripts/eda", "submission/soict_lstm_gat"):
    _ap = str(REPO / _p)
    if _ap not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _ap)

import etl_cleaning as etl  # noqa: E402
import pipeline_config as pc  # noqa: E402
from volatility_estimators import estimators_from_ohlcv  # noqa: E402

_LN2 = np.log(2.0)
_OHLC = ["open", "high", "low", "close"]
_RTOL = etl.OHLC_RTOL                 # 1e-5 -- same OHLC tolerance as the raw-data quality gate
_SPLIT = etl.SPLIT_THRESH             # |1-day return| split-jump threshold (0.50), reused from the ETL module
# volume-zscore backward-compat window: the delivered/paper result JSONs were trained on 20 (see
# pipeline_config.VOLUME_ZSCORE_WINDOW note). Kept as a named constant so reproducing them needs no code
# change; the CANONICAL window is pc.VOLUME_ZSCORE_WINDOW (22). config-ok: documented back-compat variant.
_VZ20 = 20                            # config-ok: back-compat volume window (delivered results used 20)

SCHEMA_VERSION = "enriched-1.1"

ENRICHED_COLUMNS = [
    "date", "open", "high", "low", "close", "volume",
    "parkinson_variance", "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20",
    "log_range", "daily_return", "har_daily", "har_weekly", "har_monthly", "market_pk",
    "volume_zscore_22", "volume_zscore_20", "dirty_flag", "cleaning_applied",
    "zero_range_flag", "zero_volume_flag",
]

# the 6 REAL/structural detectors composing dirty_flag (schema spec 2d)
DIRTY_CLASSES = ["high_lt_low", "open_close_outside", "nonpositive", "zero_range", "split_jump", "naninf"]

PRICE_DIRS = {
    "vn30": REPO / "data" / "raw" / "prices",
    "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
    "hose": REPO / "data" / "raw" / "prices" / "hose_vnstock",
    "hnx": REPO / "data" / "raw" / "prices" / "hnx_vnstock",
    "sp500": REPO / "data" / "raw" / "prices" / "sp500",
}
OUT_ROOT = REPO / "data" / "processed_enriched"


# --------------------------------------------------------------------------- detectors
def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def detect_dirty(df: pd.DataFrame) -> dict:
    """Per-row boolean detectors on the RAW bar. Returns a dict of the 6 structural masks + ``dirty``
    (their OR) + per-class counts. Independent of the cleaning step (schema spec: dirty_flag reflects the
    RAW bar tripping >=1 detector)."""
    o, h, lo, c = (_numeric(df, k) for k in _OHLC)
    v = _numeric(df, "volume")
    finite = np.isfinite([o, h, lo, c]).all(0)
    vbad = ("volume" in df.columns) & (~np.isfinite(v))
    naninf = (~finite) | vbad
    hi_oc, lo_oc = np.maximum(o, c), np.minimum(o, c)
    with np.errstate(invalid="ignore"):
        stack_min = np.min(np.vstack([o, h, lo, c]), axis=0)
        nonpositive = finite & (stack_min <= 0)
        high_lt_low = finite & (h < lo)
        oc_outside = finite & ((h < hi_oc * (1 - _RTOL)) | (lo > lo_oc * (1 + _RTOL)))
        zero_range = finite & (h == lo) & (h > 0)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev_c > 0, c / prev_c - 1.0, np.nan)
    split_jump = np.isfinite(ret) & (np.abs(ret) > _SPLIT)
    masks = {
        "high_lt_low": high_lt_low, "open_close_outside": oc_outside, "nonpositive": nonpositive,
        "zero_range": zero_range, "split_jump": split_jump, "naninf": naninf,
    }
    dirty = np.zeros(len(df), bool)
    for m in masks.values():
        dirty |= m
    counts = {k: int(v.sum()) for k, v in masks.items()}
    masks["dirty"] = dirty
    masks["_counts"] = counts
    return masks


# --------------------------------------------------------------------------- ETL cleaning
def _changed_dates(before: pd.DataFrame, after: pd.DataFrame) -> set:
    """Dates present in BOTH frames whose OHLC actually moved (observes a cleaner's per-row effect)."""
    m = before[["date"] + _OHLC].merge(after[["date"] + _OHLC], on="date", suffixes=("_b", "_a"))
    diff = np.zeros(len(m), bool)
    for k in _OHLC:
        diff |= ~np.isclose(m[k + "_a"].to_numpy(float), m[k + "_b"].to_numpy(float), equal_nan=True)
    return set(m.loc[diff, "date"])


def _dropped_dates(before: pd.DataFrame, after: pd.DataFrame) -> set:
    return set(before["date"]) - set(after["date"])


def clean_ohlcv(raw: pd.DataFrame) -> tuple:
    """Apply the ETL cleaners in the spec's priority order. Returns
    ``(cleaned_df, cleaning_applied[str array], rejections_df[date,reason])``.
    ``cleaning_applied`` records which rule modified each surviving bar; drops go to the manifest."""
    work = raw.copy()
    applied: dict = {d: "none" for d in work["date"]}
    rejections: list = []

    def _apply(fn, change_label, drop_reason):
        nonlocal work
        before = work
        work, _ = fn(work)
        if change_label is not None:
            for d in _changed_dates(before, work):
                applied[d] = change_label
        if drop_reason is not None:
            for d in _dropped_dates(before, work):
                rejections.append((d, drop_reason))

    _apply(etl.drop_naninf, None, "naninf")
    _apply(etl.reconstruct_nonpositive, "reconstruct_nonpositive", "nonpositive_unrecoverable")
    _apply(etl.swap_or_drop_high_low, "swap_high_low", "high_lt_low_unrecoverable")
    _apply(etl.cut_to_listing, None, "leading_backfill")
    _apply(etl.widen_range, "widen_range", None)

    # backadjust rescales ALL prior rows (a LEVEL rescale that is scale-invariant for Parkinson/GK/RS);
    # label ONLY the split-boundary day(s) actually adjusted, so the broad rescale does not clobber the
    # per-bar reconstruct/swap/widen labels on earlier rows.
    before = work
    work, info = etl.backadjust_splits(work)
    if info["n_adjusted"] > 0:
        c0 = before["close"].to_numpy(float)
        prev0 = np.concatenate([[np.nan], c0[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ret0 = np.where(prev0 > 0, c0 / prev0 - 1.0, np.nan)
        jump = np.isfinite(ret0) & (np.abs(ret0) > _SPLIT)
        for d in before.loc[jump, "date"]:
            applied[d] = "backadjust_split"

    work, _ = etl.flag_zero_range(work)
    work, _ = etl.flag_zero_volume(work)
    work = work.reset_index(drop=True)

    cleaning_applied = work["date"].map(applied).fillna("none").to_numpy()
    rej = pd.DataFrame(rejections, columns=["date", "reason"])
    return work, cleaning_applied, rej


# --------------------------------------------------------------------------- causal columns
def _volume_zscore(clean: pd.DataFrame, window: int) -> np.ndarray:
    """Trailing ``window``-day z-score of ``log1p(volume)`` (causal; ddof=1, min_periods=window)."""
    v = _numeric(clean, "volume")
    logv = pd.Series(np.log1p(v))
    mean = logv.rolling(window, min_periods=window).mean()
    std = logv.rolling(window, min_periods=window).std()
    return ((logv - mean) / std).to_numpy()


def _prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", kind="stable")
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def build_ticker(raw: pd.DataFrame) -> tuple:
    """Build the enriched per-ticker frame (``market_pk`` left NaN for the market pass) + rejection manifest."""
    n_raw = len(raw)                       # true raw row count (before parse/dedup) for honest report totals
    raw = _prepare_raw(raw)
    det = detect_dirty(raw)
    dirty_by_date = dict(zip(raw["date"], det["dirty"]))

    clean, cleaning_applied, rej = clean_ohlcv(raw)
    est = estimators_from_ohlcv(clean)

    o, h, lo, c = (_numeric(clean, k) for k in ("open", "high", "low", "close"))
    vol = _numeric(clean, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_range = np.log(h / lo)
        prev_c = np.concatenate([[np.nan], c[:-1]])
        daily_return = np.where(prev_c > 0, np.log(c / prev_c), np.nan)

    park = est["parkinson"].to_numpy(float)
    pk = pd.Series(park)
    har_weekly = pk.rolling(pc.HAR_WEEKLY_WINDOW, min_periods=pc.HAR_WEEKLY_WINDOW).mean().to_numpy()
    har_monthly = pk.rolling(pc.HAR_MONTHLY_WINDOW, min_periods=pc.HAR_MONTHLY_WINDOW).mean().to_numpy()

    dirty_col = clean["date"].map(dirty_by_date).fillna(False).to_numpy(bool)
    out = pd.DataFrame({
        "date": clean["date"].dt.strftime("%Y-%m-%d"),
        # POST-ETL CLEANED OHLCV the estimators were computed from (self-verifiable row).
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": vol,
        "parkinson_variance": park,
        "garman_klass_variance": est["garman_klass"].to_numpy(float),
        "rogers_satchell_variance": est["rogers_satchell"].to_numpy(float),
        "yang_zhang_n20": est["yang_zhang"].to_numpy(float),
        "log_range": log_range,
        "daily_return": daily_return,
        "har_daily": park,
        "har_weekly": har_weekly,
        "har_monthly": har_monthly,
        "market_pk": np.nan,
        "volume_zscore_22": _volume_zscore(clean, pc.VOLUME_ZSCORE_WINDOW),
        "volume_zscore_20": _volume_zscore(clean, _VZ20),
        "dirty_flag": dirty_col,
        "cleaning_applied": cleaning_applied,
        "zero_range_flag": clean["zero_range_flag"].to_numpy(bool),
        "zero_volume_flag": clean["zero_volume_flag"].to_numpy(bool),
    })
    counts = dict(det["_counts"])
    counts["_n_raw"] = n_raw
    return out, rej, counts


# --------------------------------------------------------------------------- market pass
def compute_market_pk(frames: dict) -> pd.Series:
    """Cross-sectional MEAN of ``parkinson_variance`` over the panel's VALID (non-NaN) tickers per date.
    Same-day, no future (schema spec 2c)."""
    cols = {tk: f.set_index("date")["parkinson_variance"] for tk, f in frames.items()}
    wide = pd.DataFrame(cols)
    return wide.mean(axis=1, skipna=True)   # NaN tickers excluded -> mean over valid only


def _read_ticker_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_ticker_from_path(path: Path) -> tuple:
    """Worker: read one raw csv and build its enriched frame. Returns ``(ticker, out_df, rej, counts)``."""
    path = Path(path)
    ticker = path.stem.replace("_ohlcv", "")
    out, rej, counts = build_ticker(_read_ticker_csv(path))
    return ticker, out, rej, counts


def build_market(market: str, price_dir: Path | None = None, out_root: Path | None = None,
                 map_fn=map, limit: int | None = None, write: bool = True,
                 regression_dir: Path | None = None) -> dict:
    """Build every ticker of ``market``, fill ``market_pk`` cross-sectionally, optionally write outputs.
    Returns a summary dict for the HTML report. ``map_fn`` lets the caller parallelise the per-ticker stage."""
    price_dir = Path(price_dir) if price_dir is not None else PRICE_DIRS[market]
    out_root = Path(out_root) if out_root is not None else OUT_ROOT
    files = sorted(price_dir.glob("*_ohlcv.csv"))
    if limit is not None:
        files = files[:limit]

    all_frames: dict = {}
    rejs: dict = {}
    dirty_counts = {k: 0 for k in DIRTY_CLASSES}
    rows_in = 0
    for ticker, out, rej, counts in map_fn(build_ticker_from_path, files):
        all_frames[ticker] = out
        rejs[ticker] = rej
        rows_in += counts["_n_raw"]                # honest raw count (incl. parse/dedup drops)
        for k in DIRTY_CLASSES:
            dirty_counts[k] += counts[k]

    # a ticker whose every bar was dropped (all naninf / unrecoverable) yields a 0-row frame; exclude it from
    # the panel + written data (a header-only CSV re-reads with object dtype and would fail the enriched
    # schema for a spurious dtype reason), but STILL write its rejection manifest so the audit trail is kept.
    frames = {tk: f for tk, f in all_frames.items() if len(f) > 0}
    n_empty = len(all_frames) - len(frames)

    market_pk = compute_market_pk(frames)
    for out in frames.values():
        out["market_pk"] = out["date"].map(market_pk).to_numpy()

    if write:
        out_dir = out_root / market
        out_dir.mkdir(parents=True, exist_ok=True)
        for ticker, out in frames.items():
            out.to_csv(out_dir / f"{ticker}.csv", index=False)
        for ticker, rej in rejs.items():
            if len(rej):
                rej.to_csv(out_dir / f"{ticker}_rejections.csv", index=False)
        (out_dir / "_schema_version.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "columns": ENRICHED_COLUMNS,
                        "market": market, "n_tickers": len(frames), "n_empty_tickers": n_empty},
                       indent=2), encoding="utf-8")

    summary = summarize_market(market, frames, rejs, dirty_counts, rows_in, market_pk)
    summary["n_empty_tickers"] = n_empty
    if regression_dir is not None:
        summary["regression"] = regression_vs_processed(frames, regression_dir)
    return summary


def regression_vs_processed(frames: dict, processed_dir: Path, cap: float = 0.1) -> dict:
    """Clean-bar regression: enriched ``parkinson_variance`` vs the existing ``data/processed`` value.
    Compares only NON-dirty bars whose existing value is NOT at the ``cap`` modeling floor -- the cap is a
    downstream QLIKE floor, not part of the causal estimator. Returns worst diff + capped count.

    ``cap`` defaults to 0.1 = the delivered ETL's known upper-clip on the Parkinson target (evidenced in the
    2026-08-31 ETL spec raw-vs-processed table); diagnostic-only threshold, never written to data."""
    processed_dir = Path(processed_dir)
    worst = 0.0
    n_capped = 0
    n_compared = 0
    for ticker, f in frames.items():
        pf = processed_dir / f"{ticker}_processed.csv"
        if not pf.exists():
            continue
        proc = pd.read_csv(pf)
        # normalize BOTH date columns to datetime before merge so a string-format mismatch cannot silently
        # collapse the overlap to zero (a false "perfect agreement"); enriched dates are ISO "%Y-%m-%d".
        f = f.assign(date=pd.to_datetime(f["date"], errors="coerce"))
        proc = proc.assign(date=pd.to_datetime(proc["date"], errors="coerce"))
        m = f[["date", "parkinson_variance", "dirty_flag"]].merge(
            proc[["date", "parkinson_variance"]], on="date", suffixes=("_enr", "_old"))
        capped = m["parkinson_variance_old"].to_numpy(float) >= cap * (1 - 1e-6)
        n_capped += int(capped.sum())
        ok = (~capped) & (~m["dirty_flag"].to_numpy(bool))
        diff = np.abs(m["parkinson_variance_enr"].to_numpy(float) - m["parkinson_variance_old"].to_numpy(float))
        d = diff[ok]
        d = d[np.isfinite(d)]
        n_compared += int(d.size)
        if d.size:
            worst = max(worst, float(d.max()))
    return {"worst_noncapped_diff": worst, "n_capped": n_capped, "n_compared": n_compared}


def summarize_market(market: str, frames: dict, rejs: dict, dirty_counts: dict,
                     rows_in: int, market_pk: pd.Series) -> dict:
    """Aggregate per-market stats for the build report."""
    rows_out = sum(len(f) for f in frames.values())
    n_dropped = sum(len(r) for r in rejs.values())
    clean_counts: dict = {}
    est_sums = {c: 0.0 for c in ("parkinson_variance", "garman_klass_variance",
                                 "rogers_satchell_variance", "yang_zhang_n20")}
    est_n = {c: 0 for c in est_sums}
    n_dirty = 0
    for f in frames.values():
        n_dirty += int(f["dirty_flag"].sum())
        for label, cnt in f["cleaning_applied"].value_counts().items():
            clean_counts[label] = clean_counts.get(label, 0) + int(cnt)
        for c in est_sums:
            v = pd.to_numeric(f[c], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            est_sums[c] += float(v.sum())
            est_n[c] += int(v.size)
    est_mean = {c: (est_sums[c] / est_n[c] if est_n[c] else float("nan")) for c in est_sums}
    mpk = market_pk.to_numpy(float)
    mpk = mpk[np.isfinite(mpk)]
    return {
        "market": market, "n_tickers": len(frames), "rows_in": rows_in, "rows_out": rows_out,
        "n_dropped": n_dropped, "n_dirty_bars": n_dirty, "dirty_by_class": dirty_counts,
        "cleaning_applied": clean_counts, "estimator_mean": est_mean,
        "market_pk": {"n_days": int(mpk.size), "min": float(mpk.min()) if mpk.size else float("nan"),
                      "mean": float(mpk.mean()) if mpk.size else float("nan"),
                      "max": float(mpk.max()) if mpk.size else float("nan")},
    }
