"""Dual-group embedding features (khach_quan + tong_hop) + emb_norm, aggregated per
(ticker, effective_trading_date).

Vendored (trimmed) from C:\\luanvan\\data_eda\\src\\modeling\\features.py (2026-07-25) — copy
only, data_eda itself is not modified. Dropped: the `PURE_NEWS_FEATURES` re-export (from
`src.modeling.pure_news`, unused by any function in this file — a re-export only) and `run()`
(data_eda's own CLI entrypoint writing to `eda_output/modeling/`; this baseline's own
`build_dual_group_panel.py` is the entrypoint instead, writing under
`data/features/dual_group_news_panel.parquet`).

Per-(ticker, date) features:
- ``kq_emb_0``..``kq_emb_31`` — mean-pooled PhoBERT embedding (PCA-reduced) from "khach_quan"
- ``th_emb_0``..``th_emb_31`` — same from "tong_hop"
- ``kq_emb_norm`` / ``th_emb_norm`` — L2 norm of the 32-dim mean-pooled embedding vector
- ``kq_topic_<category>_count`` / ``th_topic_<category>_count`` — event-type counts per group
- ``emb_0``..``emb_31`` — legacy tong_hop-only features (backward compat)
- ``ewma_kq_emb_*`` / ``ewma_th_emb_*`` (+ ``_norm``) — single 30-day-halflife EWMA (mode="ewma"/"full")
- multi-window EWMA / novelty / dispersion / max-shock (mode="full" only; NOT used by this
  baseline's `build_dual_group_panel.py`, which calls mode="ewma" — kept for a possible
  follow-up baseline per requirements.md §3)

NaN where a (ticker, date) has no news.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vendor_config import EDA_TICKERS
from vendor_data_eda.news_embeddings import (
    PCA_DIM,
    build_comparable_group_embeddings,
    build_group_embeddings,
)
from vendor_data_eda.phase04_news_helpers import TOPIC_CATEGORIES, _trading_calendar

GROUPS = ["khach_quan", "tong_hop"]
GROUP_PREFIXES = {"khach_quan": "kq", "tong_hop": "th"}

EMB_FEATURES = [f"emb_{i}" for i in range(PCA_DIM)]
TOPIC_FEATURES = [f"topic_{cat}_count" for cat in TOPIC_CATEGORIES]

# Legacy tong_hop-only feature list (backward compat)
ADV_FEATURES = EMB_FEATURES + TOPIC_FEATURES

# Dual-group features
_ADV_FEATURES_DUAL_RAW = []
for grp in GROUPS:
    px = GROUP_PREFIXES[grp]
    _ADV_FEATURES_DUAL_RAW.extend([f"{px}_emb_{i}" for i in range(PCA_DIM)])
    _ADV_FEATURES_DUAL_RAW.append(f"{px}_emb_norm")
    _ADV_FEATURES_DUAL_RAW.extend([f"{px}_{c}" for c in TOPIC_FEATURES])
ADV_FEATURES_DUAL = _ADV_FEATURES_DUAL_RAW

# EWMA features — single 30-day window
EWMA_FEATURES = []
for grp in GROUPS:
    px = GROUP_PREFIXES[grp]
    EWMA_FEATURES.extend([f"ewma_{px}_emb_{i}" for i in range(PCA_DIM)])
    EWMA_FEATURES.append(f"ewma_{px}_emb_norm")

# Multi-window EWMA features (mode="full" only) — 5,10,20,30,60 day half-lives
EWMA_WINDOWS = [5, 10, 20, 30, 60]
EWMA_MULTI_FEATURES = []
for grp in GROUPS:
    px = GROUP_PREFIXES[grp]
    for hl in EWMA_WINDOWS:
        EWMA_MULTI_FEATURES.extend([f"ewma{hl}_{px}_emb_{i}" for i in range(PCA_DIM)])
        EWMA_MULTI_FEATURES.append(f"ewma{hl}_{px}_emb_norm")

# Novelty and dispersion features (mode="full" only)
NOVELTY_FEATURES = [f"{grp}_novelty_30d" for grp in ["kq", "th"]]
DISPERSION_FEATURES = [f"{grp}_dispersion" for grp in ["kq", "th"]]
MAX_SHOCK_FEATURES = [f"{grp}_max_semantic_shock" for grp in ["kq", "th"]]

ADV_FEATURES_DUAL_FULL = (
    ADV_FEATURES_DUAL + EWMA_FEATURES + EWMA_MULTI_FEATURES
    + NOVELTY_FEATURES + DISPERSION_FEATURES + MAX_SHOCK_FEATURES
)


def aggregate_articles(rows: pd.DataFrame, emb_cols: list[str]) -> dict:
    """Aggregate one (ticker, date)'s article rows -> feature dict (mean embedding + topic
    counts + emb_norm)."""
    if rows.empty:
        return {}
    out = {}
    for c in emb_cols:
        if c in rows.columns:
            vals = rows[c].astype(float)
            out[c] = float(vals.mean())
        else:
            out[c] = np.nan
    if emb_cols:
        emb_vals = np.array([out[c] for c in emb_cols if c in out and not np.isnan(out.get(c, np.nan))])
        out["emb_norm"] = float(np.sqrt(np.sum(emb_vals ** 2))) if len(emb_vals) > 0 else np.nan
    for cat in TOPIC_CATEGORIES:
        col = f"topic_{cat}_count"
        out[col] = int(rows[col].sum()) if col in rows.columns else 0
    return out


def _aggregate_group(group: str, shared_emb: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Per-(ticker, date) aggregated features for one group, reindexed to full trading calendar.

    If ``shared_emb`` is given (from ``build_comparable_group_embeddings``), uses the
    shared-PCA-basis result instead of calling ``build_group_embeddings`` (own per-group PCA).
    """
    if shared_emb is not None and group in shared_emb:
        emb = shared_emb[group]
    else:
        emb = build_group_embeddings(group)
    if emb.empty:
        return pd.DataFrame()
    arts = emb.rename(columns={"date": "eff_date"}).dropna(subset=["eff_date"])
    td = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().sort_values()
    )
    px = GROUP_PREFIXES[group]
    emb_cols = [c for c in arts.columns if c.startswith("emb_") and c[4:].isdigit()]

    records = []
    eff_norm = arts["eff_date"].dt.normalize()
    for (ticker, date), rows_grp in arts.groupby([arts["ticker"], eff_norm]):
        drop_cols = [c for c in ["ticker", "eff_date", "source", "pca_applied"] if c in rows_grp.columns]
        agg = aggregate_articles(rows_grp.drop(columns=drop_cols, errors="ignore"), emb_cols)
        if not agg:
            continue
        rec = {"ticker": ticker, "date": date}
        for k, v in agg.items():
            if k.startswith("emb_"):
                rec[f"{px}_{k}"] = v
            elif k == "emb_norm":
                rec[f"{px}_emb_norm"] = v
            elif k.startswith("topic_"):
                rec[f"{px}_{k}"] = v
            else:
                rec[k] = v
        records.append(rec)

    agg_df = pd.DataFrame(records)
    if agg_df.empty:
        return agg_df

    out_frames = []
    for ticker in EDA_TICKERS:
        sub = agg_df[agg_df["ticker"] == ticker]
        cols = [c for c in agg_df.columns if c not in ("ticker", "date")]
        if sub.empty:
            df = pd.DataFrame(np.nan, index=td, columns=cols)
        else:
            sub = sub.copy()
            sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
            df = sub.set_index("date").reindex(td)[cols]
        df.insert(0, "ticker", ticker)
        df.index.name = "date"
        out_frames.append(df.reset_index())
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def _legacy_tong_hop_features() -> pd.DataFrame:
    """Build legacy tong_hop-only ADV_FEATURES for backward compat with 'price+news_adv'."""
    emb = build_group_embeddings("tong_hop")
    if emb.empty:
        return pd.DataFrame()
    arts = emb.rename(columns={"date": "eff_date"}).dropna(subset=["eff_date"])
    td = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().sort_values()
    )
    emb_cols = [c for c in arts.columns if c.startswith("emb_") and c[4:].isdigit()]

    records = []
    eff_norm = arts["eff_date"].dt.normalize()
    for (ticker, date), rows_grp in arts.groupby([arts["ticker"], eff_norm]):
        drop_cols = [c for c in ["ticker", "eff_date", "source", "pca_applied"] if c in rows_grp.columns]
        agg = aggregate_articles(rows_grp.drop(columns=drop_cols, errors="ignore"), emb_cols)
        if not agg:
            continue
        rec = {"ticker": ticker, "date": date}
        for k, v in agg.items():
            if k in ADV_FEATURES:
                rec[k] = v
        records.append(rec)

    agg_df = pd.DataFrame(records)
    if agg_df.empty:
        return agg_df

    out_frames = []
    for ticker in EDA_TICKERS:
        sub = agg_df[agg_df["ticker"] == ticker]
        cols = ADV_FEATURES
        if sub.empty:
            df = pd.DataFrame(np.nan, index=td, columns=cols)
        else:
            sub = sub.copy()
            sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
            df = sub.set_index("date").reindex(td)[cols]
        df.insert(0, "ticker", ticker)
        df.index.name = "date"
        out_frames.append(df.reset_index())
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def _ewma_on_series(series: pd.Series, halflife: float) -> pd.Series:
    """EWMA with (1-alpha) decay on NaN gaps, matching half-life in trading days."""
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    result = series.astype(float).copy()
    ema = np.nan
    for i in range(len(result)):
        val = result.iloc[i]
        if np.isnan(val):
            if not np.isnan(ema):
                ema = (1.0 - alpha) * ema
                result.iloc[i] = ema
        else:
            if np.isnan(ema):
                ema = val
            else:
                ema = alpha * val + (1.0 - alpha) * ema
            result.iloc[i] = ema
    return result


def ewma_embedding_features(panel_per_group: dict[str, pd.DataFrame], halflife: float = 30) -> pd.DataFrame:
    """Add EWMA-smoothed embedding columns per group.

    Input: dict of {group_name: per-(ticker,date) agg_df} where each agg_df contains
    ``{px}_emb_0..{PCA_DIM-1}`` + ``{px}_emb_norm`` columns.
    Returns a single DataFrame with ``ewma_{px}_emb_*`` + ``ewma_{px}_emb_norm``.
    """
    ewma_frames = []
    for group, df in panel_per_group.items():
        if df.empty:
            continue
        px = GROUP_PREFIXES[group]
        emb_cols = [f"{px}_emb_{i}" for i in range(PCA_DIM)]
        norm_col = f"{px}_emb_norm"
        available = [c for c in emb_cols + [norm_col] if c in df.columns]
        if not available:
            continue
        sub = df[["ticker", "date"] + available].copy().sort_values(["ticker", "date"])
        ewma_df = sub[["ticker", "date"]].copy()
        for col in available:
            ewma_df[f"ewma_{col}"] = sub.groupby("ticker")[col].transform(
                lambda s: _ewma_on_series(s, halflife)
            )
        ewma_frames.append(ewma_df.set_index(["ticker", "date"]))
    if not ewma_frames:
        return pd.DataFrame()
    merged = pd.concat(ewma_frames, axis=1).reset_index()
    merged.columns = [str(c) for c in merged.columns]
    return merged


def _multi_ewma_features(panel_per_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Multi-window EWMA: 5,10,20,30,60 day half-lives (mode="full" only)."""
    frames = []
    for hl in EWMA_WINDOWS:
        ew = ewma_embedding_features(panel_per_group, halflife=float(hl))
        if not ew.empty:
            rename_map = {c: c.replace("ewma_", f"ewma{hl}_") for c in ew.columns if c not in ("ticker", "date")}
            frames.append(ew.rename(columns=rename_map))
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        keep_cols = [c for c in frame.columns if c not in ("ticker", "date")]
        merged = merged.merge(frame[keep_cols + ["ticker", "date"]], on=["ticker", "date"], how="outer")
    return merged


def _novelty_features(panel_per_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """News novelty: how different is today's centroid from EWMA history (mode="full" only).

    novelty_t = 1 - cos(C_t, EWMA_30(C_{t-1}))
    """
    records = []
    for group, df in panel_per_group.items():
        if df.empty:
            continue
        px = GROUP_PREFIXES[group]
        emb_cols = [f"{px}_emb_{i}" for i in range(PCA_DIM)]
        available = [c for c in emb_cols if c in df.columns]
        if not available:
            continue
        sub = df[["ticker", "date"] + available].copy().sort_values(["ticker", "date"])
        for ticker in sub["ticker"].unique():
            tsub = sub[sub["ticker"] == ticker].copy()
            if len(tsub) < 2:
                continue
            emb_vals = tsub[available].values.astype(float)
            emb_vals[np.isnan(emb_vals)] = 0.0
            alpha = 1.0 - np.exp(-np.log(2) / 30.0)
            ewma_hist = emb_vals[0].copy()
            records.append({"ticker": ticker, "date": tsub.iloc[0]["date"], f"{px}_novelty_30d": 1.0})
            for i in range(1, len(tsub)):
                row = emb_vals[i]
                norm_c = np.linalg.norm(row) if np.linalg.norm(row) > 0 else 1e-10
                norm_e = np.linalg.norm(ewma_hist) if np.linalg.norm(ewma_hist) > 0 else 1e-10
                cos_sim = float(np.dot(row, ewma_hist) / (norm_c * norm_e))
                novelty = 1.0 - cos_sim
                records.append({"ticker": ticker, "date": tsub.iloc[i]["date"], f"{px}_novelty_30d": novelty})
                if np.linalg.norm(row) > 0:
                    ewma_hist = alpha * row + (1.0 - alpha) * ewma_hist
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _dispersion_features(panel_per_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Semantic dispersion: average cosine distance from articles to their daily centroid
    (mode="full" only). Higher = articles disagree on semantic direction."""
    records = []
    for group, orig_df in panel_per_group.items():
        px = GROUP_PREFIXES[group]
        emb_cols = [f"{px}_emb_{i}" for i in range(PCA_DIM)]
        disp_col = f"{px}_dispersion"
        shock_col = f"{px}_max_semantic_shock"

        available = [c for c in emb_cols if c in orig_df.columns]
        if not available:
            continue

        sub = orig_df[["ticker", "date"] + available].copy().sort_values(["ticker", "date"]).dropna(subset=available, how="all")
        if sub.empty:
            continue
        for (ticker, date), grp in sub.groupby(["ticker", "date"]):
            vals = grp[available].values.astype(float)
            vals = vals[~np.isnan(vals).any(axis=1)]
            if len(vals) < 2:
                records.append({"ticker": ticker, "date": date, disp_col: 0.0, shock_col: 0.0})
                continue
            centroid = vals.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                records.append({"ticker": ticker, "date": date, disp_col: 0.0, shock_col: 0.0})
                continue
            distances = []
            for v in vals:
                v_norm = np.linalg.norm(v)
                cos_sim = np.dot(v, centroid) / (c_norm * v_norm) if v_norm > 0 else 0
                distances.append(1.0 - cos_sim)
            records.append({
                "ticker": ticker, "date": date,
                disp_col: float(np.mean(distances)),
                shock_col: float(np.max(distances)),
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def build_advanced_features(mode: str = "ewma") -> pd.DataFrame:
    """Build the dual-group advanced-news panel (ticker, date) x various feature sets.

    Uses a single shared PCA basis for both groups (fit on pooled train-period rows) so
    kq_emb_i and th_emb_i live in the same subspace. Merges by (ticker, date). NaN where no news.
    Also includes legacy tong_hop-only columns for backward compat.

    ``mode``:
    - ``"basic"``: ADV_FEATURES_DUAL + legacy
    - ``"ewma"``: basic + single-window EWMA  (this baseline's default — see requirements.md §3)
    - ``"full"``: basic + EWMA + multi-EWMA + novelty + dispersion + shock
    """
    shared = build_comparable_group_embeddings()
    kq_df = _aggregate_group("khach_quan", shared_emb=shared)
    th_df = _aggregate_group("tong_hop", shared_emb=shared)
    # [FIX 2026-07-25] `_legacy_tong_hop_features()` fits its OWN (non-shared, tong_hop-only)
    # PCA, which after the TRAIN_CUTOFF leakage fix (2020-01-01 -> 2010-06-30, see
    # news_embeddings.py) can have too few tong_hop-only pre-cutoff rows to reach the full
    # PCA_DIM=32 components (`_reduce()`'s honest-fallback `d = min(dim, ..., n_train-1)` then
    # yields <32 emb_* columns, and ADV_FEATURES' fixed 32-column list raises KeyError). This
    # baseline never uses `leg`'s output anyway (build_dual_group_panel.py's KEEP_COLS filter
    # already drops these legacy columns, per requirements.md §3's Simplicity Gate) — skip
    # computing it rather than crashing on values nothing downstream consumes.
    panel_per_group = {"khach_quan": kq_df, "tong_hop": th_df}

    frames = []
    if not kq_df.empty:
        frames.append(kq_df.set_index(["ticker", "date"]))
    if not th_df.empty:
        frames.append(th_df.set_index(["ticker", "date"]))

    if mode in ("ewma", "full"):
        ewma_df = ewma_embedding_features(panel_per_group)
        if not ewma_df.empty:
            frames.append(ewma_df.set_index(["ticker", "date"]))

    if mode == "full":
        multi_ewma = _multi_ewma_features(panel_per_group)
        if not multi_ewma.empty:
            frames.append(multi_ewma.set_index(["ticker", "date"]))
        novelty_df = _novelty_features(panel_per_group)
        if not novelty_df.empty:
            frames.append(novelty_df.set_index(["ticker", "date"]))
        disp_df = _dispersion_features(panel_per_group)
        if not disp_df.empty:
            frames.append(disp_df.set_index(["ticker", "date"]))

    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, left_index=True, right_index=True, how="outer")
    merged = merged.reset_index()
    merged.columns = [str(c) for c in merged.columns]
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged
