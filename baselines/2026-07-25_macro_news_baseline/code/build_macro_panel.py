"""Build a market-wide (date-only, no ticker) macro news embedding panel.

Aggregates EVERY discovered source's articles (whether or not they mention a VN30 ticker) by
effective trading date, using the raw PhoBERT embeddings already cached by
`2026-07-25_expand_news_cache_baseline`'s `--include_all` run. Two streaming passes over the
data (never holds all ~7.5M embeddings in memory at once):

  Pass 1 (PCA fit): pool a bounded sample of pre-TRAIN_CUTOFF rows (across all sources) and fit
                    PCA(32). Honest fallback to raw 768-dim if too few pre-cutoff rows exist
                    (same pattern as the sibling baseline's `news_embeddings._reduce`).
  Pass 2 (aggregate): re-read each source, transform via the fitted PCA, accumulate a running
                    (sum, count) per trading date -> mean-pooled macro embedding per date.

Then EWMA(30d half-life) smooths the resulting (small, ~4890-row) date-indexed series.

Isolated (CLAUDE.md §3.F): read-only imports from
`2026-07-25_dual_group_news_embedding_baseline` (discovery, cache path, trading calendar, EWMA
helper) — no file there is modified.

Run: python build_macro_panel.py
Output: ../../../data/features/macro_news_panel.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from vendor_data_eda.discover_news import discover_source_files, load_source  # noqa: E402
from vendor_data_eda.news_embeddings import _article_cache_path, RAW_DIM, PCA_DIM, TRAIN_CUTOFF  # noqa: E402
from vendor_data_eda.phase04_news_helpers import effective_trading_date, _trading_calendar  # noqa: E402
from vendor_data_eda.dual_news_features import _ewma_on_series  # noqa: E402

MAX_PCA_FIT_ROWS = 300_000  # bounds Pass-1 memory regardless of how much pre-cutoff data exists
OUT_PATH = _ROOT / "data" / "features" / "macro_news_panel.parquet"


def _load_dated_cached_embeddings(source: str, path: Path) -> pd.DataFrame:
    """(url, eff_date, raw_0..RAW_DIM-1) for one source's articles that ARE in its cache.

    Re-reads the source CSV (for pub_date) and joins with the already-built raw_cache parquet
    (url, raw_*) on url. Returns empty frame if no cache file exists for this source or nothing
    joins."""
    cache_path = _article_cache_path(source)
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame()
    raw_cols = [c for c in cached.columns if c.startswith("raw_")]
    if len(raw_cols) != RAW_DIM or cached.empty:
        return pd.DataFrame()

    df = load_source(source, path)
    if "url" not in df.columns or "pub_date" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["url"]).drop_duplicates(subset="url")[["url", "pub_date"]]

    merged = df.merge(cached, on="url", how="inner")
    if merged.empty:
        return pd.DataFrame()

    trading = _trading_calendar()
    merged["eff_date"] = effective_trading_date(merged["pub_date"], trading)
    merged = merged.dropna(subset=["eff_date"])
    return merged[["url", "eff_date"] + raw_cols]


def _fit_pca(sources: dict[str, Path]):
    """Pass 1: pool up to MAX_PCA_FIT_ROWS pre-TRAIN_CUTOFF rows across all sources, fit PCA.

    Each source's contribution is capped at MAX_PCA_FIT_ROWS // len(sources) (random subsample
    within the source if it has more), so the pooled sample isn't dominated by whichever sources
    happen to sort alphabetically first — every source with pre-cutoff data gets a chance to
    contribute, not just the first ones processed until the budget runs out.

    Returns (pca_or_None, n_train_rows). pca is None if too few rows to fit (honest fallback —
    caller keeps the full RAW_DIM embedding instead of a mislabeled PCA)."""
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    per_source_cap = max(1, MAX_PCA_FIT_ROWS // max(1, len(sources)))
    rng = np.random.default_rng(0)
    pool = []
    n_pooled = 0
    for source, path in sorted(sources.items()):
        dated = _load_dated_cached_embeddings(source, path)
        if dated.empty:
            continue
        pre_cutoff = dated[dated["eff_date"] < cutoff]
        if pre_cutoff.empty:
            continue
        raw_cols = [c for c in dated.columns if c.startswith("raw_")]
        vals = pre_cutoff[raw_cols].to_numpy()
        if len(vals) > per_source_cap:
            sel = rng.choice(len(vals), size=per_source_cap, replace=False)
            vals = vals[sel]
        pool.append(vals)
        n_pooled += len(vals)

    if not pool or n_pooled < 2:
        return None, n_pooled
    pooled = np.concatenate(pool, axis=0)
    from sklearn.decomposition import PCA
    dim = min(PCA_DIM, pooled.shape[1], max(1, len(pooled) - 1))
    pca = PCA(n_components=dim, svd_solver="randomized").fit(pooled)
    return pca, n_pooled


def _aggregate_by_date(sources: dict[str, Path], pca, out_dim: int) -> pd.DataFrame:
    """Pass 2: re-read each source, transform via `pca` (or keep raw if pca is None), accumulate
    running (sum, count) per trading date. Returns (date, macro_emb_0..out_dim-1)."""
    trading = pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().sort_values().unique()
    date_index = {d: i for i, d in enumerate(trading)}
    sums = np.zeros((len(trading), out_dim), dtype=np.float64)
    counts = np.zeros(len(trading), dtype=np.int64)

    for source, path in sorted(sources.items()):
        dated = _load_dated_cached_embeddings(source, path)
        if dated.empty:
            continue
        raw_cols = [c for c in dated.columns if c.startswith("raw_")]
        embs = dated[raw_cols].to_numpy()
        reduced = pca.transform(embs) if pca is not None else embs
        eff_dates = dated["eff_date"].to_numpy()
        # vectorized scatter-add — np.add.at (not `sums[idx] += reduced`, which silently drops
        # all-but-one contribution when the same date index repeats within this source's batch)
        idx = np.array([date_index.get(d, -1) for d in eff_dates])
        valid = idx >= 0
        if valid.any():
            np.add.at(sums, idx[valid], reduced[valid])
            np.add.at(counts, idx[valid], 1)

    mean_emb = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    cols = {f"macro_emb_{i}": mean_emb[:, i] for i in range(out_dim)}
    out = pd.DataFrame({"date": trading, **cols})
    out["macro_emb_norm"] = np.linalg.norm(mean_emb, axis=1)
    out.loc[counts == 0, "macro_emb_norm"] = np.nan
    return out


def build_macro_panel() -> pd.DataFrame:
    sources = discover_source_files()
    print(f"[macro_panel] {len(sources)} discovered sources", flush=True)

    t0 = time.time()
    pca, n_train = _fit_pca(sources)
    out_dim = pca.n_components_ if pca is not None else RAW_DIM
    print(f"[macro_panel] PCA fit on {n_train} pre-{TRAIN_CUTOFF} rows -> "
          f"{'PCA dim=' + str(out_dim) if pca is not None else 'FALLBACK: keeping full raw dim=' + str(out_dim)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    daily = _aggregate_by_date(sources, pca, out_dim)
    n_covered = int((daily["macro_emb_norm"].notna()).sum())
    print(f"[macro_panel] aggregated {len(daily)} trading dates, {n_covered} have >=1 article "
          f"({100.0 * n_covered / max(1, len(daily)):.2f}%) ({time.time() - t0:.1f}s)", flush=True)

    emb_cols = [c for c in daily.columns if c.startswith("macro_emb_") and c != "macro_emb_norm"]
    daily = daily.sort_values("date").reset_index(drop=True)
    ewma_cols = {}
    for c in emb_cols + ["macro_emb_norm"]:
        ewma_cols[f"ewma_{c}"] = _ewma_on_series(daily[c], halflife=30.0)
    daily = pd.concat([daily, pd.DataFrame(ewma_cols)], axis=1)
    return daily


def main():
    panel = build_macro_panel()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[done] {OUT_PATH} -> {panel.shape[0]} rows, {panel.shape[1]} cols")


if __name__ == "__main__":
    main()
