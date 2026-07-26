"""Two mutually-exclusive news-source groups + PhoBERT-embedding aggregation (PCA, EWMA,
novelty, dispersion).

Vendored (trimmed) from C:\\luanvan\\data_eda\\src\\features\\news_embeddings.py (2026-07-25) —
copy only, data_eda itself is not modified. Dropped: `run()` (the data_eda CLI entrypoint that
populates per-source caches + writes to `reports/news_processing_log.md`) — this baseline's own
`build_dual_group_panel.py` is the entrypoint instead, and the per-source caches are already
populated (copied verbatim in Story 1.1, not rebuilt here).

- "khach_quan" (objective/factual reporting): mainstream press portals.
- "tong_hop" (aggregated/analyst commentary): securities firms' own research/market-wrap content.

Sources are DISCOVERED dynamically (``vendor_data_eda.discover_news.discover_source_files``).
PhoBERT [CLS] encoding is NEVER run by this module — ``_get_article_embeddings`` is a cache-ONLY
lookup keyed by ``url`` (data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet,
copied from data_eda); articles whose url isn't in that copied cache are skipped (user decision
2026-07-25 — see ``_get_article_embeddings`` docstring), not encoded.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from vendor_config import EDA_TICKERS as VN30_TICKERS
from vendor_config import FEATURES_DIR
from vendor_data_eda.discover_news import discover_source_files, load_source
from vendor_data_eda.phase04_news_helpers import TOPIC_CATEGORIES, _trading_calendar, effective_trading_date


def topic_flags(text: str) -> dict[str, int]:
    """1/0 flag per EDA-Guide category for one article (keyword match)."""
    t = str(text).lower()
    return {f"topic_{cat}_count": int(any(kw in t for kw in kws)) for cat, kws in TOPIC_CATEGORIES.items()}


TRAIN_CUTOFF = "2010-06-30"  # [FIX 2026-07-25, code review] data_eda's original value
# ("2020-01-01") assumed ONE global calendar split date shared by every ticker. This project's
# actual split (`src.lstm_gat_hybrid.dataset_with_graph_method._split_raw_data_by_date`) instead
# cuts each ticker at the SAME ROW INDEX (train_ratio=0.7 of the shortest common ticker's
# length), so each ticker's own val/test window falls on a DIFFERENT, ticker-specific calendar
# date — e.g. STB/VNM's val window starts 2010-06-30, vs. SSB's ~2024-11-11. Under the old
# "2020-01-01" cutoff, ~19 of the 30 VN30 tickers had their OWN val/test-period news rows fall
# BEFORE 2020-01-01 and therefore leaked into the "train" PCA fit — a real train/test leakage
# bug (CLAUDE.md §3.A: chronological split is CRITICAL), caught by a self code-review after the
# first training run. 2010-06-30 is the EARLIEST such per-ticker val-start date across all 30
# tickers (verified by walking each {TICKER}_processed.csv through the same split arithmetic) —
# using it as the single global PCA cutoff guarantees no ticker's val/test rows are ever
# included, at the cost of a much smaller PCA training set (~4 years of news vs ~14).
PCA_DIM = 32
RAW_DIM = 768
KHACH_QUAN_SOURCES = {
    "cafef", "hsc", "vnexpress", "thanhnien", "tuoitre", "nld", "vietnamplus",
    "thanhnien_root", "thanhnien_objective",
    "tuoitre_root", "tuoitre_objective",
    "vietnamplus_root", "vietnamplus_objective",
    "baodautu", "cafebiz", "coin68", "fica", "nhadautu",
    "nhipsongkinhdoanh", "theinvestor", "thoibaotaichinhvietnam",
    "thuonghieucongluan", "tinnhanhchungkhoan", "vietbao",
    "vietnambiz", "vietnamfinance", "vietnamnet", "vneconomy",
    "vnexpress_root", "vnexpress_objective",
    # [2026-07-25] Added after crawl_data grew (see
    # baselines/2026-07-25_expand_news_cache_baseline) — 12 newly-discovered mainstream press
    # sources now have a populated raw_cache entry (bnews has 0 ticker-mentioning articles, so
    # it contributes nothing either way, but is listed for completeness/traceability).
    "baophapluat", "bnews", "cand", "dantri", "giaoducthoidai", "hanoimoi",
    "plo", "sggp", "tapchicongthuong", "tienphong", "viettimes", "vov",
}
TONG_HOP_SOURCES = {
    "ssi", "vndirect", "vnstock", "vietstock", "vsdc",
    "forum",
    "telegram_chungkhoanf0", "telegram_chungkhoantangtruong",
    "telegram_chungkhoanvietnam2026", "telegram_chungkhoanvietnammoon",
    "telegram_financialstreetvn", "telegram_kakatachannel",
    "telegram_longshortlientuc", "telegram_vnwallstreet",
}
GROUP_SOURCES = {"khach_quan": KHACH_QUAN_SOURCES, "tong_hop": TONG_HOP_SOURCES}
TICKER_PATTERN = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b", re.IGNORECASE)
TOPIC_COLS = [f"topic_{c}_count" for c in TOPIC_CATEGORIES]
_EMB_RAW_COL_RE = re.compile(r"^raw_\d+$")


def _raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _EMB_RAW_COL_RE.match(c)]


def unclassified_sources() -> set[str]:
    """Discovered sources that aren't in either group's classification."""
    return set(discover_source_files()) - KHACH_QUAN_SOURCES - TONG_HOP_SOURCES


def _load_group(group: str) -> pd.DataFrame:
    """Raw article rows for one group, with content + source columns."""
    if group not in GROUP_SOURCES:
        raise ValueError(f"unknown group: {group}")
    wanted = GROUP_SOURCES[group]
    files = discover_source_files()
    frames = []
    for source, path in files.items():
        if source not in wanted:
            continue
        try:
            frames.append(load_source(source, path))
        except Exception:
            continue
    news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if news.empty:
        return news
    title = news.get("title", pd.Series(index=news.index)).fillna("")
    lead = news.get("lead", pd.Series(index=news.index)).fillna("")
    news["_text"] = (title.astype(str) + " " + lead.astype(str)).str.strip()
    news = news[news["_text"].str.len() > 0].reset_index(drop=True)
    if "url" not in news.columns:
        raise ValueError(f"group={group!r} news frame has no 'url' column; cannot cache incrementally")
    news = news.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)
    # PhoBERT encoding is the expensive step; `_explode_tickers` downstream discards every
    # article that never mentions a VN30 ticker anyway — filtering BEFORE encoding changes zero
    # output rows but avoids encoding text nobody will use.
    return news[news["_text"].str.contains(TICKER_PATTERN, regex=True, na=False)].reset_index(drop=True)


def _article_cache_path(source: str):
    """PER-SOURCE cache — keyed by url, one file per news source."""
    return FEATURES_DIR / f"news_emb_articles_{source}.parquet"


def _get_article_embeddings(source: str, news: pd.DataFrame) -> pd.DataFrame:
    """Cache-ONLY lookup for ONE source, keyed by ``url`` — deliberately DIFFERENT from
    data_eda's own incremental-encode behavior (which calls PhoBERT on any uncached url).

    User decision 2026-07-25: articles whose url isn't in the copied cache (crawl_data has grown
    past data_eda's 2026-07-24 21:49 cache snapshot — 316 such articles found across all 30
    sources at build time) are SKIPPED, never encoded — this baseline must never invoke PhoBERT.
    A corrupted/truncated cache file is treated as empty (all of that source's articles skipped,
    logged) rather than crashing every future run."""
    cache_path = _article_cache_path(source)
    cached = pd.DataFrame({"url": []})
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
        except Exception:
            cached = pd.DataFrame({"url": []})

    cached_raw_cols = _raw_cols(cached)
    if not cached.empty and cached_raw_cols and len(cached_raw_cols) != RAW_DIM:
        cached = pd.DataFrame({"url": []})

    known = set(cached["url"]) if not cached.empty else set()
    n_skipped = int((~news["url"].isin(known)).sum())
    if n_skipped:
        print(f"  [cache-only] {source}: skipping {n_skipped} article(s) not in the copied "
              f"cache (no PhoBERT call)", flush=True)
    return cached


def _explode_tickers(news: pd.DataFrame) -> pd.DataFrame:
    """One row per (article x mentioned ticker), with eff_date + source + raw embedding cols."""
    trading = _trading_calendar()
    eff_date = effective_trading_date(news["pub_date"], trading)
    tickers = [TICKER_PATTERN.findall(t) for t in news["_text"]]
    raw_cols = _raw_cols(news)
    records = []
    for i, tks in enumerate(tickers):
        if not tks or pd.isna(eff_date.iloc[i]):
            continue
        flags = topic_flags(news["_text"].iloc[i])
        row_raw = {c: news[c].iloc[i] for c in raw_cols}
        for t in {tk.upper() for tk in tks}:
            records.append({
                "ticker": t, "date": eff_date.iloc[i], "source": news["source"].iloc[i],
                **row_raw, **flags,
            })
    return pd.DataFrame(records)


def _build_raw(group: str) -> pd.DataFrame:
    """(date, ticker, source, raw_0..767, topic_*) — embeddings from each source's own
    incremental cache (merged per source, then combined for the group)."""
    news = _load_group(group)
    if news.empty:
        return pd.DataFrame()
    frames = []
    for source, sub_news in news.groupby("source"):
        article_embs = _get_article_embeddings(source, sub_news)
        if article_embs.empty:
            continue
        m = sub_news.merge(article_embs, on="url", how="inner")
        if not m.empty:
            frames.append(m)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        return pd.DataFrame()
    exploded = _explode_tickers(merged)
    if exploded.empty:
        return pd.DataFrame()
    exploded["date"] = exploded["date"].dt.normalize()
    return exploded


def load_or_build_raw(group: str) -> pd.DataFrame:
    """(date, ticker, source, raw_*, topic_*) for one group; PhoBERT runs only on new articles."""
    return _build_raw(group)


def _reduce(df: pd.DataFrame, dim: int = PCA_DIM) -> pd.DataFrame:
    """Apply PCA (fit on train-period rows) to a raw-embedding frame; honest fallback if too few
    train rows (keeps the full RAW embedding instead of a mislabeled 1-dim 'PCA')."""
    raw_cols = _raw_cols(df)
    embs = df[raw_cols].to_numpy()
    train_mask = (df["date"] < pd.Timestamp(TRAIN_CUTOFF)).to_numpy()
    n_train = int(train_mask.sum())
    if n_train < 2:
        reduced, out_dim, pca_applied = embs, embs.shape[1], False
    else:
        from sklearn.decomposition import PCA

        d = min(dim, embs.shape[1], max(1, n_train - 1))
        pca = PCA(n_components=d, svd_solver="randomized").fit(embs[train_mask])
        reduced, out_dim, pca_applied = pca.transform(embs).astype(np.float32), d, True
    emb_cols = {f"emb_{i}": reduced[:, i] for i in range(out_dim)}
    other = df.drop(columns=raw_cols)
    out = pd.concat([other.reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)
    out["pca_applied"] = pca_applied
    return out


def build_group_embeddings(group: str) -> pd.DataFrame:
    """(date, ticker, source, emb_0..emb_{dim-1}, topic_*) for one group — own PCA basis."""
    raw = load_or_build_raw(group)
    if raw.empty:
        return pd.DataFrame()
    return _reduce(raw)


def build_comparable_group_embeddings() -> dict[str, pd.DataFrame]:
    """Both groups reduced with a SHARED PCA (fit on pooled train-period rows), so they live in
    the same subspace and are validly comparable."""
    raws = {g: load_or_build_raw(g) for g in ("khach_quan", "tong_hop")}
    pooled_train = []
    for df in raws.values():
        if df.empty:
            continue
        raw_cols = _raw_cols(df)
        mask = (df["date"] < pd.Timestamp(TRAIN_CUTOFF)).to_numpy()
        pooled_train.append(df.loc[mask, raw_cols].to_numpy())
    pooled = np.concatenate(pooled_train, axis=0) if pooled_train else np.zeros((0, RAW_DIM))

    out = {}
    if len(pooled) >= 2:
        from sklearn.decomposition import PCA

        dim = min(PCA_DIM, pooled.shape[1], max(1, len(pooled) - 1))
        pca = PCA(n_components=dim, svd_solver="randomized").fit(pooled)
        for g, df in raws.items():
            if df.empty:
                out[g] = pd.DataFrame()
                continue
            raw_cols = _raw_cols(df)
            reduced = pca.transform(df[raw_cols].to_numpy()).astype(np.float32)
            emb_cols = {f"emb_{i}": reduced[:, i] for i in range(dim)}
            g_out = pd.concat([df.drop(columns=raw_cols).reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)
            g_out["pca_applied"] = True
            out[g] = g_out
    else:
        for g, df in raws.items():
            out[g] = _reduce(df) if not df.empty else pd.DataFrame()
    return out
