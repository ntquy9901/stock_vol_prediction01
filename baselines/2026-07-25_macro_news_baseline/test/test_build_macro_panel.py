"""Tests for build_macro_panel.py's aggregation math (_fit_pca, _aggregate_by_date).

`_load_dated_cached_embeddings` and `_trading_calendar` are monkeypatched with small synthetic
data so these tests are fast/deterministic and don't need real crawl_data or raw_cache — the
real end-to-end path (source CSV -> cache join -> PCA -> date aggregation) is covered manually
by running `build_macro_panel.py` against real data (see requirements.md go/no-go), consistent
with this project's "test I/O runner with monkeypatch + validate manually against real data"
testing convention for pipeline scripts.

Run: pytest baselines/2026-07-25_macro_news_baseline/test/test_build_macro_panel.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_macro_panel as bmp  # noqa: E402

pytestmark = pytest.mark.smoke

RAW_DIM = bmp.RAW_DIM


def _fake_dated(url_prefix: str, dates: list[str], values: list[float]) -> pd.DataFrame:
    """Synthetic (url, eff_date, raw_0..RAW_DIM-1) — raw embeddings are constant vectors (value
    repeated across all RAW_DIM dims) so aggregation math is trivially verifiable."""
    n = len(dates)
    raw = {f"raw_{i}": [v] * n for i, v in enumerate([0.0] * RAW_DIM)}
    # override: set every raw_i to the per-row value for simplicity of verification
    for i in range(RAW_DIM):
        raw[f"raw_{i}"] = list(values)
    return pd.DataFrame({
        "url": [f"{url_prefix}_{i}" for i in range(n)],
        "eff_date": pd.to_datetime(dates),
        **raw,
    })


def test_fit_pca_pools_pre_cutoff_rows_across_sources(monkeypatch):
    cutoff = pd.Timestamp(bmp.TRAIN_CUTOFF)
    pre = cutoff - pd.Timedelta(days=10)
    post = cutoff + pd.Timedelta(days=10)

    fake_data = {
        "src_a": _fake_dated("a", [pre.strftime("%Y-%m-%d")] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]),
        "src_b": _fake_dated("b", [post.strftime("%Y-%m-%d")] * 5, [10.0] * 5),  # post-cutoff, excluded
    }
    monkeypatch.setattr(bmp, "_load_dated_cached_embeddings",
                        lambda source, path: fake_data.get(source, pd.DataFrame()))

    sources = {"src_a": Path("dummy_a.csv"), "src_b": Path("dummy_b.csv")}
    pca, n_train = bmp._fit_pca(sources)
    assert n_train == 5, "only src_a's 5 pre-cutoff rows should be pooled, not src_b's post-cutoff rows"


def test_fit_pca_returns_none_when_too_few_rows(monkeypatch):
    monkeypatch.setattr(bmp, "_load_dated_cached_embeddings", lambda source, path: pd.DataFrame())
    pca, n_train = bmp._fit_pca({"src_a": Path("dummy.csv")})
    assert pca is None
    assert n_train == 0


def test_aggregate_by_date_computes_per_date_mean(monkeypatch):
    calendar = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    monkeypatch.setattr(bmp, "_trading_calendar", lambda: calendar)

    fake_data = {
        "src_a": _fake_dated("a", ["2024-01-01", "2024-01-01", "2024-01-02"], [2.0, 4.0, 10.0]),
    }
    monkeypatch.setattr(bmp, "_load_dated_cached_embeddings",
                        lambda source, path: fake_data.get(source, pd.DataFrame()))

    out = bmp._aggregate_by_date({"src_a": Path("dummy.csv")}, pca=None, out_dim=RAW_DIM)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    # 2024-01-01: mean of [2.0, 4.0] = 3.0 (every raw_i dim has this same value by construction)
    assert out.loc[0, "macro_emb_0"] == pytest.approx(3.0)
    # 2024-01-02: single value 10.0
    assert out.loc[1, "macro_emb_0"] == pytest.approx(10.0)
    # 2024-01-03: no articles -> NaN, not zero (must not silently claim "neutral news")
    assert np.isnan(out.loc[2, "macro_emb_0"])
    assert np.isnan(out.loc[2, "macro_emb_norm"])


def test_aggregate_by_date_ignores_dates_outside_trading_calendar(monkeypatch):
    """An article whose effective date isn't in the trading calendar (e.g. computed via a stale
    calendar) must be silently dropped, not crash or corrupt another date's bucket."""
    calendar = pd.to_datetime(["2024-01-01"])
    monkeypatch.setattr(bmp, "_trading_calendar", lambda: calendar)

    fake_data = {
        "src_a": _fake_dated("a", ["2024-01-01", "2099-12-31"], [5.0, 999.0]),
    }
    monkeypatch.setattr(bmp, "_load_dated_cached_embeddings",
                        lambda source, path: fake_data.get(source, pd.DataFrame()))

    out = bmp._aggregate_by_date({"src_a": Path("dummy.csv")}, pca=None, out_dim=RAW_DIM)
    assert len(out) == 1
    assert out.loc[0, "macro_emb_0"] == pytest.approx(5.0), \
        "out-of-calendar article's value (999.0) must not leak into the only in-calendar bucket"
