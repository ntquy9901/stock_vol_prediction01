"""Tests for build_incremental_cache.py — the PhoBERT call itself is monkeypatched (deterministic
fake embeddings) so these run fast and don't depend on network/model download; the aggregation,
diff, upsert, and atomic-write logic is exercised for real against tmp_path fixtures.

Run: pytest baselines/2026-07-25_expand_news_cache_baseline/test/ -v
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

import build_incremental_cache as bic  # noqa: E402
from vendor_data_eda.news_embeddings import RAW_DIM  # noqa: E402

pytestmark = pytest.mark.smoke


def _fake_embed(texts, batch_size=32):
    """Deterministic (n, RAW_DIM) array — value encodes text index so we can assert alignment."""
    n = len(texts)
    out = np.zeros((n, RAW_DIM), dtype=np.float32)
    for i in range(n):
        out[i, 0] = float(i + 1)  # distinguishing marker in raw_0
    return out


def _write_source_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ticker_mentioning_articles_filters_correctly(tmp_path):
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh hom nay", "pub_date": "01/01/2020", "lead": "chi tiet",
         "url": "u1"},
        {"title": "Thoi tiet hom nay", "pub_date": "01/01/2020", "lead": "khong lien quan",
         "url": "u2"},
        {"title": "", "pub_date": "01/01/2020", "lead": "", "url": "u3"},  # empty text
        {"title": "VNM bao cao KQKD", "pub_date": "01/01/2020", "lead": "quy 4", "url": None},
    ])
    out = bic._ticker_mentioning_articles("demo", csv_path)
    assert list(out["url"]) == ["u1"]


def test_ticker_mentioning_articles_no_url_column_returns_empty_with_url_col(tmp_path):
    """[Regression, code review 2026-07-25] a source whose load_source() output lacks a `url`
    column entirely must not crash downstream _new_rows_to_encode (which always does
    articles["url"]) — the early-return frame must still HAVE a (empty) url column."""
    csv_path = tmp_path / "demo_articles.csv"
    pd.DataFrame([{"title": "VIC tang manh", "pub_date": "01/01/2020"}]).to_csv(
        csv_path, index=False)  # no url, no article_url, no pdf_url
    out = bic._ticker_mentioning_articles("demo", csv_path)
    assert list(out.columns) == ["url", "_text"]
    assert out.empty
    # must not raise KeyError
    existing = pd.DataFrame({"url": []})
    assert bic._new_rows_to_encode(out, existing).empty


def test_all_articles_keeps_non_ticker_rows(tmp_path):
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh hom nay", "pub_date": "01/01/2020", "lead": "chi tiet",
         "url": "u1"},
        {"title": "Thoi tiet hom nay", "pub_date": "01/01/2020", "lead": "khong lien quan mã nào",
         "url": "u2"},
        {"title": "", "pub_date": "01/01/2020", "lead": "", "url": "u3"},  # still excluded (empty text)
    ])
    out = bic._all_articles("demo", csv_path)
    assert set(out["url"]) == {"u1", "u2"}, "non-ticker-mentioning article u2 must be included"


def test_new_rows_to_encode_diffs_by_url():
    articles = pd.DataFrame({"url": ["a", "b", "c"], "_text": ["t1", "t2", "t3"]})
    existing = pd.DataFrame({"url": ["a"], "raw_0": [1.0]})
    new = bic._new_rows_to_encode(articles, existing)
    assert list(new["url"]) == ["b", "c"]


def test_load_existing_cache_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: tmp_path / "nope.parquet")
    out = bic._load_existing_cache("nope")
    assert out.empty and list(out.columns) == ["url"]


def test_load_existing_cache_wrong_dim_treated_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"url": ["x"], "raw_0": [1.0], "raw_1": [2.0]}).to_parquet(path)  # dim=2, not 768
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: path)
    out = bic._load_existing_cache("bad")
    assert out.empty and list(out.columns) == ["url"]


def test_encode_rows_empty_input_no_phobert_call(monkeypatch):
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return _fake_embed(*a, **k)

    monkeypatch.setattr(bic, "extract_phobert_embeddings", _spy)
    out = bic._encode_rows(pd.DataFrame({"url": [], "_text": []}), batch_size=32)
    assert out.empty
    assert called["n"] == 0, "PhoBERT must not be called for an empty batch"


def test_atomic_write_creates_file_and_no_leftover_tmp(tmp_path):
    df = pd.DataFrame({"url": ["a"], "raw_0": [1.0]})
    path = tmp_path / "sub" / "cache.parquet"
    bic._atomic_write(df, path)
    assert path.exists()
    assert not path.with_suffix(".parquet.tmp").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_dtype=False)


def test_run_source_first_call_encodes_all_new(tmp_path, monkeypatch):
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh", "pub_date": "01/01/2020", "lead": "", "url": "u1"},
        {"title": "HPG giam nhe", "pub_date": "01/01/2020", "lead": "", "url": "u2"},
    ])
    cache_path = tmp_path / "cache" / "news_emb_articles_demo.parquet"
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: cache_path)
    monkeypatch.setattr(bic, "extract_phobert_embeddings", _fake_embed)

    stats = bic.run_source("demo", csv_path)
    assert stats["source"] == "demo"
    assert stats["n_candidates"] == 2
    assert stats["n_before"] == 0
    assert stats["n_new"] == 2
    assert stats["n_after"] == 2
    saved = pd.read_parquet(cache_path)
    assert set(saved["url"]) == {"u1", "u2"}
    assert saved.shape[1] == 1 + RAW_DIM  # url + raw_0..raw_{RAW_DIM-1}


def test_run_source_second_call_is_incremental_and_preserves_existing(tmp_path, monkeypatch):
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh", "pub_date": "01/01/2020", "lead": "", "url": "u1"},
    ])
    cache_path = tmp_path / "cache" / "news_emb_articles_demo.parquet"
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: cache_path)
    monkeypatch.setattr(bic, "extract_phobert_embeddings", _fake_embed)

    bic.run_source("demo", csv_path)
    first_cached = pd.read_parquet(cache_path)

    # crawl grows: same u1 (already cached) + a brand-new u2
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh", "pub_date": "01/01/2020", "lead": "", "url": "u1"},
        {"title": "HPG giam nhe", "pub_date": "01/01/2020", "lead": "", "url": "u2"},
    ])

    called = {"texts": None}

    def _spy(texts, batch_size=32):
        called["texts"] = list(texts)
        return _fake_embed(texts, batch_size=batch_size)

    monkeypatch.setattr(bic, "extract_phobert_embeddings", _spy)
    stats = bic.run_source("demo", csv_path)

    assert stats["n_before"] == 1
    assert stats["n_new"] == 1, "must only encode the NEW url, not re-encode u1"
    assert called["texts"] is not None and len(called["texts"]) == 1, \
        "PhoBERT must only be called with the single new article's text"

    final = pd.read_parquet(cache_path)
    assert set(final["url"]) == {"u1", "u2"}
    # u1's embedding must be UNCHANGED from the first run (existing wins on dedupe)
    u1_before = first_cached.loc[first_cached["url"] == "u1", "raw_0"].iloc[0]
    u1_after = final.loc[final["url"] == "u1", "raw_0"].iloc[0]
    assert u1_before == u1_after, "existing cached embedding must not be overwritten"


def test_run_source_include_all_encodes_non_ticker_articles(tmp_path, monkeypatch):
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": "VIC tang manh", "pub_date": "01/01/2020", "lead": "", "url": "u1"},
        {"title": "Thoi tiet hom nay khong lien quan chung khoan", "pub_date": "01/01/2020",
         "lead": "", "url": "u2"},
    ])
    cache_path = tmp_path / "cache" / "news_emb_articles_demo.parquet"
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: cache_path)
    monkeypatch.setattr(bic, "extract_phobert_embeddings", _fake_embed)

    stats_ticker_only = bic.run_source("demo", csv_path, include_all=False)
    assert stats_ticker_only["n_candidates"] == 1  # only u1 mentions a ticker
    assert stats_ticker_only["n_new"] == 1

    # reset cache and re-run with include_all=True
    cache_path.unlink()
    stats_all = bic.run_source("demo", csv_path, include_all=True)
    assert stats_all["n_candidates"] == 2  # both u1 and u2
    assert stats_all["n_new"] == 2
    saved = pd.read_parquet(cache_path)
    assert set(saved["url"]) == {"u1", "u2"}


def test_run_source_chunks_large_batches_and_writes_incrementally(tmp_path, monkeypatch):
    """With chunk_size smaller than the number of new articles, run_source must still encode
    everything (across multiple chunks) and each chunk's write must be a valid, readable cache
    (not just the final one) — verified by writing a probe that fails after the first chunk and
    confirming the first chunk's rows already persisted."""
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": f"VIC bai so {i}", "pub_date": "01/01/2020", "lead": "", "url": f"u{i}"}
        for i in range(5)
    ])
    cache_path = tmp_path / "cache" / "news_emb_articles_demo.parquet"
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: cache_path)
    monkeypatch.setattr(bic, "extract_phobert_embeddings", _fake_embed)

    stats = bic.run_source("demo", csv_path, chunk_size=2)
    assert stats["n_candidates"] == 5
    assert stats["n_new"] == 5
    assert stats["n_after"] == 5
    saved = pd.read_parquet(cache_path)
    assert set(saved["url"]) == {f"u{i}" for i in range(5)}


def test_run_source_chunking_crash_preserves_earlier_chunks(tmp_path, monkeypatch):
    """If encoding raises partway through (e.g. chunk 2 of 3), the cache file on disk must still
    contain chunk 1's already-written rows, not be empty/missing."""
    csv_path = tmp_path / "demo_articles.csv"
    _write_source_csv(csv_path, [
        {"title": f"VIC bai so {i}", "pub_date": "01/01/2020", "lead": "", "url": f"u{i}"}
        for i in range(6)
    ])
    cache_path = tmp_path / "cache" / "news_emb_articles_demo.parquet"
    monkeypatch.setattr(bic, "_article_cache_path", lambda source: cache_path)

    calls = {"n": 0}

    def _flaky_embed(texts, batch_size=32):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash on 2nd chunk")
        return _fake_embed(texts, batch_size=batch_size)

    monkeypatch.setattr(bic, "extract_phobert_embeddings", _flaky_embed)

    with pytest.raises(RuntimeError, match="simulated crash"):
        bic.run_source("demo", csv_path, chunk_size=2)

    # chunk 1 (u0, u1) must already be persisted despite chunk 2 crashing
    saved = pd.read_parquet(cache_path)
    assert set(saved["url"]) == {"u0", "u1"}
