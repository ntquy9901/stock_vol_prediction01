"""Real-data-sample smoke test for the vendored dual-group aggregation pipeline (Story 2.4).

Per CLAUDE.md Testing quality rules: a synthetic fixture would miss real-data quirks (mixed
encodings, date formats, schema drift) that this exact pipeline already hit once during
implementation (see vendor_config.py's VN30_TICKERS comment — a ticker-list mismatch surfaced
only when running against real crawl_data + the copied cache). This test runs the REAL vendored
`build_group_embeddings` against a SMALL slice of real, already-cached sources (hsc + ssi — a
few hundred KB / low-single-digit MB each, not the multi-GB `_root` sources) instead of the full
30-source aggregation, so it stays fast while still exercising real text/date/cache-join code.

Run: pytest baselines/2026-07-25_dual_group_news_embedding_baseline/test/test_build_panel_smoke.py -v

Fixture choice note: `hsc` (10 total articles) was tried first but its handful of articles
happen not to mention a VN30 ticker after cache-join, so `vnexpress_objective` (103 articles, 3
ticker-matched, all present in the copied cache — verified interactively) is used instead. This
is itself evidence for why a real-data slice (not a synthetic fixture) matters: a synthetic
"one clean article" fixture would never have surfaced that some small real sources are just too
sparse to exercise the ticker-match path at all.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vendor_config  # noqa: E402
from vendor_data_eda import news_embeddings as ne  # noqa: E402

pytestmark = pytest.mark.smoke

_RAW_CACHE = vendor_config.FEATURES_DIR


def _require_real_cache(*sources: str):
    missing = [s for s in sources if not (_RAW_CACHE / f"news_emb_articles_{s}.parquet").exists()]
    if missing:
        pytest.skip(f"real cache file(s) not present: {missing} (raw_cache not populated in this env)")


def test_load_group_khach_quan_real_slice(monkeypatch):
    """Restrict khach_quan to just `vnexpress_objective` (small, real, already-cached source,
    3 ticker-matched articles) and verify the real discover -> load -> filter -> ticker-explode
    -> cache-join -> PCA path runs end to end without exception and yields a sane (non-empty,
    correctly-shaped) frame."""
    _require_real_cache("vnexpress_objective")
    monkeypatch.setattr(ne, "KHACH_QUAN_SOURCES", {"vnexpress_objective"})
    monkeypatch.setattr(ne, "GROUP_SOURCES",
                        {"khach_quan": {"vnexpress_objective"}, "tong_hop": ne.TONG_HOP_SOURCES})

    df = ne.build_group_embeddings("khach_quan")
    assert not df.empty, "expected non-empty embeddings for real vnexpress_objective slice"
    assert {"ticker", "date", "source"} <= set(df.columns)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert len(emb_cols) > 0, "expected at least one PCA-reduced emb_* column"
    assert df[emb_cols].notna().all().all(), "PCA output should never be NaN for a matched article"
    assert (df["source"] == "vnexpress_objective").all()


def test_load_group_tong_hop_real_slice(monkeypatch):
    """Same shape check for `ssi` (tong_hop group, real cached source)."""
    _require_real_cache("ssi")
    monkeypatch.setattr(ne, "TONG_HOP_SOURCES", {"ssi"})
    monkeypatch.setattr(ne, "GROUP_SOURCES", {"khach_quan": ne.KHACH_QUAN_SOURCES, "tong_hop": {"ssi"}})

    df = ne.build_group_embeddings("tong_hop")
    assert not df.empty, "expected non-empty embeddings for real ssi slice"
    assert (df["source"] == "ssi").all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(pytest.MonkeyPatch())
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
