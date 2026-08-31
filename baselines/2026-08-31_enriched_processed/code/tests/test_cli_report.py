"""Cover the CLI ``run()`` orchestration + the HTML report builder."""
from __future__ import annotations


import cli
import enrich
import report as report_mod
from _synth import clean_frame, write_market


def test_run_builds_markets_and_writes_html(tmp_path):
    price = write_market(tmp_path / "raw", {"XA": clean_frame(n=30, seed=1),
                                            "XB": clean_frame(n=30, seed=2)})
    # point the market's price dir at our tmp raw
    enrich.PRICE_DIRS["_tmpmkt"] = price
    try:
        html = tmp_path / "r.html"
        out = cli.run(["_tmpmkt"], out_root=tmp_path / "out", jobs=1, html_path=html, limit=None)
        assert "_tmpmkt" in out
        assert html.exists()
        assert "<html" in html.read_text(encoding="utf-8")
    finally:
        del enrich.PRICE_DIRS["_tmpmkt"]


def test_run_without_html(tmp_path):
    price = write_market(tmp_path / "raw", {"XA": clean_frame(n=25, seed=1)})
    enrich.PRICE_DIRS["_tmpmkt2"] = price
    try:
        out = cli.run(["_tmpmkt2"], out_root=tmp_path / "out", jobs=1, html_path=None)
        assert out["_tmpmkt2"]["n_tickers"] == 1
    finally:
        del enrich.PRICE_DIRS["_tmpmkt2"]


def test_run_vn30_attaches_regression_to_report(tmp_path):
    import pytest
    if not list(enrich.PRICE_DIRS["vn30"].glob("*_ohlcv.csv")):  # pragma: no cover - env guard
        pytest.skip("no real vn30 raw")
    html = tmp_path / "r.html"
    out = cli.run(["vn30"], out_root=tmp_path / "out", jobs=1, html_path=html, limit=2)
    assert "regression" in out["vn30"]
    assert "Clean-bar regression" in html.read_text(encoding="utf-8")


def test_map_fn_serial_and_parallel():
    assert cli._map_fn(1) is map
    parallel = cli._map_fn(2)
    assert parallel(str, ["1", "2"]) == ["1", "2"]      # exercises the ProcessPoolExecutor branch


def _fake_summary(market):
    return {
        "market": market, "n_tickers": 2, "rows_in": 100, "rows_out": 98, "n_dropped": 2,
        "n_dirty_bars": 3, "dirty_by_class": {k: 1 for k in enrich.DIRTY_CLASSES},
        "cleaning_applied": {"none": 90, "widen_range": 8},
        "estimator_mean": {"parkinson_variance": 0.01, "garman_klass_variance": 0.01,
                           "rogers_satchell_variance": 0.01, "yang_zhang_n20": 0.01},
        "market_pk": {"n_days": 50, "min": 0.0, "mean": 0.01, "max": 0.2},
    }


def test_build_html_report_with_and_without_regression(tmp_path):
    summaries = {"vn30": _fake_summary("vn30")}
    p1 = report_mod.build_html_report(summaries, tmp_path / "a.html",
                                      regression={"worst_noncapped_diff": 1e-16, "n_capped": 1,
                                                  "n_compared": 5000})
    assert p1.exists()
    txt = p1.read_text(encoding="utf-8")
    assert "Clean-bar regression" in txt and "widen_range" in txt
    p2 = report_mod.build_html_report(summaries, tmp_path / "b.html")
    assert "Clean-bar regression" not in p2.read_text(encoding="utf-8")


def test_build_html_report_empty_summaries(tmp_path):
    p = report_mod.build_html_report({}, tmp_path / "e.html")     # covers the `if summaries else []` branch
    assert p.exists()


def test_report_fmt_variants():
    assert report_mod._fmt(1000) == "1,000"          # int path
    assert report_mod._fmt(0.012345, 3) == "0.0123"  # float path
    assert report_mod._fmt(float("nan")) == "-"       # NaN path
    assert report_mod._fmt("abc") == "abc"            # non-numeric path
    assert report_mod._fmt(True) in ("True", "1")     # bool is not int-formatted
