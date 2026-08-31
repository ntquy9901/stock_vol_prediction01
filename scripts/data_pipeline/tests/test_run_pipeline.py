"""Unit + smoke tests for the A4 data-pipeline runbook (scripts/data_pipeline/run_pipeline.py).

Covers: each phase P1..P6 callable is invoked and returns a status; --dry-run writes nothing;
--incremental recomputes only the tail with correct causal lookback (tail == full-rebuild tail, no
look-ahead); edge cases (empty new-data, first-ever build); and a real-data-sample smoke on a small
slice of one real market (vn30) asserting the enriched output schema validates.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:  # pragma: no cover - test import bootstrap
    sys.path.insert(0, str(_PKG))

import run_pipeline as rp  # noqa: E402
import enrich  # noqa: E402  (reused engine, on sys.path via run_pipeline bootstrap)
from scripts.quality_gate import data_schemas  # noqa: E402  (repo root on sys.path via rp bootstrap)


# --------------------------------------------------------------------------- fixtures
def _make_raw(n: int) -> pd.DataFrame:
    """Deterministic clean OHLCV: positive, high>low, open/close inside range, weekday dates, no splits.

    Prefix-stable: the first ``k`` rows of ``_make_raw(n)`` equal ``_make_raw(k)`` (so an 'extended' raw is a
    genuine daily append)."""
    i = np.arange(n)
    base = 100.0 + np.sin(i / 3.0)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates, "open": base - 0.1, "high": base + 0.5, "low": base - 0.5,
        "close": base + 0.1, "volume": 1000 + i,
    })


def _make_dirty(n: int) -> pd.DataFrame:
    """Clean base with one zero-range (high==low) bar so the detectors report a dirty class."""
    df = _make_raw(n)
    df.loc[3, "high"] = df.loc[3, "low"]  # zero-range -> zero_range detector fires
    return df


def _write_ohlcv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _ok(status, detail=""):
    return types.SimpleNamespace(status=status, detail=detail)


# --------------------------------------------------------------------------- helpers
def test_run_pytest_executes(tmp_path):
    t = tmp_path / "test_tmp_pass.py"
    t.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    rc, out = rp._run_pytest([str(t)], python_exe=sys.executable)
    assert rc == 0
    assert "passed" in out


def test_pytest_tail():
    assert rp._pytest_tail("a\n\n  last line  ") == "last line"
    assert rp._pytest_tail("   \n  ") == ""


def test_git_sha_real_and_empty(monkeypatch):
    assert rp._git_sha() is not None  # real git present in this repo
    monkeypatch.setattr(rp.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="  "))
    assert rp._git_sha() is None


def test_last_build_date(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert rp._last_build_date(d) is None  # empty dir
    _write_ohlcv(d / "a.csv", pd.DataFrame({"date": ["2020-01-01"]}))
    _write_ohlcv(d / "b.csv", pd.DataFrame({"date": ["2020-02-01"]}))     # later -> d>maxd True
    _write_ohlcv(d / "c.csv", pd.DataFrame({"date": ["not-a-date"]}))     # NaT -> notna False
    _write_ohlcv(d / "e.csv", pd.DataFrame({"date": ["2020-01-05"]}))     # earlier -> d>maxd False
    _write_ohlcv(d / "z_rejections.csv", pd.DataFrame({"date": ["2099-01-01"]}))  # ignored
    assert rp._last_build_date(d) == pd.Timestamp("2020-02-01")


# --------------------------------------------------------------------------- P1
def test_phase_p1_dry_run():
    r = rp.phase_p1_raw_quality(dry_run=True)
    assert r["status"] == rp.DRY_RUN and "pytest" in r["detail"]


def test_phase_p1_pass_and_fail():
    r = rp.phase_p1_raw_quality(runner=lambda paths, exe: (0, "3 passed"))
    assert r["status"] == rp.PASS and r["returncode"] == 0
    r = rp.phase_p1_raw_quality(runner=lambda paths, exe: (1, "1 failed"))
    assert r["status"] == rp.FAIL


# --------------------------------------------------------------------------- P2
def test_phase_p2_dry_run(tmp_path):
    p = tmp_path / "price"
    p.mkdir()
    _write_ohlcv(p / "AAA_ohlcv.csv", _make_raw(30))
    r = rp.phase_p2_audit("vn30", price_dir=p, dry_run=True)
    assert r["status"] == rp.DRY_RUN and r["n_files"] == 1


def test_phase_p2_counts_and_artifact(tmp_path):
    p = tmp_path / "price"
    p.mkdir()
    _write_ohlcv(p / "AAA_ohlcv.csv", _make_raw(40))       # clean -> tick_dirty == 0
    _write_ohlcv(p / "BBB_ohlcv.csv", _make_dirty(40))     # zero-range -> tick_dirty > 0
    out = tmp_path / "audit"
    r = rp.phase_p2_audit("vn30", price_dir=p, out_dir=out, limit=2)
    assert r["status"] == rp.PASS
    assert r["per_class_counts"]["zero_range"] >= 1
    assert Path(r["artifact"]).exists()


def test_phase_p2_empty_dir_skipped(tmp_path):
    r = rp.phase_p2_audit("vn30", price_dir=tmp_path, out_dir=tmp_path / "a")
    assert r["status"] == rp.SKIPPED and r["n_files"] == 0


# --------------------------------------------------------------------------- P3/P4 tail + incremental
def test_recompute_tail_equals_full_rebuild():
    """No look-ahead: tail recomputed from a lookback-warmup slice equals the full-rebuild tail."""
    raw = _make_raw(70)
    full, _r, _c = enrich.build_ticker(raw)
    last_date = pd.to_datetime(full["date"]).iloc[49]
    tail = rp._recompute_tail(enrich._prepare_raw(raw), last_date, rp.INCREMENTAL_LOOKBACK)
    full_tail = full.loc[pd.to_datetime(full["date"]) > last_date].reset_index(drop=True)
    assert len(tail) == len(full_tail) == 20
    for c in ("parkinson_variance", "garman_klass_variance", "rogers_satchell_variance",
              "yang_zhang_n20", "har_weekly", "har_monthly", "volume_zscore_22", "volume_zscore_20",
              "daily_return", "log_range"):
        pd.testing.assert_series_equal(tail[c].reset_index(drop=True),
                                       full_tail[c].reset_index(drop=True), check_names=False)


def test_recompute_tail_no_new_dates():
    raw = _make_raw(30)
    last_date = pd.to_datetime(_make_raw(30)["date"]).iloc[-1]
    tail = rp._recompute_tail(enrich._prepare_raw(raw), last_date, rp.INCREMENTAL_LOOKBACK)
    assert len(tail) == 0


def test_recompute_tail_last_date_none():
    raw = _make_raw(25)
    tail = rp._recompute_tail(enrich._prepare_raw(raw), None, rp.INCREMENTAL_LOOKBACK)
    assert len(tail) == 25


def test_append_csv_create_then_append(tmp_path):
    raw = _make_raw(40)
    out, _r, _c = enrich.build_ticker(raw)
    path = tmp_path / "AAA.csv"
    rp._append_csv(path, out.iloc[:10])
    assert len(pd.read_csv(path)) == 10
    rp._append_csv(path, out.iloc[10:15])
    assert len(pd.read_csv(path)) == 15


def _seed_full_build(price_dir, out_root, n=50):
    price_dir.mkdir(parents=True, exist_ok=True)
    _write_ohlcv(price_dir / "AAA_ohlcv.csv", _make_raw(n))
    _write_ohlcv(price_dir / "BBB_ohlcv.csv", _make_raw(n))
    enrich.build_market("vn30", price_dir=price_dir, out_root=out_root, write=True)


def test_incremental_build_appends_new_dates(tmp_path):
    price = tmp_path / "price"
    out = tmp_path / "out"
    _seed_full_build(price, out, n=50)
    # extend raw by 5 weekdays (a daily append)
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(55))
    _write_ohlcv(price / "BBB_ohlcv.csv", _make_raw(55))
    summ = rp._incremental_build("vn30", price_dir=price, out_root=out)
    assert summ["rows_appended"] == 10 and summ["n_tickers"] == 2
    aaa = pd.read_csv(out / "vn30" / "AAA.csv")
    assert len(aaa) == 55
    assert aaa["market_pk"].tail(5).notna().all()


def test_incremental_build_noop_when_no_new(tmp_path):
    price = tmp_path / "price"
    out = tmp_path / "out"
    _seed_full_build(price, out, n=40)
    summ = rp._incremental_build("vn30", price_dir=price, out_root=out, limit=2)
    assert summ["rows_appended"] == 0 and summ["n_tickers"] == 0
    assert summ["last_build_date"] is not None


def test_incremental_build_last_date_none(tmp_path):
    """market_dir exists with only a schema sidecar (no data csvs) -> last_date None -> full recompute."""
    price = tmp_path / "price"
    price.mkdir()
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(30))
    out = tmp_path / "out"
    (out / "vn30").mkdir(parents=True)
    (out / "vn30" / "_schema_version.json").write_text("{}", encoding="utf-8")
    summ = rp._incremental_build("vn30", price_dir=price, out_root=out)
    assert summ["last_build_date"] is None and summ["rows_appended"] == 30


def test_phase_p3p4_dry_run_full_and_incremental(tmp_path):
    out = tmp_path / "out"
    r = rp.phase_p3p4_enrich("vn30", out_root=out, dry_run=True)
    assert r["status"] == rp.DRY_RUN and r["mode"] == "full"
    (out / "vn30").mkdir(parents=True)
    (out / "vn30" / "_schema_version.json").write_text("{}", encoding="utf-8")
    r2 = rp.phase_p3p4_enrich("vn30", out_root=out, incremental=True, dry_run=True)
    assert r2["mode"] == "incremental" and r2["p3"]["status"] == rp.DRY_RUN


def test_phase_p3p4_full_then_incremental(tmp_path):
    price = tmp_path / "price"
    out = tmp_path / "out"
    price.mkdir()
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(50))
    r = rp.phase_p3p4_enrich("vn30", out_root=out, price_dir=price)
    assert r["status"] == rp.PASS and r["mode"] == "full" and r["p4"]["status"] == rp.PASS
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(55))
    r2 = rp.phase_p3p4_enrich("vn30", out_root=out, price_dir=price, incremental=True)
    assert r2["mode"] == "incremental" and r2["summary"]["rows_appended"] == 5


# --------------------------------------------------------------------------- P5
def test_phase_p5_dry_run():
    assert rp.phase_p5_quality_gate("vn30", dry_run=True)["status"] == rp.DRY_RUN


@pytest.mark.parametrize("enr,schema_status,expected", [
    ([("x", "pass", "ok")], rp.PASS, rp.PASS),
    ([("x", "fail", "bad")], rp.PASS, rp.FAIL),
    ([("x", "skip", "missing")], rp.PASS, rp.PASS),   # all-missing enriched -> SKIPPED, schema PASS -> PASS
    ([("x", "pass", "ok")], rp.FAIL, rp.FAIL),         # schema hard-fail
])
def test_phase_p5_status_matrix(enr, schema_status, expected):
    r = rp.phase_p5_quality_gate(
        "vn30",
        schema_fn=lambda: _ok(schema_status, "d"),
        drift_fn=lambda out_dir: _ok("INFO", "drift"),
        enriched_validate=lambda md: enr,
    )
    assert r["status"] == expected


def test_phase_p5_real_defaults(tmp_path):
    """Exercise the real qg.check_schema / check_drift / validate_enriched defaults once (empty enriched)."""
    r = rp.phase_p5_quality_gate("vn30", out_root=tmp_path, out_dir=tmp_path / "qg")
    assert r["status"] in (rp.PASS, rp.FAIL)
    assert "SKIPPED" in r["enriched"]  # tmp_path/vn30 has no enriched files


# --------------------------------------------------------------------------- P6
def test_phase_p6_dry_run(tmp_path):
    assert rp.phase_p6_freeze("vn30", out_root=tmp_path, dry_run=True)["status"] == rp.DRY_RUN


def test_phase_p6_writes_schema_and_provenance(tmp_path):
    md = tmp_path / "vn30"
    md.mkdir(parents=True)
    _write_ohlcv(md / "AAA.csv", _make_raw(5))
    r = rp.phase_p6_freeze("vn30", out_root=tmp_path, mode="full", git_sha="abc123", last_build_date="2020-03-01")
    assert r["status"] == rp.PASS
    assert Path(r["schema_version_path"]).exists() and Path(r["provenance_path"]).exists()
    import json
    prov = json.loads(Path(r["provenance_path"]).read_text(encoding="utf-8"))
    assert prov["git_sha"] == "abc123" and prov["mode"] == "full" and prov["n_tickers"] == 1


def test_phase_p6_keeps_existing_schema(tmp_path):
    md = tmp_path / "vn30"
    md.mkdir(parents=True)
    (md / "_schema_version.json").write_text('{"schema_version": "kept"}', encoding="utf-8")
    rp.phase_p6_freeze("vn30", out_root=tmp_path, git_sha="x")  # existing schema kept, not overwritten
    import json
    assert json.loads((md / "_schema_version.json").read_text())["schema_version"] == "kept"
    # default git_sha (else branch) -> real git
    r2 = rp.phase_p6_freeze("vn30", out_root=tmp_path)
    assert json.loads(Path(r2["provenance_path"]).read_text())["git_sha"] is not None


# --------------------------------------------------------------------------- report
def test_write_report_full_and_minimal(tmp_path):
    phases = {k: {"status": rp.PASS, "detail": f"d|{k}"} for k in ("P1", "P2", "P3", "P4", "P5", "P6")}
    phases["P2"]["per_class_counts"] = {"zero_range": 5, "naninf": 0}
    report = {"market": "vn30", "mode": "full", "incremental": False,
              "phases": phases, "summary": {"n_tickers": 3, "rows_out": 10, "n_dirty_bars": 1, "n_dropped": 0}}
    path = rp._write_report("vn30", report, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Build summary" in text and "Dirty-data audit" in text and "\\|" in text
    # minimal: no summary, no per-class counts -> both falsy branches
    minimal = {"market": "vn30", "mode": "full", "incremental": False,
               "phases": {k: {"status": rp.SKIPPED} for k in ("P1", "P2", "P3", "P4", "P5", "P6")},
               "summary": None}
    p2 = rp._write_report("vn30", minimal, tmp_path)
    assert "Build summary" not in p2.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- orchestrator
def test_run_pipeline_unknown_market():
    with pytest.raises(ValueError, match="unknown market"):
        rp.run_pipeline("nope")


def _stub_light(monkeypatch):
    monkeypatch.setattr(rp, "phase_p1_raw_quality",
                        lambda dry_run=False, python_exe=None: {"status": rp.PASS, "detail": "stub"})
    monkeypatch.setattr(rp, "phase_p5_quality_gate",
                        lambda *a, **k: {"status": rp.PASS, "detail": "stub"})


def test_run_pipeline_dry_run_writes_nothing(tmp_path, monkeypatch):
    _stub_light(monkeypatch)
    price = tmp_path / "price"
    price.mkdir()
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(30))
    out = tmp_path / "out"
    reports = tmp_path / "reports"
    rep = rp.run_pipeline("vn30", dry_run=True, out_root=out, price_dir=price, report_dir=reports)
    assert rep["dry_run"] is True and "report_path" not in rep
    assert not out.exists() and not reports.exists()  # nothing written


def test_run_pipeline_full_writes_report(tmp_path, monkeypatch):
    _stub_light(monkeypatch)
    price = tmp_path / "price"
    price.mkdir()
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(40))
    _write_ohlcv(price / "BBB_ohlcv.csv", _make_raw(40))
    out = tmp_path / "out"
    reports = tmp_path / "reports"
    rep = rp.run_pipeline("vn30", out_root=out, price_dir=price, report_dir=reports)
    assert rep["mode"] == "full"
    assert Path(rep["report_path"]).exists()
    assert (out / "vn30" / "AAA.csv").exists()
    assert (out / "vn30" / "_provenance.json").exists()
    assert {"P1", "P2", "P3", "P4", "P5", "P6"} == set(rep["phases"])


def test_run_pipeline_incremental(tmp_path, monkeypatch):
    _stub_light(monkeypatch)
    price = tmp_path / "price"
    price.mkdir()
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(50))
    out = tmp_path / "out"
    reports = tmp_path / "reports"
    rp.run_pipeline("vn30", out_root=out, price_dir=price, report_dir=reports)   # seed full
    _write_ohlcv(price / "AAA_ohlcv.csv", _make_raw(56))
    rep = rp.run_pipeline("vn30", incremental=True, out_root=out, price_dir=price, report_dir=reports)
    assert rep["mode"] == "incremental"
    assert pd.read_csv(out / "vn30" / "AAA.csv").shape[0] == 56


# --------------------------------------------------------------------------- real-data smoke
@pytest.mark.smoke
def test_smoke_real_data_slice(tmp_path, monkeypatch):
    """Run P1..P6 on a SMALL real vn30 slice; assert it completes + the enriched output schema validates.

    P1 (external pytest wrapper) is stubbed to keep the smoke fast; P2-P6 run on REAL data and the enriched
    output is validated with the project's Pandera enriched schema."""
    monkeypatch.setattr(rp, "phase_p1_raw_quality",
                        lambda dry_run=False, python_exe=None: {"status": rp.PASS, "detail": "smoke-stub"})
    out = tmp_path / "out"
    reports = tmp_path / "reports"
    rep = rp.run_pipeline("vn30", out_root=out, limit=2, report_dir=reports)
    for key in ("P1", "P2", "P3", "P4", "P5", "P6"):
        assert rep["phases"][key]["status"] in (rp.PASS, rp.SKIPPED)
    results = data_schemas.validate_enriched(out)  # enriched ROOT (globs <root>/<market>/<ticker>.csv)
    assert results and all(r[1] != data_schemas.INVALID for r in results)
    assert any(r[1] == data_schemas.VALID for r in results)  # the 2 real vn30 files actually validated
    assert Path(rep["report_path"]).exists()
