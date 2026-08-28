"""Tests for the OOS-suite resume guard (scripts/garch_masked/run_oos_suite.py).

External review M-06: a result counts as "GARCH complete" only if it carries a finite GARCH metric AND
a matching schema + screened-universe fingerprint, so a stale/incomplete artifact from a different
universe/horizon is recomputed rather than silently skipped.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))
import run_oos_suite as OS  # noqa: E402


def _files(names):
    return [f"/some/dir/{n}_processed.csv" for n in names]


def _write(tmp_path, res):
    rp = tmp_path / "result.json"
    rp.write_text(json.dumps(res), encoding="utf-8")
    return rp


def test_universe_fp_order_independent():
    a = OS._universe_fp(_files(["BBB", "AAA", "CCC"]))
    b = OS._universe_fp(_files(["CCC", "AAA", "BBB"]))
    assert a == b and a["n"] == 3
    assert OS._universe_fp(_files(["AAA", "BBB"])) != a   # different universe -> different fp


def test_has_garch_requires_finite_metric_schema_and_fingerprint(tmp_path):
    files = _files(["AAA", "BBB", "CCC"])
    fp = OS._universe_fp(files)
    good = {"horizon": 1, "metrics": {"GARCH": {"qlike": 1.5, "n": 100}},
            "garch_meta": {"schema": 1, "universe_fp": fp}}
    assert OS._has_garch(_write(tmp_path, good), "hnx", 1, files) is True

    # missing GARCH metric -> incomplete
    assert OS._has_garch(_write(tmp_path, {"horizon": 1, "metrics": {}}), "hnx", 1, files) is False
    # non-finite qlike -> incomplete
    bad = {"horizon": 1, "metrics": {"GARCH": {"qlike": float("nan"), "n": 100}},
           "garch_meta": {"schema": 1, "universe_fp": fp}}
    assert OS._has_garch(_write(tmp_path, bad), "hnx", 1, files) is False
    # pre-fingerprint result (no garch_meta) -> recompute to add metadata
    old = {"horizon": 1, "metrics": {"GARCH": {"qlike": 1.5, "n": 100}}}
    assert OS._has_garch(_write(tmp_path, old), "hnx", 1, files) is False
    # horizon mismatch -> stale
    assert OS._has_garch(_write(tmp_path, good), "hnx", 5, files) is False
    # screened universe changed -> stale
    assert OS._has_garch(_write(tmp_path, good), "hnx", 1, _files(["AAA", "BBB"])) is False
    # unknown ds/h/files (back-compat call) -> only checks metric presence + schema
    assert OS._has_garch(_write(tmp_path, good)) is True


def test_add_garch_writes_metrics_dm_and_garch_meta(tmp_path, monkeypatch):
    """Coverage/integration: _add_garch patches result.json with GARCH metric + DM + garch_meta (incl. the
    universe fingerprint), passing the HAR-X basis guard. Panel build + heavy compute are mocked."""
    from types import SimpleNamespace
    rp = tmp_path / "result.json"
    rp.write_text(json.dumps({"horizon": 1, "metrics": {"HAR-X": {"qlike": 1.0}},
                              "dm_date_clustered": {}}), encoding="utf-8")
    monkeypatch.setattr(OS.MR, "build_masked_rich", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(OS.CG, "_harx_pred", lambda D, cfg: {"harx": 1})
    monkeypatch.setattr(OS.CG, "_metrics", lambda pred, floor: {"qlike": 1.0, "n": 5})
    monkeypatch.setattr(OS.CG, "_dm", lambda g, h, hz, floor: {"qlike": {"p_value": 0.5}})
    def _fake_gpred(D, h, cfg, status_out=None):
        if status_out is not None:
            status_out.append({"fallback": False, "reason": "", "arch_available": True})
        return {"g": 1}
    monkeypatch.setattr(OS.CG, "_garch_pred", _fake_gpred)
    files = _files(["AAA", "BBB"])
    OS._add_garch("hnx", files, "/price", 1, SimpleNamespace(lookback=10, qlike_floor=1e-8, seed=42), rp)
    res = json.loads(rp.read_text(encoding="utf-8"))
    assert res["metrics"]["GARCH"]["qlike"] == 1.0
    assert "GARCH_vs_HARX" in res["dm_date_clustered"]
    assert res["garch_meta"]["schema"] == 1
    assert res["garch_meta"]["universe_fp"] == OS._universe_fp(files)


def test_has_garch_failsafe_on_malformed_metadata(tmp_path):
    # R-04: typed/malformed fields must return False (recompute), never raise.
    fp = OS._universe_fp(_files(["AAA", "BBB"]))
    cases = [
        {"horizon": 1, "metrics": {"GARCH": {"qlike": "n/a", "n": 100}}, "garch_meta": {"schema": 1, "universe_fp": fp}},
        {"horizon": 1, "metrics": {"GARCH": {"qlike": None, "n": 100}}, "garch_meta": {"schema": 1, "universe_fp": fp}},
        {"horizon": 1, "metrics": {"GARCH": {"qlike": 1.5, "n": "n/a"}}, "garch_meta": {"schema": 1, "universe_fp": fp}},
        {"horizon": 1, "metrics": {"GARCH": "not-a-dict"}, "garch_meta": {"schema": 1}},
        {"horizon": 1, "metrics": {"GARCH": {"qlike": 1.5, "n": 100}}, "garch_meta": "not-a-dict"},
        {"metrics": {"GARCH": {"qlike": 1.5, "n": 100}}, "garch_meta": {"schema": 1, "universe_fp": fp}},  # no horizon
    ]
    for c in cases:
        assert OS._has_garch(_write(tmp_path, c), "hnx", 1, _files(["AAA", "BBB"])) is False
    # totally corrupt JSON file
    rp = tmp_path / "bad.json"; rp.write_text("{not json", encoding="utf-8")
    assert OS._has_garch(rp, "hnx", 1, _files(["AAA", "BBB"])) is False
