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
