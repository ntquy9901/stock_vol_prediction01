"""Test compute_ablation_deltas.py's ON/OFF classification — an I/O-runner integration test
(writes fixture results.json files to a tmp dir, monkeypatches _ROOT, runs main(), asserts the
output classification matches the documented delta_qlike-sign rule: delta_qlike < 0 -> ON).

Run: pytest baselines/2026-07-25_news_usefulness_ablation/test/test_ablation_deltas.py -v
"""
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_ablation_deltas as cad  # noqa: E402 (sys.path bootstrap above)

pytestmark = pytest.mark.smoke


def _write_fixture(root: Path, har_dirname="har_only_ablation_ref_2099-01-01_000000"):
    har_dir = root / "results" / har_dirname
    har_dir.mkdir(parents=True)
    (har_dir / "results.json").write_text(json.dumps({
        "per_ticker_test_metrics": {
            "AAA": {"qlike": 0.60, "mse": 1.0e-5, "dir_acc": 50.0},
            "BBB": {"qlike": 0.50, "mse": 2.0e-5, "dir_acc": 55.0},
        }
    }), encoding="utf-8")

    (root / "results" / "all_on_dual_group_per_ticker_eval.json").write_text(json.dumps({
        "per_ticker_test_metrics": {
            "AAA": {"qlike": 0.55, "mse": 1.1e-5, "dir_acc": 52.0},  # qlike improves -> ON
            "BBB": {"qlike": 0.52, "mse": 1.9e-5, "dir_acc": 54.0},  # qlike worsens -> OFF
        }
    }), encoding="utf-8")


def test_verdict_matches_delta_qlike_sign(tmp_path, monkeypatch):
    """Property test: a ticker is classified NEWS_ON iff the all-ON model's QLIKE is lower
    (better) than the HAR-only reference's for that ticker — per compute_ablation_deltas.py's
    own documented rule (module docstring line 5-6)."""
    _write_fixture(tmp_path)
    monkeypatch.setattr(cad, "_ROOT", tmp_path)

    cad.main()

    out_path = tmp_path / "results" / "ablation_derived_ticker_classification.json"
    assert out_path.exists()
    out = json.loads(out_path.read_text(encoding="utf-8"))

    assert out["news_on_tickers"] == ["AAA"]
    assert out["news_off_tickers"] == ["BBB"]

    rows_by_ticker = {r["ticker"]: r for r in out["per_ticker_deltas"]}
    assert set(rows_by_ticker) == {"AAA", "BBB"}
    assert rows_by_ticker["AAA"]["delta_qlike"] == pytest.approx(0.55 - 0.60)
    assert rows_by_ticker["BBB"]["delta_qlike"] == pytest.approx(0.52 - 0.50)
    # Shape correctness: every row carries the full expected field set.
    for key in ("ticker", "delta_qlike", "delta_mse", "delta_diracc", "har_qlike", "on_qlike"):
        assert key in rows_by_ticker["AAA"]


def test_latest_har_only_ref_picks_most_recent_timestamp(tmp_path, monkeypatch):
    """_latest_har_only_ref() must pick the lexicographically-last (= most recent, since
    timestamps are YYYY-MM-DD_HHMMSS) results.json when multiple ref runs exist, not an
    arbitrary/first one."""
    _write_fixture(tmp_path, har_dirname="har_only_ablation_ref_2020-01-01_000000")
    newer_dir = tmp_path / "results" / "har_only_ablation_ref_2099-06-15_120000"
    newer_dir.mkdir(parents=True)
    (newer_dir / "results.json").write_text(json.dumps({
        "per_ticker_test_metrics": {"CCC": {"qlike": 0.1, "mse": 1e-6, "dir_acc": 60.0}}
    }), encoding="utf-8")
    monkeypatch.setattr(cad, "_ROOT", tmp_path)

    picked = cad._latest_har_only_ref()

    assert picked == newer_dir / "results.json"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"SKIP {fn.__name__} (needs pytest tmp_path/monkeypatch fixtures — run via pytest)")
