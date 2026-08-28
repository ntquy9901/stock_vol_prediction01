"""Smoke test for scripts/crawl_hose_hnx.py build_manifest (no network)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_hose_hnx as CR  # noqa: E402


def test_build_manifest_ok_and_missing(tmp_path):
    vals = {c: [1, 2] for c in CR.RAW_COLUMNS}
    vals["date"] = ["2020-01-01", "2020-01-02"]
    pd.DataFrame(vals, columns=CR.RAW_COLUMNS).to_csv(tmp_path / "AAA_ohlcv.csv", index=False)
    m = CR.build_manifest("HOSE", ["AAA", "ZZZ"], tmp_path)
    assert list(m["ticker"]) == ["AAA", "ZZZ"]
    assert m.loc[m.ticker == "AAA", "status"].iloc[0] == "ok"
    assert int(m.loc[m.ticker == "AAA", "n_rows"].iloc[0]) == 2
    assert m.loc[m.ticker == "ZZZ", "status"].iloc[0] == "missing"       # no file -> missing
