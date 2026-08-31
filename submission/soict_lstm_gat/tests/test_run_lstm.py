import numpy as np
import pandas as pd
import pytest

import run_lstm as R
from config import SMOKE


def _write(tmp, name, n=500):
    d = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    pk = np.abs(rng.normal(3e-4, 1e-4, n)) + 1e-6
    p = tmp / f"{name}_processed.csv"
    pd.DataFrame({"date": d, "parkinson_variance": pk}).to_csv(p, index=False)
    return str(p)


def test_build_perobs_shapes_and_alignment(tmp_path):
    files = [_write(tmp_path, f"T{i}") for i in range(4)]
    data = R.build_perobs(files, lookback=10, horizon=1, cfg=SMOKE)
    assert data["Xtr"].shape[1:] == (10, 3)
    # test windows, raw HAR test rows and test ids/dates are all aligned in length
    assert len(data["Xte"]) == len(data["har_te_f"]) == len(data["te_id"]) == len(data["te_date"])
    assert set(data["te_id"]).issubset(set(data["scalers"]))


@pytest.mark.smoke
def test_run_lstm_end_to_end(tmp_path):
    files = [_write(tmp_path, f"T{i}") for i in range(5)]
    res = R.run("vn30", files, lookback=10, horizon=1, cfg=SMOKE, out_dir=tmp_path / "out")
    assert set(res["metrics"]) == {"HAR", "GARCH", "LSTM"}
    assert res["n_test"] > 0 and res["design"] == "per-observation"
    assert "LSTM_vs_HAR" in res["dm"] and np.isfinite(res["dm"]["LSTM_vs_HAR"]["qlike"]["dm_hln"])
