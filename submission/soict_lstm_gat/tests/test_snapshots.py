import numpy as np
import pandas as pd

import snapshots as S


def _write(tmp, name, start, n):
    dates = pd.bdate_range(start, periods=n)
    pk = np.abs(np.random.default_rng(abs(hash(name)) % 2**32).normal(1e-3, 3e-4, n)) + 1e-5
    pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(
        tmp / f"{name}_processed.csv", index=False)
    return str(tmp / f"{name}_processed.csv")


def test_split_chronological_and_adj_from_train(tmp_path):
    files = [_write(tmp_path, f"T{i}", "2015-01-01", 900) for i in range(6)]
    snap = S.build_snapshots(files, lookback=10, horizon=1, min_common=300)
    # split is chronological, no overlap (train dates < val < test)
    tr_last = max(s["date"] for s in snap.train)
    va_first = min(s["date"] for s in snap.val); va_last = max(s["date"] for s in snap.val)
    te_first = min(s["date"] for s in snap.test)
    assert tr_last < va_first and va_last < te_first
    # adjacency panel uses ONLY train dates (no leakage): its rows are a subset of train target dates
    assert snap.adj_pk_train.index.max() <= pd.Timestamp(tr_last)


def test_universe_drops_recent_ipo(tmp_path):
    # 12 long-history tickers (> min_nodes=8 floor) + 1 recent IPO that alone shrinks the common window
    files = [_write(tmp_path, f"L{i:02d}", "2015-01-01", 900) for i in range(12)]
    files.append(_write(tmp_path, "NEW", "2024-06-01", 250))
    snap = S.build_snapshots(files, lookback=10, horizon=1, min_common=400)
    assert "NEW" not in snap.tickers        # recent IPO dropped to keep an adequate common window
    assert snap.num_nodes == 12
