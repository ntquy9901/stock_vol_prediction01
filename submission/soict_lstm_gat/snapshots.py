"""Common-date snapshot builder for the graph model (HAR-LSTM-GAT / LSTM w/o GAT).

A GAT needs per-date snapshots: on each date d, a set of N nodes (tickers) with their [lookback,3] HAR
windows + a shared adjacency. To keep batching simple and the graph well-defined we use a FIXED node
set (tickers covering the common date range) and build snapshots only on dates where every node has a
valid window and a target at d+h → every snapshot is [N, lookback, 3] with the same N.

Split: GLOBAL chronological 80/10/10 on the sorted snapshot dates (train dates < val < test) — no
leakage, and a common test calendar window across all nodes (deviation from the spec's per-stock
split, documented in the results report; cleaner for a cross-sectional graph). Per-ticker StandardScaler
fit on TRAIN-date feature/target rows. The graphical-lasso adjacency is estimated on TRAIN-date rows
only and frozen.

Baselines (HAR, GARCH) are evaluated on the SAME test (node, date) observations these snapshots define.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data_utils as du


@dataclass
class SnapshotData:
    tickers: list          # fixed node order
    adj_pk_train: pd.DataFrame   # TRAIN-date pk panel (columns=tickers) for glasso
    train: list            # list of snapshot dicts
    val: list
    test: list
    scalers: dict          # ticker_id -> (t_mean, t_std)
    lookback: int
    horizon: int

    @property
    def num_nodes(self) -> int:
        return len(self.tickers)


def _load_panel(files: list[str], min_common: int = 300, min_nodes: int = 8) -> pd.DataFrame:
    """Wide pk panel of common dates. To keep a usable common window for large/varied universes,
    drop the latest-listed tickers one at a time (they shrink the intersection most) until the common
    date count reaches ``min_common`` — i.e. use the largest node set with adequate shared history.
    """
    series = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        tk = f.replace("\\", "/").split("/")[-1].replace("_processed.csv", "")
        series[tk] = df.set_index("date")["parkinson_variance"]
    wide = pd.DataFrame(series)                                    # outer join (NaNs before listing)
    # order tickers by first-valid date; latest IPOs first (they limit the intersection most)
    first_valid = {tk: wide[tk].first_valid_index() for tk in wide.columns}
    order = sorted(wide.columns, key=lambda t: first_valid[t], reverse=True)
    current = list(wide.columns)
    while True:
        common = wide[current].dropna(how="any")
        if len(common) >= min_common or len(current) <= min_nodes:
            return common
        current.remove(order[0]); order.pop(0)                    # drop the latest-listed ticker


def build_snapshots(files: list[str], lookback: int, horizon: int,
                    train_frac: float = 0.80, val_frac: float = 0.10,
                    min_common: int = 300) -> SnapshotData:
    panel = _load_panel(files, min_common=min_common)
    if len(panel) < min_common:
        raise ValueError(f"only {len(panel)} common dates (< {min_common}); node set too sparse")
    tickers = list(panel.columns)
    dates = panel.index
    pk = panel.to_numpy(dtype=np.float64)                     # [T, N]
    T, N = pk.shape
    # HAR features per node: [T, N, 3]
    feats = np.stack([du.har_features(pk[:, j]) for j in range(N)], axis=1)   # [T,N,3]

    # valid anchor dates t: window [t-lookback+1..t] monthly-valid, target at t+h exists
    first_valid = 21
    anchors = [t for t in range(first_valid + lookback - 1, T - horizon)]
    n = len(anchors)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    a_tr, a_va, a_te = anchors[:i_tr], anchors[i_tr:i_va], anchors[i_va:]

    # per-ticker scalers fit on TRAIN anchor feature/target rows
    tr_feat_rows = feats[a_tr]                                # [ntr, N, 3]
    f_mean = tr_feat_rows.mean(0)                            # [N,3]
    f_std = tr_feat_rows.std(0) + 1e-8
    y_tr = pk[np.array(a_tr) + horizon]                     # [ntr, N]
    t_mean = y_tr.mean(0)                                   # [N]
    t_std = y_tr.std(0) + 1e-8
    scalers = {j: (float(t_mean[j]), float(t_std[j])) for j in range(N)}

    def snap(t):
        w = feats[t - lookback + 1: t + 1]                  # [lookback, N, 3]
        w = np.transpose(w, (1, 0, 2))                      # [N, lookback, 3]
        wn = (w - f_mean[:, None, :]) / f_std[:, None, :]
        y_raw = pk[t + horizon]                             # [N]
        y_norm = (y_raw - t_mean) / t_std
        return {"x": wn.astype(np.float32), "y_norm": y_norm.astype(np.float32),
                "y_raw": y_raw.astype(np.float64), "har_raw": feats[t].astype(np.float64),  # [N,3] raw HAR at t
                "date": dates[t + horizon].strftime("%Y-%m-%d")}

    train = [snap(t) for t in a_tr]
    val = [snap(t) for t in a_va]
    test = [snap(t) for t in a_te]
    adj_pk_train = panel.iloc[[t for t in a_tr]]            # TRAIN-date rows for glasso
    return SnapshotData(tickers, adj_pk_train, train, val, test, scalers, lookback, horizon)
