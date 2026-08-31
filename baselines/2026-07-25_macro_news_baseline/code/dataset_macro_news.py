"""Macro-augmented dual-group news dataset: returns (x_har, adj, x_news, y).

Mirrors `2026-07-25_dual_group_news_embedding_baseline/code/dataset_dual_news.py` (same repo
convention: each baseline's dataset is a focused copy+extension of the previous one, not a
shared-hook abstraction). Difference: x_news concatenates the existing per-(ticker,date)
dual-group vector with a per-date-only macro vector (SAME macro vector broadcast across all 30
tickers for a given date) built by `build_macro_panel.py`.

Isolated — no edit to src, no edit to the sibling baselines. `load_news_panel` is imported
read-only from the sibling baseline (dual-group per-ticker panel loader); this file defines its
own `load_macro_panel` (date-only, no ticker dimension).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.data_normalization import VolatilityNormalizer
from src.lstm_gat_hybrid.dataset_presplit import MultiStockDatasetWithPreSplitData
from dataset_dual_news import load_news_panel, _norm_date  # noqa: E402 (sibling, read-only)

HAR_COLS = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']


def load_macro_panel(panel_path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read macro_news_panel.parquet -> {date_str: feature_vector}, feature_cols.

    Date-only (no ticker dimension) — the same vector is looked up for every ticker at a given
    date by the dataset below. NaN filled with 0.0 (== "no market-wide news that day"), same
    convention as `dataset_dual_news.load_news_panel`.
    """
    df = pd.read_parquet(panel_path)
    feature_cols = [c for c in df.columns if c != "date"]
    df = df.copy()
    df["date"] = df["date"].astype(str).map(_norm_date)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    by_date = {row["date"]: row[feature_cols].to_numpy(dtype=np.float32) for _, row in df.iterrows()}
    return by_date, feature_cols


class MultiStockDatasetWithMacroNews(MultiStockDatasetWithPreSplitData):
    """Dataset returning (x_har, adj, x_news, y), x_news = dual-group vec ++ macro vec (broadcast).

    x_har : [seq, stocks, 3]                       (normalized in __getitem__)
    x_news: [seq, stocks, n_dual_feat+n_macro_feat] (NOT normalized — already PCA/EWMA-scaled)
    y     : [stocks]                                (normalized in __getitem__)
    """

    def __init__(self, *args, news_panel_path=None, macro_panel_path=None, **kwargs):
        self._news_panel_path = Path(news_panel_path) if news_panel_path else None
        self._macro_panel_path = Path(macro_panel_path) if macro_panel_path else None
        self._news_by_ticker: dict[str, dict[str, np.ndarray]] = {}
        self._news_feature_cols: list[str] = []
        self._macro_by_date: dict[str, np.ndarray] = {}
        self._macro_feature_cols: list[str] = []
        self._n_feat = None
        super().__init__(*args, **kwargs)

    def _create_sequences(self):
        if self._news_panel_path is not None and self._news_panel_path.exists():
            self._news_by_ticker, self._news_feature_cols = load_news_panel(self._news_panel_path)
        if self._macro_panel_path is not None and self._macro_panel_path.exists():
            self._macro_by_date, self._macro_feature_cols = load_macro_panel(self._macro_panel_path)

        n_dual = len(self._news_feature_cols) if self._news_feature_cols else 146
        n_macro = len(self._macro_feature_cols) if self._macro_feature_cols else 66
        self._n_feat = n_dual + n_macro
        if not self._news_feature_cols:
            print(f"[macro_news] no dual-group panel found -> dummy n_dual={n_dual} (smoke mode)")
        if not self._macro_feature_cols:
            print(f"[macro_news] no macro panel found -> dummy n_macro={n_macro} (smoke mode)")

        sequences = []
        min_length = min(len(df) for df in self.stock_data_with_har.values())

        all_volatility_list = []
        for stock in self.stock_names:
            vol = self.stock_data_with_har[stock]['parkinson_variance'].values[:min_length]
            all_volatility_list.append(vol)
        all_volatility = np.stack(all_volatility_list, axis=1)

        if self.graph_method == 'knn':
            from src.lstm_gat_hybrid.graph_utils_fixed import DynamicGraphBuilder
            self.graph_builder = DynamicGraphBuilder(self.config)
        elif self.graph_method != 'correlation':
            raise ValueError(f"Unknown graph_method: {self.graph_method}")

        self._total_cells = 0
        self._matched_dual_cells = 0
        self._matched_macro_cells = 0

        for i in range(min_length - self.seq_length - self.forecast_horizon):
            sequence_volatility = all_volatility[i:i + self.seq_length]
            if self.graph_method == 'correlation':
                from src.lstm_gat_hybrid.graph_correlation import construct_correlation_graph
                adj_matrix = construct_correlation_graph(
                    sequence_volatility, corr_threshold=self.graph_threshold)
            else:
                graph_data = {'volatility': sequence_volatility, 'returns': sequence_volatility}
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            x_har_all, x_news_all, y_all = [], [], []
            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                window_dates = [_norm_date(d) for d in
                                stock_feats['date'].iloc[i:i + self.seq_length].astype(str).tolist()]

                x_har_all.append(stock_feats[HAR_COLS].iloc[i:i + self.seq_length].values)

                dual_cache = self._news_by_ticker.get(stock_name) or {}
                day_feats = []
                for d in window_dates:
                    self._total_cells += 1
                    dual_vec = dual_cache.get(d)
                    if dual_vec is not None:
                        self._matched_dual_cells += 1
                    else:
                        dual_vec = np.zeros(n_dual, dtype=np.float32)
                    macro_vec = self._macro_by_date.get(d)
                    if macro_vec is not None:
                        self._matched_macro_cells += 1
                    else:
                        macro_vec = np.zeros(n_macro, dtype=np.float32)
                    day_feats.append(np.concatenate([dual_vec, macro_vec]))
                x_news_all.append(np.stack(day_feats))  # [seq, n_feat]

                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_all.append(stock_feats['parkinson_variance'].iloc[target_idx])

            x_har = np.stack(x_har_all, axis=1).astype(np.float32)
            x_news = np.stack(x_news_all, axis=1).astype(np.float32)
            y = np.array(y_all, dtype=np.float32)
            sequences.append((x_har, adj_matrix, x_news, y))

        cov_dual = 100.0 * self._matched_dual_cells / max(1, self._total_cells)
        cov_macro = 100.0 * self._matched_macro_cells / max(1, self._total_cells)
        print(f"[macro_news] date-match coverage: dual={self._matched_dual_cells}/{self._total_cells} "
              f"({cov_dual:.2f}%), macro={self._matched_macro_cells}/{self._total_cells} "
              f"({cov_macro:.2f}%)")
        if self._news_by_ticker and self._matched_dual_cells == 0:
            raise RuntimeError("ZERO dual-group matches despite panel loaded — date format mismatch?")
        if self._macro_by_date and self._matched_macro_cells == 0:
            raise RuntimeError("ZERO macro matches despite panel loaded — date format mismatch?")
        return sequences

    def __getitem__(self, idx):
        x_har, adj, x_news, y = self.sequences[idx]

        if self.normalize:
            x_har_n = np.zeros_like(x_har, dtype=np.float32)
            for s_idx, sname in enumerate(self.stock_names):
                if sname in self.feature_normalizers:
                    for f_idx in range(x_har.shape[2]):
                        x_har_n[:, s_idx, f_idx] = self.feature_normalizers[sname].transform(
                            x_har[:, s_idx, f_idx:f_idx + 1]).flatten()
                else:
                    x_har_n[:, s_idx, :] = x_har[:, s_idx, :]
            x_har = x_har_n

            y_n = np.zeros_like(y, dtype=np.float32)
            for s_idx, sname in enumerate(self.stock_names):
                if sname in self.target_normalizers:
                    y_n[s_idx] = self.target_normalizers[sname].transform(
                        y[s_idx:s_idx + 1].reshape(1, -1)).flatten()[0]
                else:
                    y_n[s_idx] = y[s_idx]
            y = np.clip(y_n, -10.0, 10.0)

        return (torch.FloatTensor(x_har), torch.FloatTensor(adj),
                torch.FloatTensor(x_news), torch.FloatTensor(y))


def create_macro_news_dataloaders(
    data_dir, news_panel_path, macro_panel_path, seq_length=22, forecast_horizon=5,
    graph_method='knn', graph_threshold=0.7, k_neighbors=8, batch_size=32, train_ratio=0.7,
    val_ratio=0.15, test_ratio=0.15, num_workers=0, normalize=True,
    remove_outliers=True, n_std=3.0, config=None,
):
    """Mirror of create_dual_news_dataloaders, using MultiStockDatasetWithMacroNews +
    both panel paths. No edits to original."""
    from src.lstm_gat_hybrid.dataset_with_graph_method import (
        _load_raw_stock_data, _split_raw_data_by_date, _generate_har_for_split,
    )
    print(f"\n[create_macro_news_dataloaders] graph_method={graph_method}, "
          f"news_panel_path={news_panel_path}, macro_panel_path={macro_panel_path}")

    stock_data_raw = _load_raw_stock_data(data_dir=data_dir, remove_outliers=remove_outliers, n_std=n_std)
    train_raw, val_raw, test_raw, _, _, min_length = \
        _split_raw_data_by_date(stock_data_raw, train_ratio, val_ratio, test_ratio)
    train_har = _generate_har_for_split(train_raw, 'train')
    val_har = _generate_har_for_split(val_raw, 'val')
    test_har = _generate_har_for_split(test_raw, 'test')

    common_stocks = sorted(set(train_har.keys()))
    train_har_c = {k: v for k, v in train_har.items() if k in common_stocks}
    train_raw_c = {k: v for k, v in train_raw.items() if k in common_stocks}
    val_har_c = {k: v for k, v in val_har.items() if k in common_stocks}
    val_raw_c = {k: v for k, v in val_raw.items() if k in common_stocks}
    test_har_c = {k: v for k, v in test_har.items() if k in common_stocks}
    test_raw_c = {k: v for k, v in test_raw.items() if k in common_stocks}
    print(f"[macro_news] common stocks: {len(common_stocks)}")

    ds_kwargs = dict(seq_length=seq_length, forecast_horizon=forecast_horizon,
                     graph_method=graph_method, graph_threshold=graph_threshold,
                     k_neighbors=k_neighbors, normalize=normalize, config=config,
                     news_panel_path=news_panel_path, macro_panel_path=macro_panel_path)
    train_ds = MultiStockDatasetWithMacroNews(train_raw_c, train_har_c, common_stocks,
                                              train_mode=True, **ds_kwargs)
    val_ds = MultiStockDatasetWithMacroNews(val_raw_c, val_har_c, common_stocks,
                                            train_mode=False, **ds_kwargs)
    test_ds = MultiStockDatasetWithMacroNews(test_raw_c, test_har_c, common_stocks,
                                             train_mode=False, **ds_kwargs)
    print(f"[macro_news] sequences - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[macro_news] detected n_feat={train_ds._n_feat}")

    if normalize:
        for sn in common_stocks:
            for ds in (train_ds, val_ds, test_ds):
                ds.feature_normalizers[sn] = VolatilityNormalizer()
                ds.target_normalizers[sn] = VolatilityNormalizer()
        for s_idx, sn in enumerate(train_ds.stock_names):
            feats, targs = [], []
            for x_har, _adj, _xnews, y in train_ds.sequences:
                feats.append(x_har[:, s_idx, :])
                targs.append(y[s_idx])
            feats = np.concatenate(feats, axis=0)
            targs = np.array(targs)
            train_ds.feature_normalizers[sn].fit(feats)
            train_ds.target_normalizers[sn].fit(targs.reshape(-1, 1))
            val_ds.feature_normalizers[sn] = train_ds.feature_normalizers[sn]
            val_ds.target_normalizers[sn] = train_ds.target_normalizers[sn]
            test_ds.feature_normalizers[sn] = train_ds.feature_normalizers[sn]
            test_ds.target_normalizers[sn] = train_ds.target_normalizers[sn]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("[create_macro_news_dataloaders] COMPLETE")
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
