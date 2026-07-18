"""Pure-market dataset: returns (x_har, adj, x_market, market_mask, y) — 5-tuple.

Subclass of MultiStockDatasetWithPreSplitData (read-only import). NO per-stock news branch
at all — market embeddings (data/sentiment_embedding/market_emb.npz, produced by the sibling
2026-07-08_market_fallback baseline's extract_market_embeddings.py, READ-ONLY reuse) are loaded
by DATE ONLY and are IDENTICAL for every stock in a given window (no ticker matching, no gate).
See ../requirements/requirements.md and ../design/design.md.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.lstm_gat_hybrid.dataset_presplit import MultiStockDatasetWithPreSplitData
from src.common.data_normalization import VolatilityNormalizer

HAR_COLS = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
MAX_MARKET = 15  # market articles per day cap (matches 2026-07-08_market_fallback's percentile-99 choice)


def _norm_date(s):
    s = str(s).strip()
    for sep in (' ', 'T'):
        if sep in s:
            s = s.split(sep)[0]
    return s[:10]


class PureMarketDataset(MultiStockDatasetWithPreSplitData):
    """Dataset returning (x_har, adj, x_market, market_mask, y).

    x_har      : [seq, stocks, 3]
    x_market   : [seq, MAX_MARKET, emb_dim]   (per-day, SAME for every stock — no ticker match)
    market_mask: [seq, MAX_MARKET]
    y          : [stocks]
    """

    def __init__(self, *args, market_cache_path=None, max_market=MAX_MARKET, **kwargs):
        self._market_cache_path = Path(market_cache_path) if market_cache_path else None
        self._max_market = max_market
        self._market = None
        self._emb_dim = None
        super().__init__(*args, **kwargs)  # calls _create_sequences (our override)

    def _load_market(self):
        if self._market_cache_path is None or not self._market_cache_path.exists():
            print(f"[market] no market cache at {self._market_cache_path} (smoke mode)")
            self._market = {}
            return
        data = np.load(self._market_cache_path, allow_pickle=False)
        self._market = {_norm_date(k): data[k] for k in data.files}
        print(f"[market] loaded {len(self._market)} dates from {self._market_cache_path.name}")

    def _pad_articles(self, arr):
        dim = self._emb_dim or 64
        out = np.zeros((self._max_market, dim), dtype=np.float32)
        mask = np.zeros(self._max_market, dtype=np.float32)
        if arr is not None and len(arr) > 0:
            if self._emb_dim is not None and arr.shape[1] != self._emb_dim:
                # [lesson from sibling 2026-07-08_market_fallback HIGH-2] fail loud with an
                # actionable message instead of a bare numpy broadcast-shape error.
                raise ValueError(
                    f"Embedding dim mismatch: cache array dim={arr.shape[1]} but expected "
                    f"_emb_dim={self._emb_dim}. Re-extract market_emb.npz with a consistent --dim.")
            a = arr.astype(np.float32, copy=False)
            finite = np.isfinite(a).all(axis=1)  # skip non-finite articles (same lesson as siblings)
            j = 0
            for i in range(min(len(a), self._max_market)):
                if finite[i]:
                    out[j] = a[i]
                    mask[j] = 1.0
                    j += 1
        return out, mask

    def _create_sequences(self):
        self._load_market()
        self._emb_dim = None
        for arr in self._market.values():
            if arr is not None and len(arr) > 0:
                self._emb_dim = arr.shape[1]
                break
        if self._emb_dim is None:
            self._emb_dim = 64
            print(f"[market] no cache found -> dummy emb_dim={self._emb_dim} (smoke mode)")

        sequences = []
        min_length = min(len(df) for df in self.stock_data_with_har.values())
        all_volatility = np.stack(
            [self.stock_data_with_har[s]['parkinson_volatility'].values[:min_length]
             for s in self.stock_names], axis=1)

        if self.graph_method == 'knn':
            from src.lstm_gat_hybrid.graph_utils_fixed import DynamicGraphBuilder
            self.graph_builder = DynamicGraphBuilder(self.config)
        elif self.graph_method != 'correlation':
            raise ValueError(f"Unknown graph_method: {self.graph_method}")

        self._market_total = 0
        self._market_matched = 0

        for i in range(min_length - self.seq_length - self.forecast_horizon):
            sequence_volatility = all_volatility[i:i + self.seq_length]
            if self.graph_method == 'correlation':
                from src.lstm_gat_hybrid.graph_correlation import construct_correlation_graph
                adj_matrix = construct_correlation_graph(
                    sequence_volatility, corr_threshold=self.graph_threshold)
            else:
                graph_data = {'volatility': sequence_volatility, 'returns': sequence_volatility}
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            x_har_all, y_all = [], []
            window_dates = None
            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                if window_dates is None:
                    window_dates = [_norm_date(d) for d in
                                    stock_feats['date'].iloc[i:i + self.seq_length].astype(str).tolist()]
                x_har_all.append(stock_feats[HAR_COLS].iloc[i:i + self.seq_length].values)
                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_all.append(stock_feats['parkinson_volatility'].iloc[target_idx])

            # market embeddings per day — computed ONCE per window (shared across ALL stocks,
            # no per-stock loop needed since there's no ticker matching at all)
            market_day_embs, market_day_masks = [], []
            for d in window_dates:
                marr = self._market.get(d)
                self._market_total += 1
                if marr is not None and len(marr) > 0:
                    self._market_matched += 1
                me, mm = self._pad_articles(marr)
                market_day_embs.append(me)
                market_day_masks.append(mm)
            x_market = np.stack(market_day_embs)      # [seq, MAX_MARKET, dim]
            market_mask = np.stack(market_day_masks)  # [seq, MAX_MARKET]

            x_har = np.stack(x_har_all, axis=1).astype(np.float32)  # [seq, stocks, 3]
            y = np.array(y_all, dtype=np.float32)
            sequences.append((x_har, adj_matrix, x_market, market_mask, y))

        cov = 100.0 * self._market_matched / max(1, self._market_total)
        print(f"[market] date-match coverage: {self._market_matched}/{self._market_total} "
              f"days ({cov:.2f}%)")
        if self._market and self._market_matched == 0:
            raise RuntimeError(
                "ZERO market news matches despite market cache loaded — date format mismatch? "
                "Market branch would be silently inert.")
        return sequences

    def __getitem__(self, idx):
        x_har, adj, x_market, market_mask, y = self.sequences[idx]

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
                torch.FloatTensor(x_market), torch.FloatTensor(market_mask),
                torch.FloatTensor(y))


def create_pure_market_dataloaders(
    data_dir, market_cache_path, seq_length=22, forecast_horizon=5, graph_method='knn',
    graph_threshold=0.7, k_neighbors=8, batch_size=32, train_ratio=0.7,
    val_ratio=0.15, test_ratio=0.15, num_workers=0, normalize=True,
    remove_outliers=True, n_std=3.0, config=None,
):
    """Mirror of create_embedding_dataloaders (sibling), using PureMarketDataset. No edits
    to any sibling baseline."""
    from src.lstm_gat_hybrid.dataset_with_graph_method import (
        _load_raw_stock_data, _split_raw_data_by_date, _generate_har_for_split,
    )
    print(f"\n[create_pure_market_dataloaders] graph_method={graph_method}, "
          f"market_cache_path={market_cache_path}")

    stock_data_raw = _load_raw_stock_data(data_dir=data_dir, remove_outliers=remove_outliers, n_std=n_std)
    train_raw, val_raw, test_raw, _, _, _ = \
        _split_raw_data_by_date(stock_data_raw, train_ratio, val_ratio, test_ratio)
    train_har = _generate_har_for_split(train_raw, 'train')
    val_har = _generate_har_for_split(val_raw, 'val')
    test_har = _generate_har_for_split(test_raw, 'test')

    common_stocks = sorted(set(train_har.keys()))
    def _c(d): return {k: v for k, v in d.items() if k in common_stocks}
    train_har_c, val_har_c, test_har_c = _c(train_har), _c(val_har), _c(test_har)
    train_raw_c, val_raw_c, test_raw_c = _c(train_raw), _c(val_raw), _c(test_raw)
    print(f"[pure_market] common stocks: {len(common_stocks)}")

    ds_kwargs = dict(seq_length=seq_length, forecast_horizon=forecast_horizon,
                     graph_method=graph_method, graph_threshold=graph_threshold,
                     k_neighbors=k_neighbors, normalize=normalize, config=config,
                     market_cache_path=market_cache_path)
    train_ds = PureMarketDataset(train_raw_c, train_har_c, common_stocks, train_mode=True, **ds_kwargs)
    val_ds = PureMarketDataset(val_raw_c, val_har_c, common_stocks, train_mode=False, **ds_kwargs)
    test_ds = PureMarketDataset(test_raw_c, test_har_c, common_stocks, train_mode=False, **ds_kwargs)
    print(f"[pure_market] sequences - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[pure_market] detected emb_dim={train_ds._emb_dim}")

    if normalize:
        for sn in common_stocks:
            for ds in (train_ds, val_ds, test_ds):
                ds.feature_normalizers[sn] = VolatilityNormalizer()
                ds.target_normalizers[sn] = VolatilityNormalizer()
        for s_idx, sn in enumerate(train_ds.stock_names):
            feats, targs = [], []
            for x_har, _adj, _xm, _mm, y in train_ds.sequences:
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
    print("[create_pure_market_dataloaders] COMPLETE")
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
