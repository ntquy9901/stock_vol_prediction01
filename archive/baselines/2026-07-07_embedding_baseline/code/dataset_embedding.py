"""Embedding dataset: returns (x_har, adj, x_emb, mask, y).

Subclass of MultiStockDatasetWithPreSplitData (read-only import). Loads HAR (3 cols)
+ cached news embeddings (data/sentiment_embedding/{TICKER}_emb.npz), pads articles
to MAX_ARTICLES per (stock, day). Isolated — no edit to src.
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
# [MEDIUM-10] raised from 5 to 10 — covers >99% of days; reduces arbitrary truncation.
# Trade-off: +~2x RAM on x_emb. Set lower if memory-constrained.
MAX_ARTICLES = 10


def _norm_date(s):
    """Normalize a date string to YYYY-MM-DD (strip time component). [HIGH-2 fix]"""
    s = str(s).strip()
    for sep in (' ', 'T'):
        if sep in s:
            s = s.split(sep)[0]
    return s[:10]


class MultiStockDatasetWithEmbedding(MultiStockDatasetWithPreSplitData):
    """Dataset returning (x_har, adj, x_emb, mask, y).

    x_har: [seq, stocks, 3]   (normalized in __getitem__)
    x_emb: [seq, stocks, MAX_ARTICLES, emb_dim]  (NOT normalized — model's proj handles scale)
    mask : [seq, stocks, MAX_ARTICLES]
    y    : [stocks]           (normalized in __getitem__)
    """

    def __init__(self, *args, emb_dir=None, max_articles=MAX_ARTICLES, **kwargs):
        self._emb_dir = Path(emb_dir) if emb_dir else None
        self._max_articles = max_articles
        self._emb_cache = {}
        self._emb_dim = None
        # parent __init__ calls _create_sequences -> our override (cache loaded there)
        super().__init__(*args, **kwargs)

    # ---- embedding cache helpers ----
    def _load_emb_for_ticker(self, ticker):
        if ticker in self._emb_cache:
            return self._emb_cache[ticker]
        result = None
        if self._emb_dir is not None:
            path = self._emb_dir / f"{ticker}_emb.npz"
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=False)
                    # [HIGH-2 fix] normalize cache keys to YYYY-MM-DD (match window_dates)
                    result = {_norm_date(k): data[k] for k in data.files}
                except Exception as e:
                    print(f"[emb] warn: cannot load {path.name}: {e}")
                    result = None
        self._emb_cache[ticker] = result
        return result

    def _pad_articles(self, arr):
        """arr: [n, dim] (or None) -> (padded [MAX, dim] float32, mask [MAX] float32).

        [MEDIUM-6 fix] only set mask=1 for finite rows (skip NaN/Inf articles silently).
        """
        MAX = self._max_articles
        dim = self._emb_dim or 64
        out = np.zeros((MAX, dim), dtype=np.float32)
        mask = np.zeros(MAX, dtype=np.float32)
        if arr is not None and len(arr) > 0:
            a = arr.astype(np.float32, copy=False)
            finite = np.isfinite(a).all(axis=1)   # per-article finiteness
            j = 0
            for i in range(min(len(a), MAX)):
                if finite[i]:
                    out[j] = a[i]
                    mask[j] = 1.0
                    j += 1
                # non-finite article: skipped (mask stays 0)
        return out, mask

    def _create_sequences(self):
        # [HIGH-2 fix] coverage tracking — detect silent zero-embedding
        self._total_cells = 0
        self._matched_cells = 0
        # Load caches + detect emb_dim
        for s in self.stock_names:
            self._load_emb_for_ticker(s)
        for s in self.stock_names:
            d = self._emb_cache.get(s)
            if d:
                for arr in d.values():
                    if arr is not None and len(arr) > 0:
                        self._emb_dim = arr.shape[1]
                        break
            if self._emb_dim:
                break
        if self._emb_dim is None:
            self._emb_dim = 64
            print(f"[emb] no cache found -> dummy emb_dim={self._emb_dim} (smoke mode)")

        sequences = []
        min_length = min(len(df) for df in self.stock_data_with_har.values())

        all_volatility_list = []
        for stock in self.stock_names:
            vol = self.stock_data_with_har[stock]['parkinson_volatility'].values[:min_length]
            all_volatility_list.append(vol)
        all_volatility = np.stack(all_volatility_list, axis=1)

        if self.graph_method == 'knn':
            from src.lstm_gat_hybrid.graph_utils_fixed import DynamicGraphBuilder
            self.graph_builder = DynamicGraphBuilder(self.config)
        elif self.graph_method != 'correlation':
            raise ValueError(f"Unknown graph_method: {self.graph_method}")

        for i in range(min_length - self.seq_length - self.forecast_horizon):
            sequence_volatility = all_volatility[i:i + self.seq_length]
            if self.graph_method == 'correlation':
                from src.lstm_gat_hybrid.graph_correlation import construct_correlation_graph
                adj_matrix = construct_correlation_graph(
                    sequence_volatility, corr_threshold=self.graph_threshold)
            else:  # knn
                graph_data = {'volatility': sequence_volatility, 'returns': sequence_volatility}
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            x_har_all, x_emb_all, mask_all, y_all = [], [], [], []
            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                # [HIGH-2 fix] normalize dates to YYYY-MM-DD to match cache keys
                window_dates = [_norm_date(d) for d in
                                stock_feats['date'].iloc[i:i + self.seq_length].astype(str).tolist()]

                x_har_all.append(stock_feats[HAR_COLS].iloc[i:i + self.seq_length].values)

                emb_cache = self._emb_cache.get(stock_name) or {}
                day_embs, day_masks = [], []
                for d in window_dates:
                    arr = emb_cache.get(d)
                    self._total_cells += 1
                    if arr is not None and len(arr) > 0:
                        self._matched_cells += 1
                    e, m = self._pad_articles(arr)
                    day_embs.append(e)
                    day_masks.append(m)
                x_emb_all.append(np.stack(day_embs))   # [seq, MAX, dim]
                mask_all.append(np.stack(day_masks))   # [seq, MAX]

                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_all.append(stock_feats['parkinson_volatility'].iloc[target_idx])

            x_har = np.stack(x_har_all, axis=1).astype(np.float32)   # [seq, stocks, 3]
            x_emb = np.stack(x_emb_all, axis=1)                       # [seq, stocks, MAX, dim]
            mask = np.stack(mask_all, axis=1)                         # [seq, stocks, MAX]
            y = np.array(y_all, dtype=np.float32)
            sequences.append((x_har, adj_matrix, x_emb, mask, y))

        # [HIGH-2 fix] coverage check — fail loud if news branch would be silently empty
        cov = 100.0 * self._matched_cells / max(1, self._total_cells)
        any_cache = any(self._emb_cache.get(s) for s in self.stock_names)
        print(f"[emb] date-match coverage: {self._matched_cells}/{self._total_cells} "
              f"cells ({cov:.2f}%)")
        if any_cache and self._matched_cells == 0:
            raise RuntimeError(
                "ZERO news-to-stockday matches despite caches loaded — date format mismatch? "
                "News branch would be silently empty. Check _norm_date vs cache keys.")
        return sequences

    def __getitem__(self, idx):
        x_har, adj, x_emb, mask, y = self.sequences[idx]

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
                torch.FloatTensor(x_emb), torch.FloatTensor(mask), torch.FloatTensor(y))


def create_embedding_dataloaders(
    data_dir, emb_dir, seq_length=22, forecast_horizon=5, graph_method='knn',
    graph_threshold=0.7, k_neighbors=8, batch_size=32, train_ratio=0.7,
    val_ratio=0.15, test_ratio=0.15, num_workers=0, normalize=True,
    remove_outliers=True, n_std=3.0, config=None,
):
    """Mirror of create_multi_stock_dataloaders_with_graph_method_fixed, using
    MultiStockDatasetWithEmbedding + emb_dir. No edits to original."""
    from src.lstm_gat_hybrid.dataset_with_graph_method import (
        _load_raw_stock_data, _split_raw_data_by_date, _generate_har_for_split,
    )
    print(f"\n[create_embedding_dataloaders] graph_method={graph_method}, emb_dir={emb_dir}")

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
    print(f"[emb] common stocks: {len(common_stocks)}")

    ds_kwargs = dict(seq_length=seq_length, forecast_horizon=forecast_horizon,
                     graph_method=graph_method, graph_threshold=graph_threshold,
                     k_neighbors=k_neighbors, normalize=normalize, config=config,
                     emb_dir=emb_dir)
    train_ds = MultiStockDatasetWithEmbedding(train_raw_c, train_har_c, common_stocks,
                                              train_mode=True, **ds_kwargs)
    val_ds = MultiStockDatasetWithEmbedding(val_raw_c, val_har_c, common_stocks,
                                            train_mode=False, **ds_kwargs)
    test_ds = MultiStockDatasetWithEmbedding(test_raw_c, test_har_c, common_stocks,
                                             train_mode=False, **ds_kwargs)
    print(f"[emb] sequences - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[emb] detected emb_dim={train_ds._emb_dim}")

    # Fit normalizers on TRAINING data only (scalar; HAR 3 cols)
    if normalize:
        for sn in common_stocks:
            for ds in (train_ds, val_ds, test_ds):
                ds.feature_normalizers[sn] = VolatilityNormalizer()
                ds.target_normalizers[sn] = VolatilityNormalizer()
        for s_idx, sn in enumerate(train_ds.stock_names):
            feats, targs = [], []
            for x_har, _adj, _xemb, _m, y in train_ds.sequences:
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
    print("[create_embedding_dataloaders] COMPLETE")
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
