"""Market-fallback dataset: returns 7-tuple (x_har, adj, x_emb, mask, x_market, market_mask, y).

Subclass of MultiStockDatasetWithPreSplitData (read-only import). Loads:
  - per-ticker news embeddings (data/sentiment_embedding/{TICKER}_emb.npz) — sparse, stock-specific
  - market news embeddings (data/sentiment_embedding/market_emb.npz) — dense, all articles by date
Pads articles to MAX_ARTICLES (per-stock) and MAX_MARKET (per-day market). Isolated.
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
MAX_ARTICLES = 10   # per-stock cap per day
MAX_MARKET = 15     # market articles per day (vĩ mô days are crowded; percentile 99)


def _norm_date(s):
    """Normalize a date string to YYYY-MM-DD (strip time component). [HIGH-2 lesson]"""
    s = str(s).strip()
    for sep in (' ', 'T'):
        if sep in s:
            s = s.split(sep)[0]
    return s[:10]


class MultiStockDatasetWithMarket(MultiStockDatasetWithPreSplitData):
    """Dataset returning (x_har, adj, x_emb, mask, x_market, market_mask, y).

    x_har      : [seq, stocks, 3]
    x_emb      : [seq, stocks, MAX_ARTICLES, emb_dim]   (per-stock, sparse)
    mask       : [seq, stocks, MAX_ARTICLES]
    x_market   : [seq, MAX_MARKET, emb_dim]             (per-day, dense, shared across stocks)
    market_mask: [seq, MAX_MARKET]
    y          : [stocks]
    """

    def __init__(self, *args, emb_dir=None, market_cache_path=None,
                 max_articles=MAX_ARTICLES, max_market=MAX_MARKET, **kwargs):
        self._emb_dir = Path(emb_dir) if emb_dir else None
        self._market_cache_path = Path(market_cache_path) if market_cache_path else None
        self._max_articles = max_articles
        self._max_market = max_market
        self._emb_cache = {}
        self._market = None
        self._emb_dim = None
        super().__init__(*args, **kwargs)   # calls _create_sequences (our override)

    # ---- per-ticker cache (same as embedding baseline) ----
    def _load_emb_for_ticker(self, ticker):
        if ticker in self._emb_cache:
            return self._emb_cache[ticker]
        result = None
        if self._emb_dir is not None:
            path = self._emb_dir / f"{ticker}_emb.npz"
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=False)
                    result = {_norm_date(k): data[k] for k in data.files}
                except Exception as e:
                    print(f"[emb] warn: cannot load {path.name}: {e}")
        self._emb_cache[ticker] = result
        return result

    def _load_market(self):
        if self._market_cache_path is None or not self._market_cache_path.exists():
            print(f"[market] no market cache at {self._market_cache_path} (smoke mode)")
            self._market = {}
            return
        try:
            data = np.load(self._market_cache_path, allow_pickle=False)
            self._market = {_norm_date(k): data[k] for k in data.files}
            print(f"[market] loaded {len(self._market)} dates from {self._market_cache_path.name}")
        except Exception as e:
            print(f"[market] warn: cannot load market cache: {e}")
            self._market = {}

    def _pad_articles(self, arr, max_articles):
        """arr: [n, dim] (or None) -> (padded [max_articles, dim] float32, mask float32).

        [MEDIUM-6 lesson] only set mask=1 for finite rows.
        """
        dim = self._emb_dim or 64
        out = np.zeros((max_articles, dim), dtype=np.float32)
        mask = np.zeros(max_articles, dtype=np.float32)
        if arr is not None and len(arr) > 0:
            if self._emb_dim is not None and arr.shape[1] != self._emb_dim:
                # [HIGH-2] fail loud if per-stock dim != market dim -> would crash MarketBranch forward
                raise ValueError(
                    f"Embedding dim mismatch: cache array dim={arr.shape[1]} but expected "
                    f"_emb_dim={self._emb_dim}. Re-extract caches with consistent --dim.")
            a = arr.astype(np.float32, copy=False)
            finite = np.isfinite(a).all(axis=1)
            j = 0
            for i in range(min(len(a), max_articles)):
                if finite[i]:
                    out[j] = a[i]
                    mask[j] = 1.0
                    j += 1
        return out, mask

    def _create_sequences(self):
        self._total_cells = 0
        self._matched_cells = 0
        self._market_total = 0      # [HIGH-1] market coverage tracking
        self._market_matched = 0
        self._market_truncated = 0  # [MED-5] count days > MAX_M
        for s in self.stock_names:
            self._load_emb_for_ticker(s)
        self._load_market()
        # detect emb_dim from any non-empty cache (per-stock or market)
        for s in self.stock_names:
            d = self._emb_cache.get(s)
            if d:
                for arr in d.values():
                    if arr is not None and len(arr) > 0:
                        self._emb_dim = arr.shape[1]
                        break
            if self._emb_dim:
                break
        if self._emb_dim is None and self._market:
            for arr in self._market.values():
                if arr is not None and len(arr) > 0:
                    self._emb_dim = arr.shape[1]
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
            else:
                graph_data = {'volatility': sequence_volatility, 'returns': sequence_volatility}
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            x_har_all, x_emb_all, mask_all, y_all = [], [], [], []
            window_dates = None
            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                if window_dates is None:
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
                    e, m = self._pad_articles(arr, self._max_articles)
                    day_embs.append(e)
                    day_masks.append(m)
                x_emb_all.append(np.stack(day_embs))
                mask_all.append(np.stack(day_masks))

                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_all.append(stock_feats['parkinson_volatility'].iloc[target_idx])

            # market embeddings per day (shared across all stocks)
            market_day_embs, market_day_masks = [], []
            for d in window_dates:
                marr = self._market.get(d)
                self._market_total += 1
                if marr is not None and len(marr) > 0:
                    self._market_matched += 1
                    if len(marr) > self._max_market:
                        self._market_truncated += 1   # [MED-5]
                me, mm = self._pad_articles(marr, self._max_market)
                market_day_embs.append(me)
                market_day_masks.append(mm)
            x_market = np.stack(market_day_embs)      # [seq, MAX_MARKET, dim]
            market_mask = np.stack(market_day_masks)  # [seq, MAX_MARKET]

            x_har = np.stack(x_har_all, axis=1).astype(np.float32)
            x_emb = np.stack(x_emb_all, axis=1)
            mask = np.stack(mask_all, axis=1)
            y = np.array(y_all, dtype=np.float32)
            sequences.append((x_har, adj_matrix, x_emb, mask, x_market, market_mask, y))

        cov = 100.0 * self._matched_cells / max(1, self._total_cells)
        any_cache = any(self._emb_cache.get(s) for s in self.stock_names)
        print(f"[emb] per-stock date-match coverage: {self._matched_cells}/{self._total_cells} "
              f"cells ({cov:.2f}%)")
        if any_cache and self._matched_cells == 0:
            raise RuntimeError("ZERO per-stock news matches despite caches — date format mismatch?")
        # [HIGH-1] market coverage assert — fail loud if market branch would be silently inert
        m_cov = 100.0 * self._market_matched / max(1, self._market_total)
        print(f"[market] date-match coverage: {self._market_matched}/{self._market_total} "
              f"days ({m_cov:.2f}%)")
        if self._market and self._market_matched == 0:
            raise RuntimeError(
                "ZERO market news matches despite market cache loaded — date format mismatch? "
                "Market branch would be silently inert.")
        if self._market_truncated > 0:   # [MED-5]
            print(f"[market][info] {self._market_truncated} day-entries truncated at "
                  f"MAX_MARKET={self._max_market} (keep first). Consider percentile-based cap.")
        return sequences

    def __getitem__(self, idx):
        x_har, adj, x_emb, mask, x_market, market_mask, y = self.sequences[idx]

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
                torch.FloatTensor(x_emb), torch.FloatTensor(mask),
                torch.FloatTensor(x_market), torch.FloatTensor(market_mask),
                torch.FloatTensor(y))


def create_market_dataloaders(
    data_dir, emb_dir, market_cache_path, seq_length=22, forecast_horizon=5,
    graph_method='knn', graph_threshold=0.7, k_neighbors=8, batch_size=32,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_workers=0, normalize=True,
    remove_outliers=True, n_std=3.0, config=None,
):
    from src.lstm_gat_hybrid.dataset_with_graph_method import (
        _load_raw_stock_data, _split_raw_data_by_date, _generate_har_for_split,
    )
    print(f"\n[create_market_dataloaders] graph={graph_method}, emb_dir={emb_dir}, "
          f"market={market_cache_path}")

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
    print(f"[market] common stocks: {len(common_stocks)}")

    ds_kwargs = dict(seq_length=seq_length, forecast_horizon=forecast_horizon,
                     graph_method=graph_method, graph_threshold=graph_threshold,
                     k_neighbors=k_neighbors, normalize=normalize, config=config,
                     emb_dir=emb_dir, market_cache_path=market_cache_path)
    train_ds = MultiStockDatasetWithMarket(train_raw_c, train_har_c, common_stocks, train_mode=True, **ds_kwargs)
    val_ds = MultiStockDatasetWithMarket(val_raw_c, val_har_c, common_stocks, train_mode=False, **ds_kwargs)
    test_ds = MultiStockDatasetWithMarket(test_raw_c, test_har_c, common_stocks, train_mode=False, **ds_kwargs)
    print(f"[market] sequences - train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[market] detected emb_dim={train_ds._emb_dim}")

    if normalize:
        for sn in common_stocks:
            for ds in (train_ds, val_ds, test_ds):
                ds.feature_normalizers[sn] = VolatilityNormalizer()
                ds.target_normalizers[sn] = VolatilityNormalizer()
        for s_idx, sn in enumerate(train_ds.stock_names):
            feats, targs = [], []
            for x_har, _a, _e, _m, _mk, _mm, y in train_ds.sequences:
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
    print("[create_market_dataloaders] COMPLETE")
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
