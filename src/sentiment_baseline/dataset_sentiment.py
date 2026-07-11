"""ISOLATED sentiment-augmented dataset (subclass of existing - no edits to original).

Adds 2 daily sentiment features to the 3 HAR features -> 5 features total.
Reuses the parent's __getitem__ / normalization (VolatilityNormalizer is scalar,
so 5 features work identically to 3).

Imports helpers from src.lstm_gat_hybrid.dataset_with_graph_method (read-only use).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.lstm_gat_hybrid.dataset_presplit import MultiStockDatasetWithPreSplitData
from src.lstm_gat_hybrid.dataset_with_graph_method import (
    _load_raw_stock_data,
    _split_raw_data_by_date,
    _generate_har_for_split,
)
from src.common.data_normalization import VolatilityNormalizer

# 2 daily sentiment features (user-requested simple design: no rolling, no flag)
SENTIMENT_COLS = ['sentiment_1d', 'news_count_1d']
# Scale sentiment to HAR-volatility magnitude (~1e-3) so the shared scalar
# feature normalizer pools comparable-scale features. Heuristic, baseline-only.
SENTIMENT_SCALE = {'sentiment_1d': 0.005, 'news_count_1d': 0.0005}
FEATURE_COLS = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'] + SENTIMENT_COLS


class MultiStockDatasetWithSentiment(MultiStockDatasetWithPreSplitData):
    """Subclass: merges daily sentiment columns then builds sequences with FEATURE_COLS."""

    def __init__(self, *args, sentiment_dir=None, **kwargs):
        self._sentiment_dir = sentiment_dir
        self._sentiment_merged = False
        # parent __init__ calls _create_sequences -> our override (sentiment merged there)
        super().__init__(*args, **kwargs)

    def _merge_sentiment(self):
        if self._sentiment_merged:
            return
        sent_dir = Path(self._sentiment_dir) if self._sentiment_dir else None
        for stock in self.stock_names:
            df = self.stock_data_with_har[stock]
            if 'date' not in df.columns:
                df = df.reset_index()  # bring date index (if any) to a column
            df = df.copy()

            if 'date' not in df.columns or sent_dir is None:
                for c in SENTIMENT_COLS:
                    df[c] = 0.0
                self.stock_data_with_har[stock] = df
                continue

            df['date'] = df['date'].astype(str)
            sent_path = sent_dir / f'{stock}_sentiment.csv'
            if sent_path.exists():
                sent = pd.read_csv(sent_path)
                sent['date'] = sent['date'].astype(str)
                df = df.merge(sent[['date'] + SENTIMENT_COLS], on='date', how='left')
                for c in SENTIMENT_COLS:
                    df[c] = df[c].fillna(0.0) * SENTIMENT_SCALE[c]
            else:
                for c in SENTIMENT_COLS:
                    df[c] = 0.0
            self.stock_data_with_har[stock] = df
        self._sentiment_merged = True

    def _create_sequences(self):
        """Same as parent but merges sentiment first and uses FEATURE_COLS (5 cols)."""
        self._merge_sentiment()

        sequences = []
        min_length = min(len(df) for df in self.stock_data_with_har.values())

        all_volatility_list = []
        for stock in self.stock_names:
            vol_data = self.stock_data_with_har[stock]['parkinson_volatility'].values
            vol_data_truncated = vol_data[:min_length]
            all_volatility_list.append(vol_data_truncated)
        all_volatility = np.stack(all_volatility_list, axis=1)

        if self.graph_method == 'knn':
            from src.lstm_gat_hybrid.graph_utils_fixed import DynamicGraphBuilder
            self.graph_builder = DynamicGraphBuilder(self.config)
        elif self.graph_method == 'correlation':
            pass
        else:
            raise ValueError(f"Unknown graph_method: {self.graph_method}")

        for i in range(min_length - self.seq_length - self.forecast_horizon):
            sequence_volatility = all_volatility[i:i + self.seq_length]

            if self.graph_method == 'correlation':
                from src.lstm_gat_hybrid.graph_correlation import construct_correlation_graph
                adj_matrix = construct_correlation_graph(
                    sequence_volatility, corr_threshold=self.graph_threshold)
            elif self.graph_method == 'knn':
                graph_data = {'volatility': sequence_volatility, 'returns': sequence_volatility}
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')

            x_all_stocks = []
            y_all_stocks = []
            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                x_seq = stock_feats[FEATURE_COLS].iloc[i:i + self.seq_length].values
                x_all_stocks.append(x_seq)
                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_target = stock_feats['parkinson_volatility'].iloc[target_idx]
                y_all_stocks.append(y_target)

            x = np.stack(x_all_stocks, axis=1)
            y = np.array(y_all_stocks)
            sequences.append((x, adj_matrix, y))

        return sequences


def create_sentiment_dataloaders(
    data_dir: str,
    sentiment_dir: str,
    seq_length: int = 22,
    forecast_horizon: int = 5,
    graph_method: str = 'knn',
    graph_threshold: float = 0.7,
    k_neighbors: int = 8,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    normalize: bool = True,
    remove_outliers: bool = True,
    n_std: float = 3.0,
    config=None,
):
    """Isolated copy of create_multi_stock_dataloaders_with_graph_method_fixed,
    using MultiStockDatasetWithSentiment + sentiment_dir. No edits to original."""
    print(f"\n[create_sentiment_dataloaders] graph_method={graph_method}, "
          f"sentiment_dir={sentiment_dir}")

    # Step 1: load raw stock data (reuses existing helper)
    stock_data_raw = _load_raw_stock_data(
        data_dir=data_dir, remove_outliers=remove_outliers, n_std=n_std)

    # Step 2: chronological split (reuses existing helper)
    train_raw, val_raw, test_raw, train_end_idx, val_end_idx, min_length = \
        _split_raw_data_by_date(stock_data_raw, train_ratio, val_ratio, test_ratio)

    # Step 3: HAR features separately per split (reuses existing helper)
    train_har = _generate_har_for_split(train_raw, 'train')
    val_har = _generate_har_for_split(val_raw, 'val')
    test_har = _generate_har_for_split(test_raw, 'test')

    # Step 4: common stocks (train-defined)
    common_stocks = sorted(set(train_har.keys()))
    train_har_common = {k: v for k, v in train_har.items() if k in common_stocks}
    train_raw_common = {k: v for k, v in train_raw.items() if k in common_stocks}
    val_har_common = {k: v for k, v in val_har.items() if k in common_stocks}
    val_raw_common = {k: v for k, v in val_raw.items() if k in common_stocks}
    test_har_common = {k: v for k, v in test_har.items() if k in common_stocks}
    test_raw_common = {k: v for k, v in test_raw.items() if k in common_stocks}

    print(f"[sentiment] common stocks: {len(common_stocks)} "
          f"(val={len(val_har_common)}, test={len(test_har_common)})")

    train_dataset = MultiStockDatasetWithSentiment(
        stock_data=train_raw_common, stock_data_with_har=train_har_common,
        stock_names=common_stocks, seq_length=seq_length,
        forecast_horizon=forecast_horizon, graph_method=graph_method,
        graph_threshold=graph_threshold, k_neighbors=k_neighbors,
        normalize=normalize, train_mode=True, config=config,
        sentiment_dir=sentiment_dir,
    )
    val_dataset = MultiStockDatasetWithSentiment(
        stock_data=val_raw_common, stock_data_with_har=val_har_common,
        stock_names=common_stocks, seq_length=seq_length,
        forecast_horizon=forecast_horizon, graph_method=graph_method,
        graph_threshold=graph_threshold, k_neighbors=k_neighbors,
        normalize=normalize, train_mode=False, config=config,
        sentiment_dir=sentiment_dir,
    )
    test_dataset = MultiStockDatasetWithSentiment(
        stock_data=test_raw_common, stock_data_with_har=test_har_common,
        stock_names=common_stocks, seq_length=seq_length,
        forecast_horizon=forecast_horizon, graph_method=graph_method,
        graph_threshold=graph_threshold, k_neighbors=k_neighbors,
        normalize=normalize, train_mode=False, config=config,
        sentiment_dir=sentiment_dir,
    )

    print(f"[sentiment] sequences - train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    # Step 5: fit normalizers on TRAINING data only (scalar normalizer; works for 5 feats)
    if normalize:
        for stock_name in common_stocks:
            train_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            train_dataset.target_normalizers[stock_name] = VolatilityNormalizer()
            val_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            val_dataset.target_normalizers[stock_name] = VolatilityNormalizer()
            test_dataset.feature_normalizers[stock_name] = VolatilityNormalizer()
            test_dataset.target_normalizers[stock_name] = VolatilityNormalizer()

        for stock_idx, stock_name in enumerate(train_dataset.stock_names):
            train_features = []
            train_targets = []
            for seq in train_dataset.sequences:
                x, adj_matrix, y = seq
                train_features.append(x[:, stock_idx, :])
                train_targets.append(y[stock_idx])
            train_features = np.concatenate(train_features, axis=0)
            train_targets = np.array(train_targets)
            train_dataset.feature_normalizers[stock_name].fit(train_features)
            train_dataset.target_normalizers[stock_name].fit(train_targets.reshape(-1, 1))
            val_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            val_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]
            test_dataset.feature_normalizers[stock_name] = train_dataset.feature_normalizers[stock_name]
            test_dataset.target_normalizers[stock_name] = train_dataset.target_normalizers[stock_name]

    # Step 6: dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[create_sentiment_dataloaders] COMPLETE")
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
