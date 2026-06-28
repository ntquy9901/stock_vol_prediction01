"""
Lightweight Dataset Class that accepts pre-computed HAR data
Used for split-first approach to prevent HAR leakage
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict
from src.common.data_normalization import VolatilityNormalizer
from .config import LSTMGATConfig


class MultiStockDatasetWithPreSplitData(Dataset):
    """
    Dataset that uses pre-split HAR data (no data leakage)

    This dataset accepts pre-computed HAR features and raw data,
    bypassing the __init__ data loading to prevent leakage.
    """

    def __init__(
        self,
        stock_data: Dict[str, np.ndarray],
        stock_data_with_har: Dict[str, np.ndarray],
        stock_names: list,
        seq_length: int = 22,
        forecast_horizon: int = 5,
        graph_method: str = 'knn',
        graph_threshold: float = 0.7,
        k_neighbors: int = 8,
        normalize: bool = True,
        train_mode: bool = False,
        config: LSTMGATConfig = None
    ):
        """
        Initialize with pre-split data (no loading from directory)

        Args:
            stock_data: Raw stock data dictionary
            stock_data_with_har: HAR features dictionary
            stock_names: List of stock names
            seq_length: Input sequence length
            forecast_horizon: Forecast horizon
            graph_method: 'correlation' or 'knn'
            graph_threshold: Correlation threshold
            k_neighbors: k neighbors for k-NN
            normalize: Whether to normalize
            train_mode: Training mode flag
            config: Configuration object
        """
        # Use provided pre-split data (NO loading from directory)
        self.stock_data = stock_data
        self.stock_data_with_har = stock_data_with_har
        self.stock_names = stock_names

        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.graph_method = graph_method
        self.graph_threshold = graph_threshold
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        self.train_mode = train_mode
        self.config = config if config else LSTMGATConfig()

        # Initialize normalizers (empty, will fit later)
        self.feature_normalizers = {}
        self.target_normalizers = {}

        # Create sequences IMMEDIATELY from pre-split HAR data
        self.sequences = self._create_sequences()

        print(f"[MultiStockDatasetWithPreSplitData] Created {len(self.sequences)} sequences from PRE-SPLIT data")

    def _create_sequences(self) -> list:
        """Create multi-stock sequences with PER-SEQUENCE graph construction"""
        sequences = []

        # Get minimum length
        min_length = min(len(df) for df in self.stock_data_with_har.values())

        # Prepare volatility data for per-sequence graph construction
        all_volatility_list = []
        for stock in self.stock_names:
            vol_data = self.stock_data_with_har[stock]['parkinson_volatility'].values
            vol_data_truncated = vol_data[:min_length]
            all_volatility_list.append(vol_data_truncated)

        all_volatility = np.stack(all_volatility_list, axis=1)

        # Initialize graph builder if needed
        if self.graph_method == 'knn':
            from .graph_utils_fixed import DynamicGraphBuilder
            self.graph_builder = DynamicGraphBuilder(self.config)
        elif self.graph_method == 'correlation':
            from .graph_correlation import construct_correlation_graph
        else:
            raise ValueError(f"Unknown graph_method: {self.graph_method}")

        # Create sequences with PER-SEQUENCE graph construction
        for i in range(min_length - self.seq_length - self.forecast_horizon):
            # Build graph using ONLY this sequence's data window
            sequence_volatility = all_volatility[i:i+self.seq_length]

            if self.graph_method == 'correlation':
                from .graph_correlation import construct_correlation_graph
                adj_matrix = construct_correlation_graph(
                    sequence_volatility,
                    corr_threshold=self.graph_threshold
                )
            elif self.graph_method == 'knn':
                graph_data = {
                    'volatility': sequence_volatility,
                    'returns': sequence_volatility
                }
                adj_matrix = self.graph_builder.build_graph_from_data(graph_data, 'correlation')
            else:
                raise ValueError(f"Unknown graph_method: {self.graph_method}")

            # Create sequence features
            x_all_stocks = []
            y_all_stocks = []

            for stock_name in self.stock_names:
                stock_feats = self.stock_data_with_har[stock_name]
                x_seq = stock_feats[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+self.seq_length].values
                x_all_stocks.append(x_seq)

                target_idx = i + self.seq_length + self.forecast_horizon - 1
                y_target = stock_feats['parkinson_volatility'].iloc[target_idx]
                y_all_stocks.append(y_target)

            x = np.stack(x_all_stocks, axis=1)
            y = np.array(y_all_stocks)

            sequences.append((x, adj_matrix, y))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sequence"""
        x, adj_matrix, y = self.sequences[idx]

        # DEBUG: Log raw target statistics
        if idx == 0 and not hasattr(self, '_debug_logged'):
            self._debug_logged = 'raw'
            print(f"[DEBUG __getitem__] idx=0, normalize={self.normalize}, num_normalizers={len(self.target_normalizers)}")
            print(f"[DEBUG __getitem__] Raw y mean: {y.mean():.6f}, std: {y.std():.6f}, range: [{y.min():.6f}, {y.max():.6f}]")

        # Normalize if enabled
        if self.normalize:
            x_normalized = np.zeros_like(x)
            for stock_idx in range(x.shape[1]):
                stock_name = self.stock_names[stock_idx]
                if stock_name in self.feature_normalizers:
                    for feat_idx in range(x.shape[2]):
                        x_normalized[:, stock_idx, feat_idx] = \
                            self.feature_normalizers[stock_name].transform(
                                x[:, stock_idx, feat_idx:feat_idx+1]
                            ).flatten()
                else:
                    x_normalized[:, stock_idx, :] = x[:, stock_idx, :]
                    if idx == 0 and hasattr(self, '_debug_logged'):
                        print(f"[DEBUG __getitem__] WARNING: {stock_name} not in feature_normalizers!")

            x = x_normalized

            y_normalized = np.zeros_like(y)
            stocks_normalized = 0
            for stock_idx, stock_name in enumerate(self.stock_names):
                if stock_name in self.target_normalizers:
                    y_normalized[stock_idx] = \
                        self.target_normalizers[stock_name].transform(
                            y[stock_idx:stock_idx+1].reshape(1, -1)
                        ).flatten()[0]
                    stocks_normalized += 1
                else:
                    y_normalized[stock_idx] = y[stock_idx]
                    if idx == 0:
                        print(f"[DEBUG __getitem__] WARNING: {stock_name} not in target_normalizers!")

            y = y_normalized

            # DEBUG: Log normalized target statistics
            if idx == 0:
                print(f"[DEBUG __getitem__] After normalization: {stocks_normalized}/{len(self.stock_names)} stocks normalized")
                print(f"[DEBUG __getitem__] Normalized y mean: {y.mean():.6f}, std: {y.std():.6f}, range: [{y.min():.6f}, {y.max():.6f}]")

        x = torch.FloatTensor(x)
        adj_matrix = torch.FloatTensor(adj_matrix)
        y = torch.FloatTensor(y)

        return x, adj_matrix, y, {}
