"""
LSTM-GAT Hybrid Training - Simplified (5 stocks for testing)

Modified version to test with fewer stocks for faster iteration
"""
import sys
sys.path.append('.')

# Modified config for 5 stocks
from src.lstm_gat_hybrid.config import LSTMGATConfig

class LSTMGATSimplifiedConfig(LSTMGATConfig):
    """Simplified config for 5 stocks testing"""

    def __init__(self):
        super().__init__()
        # Override: only use 5 stocks instead of 30
        self.num_stocks = 5
        print(f"[Simplified Config] Using {self.num_stocks} stocks instead of 30 for faster training")

# Monkey patch the config AND the dataset
import src.lstm_gat_hybrid.dataset as dataset_module
import src.lstm_gat_hybrid.config as config_module

# Patch config
config_module.LSTMGATConfig = LSTMGATSimplifiedConfig

# Save original class
OriginalMultiStockDataset = dataset_module.MultiStockDataset

# Create patched dataset that limits to 5 stocks
class SimplifiedMultiStockDataset(OriginalMultiStockDataset):
    def __init__(self, *args, **kwargs):
        # Call parent init first
        super().__init__(*args, **kwargs)

        # Limit to first 5 stocks
        print(f"[Simplified Dataset] Limiting from {len(self.stock_names)} to 5 stocks...")
        self.stock_names = self.stock_names[:5]

        # Update stock data dictionaries to only include first 5 stocks
        limited_stock_data = {}
        limited_stock_data_with_har = {}

        for stock_name in self.stock_names:
            limited_stock_data[stock_name] = self.stock_data[stock_name]
            limited_stock_data_with_har[stock_name] = self.stock_data_with_har[stock_name]

        self.stock_data = limited_stock_data
        self.stock_data_with_har = limited_stock_data_with_har

        # Recreate graph builder with limited stocks
        from src.lstm_gat_hybrid.config import LSTMGATConfig
        simplified_config = LSTMGATConfig()
        simplified_config.num_stocks = len(self.stock_names)  # Should be 5
        self.graph_builder = dataset_module.DynamicGraphBuilder(simplified_config)

        # Recreate sequences with limited stocks
        print(f"[Simplified Dataset] Recreating sequences with {len(self.stock_names)} stocks...")
        self.sequences = self._create_sequences()

        # Recreate normalizers
        self.feature_normalizers = {}
        self.target_normalizers = {}
        if self.normalize:
            self._initialize_normalizers()

        print(f"[Simplified Dataset] Final: {len(self.stock_names)} stocks, {len(self.sequences)} sequences")

# Replace dataset class
dataset_module.MultiStockDataset = SimplifiedMultiStockDataset

# Now import train
from src.lstm_gat_hybrid.train import train_lstm_gat_hybrid

print("="*80)
print("LSTM-GAT HYBRID - SIMPLIFIED VERSION (5 STOCKS)")
print("="*80)
print("This version uses only 5 stocks for faster training and testing.")
print("Once validated, we can scale up to all 30 stocks.")
print("="*80)

# Train with simplified config
results = train_lstm_gat_hybrid()
