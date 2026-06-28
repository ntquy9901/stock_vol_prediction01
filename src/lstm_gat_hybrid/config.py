"""
LSTM-GAT Hybrid Configuration

Combines LSTM temporal encoder with Graph Attention Network spatial modeling
for multi-stock volatility forecasting.
"""

class LSTMGATConfig:
    """
    LSTM-GAT Hybrid configuration for volatility prediction

    Architecture:
    - Per-stock LSTM encoder for temporal patterns
    - Dynamic graph construction for cross-stock relationships
    - Multi-head GAT for spatial dependency modeling
    - Fusion strategy for final prediction

    Training follows project standardized hyperparameters.
    """

    # ===== Architecture Parameters =====

    # Data dimensions
    num_stocks = 30              # VN30 stocks
    seq_length = 22              # Input sequence length (1 month)
    forecast_horizon = 5        # 5-day ahead forecast
    num_features_per_stock = 3  # HAR features: daily, weekly, monthly

    # LSTM Encoder (temporal branch)
    lstm_hidden_dim = 64         # LSTM hidden state dimension
    lstm_num_layers = 2         # Number of LSTM layers
    lstm_dropout = 0.2           # LSTM dropout rate

    # Graph Construction
    graph_method = 'correlation'  # 'correlation', 'spillover', 'hybrid'
    correlation_window = 22      # Window for correlation calculation
    top_k_neighbors = 8          # Top-k neighbors for graph construction
    edge_threshold = 0.5         # Correlation threshold for edges

    # Graph Attention Network (spatial branch)
    gat_hidden_dim = 64          # GAT hidden dimension
    gat_num_heads = 4            # Number of attention heads
    gat_num_layers = 2           # Number of GAT layers
    gat_dropout = 0.1           # GAT dropout rate
    gat_alpha = 0.2              # LeakyReLU alpha for attention

    # Fusion Strategy
    fusion_method = 'concat'      # 'concat', 'add', 'attention'
    fusion_hidden_dim = 128      # Hidden dimension for fusion layer
    fusion_dropout = 0.3          # Dropout rate for fusion layers

    # Output projection
    output_dim = 1               # Single volatility prediction per stock

    # ===== Training Parameters (Standardized) =====
    # Project standards - same for all models
    num_epochs = 70              # Maximum training epochs (MANDATORY)
    patience = 15               # Early stopping patience (MANDATORY)
    learning_rate = 0.0005      # Moderate learning rate to balance stability and learning
    weight_decay = 1e-5         # L2 regularization

    # Training optimization
    batch_size = 32              # Batch size
    num_workers = 0              # DataLoader workers (0 for Windows)
    pin_memory = False           # Pin memory for GPU transfer

    # Regularization
    gradient_clip = 1.0          # Gradient clipping threshold

    # ===== Early Stopping Parameters =====
    min_epochs = 20              # Minimum epochs before early stopping
    min_delta = 1e-6            # Minimum change to qualify as improvement

    # ===== Graph-specific Parameters =====
    # Dynamic graph update frequency
    graph_update_freq = 1        # Update graph every N epochs (1 = every epoch)

    # Attention visualization
    save_attention_maps = True   # Save attention weights for analysis
    attention_visualization_freq = 10  # Save every N epochs

    # ===== Device Configuration =====
    device = 'cpu'               # Device for training ('cpu' or 'cuda')

    def __init__(self):
        """Initialize configuration with validation"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate positive parameters
        assert self.num_stocks > 0, "num_stocks must be positive"
        assert self.seq_length > 0, "seq_length must be positive"
        assert self.lstm_hidden_dim > 0, "lstm_hidden_dim must be positive"
        assert self.gat_hidden_dim > 0, "gat_hidden_dim must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

        # Validate ranges
        assert 0 <= self.lstm_dropout <= 1, "lstm_dropout must be between 0 and 1"
        assert 0 <= self.gat_dropout <= 1, "gat_dropout must be between 0 and 1"
        assert self.patience < self.num_epochs, "patience must be less than num_epochs"
        assert self.gat_num_heads > 0, "gat_num_heads must be positive"

        # Validate compatibility
        assert self.gat_hidden_dim % self.gat_num_heads == 0, \
            "gat_hidden_dim must be divisible by gat_num_heads"

    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            # Architecture
            'num_stocks': self.num_stocks,
            'seq_length': self.seq_length,
            'forecast_horizon': self.forecast_horizon,
            'num_features_per_stock': self.num_features_per_stock,

            # LSTM
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_dropout': self.lstm_dropout,

            # Graph
            'graph_method': self.graph_method,
            'correlation_window': self.correlation_window,
            'top_k_neighbors': self.top_k_neighbors,
            'edge_threshold': self.edge_threshold,

            # GAT
            'gat_hidden_dim': self.gat_hidden_dim,
            'gat_num_heads': self.gat_num_heads,
            'gat_num_layers': self.gat_num_layers,
            'gat_dropout': self.gat_dropout,
            'gat_alpha': self.gat_alpha,

            # Fusion
            'fusion_method': self.fusion_method,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'output_dim': self.output_dim,

            # Training
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,

            # Regularization
            'gradient_clip': self.gradient_clip,
            'min_epochs': self.min_epochs,
            'min_delta': self.min_delta,

            # Device
            'device': self.device
        }

    def __repr__(self):
        """String representation of configuration"""
        return f"LSTMGATConfig(num_stocks={self.num_stocks}, " \
               f"lstm_hidden_dim={self.lstm_hidden_dim}, " \
               f"gat_hidden_dim={self.gat_hidden_dim}, " \
               f"gat_num_heads={self.gat_num_heads}, " \
               f"num_epochs={self.num_epochs}, " \
               f"learning_rate={self.learning_rate})"


# Global configuration instance
config = LSTMGATConfig()
