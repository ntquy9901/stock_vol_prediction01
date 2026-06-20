"""
TimesNet Configuration for Volatility Prediction

Centralized hyperparameters following TimesNet (ICLR 2023) paper adaptations
for cryptocurrency volatility forecasting with HAR features.
"""

class TimesNetConfig:
    """
    TimesNet configuration for volatility prediction

    Architecture parameters adapted from TimesNet paper:
    - seq_len: 22 trading days (1 month) - matches LSTM-HAR Enhanced
    - pred_len: 1 (single-step prediction)
    - enc_in: 3 HAR features (daily, weekly, monthly volatility)
    - d_model: 64 (embedding dimension)
    - num_kernels: 6 (2D convolution kernels for multi-periodicity)

    Training parameters following project standards:
    - num_epochs: 70 (mandatory standard for all models)
    - patience: 15 (mandatory standard for early stopping)
    - learning_rate: 0.001 (conservative, learned from CryptoMamba issues)
    - weight_decay: 1e-5 (L2 regularization)
    """

    # ===== Architecture Parameters =====
    # TimesNet-specific (from ICLR 2023 paper)
    seq_len = 22              # Input sequence length (1 month of trading days)
    pred_len = 1              # Prediction length (single-step)
    enc_in = 3                # Number of input features (HAR: daily, weekly, monthly)
    d_model = 64              # Embedding dimension
    embed_type = 'timeF'      # Temporal embedding type (timeFeatures)
    freq = 'd'                # Data frequency (daily)
    dropout = 0.1             # Dropout rate
    num_kernels = 6           # Number of 2D convolution kernels (TimesNet core)

    # TimesNet 2D convolution parameters
    top_k = 3                 # Top k kernels for multi-periodicity
    d_ff = 256                # Feed-forward dimension
    activation = 'gelu'       # Activation function
    e_layers = 3              # Number of encoder layers

    # ===== Training Parameters =====
    # Standardized across all models (CLAUDE.md Section 6)
    num_epochs = 70           # Maximum training epochs (MANDATORY standard)
    patience = 15             # Early stopping patience (MANDATORY standard)
    learning_rate = 0.001     # Learning rate (conservative, prevents instability)
    weight_decay = 1e-5       # L2 regularization

    # Training optimization
    batch_size = 32           # Batch size for training
    num_workers = 0           # DataLoader workers (0 for Windows compatibility)
    pin_memory = False        # Pin memory for GPU transfer (False for CPU)

    # ===== Model-Specific Parameters =====
    # Volatility prediction specific
    forecast_horizon = 5      # 5-day ahead forecast (matches project target)

    # Regularization
    gradient_clip = 1.0       # Gradient clipping threshold

    # ===== Early Stopping Parameters =====
    min_epochs = 20           # Minimum epochs before early stopping can trigger
    min_delta = 1e-6          # Minimum change to qualify as improvement

    # ===== Evaluation Parameters =====
    # 6 mandatory metrics
    eval_metrics = ['mse', 'rmse', 'mae', 'r2', 'qlike', 'directional_accuracy']

    # ===== Output Parameters =====
    save_best_only = True     # Only save best model (by validation loss)
    save_learning_curves = True  # Plot learning curves every 10 epochs

    # ===== Device Configuration =====
    device = 'cpu'            # Device for training ('cpu' or 'cuda')

    def __init__(self):
        """Initialize configuration with validation"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate positive parameters
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.pred_len > 0, "pred_len must be positive"
        assert self.enc_in > 0, "enc_in must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

        # Validate ranges
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.patience < self.num_epochs, "patience must be less than num_epochs"

        # Validate TimesNet-specific parameters
        assert self.num_kernels > 0, "num_kernels must be positive"
        assert self.top_k <= self.num_kernels, "top_k must be <= num_kernels"

    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            # Architecture
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'd_model': self.d_model,
            'embed_type': self.embed_type,
            'freq': self.freq,
            'dropout': self.dropout,
            'num_kernels': self.num_kernels,
            'top_k': self.top_k,
            'd_ff': self.d_ff,
            'activation': self.activation,
            'e_layers': self.e_layers,

            # Training
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,

            # Model-specific
            'forecast_horizon': self.forecast_horizon,
            'gradient_clip': self.gradient_clip,

            # Early stopping
            'min_epochs': self.min_epochs,
            'min_delta': self.min_delta,

            # Device
            'device': self.device
        }

    def __repr__(self):
        """String representation of configuration"""
        return f"TimesNetConfig(seq_len={self.seq_len}, pred_len={self.pred_len}, " \
               f"enc_in={self.enc_in}, d_model={self.d_model}, num_kernels={self.num_kernels}, " \
               f"num_epochs={self.num_epochs}, learning_rate={self.learning_rate})"


# Global configuration instance
config = TimesNetConfig()
