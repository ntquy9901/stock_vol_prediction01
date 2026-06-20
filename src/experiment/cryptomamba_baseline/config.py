"""
CryptoMamba Configuration

Simplified hyperparameters for Phase 1 proof-of-concept.
Based on CryptoMamba paper with adaptations for volatility prediction.

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

# Model architecture
CRYPTOMAMBA_CONFIG = {
    # Architecture
    'hidden_dim': 64,            # Hidden dimension (moderate size)
    'num_layers': 3,             # Number of Mamba blocks
    'd_state': 16,               # State dimension (default from paper)
    'd_conv': 4,                 # Convolution kernel size
    'expand': 2,                 # Expansion factor
    'dropout': 0.1,              # Dropout rate

    # Data
    'seq_length': 22,            # Input sequence length (HAR monthly)
    'forecast_horizon': 5,       # 5-day ahead forecast
    'use_volume': False,         # No volume in current processed data

    # Training
    'batch_size': 32,            # Batch size
    'learning_rate': 0.001,     # Learning rate (same as LSTM fix)
    'weight_decay': 1e-6,       # Weight decay (reduced regularization)
    'num_epochs': 100,           # Maximum epochs
    'patience': 10,              # Early stopping patience
    'min_epochs': 15,            # Minimum epochs before stopping

    # Evaluation
    'val_split': 0.15,           # Validation split ratio
    'test_split': 0.15,          # Test split ratio
}

# Training config (for easy access)
TRAINING_CONFIG = {
    'num_epochs': CRYPTOMAMBA_CONFIG['num_epochs'],
    'patience': CRYPTOMAMBA_CONFIG['patience'],
    'min_epochs': CRYPTOMAMBA_CONFIG['min_epochs'],
    'learning_rate': CRYPTOMAMBA_CONFIG['learning_rate'],
    'weight_decay': CRYPTOMAMBA_CONFIG['weight_decay'],
}

# Model config
MODEL_CONFIG = {
    'num_features': 3 if not CRYPTOMAMBA_CONFIG['use_volume'] else 4,  # HAR features only
    'hidden_dim': CRYPTOMAMBA_CONFIG['hidden_dim'],
    'num_layers': CRYPTOMAMBA_CONFIG['num_layers'],
    'dropout': CRYPTOMAMBA_CONFIG['dropout'],
    'seq_length': CRYPTOMAMBA_CONFIG['seq_length'],
}

print("CryptoMamba Configuration Loaded:")
print(f"  Hidden dim: {CRYPTOMAMBA_CONFIG['hidden_dim']}")
print(f"  Num layers: {CRYPTOMAMBA_CONFIG['num_layers']}")
print(f"  Dropout: {CRYPTOMAMBA_CONFIG['dropout']}")
print(f"  Learning rate: {CRYPTOMAMAMBA_CONFIG['learning_rate']}")
print(f"  Sequence length: {CRYPTOMAMBA_CONFIG['seq_length']}")
print(f"  Use volume: {CRYPTOMAMBA_CONFIG['use_volume']}")
print(f"  Num features: {MODEL_CONFIG['num_features']} (HAR features only)")
