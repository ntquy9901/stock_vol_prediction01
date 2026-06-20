"""
CryptoMamba Configuration (v2) - Original Hyperparameters

Based on original CryptoMamba paper implementation.
Optimized for volatility prediction with HAR features.

Key changes from v1:
- Learning rate: 0.001 → 0.01 (10× higher)
- Max epochs: 100 → 1000 (10× longer)
- Weight decay: 1e-6 → 0.0005 (500× stronger)
- Added learning rate scheduling
- Hidden dim: 64 → 14 (original setting)
- Dropout: 0.1 → 0.0 (original uses 0.0)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

# Model architecture (v2 - matches original CryptoMamba)
CRYPTOMAMBA_CONFIG_V2 = {
    # Architecture (original settings)
    'hidden_dim': 14,            # Original: hidden_dims=[14, 1]
    'num_layers': 1,             # Single hierarchical transition
    'd_state': 16,               # State dimension (original)
    'd_conv': 4,                 # Convolution kernel size (original)
    'expand': 2,                 # Expansion factor (original)
    'dropout': 0.0,              # Original: 0.0 (no dropout)

    # Data
    'seq_length': 22,            # Input sequence length (HAR monthly)
    'forecast_horizon': 5,       # 5-day ahead forecast
    'use_volume': False,         # No volume in current processed data

    # Training (original CryptoMamba settings)
    'batch_size': 32,            # Batch size
    'learning_rate': 0.01,      # Original: 0.01 (10× higher than v1!)
    'weight_decay': 0.0005,      # Original: 0.0005 (500× stronger!)
    'num_epochs': 1000,          # Original: 1000-1100 epochs
    'patience': 50,              # Early stopping patience (increased for longer training)
    'min_epochs': 100,           # Minimum epochs before stopping (10% of max)

    # Learning rate scheduling (original)
    'lr_step_size': 100,         # LR decay every 100 epochs
    'lr_gamma': 0.5,             # LR decay factor (halves LR)
}

# Training config
TRAINING_CONFIG_V2 = {
    'num_epochs': CRYPTOMAMBA_CONFIG_V2['num_epochs'],
    'patience': CRYPTOMAMBA_CONFIG_V2['patience'],
    'min_epochs': CRYPTOMAMBA_CONFIG_V2['min_epochs'],
    'learning_rate': CRYPTOMAMBA_CONFIG_V2['learning_rate'],
    'weight_decay': CRYPTOMAMBA_CONFIG_V2['weight_decay'],
    'lr_step_size': CRYPTOMAMBA_CONFIG_V2['lr_step_size'],
    'lr_gamma': CRYPTOMAMBA_CONFIG_V2['lr_gamma'],
}

# Model config
MODEL_CONFIG_V2 = {
    'num_features': 3 if not CRYPTOMAMBA_CONFIG_V2['use_volume'] else 4,  # HAR features only
    'hidden_dim': CRYPTOMAMBA_CONFIG_V2['hidden_dim'],
    'num_layers': CRYPTOMAMBA_CONFIG_V2['num_layers'],
    'dropout': CRYPTOMAMBA_CONFIG_V2['dropout'],
    'seq_length': CRYPTOMAMBA_CONFIG_V2['seq_length'],
    'd_state': CRYPTOMAMBA_CONFIG_V2['d_state'],
    'd_conv': CRYPTOMAMBA_CONFIG_V2['d_conv'],
    'expand': CRYPTOMAMBA_CONFIG_V2['expand'],
}

print("CryptoMamba Configuration V2 Loaded (Original Hyperparameters):")
print(f"  Hidden dim: {CRYPTOMAMBA_CONFIG_V2['hidden_dim']} (original: 14)")
print(f"  Num layers: {CRYPTOMAMBA_CONFIG_V2['num_layers']} (hierarchical)")
print(f"  Dropout: {CRYPTOMAMBA_CONFIG_V2['dropout']} (original: 0.0)")
print(f"  Learning rate: {CRYPTOMAMBA_CONFIG_V2['learning_rate']} (10× higher!)")
print(f"  Weight decay: {CRYPTOMAMBA_CONFIG_V2['weight_decay']} (500× stronger!)")
print(f"  Max epochs: {CRYPTOMAMBA_CONFIG_V2['num_epochs']} (10× longer!)")
print(f"  LR scheduling: StepLR (step={CRYPTOMAMBA_CONFIG_V2['lr_step_size']}, gamma={CRYPTOMAMBA_CONFIG_V2['lr_gamma']})")
print(f"  Sequence length: {CRYPTOMAMBA_CONFIG_V2['seq_length']}")
print(f"  Use volume: {CRYPTOMAMBA_CONFIG_V2['use_volume']}")
print(f"  Num features: {MODEL_CONFIG_V2['num_features']} (HAR features only)")
