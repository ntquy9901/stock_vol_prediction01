"""
CryptoMamba Configuration (Full) - Original Hyperparameters with mamba_ssm

Based on original CryptoMamba paper implementation using official mamba_ssm package.
Optimized for volatility prediction with HAR features.

Key features:
- Full selective state space mechanism (via mamba_ssm)
- Original hyperparameters: LR=0.01, epochs=1000, weight_decay=0.0005
- Hierarchical architecture: hidden_dims=[14, 1]
- 136K parameters (vs 2,787 in simplified V2)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

# Model architecture (full - matches original CryptoMamba with mamba_ssm)
CRYPTOMAMBA_CONFIG_FULL = {
    # Architecture (original settings with hierarchical hidden_dims)
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
    'learning_rate': 0.01,      # Original: 0.01 (10× higher than conservative!)
    'weight_decay': 0.0005,      # Original: 0.0005 (500× stronger!)
    'num_epochs': 1000,          # Original: 1000-1100 epochs
    'patience': 50,              # Early stopping patience (increased for longer training)
    'min_epochs': 100,           # Minimum epochs before stopping (10% of max)

    # Learning rate scheduling (original)
    'lr_step_size': 100,         # LR decay every 100 epochs
    'lr_gamma': 0.5,             # LR decay factor (halves LR)
}

# Training config
TRAINING_CONFIG_FULL = {
    'num_epochs': CRYPTOMAMBA_CONFIG_FULL['num_epochs'],
    'patience': CRYPTOMAMBA_CONFIG_FULL['patience'],
    'min_epochs': CRYPTOMAMBA_CONFIG_FULL['min_epochs'],
    'learning_rate': CRYPTOMAMBA_CONFIG_FULL['learning_rate'],
    'weight_decay': CRYPTOMAMBA_CONFIG_FULL['weight_decay'],
    'lr_step_size': CRYPTOMAMBA_CONFIG_FULL['lr_step_size'],
    'lr_gamma': CRYPTOMAMBA_CONFIG_FULL['lr_gamma'],
}

# Model config
MODEL_CONFIG_FULL = {
    'num_features': 3 if not CRYPTOMAMBA_CONFIG_FULL['use_volume'] else 4,  # HAR features only
    'hidden_dim': CRYPTOMAMBA_CONFIG_FULL['hidden_dim'],
    'num_layers': CRYPTOMAMBA_CONFIG_FULL['num_layers'],
    'dropout': CRYPTOMAMBA_CONFIG_FULL['dropout'],
    'seq_length': CRYPTOMAMBA_CONFIG_FULL['seq_length'],
    'd_state': CRYPTOMAMBA_CONFIG_FULL['d_state'],
    'd_conv': CRYPTOMAMBA_CONFIG_FULL['d_conv'],
    'expand': CRYPTOMAMBA_CONFIG_FULL['expand'],
}

print("CryptoMamba Configuration FULL Loaded (Original Hyperparameters with mamba_ssm):")
print(f"  Hidden dim: {CRYPTOMAMBA_CONFIG_FULL['hidden_dim']} (original: 14)")
print(f"  Num layers: {CRYPTOMAMBA_CONFIG_FULL['num_layers']} (hierarchical)")
print(f"  Dropout: {CRYPTOMAMBA_CONFIG_FULL['dropout']} (original: 0.0)")
print(f"  Learning rate: {CRYPTOMAMBA_CONFIG_FULL['learning_rate']} (original!)")
print(f"  Weight decay: {CRYPTOMAMBA_CONFIG_FULL['weight_decay']} (original!)")
print(f"  Max epochs: {CRYPTOMAMBA_CONFIG_FULL['num_epochs']} (original!)")
print(f"  LR scheduling: StepLR (step={CRYPTOMAMBA_CONFIG_FULL['lr_step_size']}, gamma={CRYPTOMAMBA_CONFIG_FULL['lr_gamma']})")
print(f"  Sequence length: {CRYPTOMAMBA_CONFIG_FULL['seq_length']}")
print(f"  Use volume: {CRYPTOMAMBA_CONFIG_FULL['use_volume']}")
print(f"  Num features: {MODEL_CONFIG_FULL['num_features']} (HAR features only)")
print(f"  Expected parameters: ~136K (full selective scan)")
