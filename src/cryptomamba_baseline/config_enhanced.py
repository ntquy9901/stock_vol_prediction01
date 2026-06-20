"""
CryptoMamba Configuration (Enhanced V2) - Root Cause Fixes

Enhanced V2 architecture to fix overfitting with proper implementation.
Root cause fixes from adversarial review:
- HAR features will be generated (not assumed to exist)
- Model has ReLU output (prevents negative predictions)
- Performance claims are honest (hypothesis, not promise)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

# Model architecture (enhanced V2 - increased capacity)
CRYPTOMAMBA_CONFIG_ENHANCED = {
    # Architecture (enhanced for capacity)
    'hidden_dim': 64,             # Enhanced: 64 (vs 14 in V2) - 4.6× larger
    'num_layers': 3,              # Enhanced: 3 layers (vs 1 in V2) - 3× deeper
    'd_state': 32,                # Enhanced: 32 (vs 16 in V2) - 2× larger state
    'd_conv': 4,                  # Same: Convolution kernel size
    'expand': 2,                  # Same: Expansion factor
    'dropout': 0.1,               # Enhanced: 0.1 (vs 0.0 in V2) - prevent overfitting

    # Data
    'seq_length': 22,            # Input sequence length (HAR monthly)
    'forecast_horizon': 5,       # 5-day ahead forecast
    'use_volume': False,         # No volume in current processed data

    # Training (FIXED: Data normalization + reasonable learning rate)
    'batch_size': 32,            # Batch size
    'learning_rate': 0.001,     # FIXED: 0.001 (works with normalized data)
    'weight_decay': 0.0005,      # Original: 0.0005 (500× stronger!)
    'num_epochs': 1000,          # Original: 1000-1100 epochs
    'patience': 50,              # Early stopping patience (increased for longer training)
    'min_epochs': 100,           # Minimum epochs before stopping (10% of max)

    # Learning rate scheduling (original)
    'lr_step_size': 100,         # LR decay every 100 epochs
    'lr_gamma': 0.5,             # LR decay factor (halves LR)
}

# Training config
TRAINING_CONFIG_ENHANCED = {
    'num_epochs': CRYPTOMAMBA_CONFIG_ENHANCED['num_epochs'],
    'patience': CRYPTOMAMBA_CONFIG_ENHANCED['patience'],
    'min_epochs': CRYPTOMAMBA_CONFIG_ENHANCED['min_epochs'],
    'learning_rate': CRYPTOMAMBA_CONFIG_ENHANCED['learning_rate'],
    'weight_decay': CRYPTOMAMBA_CONFIG_ENHANCED['weight_decay'],
    'lr_step_size': CRYPTOMAMBA_CONFIG_ENHANCED['lr_step_size'],
    'lr_gamma': CRYPTOMAMBA_CONFIG_ENHANCED['lr_gamma'],
}

# Model config
MODEL_CONFIG_ENHANCED = {
    'num_features': 3 if not CRYPTOMAMBA_CONFIG_ENHANCED['use_volume'] else 4,  # HAR features only
    'hidden_dim': CRYPTOMAMBA_CONFIG_ENHANCED['hidden_dim'],
    'num_layers': CRYPTOMAMBA_CONFIG_ENHANCED['num_layers'],
    'dropout': CRYPTOMAMBA_CONFIG_ENHANCED['dropout'],
    'seq_length': CRYPTOMAMBA_CONFIG_ENHANCED['seq_length'],
    'd_state': CRYPTOMAMBA_CONFIG_ENHANCED['d_state'],
    'd_conv': CRYPTOMAMBA_CONFIG_ENHANCED['d_conv'],
    'expand': CRYPTOMAMBA_CONFIG_ENHANCED['expand'],
}

print("CryptoMamba Configuration ENHANCED Loaded (Root Cause Fixes):")
print(f"  Hidden dim: {CRYPTOMAMBA_CONFIG_ENHANCED['hidden_dim']} (enhanced: 64 vs 14 in V2)")
print(f"  Num layers: {CRYPTOMAMBA_CONFIG_ENHANCED['num_layers']} (enhanced: 3 vs 1 in V2)")
print(f"  D State: {CRYPTOMAMBA_CONFIG_ENHANCED['d_state']} (enhanced: 32 vs 16 in V2)")
print(f"  Dropout: {CRYPTOMAMBA_CONFIG_ENHANCED['dropout']} (enhanced: 0.1 vs 0.0 in V2)")
print(f"  Learning rate: {CRYPTOMAMBA_CONFIG_ENHANCED['learning_rate']} (original)")
print(f"  Weight decay: {CRYPTOMAMBA_CONFIG_ENHANCED['weight_decay']} (original)")
print(f"  Max epochs: {CRYPTOMAMBA_CONFIG_ENHANCED['num_epochs']} (original)")
print(f"  LR scheduling: StepLR (step={CRYPTOMAMBA_CONFIG_ENHANCED['lr_step_size']}, gamma={CRYPTOMAMBA_CONFIG_ENHANCED['lr_gamma']})")
print(f"  Sequence length: {CRYPTOMAMBA_CONFIG_ENHANCED['seq_length']}")
print(f"  Use volume: {CRYPTOMAMBA_CONFIG_ENHANCED['use_volume']}")
print(f"  Num features: {MODEL_CONFIG_ENHANCED['num_features']} (HAR features only)")
print(f"  Expected parameters: 50-100K (vs 2,787 in V2)")
print(f"  Hypothesized Dir Acc: 50-52% (vs 47.78% in V2, requires validation)")

print("\n[!] CRITICAL FIXES APPLIED:")
print("  [+] Model includes ReLU output (ensures non-negative predictions)")
print("  [+] HAR features will be generated from parkinson_volatility")
print("  [+] Performance claims are hypotheses (not guarantees)")
print("  [+] Integration tests will validate actual behavior")
