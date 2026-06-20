"""
LSTM Architecture Visualization

Visualize the Simple LSTM model architecture for review.

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lstm_baseline.model import SimpleVolatilityLSTM


def visualize_model_architecture(model, save_path='lstm_architecture.png'):
    """
    Visualize LSTM model architecture.

    Args:
        model: SimpleVolatilityLSTM instance
        save_path: Path to save the architecture diagram
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Simple LSTM Architecture for Volatility Prediction',
            fontsize=16, fontweight='bold', ha='center')

    # Colors
    color_input = '#E3F2FD'  # Light blue
    color_lstm = '#FFF3E0'   # Light orange
    color_output = '#E8F5E9'  # Light green
    color_arrow = '#1976D2'  # Blue

    # Input Layer
    input_box = FancyBboxPatch((0.5, 7), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7.6, 'Input Layer', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, '(batch, 22, 1)', fontsize=10, ha='center', style='italic')

    # LSTM Layer
    lstm_box = FancyBboxPatch((3.5, 7), 2.5, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_lstm, linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(4.75, 7.6, 'LSTM Layer', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, 7.2, '1 layer, 32 hidden', fontsize=10, ha='center', style='italic')

    # Extract Last Timestep
    extract_box = FancyBboxPatch((3.5, 5.5), 2.5, 1, boxstyle="round,pad=0.1",
                                 edgecolor='gray', facecolor='#F5F5F5', linewidth=1.5)
    ax.add_patch(extract_box)
    ax.text(4.75, 6, 'Extract Last', fontsize=11, ha='center')
    ax.text(4.75, 5.7, 'Timestep', fontsize=11, ha='center')

    # Output Layer (FC)
    output_box = FancyBboxPatch((3.5, 4), 2.5, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(output_box)
    ax.text(4.75, 4.6, 'FC Layer', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, 4.2, '32 -> 1', fontsize=10, ha='center', style='italic')

    # ReLU Activation
    relu_box = FancyBboxPatch((3.5, 2.5), 2.5, 1, boxstyle="round,pad=0.1",
                              edgecolor='gray', facecolor='#F5F5F5', linewidth=1.5)
    ax.add_patch(relu_box)
    ax.text(4.75, 3, 'ReLU', fontsize=11, fontweight='bold', ha='center')
    ax.text(4.75, 2.7, '(x -> max(0, x))', fontsize=9, ha='center', style='italic')

    # Output
    final_output = FancyBboxPatch((3.5, 1), 2.5, 1, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='#FFEBEE', linewidth=2)
    ax.add_patch(final_output)
    ax.text(4.75, 1.5, 'Output', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, 1.2, '(batch, 1)', fontsize=10, ha='center', style='italic')

    # Arrows
    def add_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               mutation_scale=20, linewidth=2, color=color_arrow)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.15, label, fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    add_arrow(2.5, 7.6, 3.5, 7.6, '22 timesteps')
    add_arrow(4.75, 7, 4.75, 6.5, 'last output')
    add_arrow(4.75, 5.5, 4.75, 5.2, 'h_t')
    add_arrow(4.75, 4, 4.75, 3.5, 'linear')
    add_arrow(4.75, 2.5, 4.75, 2, 'activation')

    # LSTM Details Box
    details_box = FancyBboxPatch((7, 7), 2.5, 3.5, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(details_box)
    ax.text(8.25, 9.8, 'LSTM Cell Details', fontsize=12, fontweight='bold', ha='center')

    details_text = 'Input: 1 feature\n'
    details_text += 'Hidden: 32 units\n'
    details_text += 'Layers: 1\n'
    details_text += 'Direction: Forward\n'
    details_text += 'Batch_first: True\n\n'
    details_text += 'Parameters:\n'
    details_text += f'{sum(p.numel() for p in model.parameters()):,} total'

    ax.text(8.25, 8.5, details_text, fontsize=10, ha='left', va='top',
            family='monospace')

    # Data Flow Box
    flow_box = FancyBboxPatch((7, 3), 2.5, 3.5, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(flow_box)
    ax.text(8.25, 6.3, 'Data Flow', fontsize=12, fontweight='bold', ha='center')

    flow_text = '1. Load 22-day window\n'
    flow_text += '2. Scale features\n'
    flow_text += '3. LSTM processes\n'
    flow_text += '   sequence\n'
    flow_text += '4. Extract last\n'
    flow_text += '   hidden state\n'
    flow_text += '5. Linear projection\n'
    flow_text += '6. ReLU (non-negative)\n'
    flow_text += '7. Output prediction'

    ax.text(8.25, 5.5, flow_text, fontsize=10, ha='left', va='top', family='monospace')

    # Loss & Optimization
    loss_box = FancyBboxPatch((0.5, 1), 2, 1.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#FFCCBC', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(1.5, 2.5, 'Training Config', fontsize=11, fontweight='bold', ha='center')

    training_text = 'Loss: MSE\n'
    training_text += 'Optimizer: Adam\n'
    training_text += 'LR: 0.001\n'
    training_text += 'Batch: 64\n'
    training_text += 'Precision: FP16'

    ax.text(1.5, 2.1, training_text, fontsize=9, ha='center', va='top', family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Architecture diagram saved to: {save_path}")

    return fig


def print_model_summary(model):
    """Print detailed model summary."""
    print("\n" + "=" * 80)
    print("LSTM MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)

    print("\n[Model Configuration]")
    print(f"  Input: (batch_size, seq_length=22, 1)")
    print(f"  LSTM: 1 layer, 32 hidden units")
    print(f"  Output: (batch_size, 1)")

    print("\n[Layer Details]")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\n[Parameter Count]")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n[Memory Estimate]")
    # Approximate memory per batch
    batch_size = 64
    seq_len = 22
    hidden_size = 32

    input_memory = batch_size * seq_len * 1 * 4  # float32
    lstm_memory = 4 * batch_size * hidden_size * hidden_size  # 4 gates
    output_memory = batch_size * 1 * 4

    total_mb = (input_memory + lstm_memory + output_memory) / (1024**2)
    print(f"  Per batch (64 samples): ~{total_mb:.2f} MB")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("LSTM ARCHITECTURE VISUALIZATION")
    print("=" * 80)

    # Initialize model
    model = SimpleVolatilityLSTM(hidden_size=32)

    # Print summary
    print_model_summary(model)

    # Create visualization
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'lstm_architecture.png')
    visualize_model_architecture(model, save_path)

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
