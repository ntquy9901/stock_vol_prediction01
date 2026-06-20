"""
TimesNet Model for Volatility Prediction

Adapts TimesNet (ICLR 2023) architecture for cryptocurrency volatility forecasting.
Uses HAR features (daily, weekly, monthly) with temporal embeddings.

Core innovation: 2D convolution for multi-periodicity capture via FFT-based period detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import TimesNetConfig


class FFTPeriodDetector:
    """
    FFT-based period detection for TimesNet

    Detects dominant periods in time series using Fast Fourier Transform.
    TimesNet uses this to reshape 1D sequences into 2D tensors for convolution.
    """

    @staticmethod
    def detect_periods(x, k=2):
        """
        Detect top-k dominant periods using FFT

        Args:
            x: Input tensor [B, T, C] (batch, time, channels)
            k: Number of top periods to detect

        Returns:
            periods: Array of period lengths
            weights: Amplitude weights for each period
        """
        # FFT along time dimension
        xf = torch.fft.rfft(x, dim=1)

        # Calculate mean amplitude across batch and channels
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # Remove DC component

        # Find top-k frequencies
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()

        # Convert frequencies to periods
        periods = x.shape[1] // top_list

        # Ensure periods are in valid range [2, seq_len//2]
        periods = np.clip(periods, 2, x.shape[1] // 2)

        # Get corresponding amplitudes
        weights = abs(xf).mean(-1)[:, top_list]

        return periods, weights


class InceptionConvBlock(nn.Module):
    """
    Inception-style 2D convolution block for TimesNet

    Uses multiple kernel sizes to capture patterns at different scales.
    Adapted from TSLib Inception_Block_V1.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6):
        super(InceptionConvBlock, self).__init__()
        self.num_kernels = num_kernels

        # Create multiple 2D convolutions with different kernel sizes
        kernels = []
        for i in range(num_kernels):
            kernel_size = 2 * i + 1  # [1, 3, 5, 7, 9, 11]
            padding = i
            kernels.append(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, padding=padding)
            )

        self.kernels = nn.ModuleList(kernels)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Apply multiple convolutions and average results

        Args:
            x: Input tensor [B, N, H, W] (batch, channels, height, width)

        Returns:
            Average output across all kernel sizes
        """
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))

        # Average across all kernels
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """
    TimesNet block with period detection and 2D convolution

    Core TimesNet innovation:
    1. Use FFT to detect dominant periods
    2. Reshape 1D sequence to 2D based on period
    3. Apply 2D convolution to capture temporal variations
    4. Aggregate multiple periods adaptively
    """

    def __init__(self, config):
        super(TimesBlock, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.k = config.top_k  # Number of periods to detect

        # Parameter-efficient design with two Inception blocks
        self.conv = nn.Sequential(
            InceptionConvBlock(config.d_model, config.d_ff, num_kernels=config.num_kernels),
            nn.GELU(),
            InceptionConvBlock(config.d_ff, config.d_model, num_kernels=config.num_kernels)
        )

    def forward(self, x):
        """
        Forward pass with multi-periodicity modeling

        Args:
            x: Input tensor [B, T, C] (batch, time, channels)

        Returns:
            Output with multi-scale temporal modeling [B, T, C]
        """
        B, T, N = x.size()

        # Detect dominant periods
        period_list, period_weight = FFTPeriodDetector.detect_periods(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # Pad sequence if not evenly divisible by period
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            # Reshape 1D to 2D: [B, T, C] -> [B, C, T//period, period]
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

            # Apply 2D convolution
            out = self.conv(out)

            # Reshape back to 1D: [B, C, H, W] -> [B, T, C]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # Stack and aggregate with learned weights
        res = torch.stack(res, dim=-1)

        # Softmax weighting for adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # Residual connection
        res = res + x
        return res


class TemporalEmbedding(nn.Module):
    """
    Temporal feature embedding for time information

    Encodes month, day, weekday information for temporal awareness.
    """

    def __init__(self, d_model, dropout=0.1):
        super(TemporalEmbedding, self).__init__()

        # Feature dimensions: month (12), day (31), weekday (7)
        month_size = 13
        day_size = 32
        weekday_size = 7

        self.month_embed = nn.Embedding(month_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_mark):
        """
        Args:
            x_mark: Temporal features [B, T, 3] (month, day, weekday)

        Returns:
            Temporal embeddings [B, T, d_model]
        """
        x_mark = x_mark.long()
        month_x = self.month_embed(x_mark[:, :, 0])
        day_x = self.day_embed(x_mark[:, :, 1])
        weekday_x = self.weekday_embed(x_mark[:, :, 2])

        return self.dropout(month_x + day_x + weekday_x)


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional encoding
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                   -(torch.log(torch.tensor(10000.0)) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]

        Returns:
            Positional encoding [B, T, d_model]
        """
        return self.pe[:, :x.size(1)]


class TimesNetForVolatility(nn.Module):
    """
    TimesNet model adapted for volatility prediction

    Architecture:
    1. HAR feature embedding (Conv1D)
    2. Temporal embedding (month, day, weekday)
    3. Positional encoding
    4. Multiple TimesNet blocks (FFT + 2D conv)
    5. Projection to scalar volatility output

    Input: HAR features [B, seq_len, 3] + temporal features [B, seq_len, 3]
    Output: Predicted volatility [B, 1]
    """

    def __init__(self, config: TimesNetConfig):
        super(TimesNetForVolatility, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.d_model = config.d_model

        # Feature embedding (HAR features via Conv1D)
        self.value_embedding = nn.Conv1d(
            in_channels=config.enc_in,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )

        # Temporal embedding (month, day, weekday)
        self.temporal_embedding = TemporalEmbedding(
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Positional encoding
        self.position_embedding = PositionalEmbedding(
            d_model=config.d_model,
            max_len=config.seq_len + config.pred_len
        )

        # TimesNet blocks
        self.times_blocks = nn.ModuleList([
            TimesBlock(config) for _ in range(config.e_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Output projection (for single-step prediction)
        self.predict_linear = nn.Linear(config.seq_len, config.pred_len + config.seq_len)
        self.volatility_projection = nn.Linear(config.d_model, 1, bias=True)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, x_mark):
        """
        Forward pass for volatility prediction

        Args:
            x: HAR features [B, seq_len, 3] (daily, weekly, monthly)
            x_mark: Temporal features [B, seq_len, 3] (month, day, weekday)

        Returns:
            Predicted volatility [B, 1]
        """
        # Feature embedding
        x = self.value_embedding(x.permute(0, 2, 1)).transpose(1, 2)  # [B, T, d_model]

        # Add temporal and positional embeddings
        x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)
        x = self.dropout(x)

        # Align temporal dimension for prediction
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, seq_len+pred_len, d_model]

        # TimesNet blocks with residual connections
        for block in self.times_blocks:
            x = self.layer_norm(block(x))

        # Extract prediction (take last pred_len timesteps)
        x = x[:, -self.pred_len:, :]

        # Project to scalar volatility
        volatility_pred = self.volatility_projection(x).squeeze(-1)  # [B, pred_len]

        return volatility_pred


def create_timesnet_model(config: TimesNetConfig = None):
    """
    Factory function to create TimesNet model for volatility prediction

    Args:
        config: TimesNetConfig object (uses default if None)

    Returns:
        TimesNetForVolatility model
    """
    if config is None:
        config = TimesNetConfig()

    model = TimesNetForVolatility(config)
    return model
