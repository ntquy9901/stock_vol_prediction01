"""PatchTST-specific tunable constants for the 2026-09-04 PatchTST+GAT baseline.

Single source of truth for the PatchTST-only hyperparameters that do NOT exist in the shared
``submission/soict_lstm_gat/pipeline_config.py`` (which must not be edited from a baseline). Every
SHARED tunable (hidden, heads, dropout, lr, weight_decay, grad_clip, epochs, patience, seeds,
floors, windows, lookback, edge params) still comes from ``pipeline_config`` via ``Config`` /
``training_config`` — those are NOT redefined here.

Defaults are parsimony-biased (small d_model, depth=2, few patches) because the target panel is a
small VN universe on which transformers overfit easily, and the standing project finding is that HAR
is very hard to beat (see requirements §6). ``PATCH_LEN=6, STRIDE=4`` tile the experiment
``lookback=22`` exactly so the most recent day sits inside the last patch (no trailing days dropped);
see design §2.
"""
from __future__ import annotations

from dataclasses import dataclass

# ============================ PATCHTST HYPERPARAMETERS (NEW tunables) ============================
PATCH_LEN: int = 6        # length of each subseries patch
STRIDE: int = 4           # patch stride (overlap = PATCH_LEN - STRIDE)
D_MODEL: int = 64         # transformer model width (per patch token)
N_HEADS: int = 4          # multi-head self-attention heads (D_MODEL % N_HEADS == 0)
DEPTH: int = 2            # number of TransformerEncoder layers
FF_DIM: int = 128         # feed-forward hidden dim (2*D_MODEL; smaller than the usual 4x to curb overfit)
POOL: str = "flatten"     # patch-token pooling: "flatten" (faithful PatchTST) | "mean" (parsimony lever)


@dataclass(frozen=True)
class PatchTSTHParams:
    """Bundle of the PatchTST-only knobs, defaulting to the module constants above."""
    patch_len: int = PATCH_LEN
    stride: int = STRIDE
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    depth: int = DEPTH
    ff_dim: int = FF_DIM
    pool: str = POOL

    def __post_init__(self):
        if self.patch_len < 1 or self.stride < 1:
            raise ValueError(f"patch_len/stride must be >= 1; got {self.patch_len}/{self.stride}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.pool not in ("flatten", "mean"):
            raise ValueError(f"pool must be 'flatten' or 'mean'; got {self.pool!r}")


def num_patches(seq_len: int, patch_len: int = PATCH_LEN, stride: int = STRIDE) -> int:
    """Patch count for a lookback of ``seq_len``: floor((seq_len - patch_len)/stride) + 1."""
    if seq_len < patch_len:
        raise ValueError(f"seq_len ({seq_len}) must be >= patch_len ({patch_len})")
    return (seq_len - patch_len) // stride + 1
