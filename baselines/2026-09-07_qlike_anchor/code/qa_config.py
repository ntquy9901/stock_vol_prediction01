"""Config constants for the QLIKE-loss / HAR-X-anchor experiment (single source of truth; no scattered
literals). Training hyperparameters come from the delivered ``training_config``; only the anchor clip and
the OOF-HAR-X split parameters are new here."""

ANCHOR_CLIP: float = 0.5        # bound on the residual: yhat = HAR-X * exp(clip(z, -c, c)); |z|<=c
OOF_SPLITS: int = 4            # chronological blocks for the leakage-safe OOF HAR-X (anchor training target)
OOF_WARMUP_FRAC: float = 0.30  # first fraction of train anchors used only to warm the OOF fits
LOSSES = ("mse", "qlike")
ANCHORS = ("none", "harx")
