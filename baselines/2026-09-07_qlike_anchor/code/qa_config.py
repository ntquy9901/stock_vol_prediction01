"""Config constants for the QLIKE-loss / HAR-X-anchor experiment (single source of truth; no scattered
literals). Training hyperparameters come from the delivered ``training_config``; only the anchor clip and
the OOF-HAR-X split parameters are new here."""

ANCHOR_CLIP: float = 0.5        # bound on the residual: yhat = HAR-X * exp(clip(z, -c, c)); |z|<=c
# The anchor's OOF HAR-X (leakage-safe residual target) reuses xgb_oof, whose split/warmup/eps live in
# xgb_config -- the single source of truth. Do NOT redeclare them here (avoids silent config drift).
LOSSES = ("mse", "qlike")
ANCHORS = ("none", "harx")
