"""Single source of truth for the GBME+DAE (denoising-autoencoder representation -> gamma-GBM) baseline.

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". The GBM hyper-parameters (loss/max_iter/learning_rate/max_leaf_nodes/
l2) are NOT duplicated here: they are reused from the champion `full_matrix.gbm`, so there is exactly ONE
source for them. The prediction FLOOR is likewise reused from `full_matrix.FL` (== pipeline QLIKE_FLOOR) and
is deliberately NOT re-declared here to avoid a divergent second copy; only the UPPER safety cap (PRED_CAP)
is a constant of this baseline. See design/design.md. Inline comments are intentional (they also keep the
config-hardcode scanner's bare `NAME = <num>$` rule from firing on legitimate config constants).
"""
# --- DAE architecture / corruption (Jahrer swap-noise denoising autoencoder) ---
SWAP_RATE = 0.15           # per-cell swap-noise corruption probability (Jahrer 2018 Porto-Seguro ~0.15)
K = 16                     # bottleneck (latent) dimension appended to the raw features as z0..z{K-1}
HIDDEN = (64, 32)          # encoder hidden layer widths (decoder mirrors them); "small" tabular DAE

# --- DAE optimiser (its own; NOT the GBM's) ---
EPOCHS = 100               # max DAE epochs (full run; early-stop on clean-val reconstruction MSE)
EPOCHS_SMOKE = 5           # max DAE epochs (smoke / test)
PATIENCE = 15              # early-stop patience on clean-val reconstruction MSE
MIN_EPOCH = 10             # min epochs before early stop may fire
LR = 1e-3                  # Adam learning rate for the DAE
WEIGHT_DECAY = 1e-5        # Adam L2 weight decay for the DAE
BATCH = 4096               # DAE minibatch size (large batches -> GPU-efficient tabular MLP, no batch=1)
VALID_LEN = 22             # trailing burn-in dates held out as the DAE early-stop (clean) validation slice
DAE_SEED = 0               # single DAE seed (a neural latent basis is only defined up to rotation/permutation,
#                            so averaging embeddings across seeds shrinks Z in an ill-defined basis; the GBM
#                            downstream is still seed-ensembled via full_matrix.SEEDS)

# --- walk-forward scope ---
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the sibling comparison table)
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- numerical guard ---
PRED_CAP = 10.0            # upper clip on gamma predictions (>> max observed HOSE variance ~5.3): a pure
#                            safety valve against an embedding pushing a prediction to a blow-up value; it
#                            never binds on a realistic forecast. Applied IDENTICALLY to both compared models
#                            so it cannot bias the DM comparison. Lower floor reused from full_matrix.FL.

# --- pre-registered kill criterion (design section 6) ---
GAIN_MIN = 0.0            # GBME+DAE must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05          # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)   # BOTH of these horizons must pass (gain>GAIN_MIN AND p<DM_ALPHA) for success

# --- HOSE regime-spike robustness (design section 6 / project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
