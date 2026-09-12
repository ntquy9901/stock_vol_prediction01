"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

Constants replicate the Complexity-2026 complex-network method (Nguyen et al., DOI 10.1155/cplx/5670093),
aligned to the project's 22-trading-day month convention. See
``docs/reports/2026-09-12_complex_network_gbm_design.md`` for the full design and every deviation from the
paper (window sizes, volume proxy).
"""
from pathlib import Path

# repo root: code/ -> 2026-09-12_complex_network -> baselines -> repo
REPO = Path(__file__).resolve().parents[3]

WIN = 66            # trailing correlation window DeltaT = 3 months x 22 trading days (paper used 125 ~ 6 months)
WIN_ROBUST = 132    # 6-month robustness window = 6 x 22 (matches the paper's ~6-month DeltaT)
STEP = 22           # slide the window by one month = 22 trading days
L = 22              # future target horizon = one project-month (paper eq. 2-4 used L = 21)
THR = 0.5           # threshold tau on |combined corr| (paper "Threshold" network; best for volatility)
ALPHA = 0.7         # combined = ALPHA*return-corr + (1-ALPHA)*volume-corr (paper's optimal alpha for volatility)
ALPHA_GRID = (0.0, 0.5, 0.7, 1.0)   # robustness grid over the blend weight (paper Table 1 structure)
MIN_COMMON_TICKERS = 20             # minimum tickers present in BOTH return and volume windows
WIN_MIN_FRAC = 0.8                  # a ticker must have >= WIN_MIN_FRAC of the window's rows to enter
MIN_TRAIN_WINDOWS = 24             # minimum past windows before a walk-forward test point is scored
# shallow forest: the Experiment-A sample is tiny (~90 monthly windows), so cap depth to avoid overfitting
RF_KW = dict(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0, n_jobs=-1)  # n_jobs: all cores, no result change
HORIZONS = (1, 5, 10, 22)          # per-stock GBM forecast horizons (Experiment B)
# 6 global topology metrics. Eigenvector centrality ("eig") was dropped: the causal feature screen found it
# carries ~0 mutual information with the target and is perfectly collinear (VIF=inf) with the other metrics.
TOPO = ["dens", "avg_deg", "clus", "avg_w", "diam", "betw"]
# raw OHLCV directory per market for REAL ln(volume) (the paper's exact volume transform, not volume_zscore_22)
RAW_VOL_DIR = {"hose": "data/raw/prices/hose_vnstock", "sp500": "data/raw/prices/sp500_clean"}

# --- causal per-fold feature screen (Variance -> Pearson/Spearman -> Mutual Information -> VIF) ---
SCREEN_MI_CAP = 40000      # max train rows sampled per fold for mutual_info_regression (kNN cost bound)
SCREEN_MI_SEED = 20260912  # RNG seed for the MI subsample (reproducible; no wall-clock randomness)
SCREEN_CORR_LO = 0.03      # |Pearson| and |Spearman| below this -> negligible linear/monotone signal
SCREEN_MI_LO = 0.01        # mean MI (nats) below this -> negligible non-linear signal
SCREEN_VIF_HI = 10.0       # mean VIF above this -> feature information is duplicated by the others (redundant)
# minimum causal training rows before a fold is screened -- mirrors run_gbm's fold gate so the screened
# windows are exactly the windows the GBM fits (30000 for the large sp500 panel, 3000 elsewhere).
SCREEN_MIN_TRAIN_ROWS = {"sp500": 30000, "default": 3000}
