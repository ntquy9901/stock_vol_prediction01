# Design spec — SOICT paper experiment suite (HAR-LSTM-GAT for volatility forecasting)

Date: 2026-08-21. Source plan: `requirement/requirements_soict_paper.md`. Approach: **A**
(extract tested components into ONE minimal self-contained submission folder, TDD the new glue,
train from that folder). This spec is the source of truth; the implementation plan (writing-plans)
follows it.

## 1. Goal & success criteria

Build a reproducible experiment suite + paper for SOICT that forecasts Parkinson volatility (variance)
with a proposed **HAR-LSTM-GAT** model, compared against HAR and GARCH baselines.

- **Success (per user):** HAR-LSTM-GAT beats HAR **and** GARCH on **QLIKE** at **h1 and/or h5** with
  Diebold–Mariano p < 0.05. Reporting is honest: if only one horizon wins (or none), report all
  horizons and the DM verdicts as-is. Beating both baselines at h1 is the primary target (project
  evidence: the deep edge is a short-horizon effect; graph value is uncertain — a null/partial result
  is an acceptable, honestly-reported outcome).

## 2. Analysis of the original plan (right / wrong / missing)

- **Right:** pooled model + per-stock 80/10/10; baselines GARCH+HAR (both already implemented);
  metrics MSE/RMSE/MAE/QLIKE/R2 + DM (no DirAcc); 20-epoch early-stop; 5 seeds; GPU + parallel;
  learning curves; LSTM+GAT + graphical-lasso edges (both exist and are tested in-repo).
- **Fixed via brainstorming decisions:**
  - §1.2 mislabels config-variations as "ablation". **Decision:** the main ablation is a true
    LEAVE-ONE-OUT (HAR-LSTM-GAT vs LSTM w/o GAT) per CLAUDE.md; the other 3 (lookback 22, VN100,
    S&P500) are renamed **sensitivity/variation studies**. §1.2 item (1) IS the leave-one-out.
  - "openleaf" → **Overleaf/LaTeX**. **Decision:** Markdown draft first, then LaTeX per the SOICT
    template (to be fetched from soict.org).
  - Loss: **MSE** for training + early-stop (NOT QLIKE). QLIKE remains an evaluation metric + the
    success/DM criterion.
  - Model name: **"Full" → "HAR-LSTM-GAT (Ours)"**; ablation variant labelled **"LSTM (w/o GAT)"**.
- **Resolved gaps:** 5 seeds (added by user); QLIKE positivity floor made identical across compared
  models; success criteria defined; glasso edges estimated on TRAIN-only (leakage control);
  hyperparameters specified below.

## 3. Submission folder structure (minimal, self-contained)

```
submission/soict_lstm_gat/
├── README.md              # what it is + how to run
├── REPRODUCE.md           # reviewer reproduction guide
├── reproduce.sh           # ONE command: train + test the whole suite from data/ below
├── EXTRACTION_LOG.md      # detailed provenance log: <src file> from <src folder> -> <dest> (ENFORCE)
├── requirements.txt       # pinned deps
├── config.py              # all hyperparameters, seeds, dataset paths
├── data/                  # SHIPPED processed data (derived; NOT raw OHLCV)
│   ├── vn30/<TICKER>_processed.csv     # columns: date, parkinson_volatility
│   ├── vn100/<TICKER>_processed.csv    # vnstock (clean)
│   └── sp500/<TICKER>_processed.csv    # derived from Yahoo (non-commercial research)
├── data_utils.py          # load, 3-HAR-feature windows, per-stock 80/10/10, per-ticker scalers
├── edges.py               # graphical-lasso partial-corr adjacency (TRAIN-only, frozen)
├── model.py               # HAR-LSTM-GAT (3-feat nodes) + use_graph toggle for the w/o-GAT variant
├── baselines.py           # HAR (OLS) + GARCH(1,1)
├── metrics.py             # MSE/RMSE/MAE/QLIKE(floor 1e-8)/R2 + Diebold-Mariano
├── train.py               # pooled loop: MSE loss, 5 seeds, 20ep early-stop, GPU, parallel, curves, log
├── evaluate.py            # test eval (all metrics) + DM
├── run_all.py             # orchestrate: main + ablation + 3 variations, all horizons/seeds
└── tests/                 # TDD tests (data/edges/model/metrics/baselines/train-smoke/integration)
```

Data policy: only **processed** (derived Parkinson variance) is shipped — the plan uses only 3 HAR
features, so raw OHLCV is unnecessary; this avoids redistributing raw Yahoo OHLCV (ToS-restricted) on
a public repo. README states provenance (VN100=vnstock, S&P500 derived-from-Yahoo, non-commercial
research). `reproduce.sh` can optionally regenerate S&P500 from source (yfinance) if a reviewer wants.

## 4. Data & splits

- **Datasets:** VN30 (33 tickers), VN100 (vnstock, ~104), S&P500 (~500). Target = Parkinson VARIANCE
  at t+h (point forecast), h ∈ {1, 5} (1 day, 1 trading week). Lookback = 10 (main), 22 (variation).
- **Features (3):** HAR = [parkinson(t), rolling-5 mean, rolling-22 mean]. SHARED as the LSTM input
  sequence and the GAT node features (no 5-feature set, no news, no volume).
- **Split:** per-stock chronological **80/10/10** (train/val/test); ONE pooled model; per-ticker
  StandardScaler fit on TRAIN rows only, applied to that ticker's windows; inverse-transform at eval.
- **Windows:** `[lookback, 3]` per anchor (monthly-valid), pooled across tickers + shuffled. No
  leakage: windows within-split, scalers train-only, test read once.

## 5. Model, ablation, variations

- **HAR-LSTM-GAT (proposed):** LSTM (temporal, 2-layer, hidden 64) over the `[lookback,3]` sequence +
  a GAT branch over the 3-feature nodes at day t; **edges = graphical-lasso partial-correlation Top-K**
  adjacency estimated on TRAIN rows only and frozen for val/test (leakage-safe); branches concatenated
  → head. Linear output on normalized scale + positivity floor at inverse-transform.
- **Leave-one-out ablation (main; VN30, lb=10, h1/h5, 5 seeds):** HAR-LSTM-GAT vs **LSTM (w/o GAT)**
  (only removable component, since no news/gate). Effect = QLIKE(w/o GAT) − QLIKE(full), DM-tested.
- **Sensitivity / variation studies (lstm+gat):** (i) lookback 22 vs 10 (VN30); (ii) VN100; (iii)
  S&P500 — each × {h1,h5} × 5 seeds.
- **Baselines:** HAR (pooled OLS on the 3 HAR features), GARCH(1,1). Both "to be beaten" on QLIKE.

## 6. Training

MSE loss; Adam(lr 1e-3, weight_decay 1e-5); grad-clip 1.0; dropout 0.2; ReduceLROnPlateau; 20 epochs
max; early-stop on **val MSE** (patience 3, min_epochs 5). **5 seeds** {42, 123, 2026, 7, 2024}. GPU;
parallelism via concurrent processes across (seed × config) + batched tensors / DataLoader workers.
Learning curves (train/val loss) saved as PNG every 5 epochs. Startup prints hyperparameters; per-epoch
prints all val metrics; a debug log is written to file.

## 7. Evaluation

All 5 metrics (MSE, RMSE, MAE, QLIKE, R2) on the held-out test, seed-averaged (mean ± std over 5
seeds). **QLIKE positivity floor = 1e-8, identical across every compared model** (HAR-LSTM-GAT, LSTM
w/o GAT, HAR, GARCH). Diebold–Mariano (HLN small-sample correction, HAC lag h−1), seed-ensembled
predictions: HAR-LSTM-GAT vs {HAR, GARCH, LSTM-w/o-GAT} on QLIKE (+ squared-error). Outputs:
`results.json`, per-horizon comparison tables (row order HAR → GARCH → LSTM w/o GAT → HAR-LSTM-GAT
(Ours)), learning-curve PNGs.

## 8. Paper

Markdown draft (objective style: no DirAcc, all horizons reported, VN-market scope, no e-notation, no
internal jargon), then LaTeX per the SOICT template fetched from soict.org/submission. Sections:
Abstract, Introduction, Related Work, Method (HAR features, LSTM-GAT, graphical-lasso edge), Data,
Experiments (main + ablation + variations), Results (tables + DM + learning curves), Discussion,
Conclusion.

## 9. Reproducibility (ENFORCE, §1.6)

`EXTRACTION_LOG.md` records every extracted file's origin. `reproduce.sh` runs the full suite from the
shipped `data/`. `REPRODUCE.md` documents environment + commands for reviewers. Training starts FROM
the submission folder (not the main repo).

## 10. TDD test plan (write test → confirm FAIL → implement)

- **data_utils:** window shape `[lookback,3]`; per-stock 80/10/10 boundary counts; scaler fit
  train-only (a test-row change must NOT move train scaler); anchor monthly-valid start; target=pk[t+h].
- **edges (glasso):** adjacency `[N,N]`, symmetric partial-corr Top-K, diagonal self-loop; estimated
  on TRAIN rows only (appending test rows must not change the frozen edge); non-convergence bumps alpha.
- **model:** forward output `[B,N]`; `use_graph=False` removes the GAT branch (fewer params, no adjacency
  use); input feature dim = 3; inverse-transform yields positive predictions.
- **metrics:** QLIKE floor applied identically to target & prediction; MSE/RMSE/MAE/R2 match formulas;
  DM sign convention (negative favors A) + reproduces a known value.
- **baselines:** HAR OLS fit/predict shape + coefficients on a linear fixture; GARCH(1,1) fit+forecast smoke.
- **train (smoke):** 2-epoch run on tiny synthetic → checkpoint + learning-curve PNG + metrics JSON;
  early-stop triggers; 5-seed loop produces 5 result sets.
- **integration (smoke):** `run_all` on a tiny subset → results.json with main + ablation + DM cells.

## 11. Defaults chosen (were unspecified) & out of scope

- **Defaults:** glasso Top-K = 5, alpha auto-raised until convergence; seeds {42,123,2026,7,2024};
  h5 target = point forecast at t+5 (consistent with HAR, NOT a 5-day sum); GARCH per-ticker(1,1).
- **Out of scope:** news/gate branches; 5-feature node set; DirAcc; rolling recalibration; raw-OHLCV
  redistribution.
