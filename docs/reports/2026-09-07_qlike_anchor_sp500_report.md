# QLIKE-loss on S&P 500 (Colab A100) — results, log/issue analysis, paper update

SP500-clean panel (480 screened tickers, 320,141 daily obs at h1, 7 folds, 5 seeds, lookback 10).
Trained on Colab A100; results pushed to git (`d525b2d`), pulled locally. Configs: loss=qlike ×
anchor∈{none, harx}, all horizons (8 result JSONs, all with train/val/fit evidence, fit=ok). MSE-loss
baseline is the delivered `results/edge_hmatched/edgehm_sp500_clean_h*.json`.

## Outcome: GO — strongest QLIKE-loss result of the three panels
On the large, liquid SP500 panel the MSE-loss deep models are far worse than HAR-X on QLIKE (the
documented spike-collapse); training on the QLIKE loss reverses this, cutting QLIKE below HAR-X at
every horizon and significantly so at h1 and h22.

## QLIKE by horizon (SP500, anchor=none)
| h | HAR-X | LSTM (MSE) | LSTM (QLIKE) | VolGA (MSE) | VolGA (QLIKE) | DM QLIKE-loss vs HAR-X (LSTM / VolGA) |
|---|---|---|---|---|---|---|
| 1 | 0.4061 | 0.6141 | **0.3603** | 0.5492 | **0.3598** | **p=0.001** / **p=0.001** |
| 5 | 0.4550 | 0.5640 | **0.4448** | 0.5531 | **0.4468** | p=0.132 / 0.298 (n.s.) |
| 10 | 0.4797 | 0.6821 | **0.4714** | 0.6375 | **0.4695** | p=0.461 / 0.401 (n.s.) |
| 22 | 0.4959 | 0.6066 | **0.4699** | 0.6026 | **0.4696** | **p=0.010** / **p=0.013** |

- **h1:** QLIKE-loss cuts the deep-model QLIKE from ~0.55–0.61 (MSE-loss) to ~0.36, beating HAR-X
  (0.4061) with DM p=0.001. A ~34–41% QLIKE reduction versus the MSE-loss models.
- **h22:** significant win (p=0.010/0.013).
- **h5/h10:** below HAR-X but not significant.
- **anchor=harx** (secondary): same direction, slightly weaker; anchor=none is the lever.

## MSE / RMSE / MAE cost — none (slight gain)
On SP500, QLIKE-loss changes MSE/RMSE by ≤1.2% and MAE by ≤1.2%, mostly negative (a small
improvement). Unlike the small change on VN, SP500 pays no squared-error cost at all.

## Log / issue analysis (from result JSONs + the h1 cell dump)
1. **The QLIKE damage under MSE-loss is a heavy under-forecast TAIL, not the median.** On the top-1%
   realized-variance cells the median forecast/realized ratio is similar across models (HAR-X 0.238,
   VolGA 0.248, LSTM 0.259). The QLIKE gap comes from the extreme tail: catastrophic under-forecasts
   (f/y < 0.02) occur in **0.112% / 0.083%** of LSTM / VolGA test cells versus **0.016%** for HAR-X
   (~5–7×), and the **worst 1% of cells contribute 42% / 39%** of total QLIKE for LSTM / VolGA versus
   24% for HAR-X. QLIKE (asymmetric, punishes under-forecast) is dominated by this tail — which is
   exactly what QLIKE-loss training removes (QLIKE → 0.36). Cell recompute reproduces the JSON QLIKE
   (HAR-X 0.4061, LSTM 0.6141, VolGA 0.5492).
2. **No limit-lock regime on SP500** (`n_limitlock=None`), unlike VN30 — the SP500 QLIKE issue is pure
   spike under-forecast, with no floored zero-range days.
3. **Fit is healthy:** `fit_diagnostics=ok` for LSTM/VolGA at every horizon; val→test QLIKE gap small
   and negative (test no worse than val); train→test R² drop 8–15% (modest, within threshold). No
   over/under-fit.
4. **Minor logging gap (no result impact):** the Colab SP500 qlike runs left `qlike_robust=nan`
   (the robust-QLIKE field was not populated). Since SP500 has no limit-lock cells to exclude, robust
   ≈ raw here; noted for reproducibility, does not affect any reported number.

## Cross-panel summary (QLIKE-loss anchor=none, h1)
| panel | N | HAR-X | LSTM (MSE→QL) | VolGA (MSE→QL) | DM QL vs HAR-X |
|---|---|---|---|---|---|
| SP500 | 480 | 0.4061 | 0.6141 → **0.3603** | 0.5492 → **0.3598** | p=0.001 / 0.001 |
| VN100 | 102 | 0.5000 | 0.5155 → **0.4840** | 0.4879 → **0.4833** | p=0.003 / 0.004 |
| VN30 | 33 | 0.4801 | 0.4787 → **0.4634** | 0.4740 → **0.4639** | p=0.004 / 0.006 |

The QLIKE-loss finding holds on all three panels and is largest where the MSE-loss models were
weakest (SP500).

## Paper update
`docs/paper/soict_harlstmgat_2026-09-07_final.tex` §Ablation "Training objective: QLIKE-loss training":
added the SP500 panel to Table~\ref{tab:qlikeloss} (as an out-of-Vietnam robustness check) + a
sentence; caption and Limitations/scope updated to state SP500 enters only for this ablation. PDF
recompiled (12 pages) and re-uploaded to Drive `gdrive:luanvan_papers/`. SP500 result JSONs are on
master (`d525b2d`); cell parquets remain on Drive (`gdrive:luanvan_data/cells/`, gitignored).
