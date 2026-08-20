# SOICT HAR-LSTM-GAT — experiment results (honest)

Date: 2026-08-21. Suite: `submission/soict_lstm_gat/run_all.py`, 20 epochs max + early-stop (val MSE),
5 seeds {42,123,2026,7,2024}, per-ticker StandardScaler (train-only), graphical-lasso Top-5 edges
(train-only), QLIKE floor 1e-8 identical across models. Snapshots = common-date fixed-N, GLOBAL-date
80/10/10 (deviation from the spec's per-stock split — see §Caveats). Decision metric = QLIKE + DM.

## Headline result (honest, negative)

Across every completed configuration, **HAR is the best model**; the proposed **HAR-LSTM-GAT does NOT
beat HAR (or GARCH-beating aside) and the GAT graph consistently HURTS** — removing it (LSTM w/o GAT)
always lowers QLIKE. All deep models beat GARCH. This is a consistent negative result for the graph
model, in line with the project's prior evidence that the graph adds no out-of-sample value for VN
volatility.

## Test QLIKE (lower is better; seed-averaged over 5 seeds; row order = paper order)

| dataset | h | HAR | GARCH | LSTM (w/o GAT) | HAR-LSTM-GAT (Ours) | best |
|---|---|---|---|---|---|---|
| VN30 (lb10) | 1 | **0.3946** | 0.6500 | 0.4120 | 0.4528 | HAR |
| VN30 (lb10) | 5 | **0.4531** | 0.6420 | 0.4663 | 0.4991 | HAR |
| VN30 (lb22) | 1 | **0.3969** | 0.6034 | 0.4137 | 0.4714 | HAR |
| VN30 (lb22) | 5 | **0.4547** | 0.5980 | 0.4692 | 0.5166 | HAR |
| VN100 (lb10) | 1 | **0.4843** | 0.6210 | 0.5204 | 0.5296 | HAR |
| VN100 (lb10) | 5 | **0.5442** | 0.6157 | 0.5552 | 0.5588 | HAR |
| S&P500 (lb10) | 1,5 | — | — | — | — | excluded (OOM) |

R2 (VN30 h1): HAR 0.311, LSTM w/o GAT 0.265, Ours 0.205, GARCH 0.028 — all positive; HAR explains the
most variance. RMSE/MAE track the same ordering (values ~5e-4 as volatility is a small variance).

## Diebold–Mariano (QLIKE; negative dm favors Ours; * p<0.05)

| dataset | h | Ours vs HAR | Ours vs GARCH | Ours vs LSTM (w/o GAT) |
|---|---|---|---|---|
| VN30 lb10 | 1 | +9.11 (p=1e-19) → HAR* | −10.22 (p=3e-24) → Ours* | +6.40 (p=2e-10) → w/o-GAT* |
| VN30 lb10 | 5 | +6.09 (p=1e-9) → HAR* | −5.00 (p=6e-7) → Ours* | +4.94 (p=8e-7) → w/o-GAT* |
| VN30 lb22 | 1 | +10.86 (p=4e-27) → HAR* | −10.08 → Ours* | +8.80 → w/o-GAT* |
| VN30 lb22 | 5 | +7.44 (p=1e-13) → HAR* | −4.37 → Ours* | +5.92 → w/o-GAT* |
| VN100 lb10 | 1 | +4.01 (p=6e-5) → HAR* | −7.39 → Ours* | +0.93 (p=0.35) → w/o-GAT |
| VN100 lb10 | 5 | +1.11 (p=0.27) → HAR (tie) | −3.66 → Ours* | +0.38 (p=0.70) → w/o-GAT |

**Reading:** (1) HAR significantly beats Ours at every horizon except VN100-h5 (a tie). (2) Ours
significantly beats GARCH everywhere. (3) The ablation **Ours vs LSTM (w/o GAT) always favors w/o-GAT**
(significantly on VN30) — the graphical-lasso GAT branch consistently hurts.

## Findings

1. **HAR is the strongest volatility forecaster here.** Neither deep variant beats it at h1/h5.
2. **The graph (GAT over a graphical-lasso graph) hurts.** Leave-one-out removing the GAT lowers QLIKE
   in all six configs — the cross-sectional graph adds noise, not signal, for this target.
3. **All learned models beat GARCH** (the classical conditional-variance baseline is the weakest).
4. **Lookback 22 vs 10:** no benefit for the deep model (slightly worse at lb22); HAR ~unchanged.

## Caveats (important — flagged for review)

- **Snapshot / global-date split.** A GAT needs per-date snapshots, so the deep models use common-date
  fixed-N snapshots with a GLOBAL chronological 80/10/10 split (train=older, test=recent). Training
  logs show **val MSE ≈ 1.2 > 1.0** on the standardized target — the deep model regresses toward the
  train-regime mean and generalizes poorly across the volatility regime shift, while HAR (raw-scale,
  current-feature-driven) adapts. This handicaps the deep model relative to HAR.
- **Contrast with the per-observation setup.** The earlier cross-market study
  (`scripts/sp500_crossmarket`, per-ticker split, per-observation pooling, ~500 tickers) found a
  price-only LSTM *did* beat HAR at h1/h5. The difference is the data/split design, not the LSTM
  itself. A per-ticker per-observation design (spec's stated split) would give the deep model a fairer
  test; the snapshot design was adopted to support the graph.
- **S&P500 excluded:** the GAT attention is O(N²) in nodes; at N=500 it exhausts 8 GB VRAM even at
  batch 16. The graph model does not scale to 500 nodes on this GPU.
- **Honesty:** these are the true DM verdicts; no result was fabricated. The proposed model did not
  meet the "beat HAR at h1/h5" success target.

## Recommendation for the morning

Options: (a) accept the honest negative result and reframe the paper as an empirical study — *"HAR
remains a strong baseline for Vietnamese volatility; a graphical-lasso GAT does not improve a HAR-LSTM
and consistently hurts"* (a valid contribution); (b) re-run the deep models under the spec's per-stock
per-observation split (which beat HAR on S&P500) for a fairer deep-vs-HAR test before finalizing; (c) a
combination. No fabricated win is presented.

## Artifacts

- Code: `submission/soict_lstm_gat/` (32 tests pass). Results: `results/soict/*/result.json` + learning
  curves. Row order in tables: HAR → GARCH → LSTM (w/o GAT) → HAR-LSTM-GAT (Ours).
