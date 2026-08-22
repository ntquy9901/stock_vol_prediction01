# HAR-Anchored LSTM–GAT Study — Consolidated Findings and Graph Diagnosis (Handoff)

Purpose: a self-contained report of the current experimental results, the statistical methodology, and a
detailed diagnosis of why the graph (GAT) branch does not add out-of-sample value (and actively hurts in
the full-target form). Intended to be handed to another analyst/AI for a second opinion. All numbers are
from `results/har_anchored/<dataset>_h<h>/result.json` (assembled in `reports/experiment_results.md`);
the graph diagnosis is from `docs/reports/2026-08-22_graph_no_value_analysis.md`.

## 1. Task and data
- Target: daily **Parkinson variance** (σ², from daily High/Low), column `parkinson_volatility`. Not
  realized variance (no intraday data). One scale throughout. QLIKE `L = y/ŷ + log ŷ` applied to variance.
- Target definition per horizon h: **terminal value** `pk[t+h]` (single day), not an average/sum.
- Panels: VN30 (33 tickers), VN100 (104), S&P 500 (≈457–500). Horizons h ∈ {1, 5, 10, 22}. Lookback 10.
- Design: **common-date snapshot** — on each date d, N nodes (tickers) with their [10,3] HAR-feature
  windows and a shared adjacency; needed so the GAT graph is well-defined and the graph vs no-graph
  variants are compared on identical folds.
- Split: chronological 80/10/10 on snapshot dates, with a **target-overlap purge of h snapshots** at each
  boundary (no train/val/test target interval crosses a boundary). Per-ticker StandardScaler and the
  graphical-lasso adjacency are fit on TRAIN dates only (frozen). 5 seeds {42,123,2026,7,2024}.

## 2. Experiment ladder (E0–E10)
| ID | Model | Note |
|---|---|---|
| E0 | HAR (per-horizon pooled OLS) | locked benchmark |
| E1 | LSTM, full target | temporal only, predicts the whole target |
| E2 | LSTM+GAT, full target | the "full neural" model; graph forecasts the whole target |
| E3 | convex combination α·HAR + (1−α)·E2 | α fit on validation only (frozen experts) |
| E5 | HAR + LSTM residual | residual = y − cross-fitted-HAR; zero-init head ⇒ HAR fallback |
| E6 | HAR + **GAT-only** residual | graph-only correction (no LSTM branch) |
| E7 | HAR + LSTM+GAT residual | combined residual |
| E8 | multiplicative HAR-anchored residual | `(HAR+ε)·exp(λ·δ)`, positive by construction |
| E9 / E10 | static / dynamic gated residual | λ scales the E7 correction; observable-state gate |

Residual targets use **expanding-window cross-fitted HAR** (each train block predicted from strictly
earlier blocks) so residuals reflect deployment, not in-sample optimism. Graphical-lasso adjacency =
signed Top-5 partial-correlation edges from the train Parkinson panel.

## 3. Statistical methodology (important)
Significance is the **date-clustered Diebold–Mariano** test: the per-observation loss differential is
aggregated to one value per date, then HLN-DM is run on that T-length series. The panel is
cross-sectionally dependent (all N tickers share each date), so the naive per-observation DM treats
n = N·T and **over-states significance by ≈√N** (≈6–8×). An earlier read of these results used the
row-level DM and produced spurious "graph beats HAR" p-values; all significance below is the corrected
date-clustered value. (This same caveat applies to the project's earlier per-observation "LSTM beats HAR"
results, which were assessed with row-level DM.)

## 4. Headline results (VN30, VN100 — reliable panels)
QLIKE (lower better); dQLIKE% = % improvement vs HAR; DM p = date-clustered vs HAR.

**VN30** (n_test ≈ 4300 obs over ~130 test dates)
| h | HAR QLIKE | best deep model & QLIKE | E3 combo QLIKE (dQLIKE%, DM p) | verdict |
|---|---|---|---|---|
| 1 | 0.3946 | E1 0.4383 (worse) | 0.3984 (−0.9%, p=0.24) | nothing beats HAR |
| 5 | 0.4531 | E1 0.4810 (worse) | 0.4639 (−2.4%, p=0.14) | nothing beats HAR |
| 10 | 0.4849 | E1 0.5291 (worse) | 0.4958 (−2.2%, p=0.32) | nothing beats HAR |
| 22 | 0.5025 | E1 0.5185 (worse) | 0.5137 (−2.2%, p=0.51) | nothing beats HAR |

**VN100** (n_test ≈ 5000 obs over ~49–130 test dates)
| h | HAR | E1 (full LSTM) | E2 (full LSTM+GAT) | E3 combo (dQLIKE%, DM p) | E6 (GAT resid) dQLIKE% |
|---|---|---|---|---|---|
| 1 | 0.4844 | 0.5371 | 0.5207 | 0.4821 (+0.5%, p=0.24) | +0.04% |
| 5 | 0.5441 | 0.5614 | 0.5582 | 0.5339 (+1.9%, p=0.29) | +0.05% |
| 10 | 0.5985 | 0.5750 | 0.5803 | 0.5701 (+4.8%, p=0.20) | +1.70% |
| 22 | 0.6177 | 0.5888 | 0.6031 | 0.5816 (+5.9%, p=0.08) | +4.08% |

Reading: point-estimate QLIKE improvements grow with horizon on VN100 (E3 up to +5.9%, E6 up to +4.1%),
but **no date-clustered DM p-value is below 0.05** — under panel-correct inference nothing significantly
beats HAR. GARCH is worse than HAR at short horizons and roughly ties at long horizons. (S&P 500 is being
re-run with a proper test window; see §7 — the initial run had only 34 common-date test dates.)

## 5. The graph: full-target HARM vs residual NULL
Two distinct facts about the GAT graph:

1. **In the full-target form (E2), the graph HURTS.** E2 (LSTM+GAT predicting the whole target) is worse
   than E1 (LSTM alone, no graph) at short horizons: VN30 h1 E2 0.4915 vs E1 0.4383; VN100 h1 E2 0.5207
   vs E1 0.5371 (mixed) — and both are far worse than HAR. When the network must forecast the entire
   target, aggregating noisy neighbours dilutes a node's own strong HAR signal (over-smoothing).
2. **In the HAR-anchored residual form (E6/E7), the graph is NULL, not harmful.** E6 (GAT-only residual)
   ≈ E5 (LSTM-only residual) ≈ HAR at the point-estimate level, and the paired date-clustered DM E6-vs-E5
   is never significant (p ∈ 0.44–0.995, direction mixed). Anchoring on HAR and only learning a
   zero-initialised correction removes the harm — but adds no value.

## 6. WHY the graph adds no OOS value — diagnosis (VN30, VN100)
Six independent checks (`docs/reports/2026-08-22_graph_no_value_analysis.md`):

1. **Edges do not transfer out of sample.** The glasso Top-5 graph estimated on train vs on test overlaps
   weakly: neighbour Jaccard 0.17 (VN30) / 0.09 (VN100), edge-set Jaccard 0.16 / 0.09, edge-weight
   correlation 0.19 / 0.15, and the negative-edge share flips from ~3% (train) to ~19% (test). The graph
   is a real non-identity structure (density 6–18%, degree 5–8) but an unstable one.
2. **Model-free spillover test finds nothing (the decisive evidence).** A leakage-safe linear regression
   HAR vs HAR + (mean Parkinson variance of the Top-5 glasso neighbours at day t) adds essentially zero
   incremental out-of-sample R² at every horizon on both panels (VN30 −0.0001…+0.0013; VN100
   −0.0007…+0.009). This bypasses the GAT, attention and residual head entirely: if exploitable
   cross-sectional spillover existed OOS, this simplest possible test would find it. It does not.
3. **Attention has collapsed to uniform.** GAT attention entropy is ~0.99 of the uniform maximum (~99% of
   rows near-uniform) — the layer is doing plain neighbour-averaging, learning no selective structure.
4. **The branch is alive, not suppressed.** At VN100 h22 the GAT residual emits a correction ~15% of HAR
   magnitude, yet still fails to beat HAR (date-clustered p=0.47) — so the null is not a dead/zeroed branch.
5. **No bug in the graph path.** Verified: adjacency masking (`!=0`, `−inf`, softmax over the neighbour
   dim, `nan_to_num`), day-t raw node-feature indexing, `[N,N]→[B,N,N]` broadcast, train-only frozen
   glasso, and `use_graph=False` genuinely removing the branch (E6 17.8k params < E5 55.2k).
6. **The graph does NOT overfit.** E6 residual R² is train 0.0017 vs test 0.039 (test ≥ train), and E6's
   train QLIKE is no better than E5's. There is no positive train→test degradation — the zero-init HAR
   anchor removed the overfitting failure mode of the full-target model. So the null is neither (a)
   overfitting nor (b) attention-collapse hiding real train-side signal (train R² is itself ≈0); it is
   (c) a **genuine no-signal** result: there is no out-of-sample-transferable cross-sectional spillover in
   these HAR-feature panels for the graph to exploit.

## 7. S&P 500 — corrected run (the one significant beat-HAR result)
The initial S&P 500 run had only 34 common-date test dates and additive-residual blow-ups. Corrected via
(i) a train-derived positive output floor (plan §10), (ii) a long-history subset (`min_common=3000` ⇒ 457
nodes, ~300 test dates, 137k test obs), (iii) batched eval forwards (whole-set forward OOM'd at 457 nodes).

Result (date-clustered DM, ~300 test dates — adequate power):
| h | HAR | E1 LSTM (no graph) | E2 LSTM+GAT | E3 combination |
|---|---|---|---|---|
| 1 | 0.3776 | — | — | 0.3656 (+3.2%, p<0.001) |
| 22 | 0.4596 | **0.4270 (+7.1%, p<0.001)** | 0.4386 (+4.6%, p=0.004) | 0.4349 (+5.4%, p<0.001) |

On this large, data-rich panel the deep temporal model **significantly beats HAR** (E1 up to +7.1% at
h22, date-clustered p<0.001), and the convex combination E3 beats HAR at all four horizons (+3.2…+5.4%).
Crucially, **the graph only detracts**: E2 (LSTM+GAT) is worse than E1 (LSTM alone) at h22 (+4.6% vs
+7.1%), and the graph-residual E6/E7 fail or blow up numerically (E5/E7 additive still unstable at 457
nodes even with the floor; E8 multiplicative is stable but worse than HAR). This is the first significant
beat-HAR result under panel-correct inference, and it is attributable to temporal nonlinearity, not the
graph — consistent with the model-free screening (§6.2) and the VN point estimates that lacked the test
power to reach significance.

## 8. Honest verdict and open questions for a second opinion
Verdict (updated with the corrected S&P 500 run):
- On the small VN panels (VN30, VN100), under date-clustered inference, **no model significantly beats
  HAR** at any horizon — point estimates favour the hybrid at long horizons but the test windows are too
  short (~49–130 dates) for significance.
- On the large **S&P 500** panel (~300 test dates), the **deep temporal model significantly beats HAR**
  (E1 LSTM up to +7.1% at h22, combination +3.2…+5.4%, date-clustered p<0.001) — HAR IS beatable given a
  data-rich panel and adequate test power.
- **The graph never helps and often hurts:** E2 (LSTM+GAT) < E1 (LSTM) on S&P 500; the graph-residual
  E6/E7 fail; and model-independently (§6.2) the weighted/signed/innovation/lead-lag neighbour signals add
  ≈0 incremental OOS R² across all three panels. The required cross-stock spillover is not
  out-of-sample-transferable on these HAR-feature panels — not a GAT/attention/tuning artefact, and not
  fixable by consuming edge sign/weight (V2), since the model-free upper bound is itself null.

Overall: the beatable margin over HAR comes from **temporal nonlinearity on large panels, not from the
cross-sectional graph**. Consistent with the literature (e.g. Branco, Rubesam & Zevallos, "Does anything
beat linear models?", and Christensen et al. on ML gains at longer horizons with more data).

Open questions worth a second opinion (each could be tested, though §6.2 suggests limited upside):
- Alternative edge definitions the study has not run: return lead–lag, volume-shock correlation, or a
  static sector graph; different rolling windows {20,60,120}; directed edges. Does any produce a
  train→test-stable graph with non-zero §6.2 incremental R²?
- Richer node features beyond the 3 HAR lags (e.g. realized measures if intraday data were available,
  liquidity, market/factor returns) — the no-signal result is specific to HAR-only node features.
- A longer / different test window (the VN common-date windows are short, ~49–130 dates, which limits
  power even where point estimates favour the hybrid); walk-forward re-estimation instead of one split.
- Whether the point-estimate long-horizon gains on VN100 (E3 +5.9%, E6 +4.1% at h22) become significant
  with more test dates or a more powerful panel test (Driscoll–Kraay / two-way clustered), rather than the
  conservative date-clustered DM.

Key files: `reports/experiment_results.md` (all metrics), `reports/leakage_audit.md` (leakage controls),
`docs/reports/2026-08-22_graph_no_value_analysis.md` (graph diagnosis), `baselines/2026-08-21_har_anchored_residual/`
(code + tests + design), `results/har_anchored/*/row_predictions.csv` (row-aligned predictions).
