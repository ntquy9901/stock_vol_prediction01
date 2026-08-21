# HAR-Anchored LSTM–GAT Experiment Plan

## 1. Purpose

This document is an implementation handoff for testing whether a neural cross-sectional model can improve upon a Heterogeneous Autoregressive (HAR) baseline for stock-volatility forecasting.

The current neural model concatenates an LSTM with a GAT and uses the three standard HAR inputs, but it has not beaten HAR out of sample. The experiments below test forecast blending, residual correction, horizon-specific weighting, and regime-aware gating while preserving HAR's strong persistence structure.

The intended target is **daily Parkinson variance or Parkinson volatility computed from daily High/Low prices**. There is no intraday data, so do not rename the target `realized volatility` in code, tables, or conclusions. Papers based on intraday realized variance provide architectural precedents only; their empirical results do not directly validate this dataset.

## 2. Prediction contract — complete before implementation

Record these values in the experiment configuration and final report:

- Prediction timestamp: before or after market close on day `t`.
- Permitted information cutoff at prediction time.
- Target type: Parkinson variance or Parkinson volatility; do not mix them.
- Target definition for each horizon `h ∈ {1, 5, 10}`.
- Whether a multi-day target is an average, sum, terminal value, or another aggregation.
- Entity key: ticker and trading date.
- Historical universe policy, including delisted/replaced VN30 constituents if available.
- Train, validation, and locked test dates.
- Retraining schedule: fixed, expanding-window, sliding-window, or walk-forward.

For daily Parkinson variance:

$$
PK^2_{i,t}=\frac{1}{4\ln 2}\left[\ln\left(\frac{H_{i,t}}{L_{i,t}}\right)\right]^2.
$$

Parkinson volatility is:

$$
PK_{i,t}=\sqrt{PK^2_{i,t}}.
$$

Use exactly one target scale throughout a run. If log variance is modeled, document the transformation and inverse transformation.

## 3. Research hypotheses

- **H1 — Static combination:** HAR and LSTM–GAT errors contain complementary information, so a validation-fitted convex combination beats both experts.
- **H2 — Horizon specialization:** the optimal HAR weight differs for T+1, T+5, and T+10.
- **H3 — Residual learnability:** LSTM and/or GAT can predict the portion of the target not explained by HAR.
- **H4 — Cross-sectional value:** GAT contributes incremental predictive information beyond a ticker-local LSTM and HAR.
- **H5 — State dependence:** the neural correction is useful only in observable market states such as high current volatility, high dispersion, or high correlation.
- **H6 — Safe anchoring:** initializing the hybrid exactly at HAR and learning only a correction is more stable and generalizes better than training a full LSTM–GAT forecast from random initialization.

## 4. Literature motivation

### 4.1 HARNet

HARNet uses a hierarchy of dilated convolutional layers and provides an initialization under which the network initially produces the same prediction as its HAR baseline. The authors report that HAR initialization, particularly with QLIKE optimization, stabilizes training and can improve forecasting accuracy relative to HAR. This motivates a zero-initialized neural correction anchored at HAR.

- Reisenhofer, Bayer, and Hautsch, *HARNet: A Convolutional Neural Network for Realized Volatility Forecasting*: https://arxiv.org/abs/2205.07719

### 4.2 Machine learning against the HAR lineage

Christensen, Siggaard, and Veliyev report that machine-learning models can beat HAR-family benchmarks even when the predictors are limited to daily, weekly, and monthly variance lags. Gains are more pronounced at longer horizons. This supports testing nonlinear mappings of the HAR inputs, but it does not imply that every deep model will beat HAR.

- *A Machine Learning Approach to Volatility Forecasting*: https://arxiv.org/abs/2601.13014

### 4.3 GNN and volatility spillovers

GNN studies motivate learning cross-asset effects that a univariate HAR cannot represent. Their relevance depends on whether graph edges are causal/predictive at the forecast cutoff rather than contemporaneous artifacts.

- Zhang et al., *Forecasting Realized Volatility with Spillover Effects*: https://ora.ox.ac.uk/objects/uuid%3A316ea7da-8ed5-4833-8573-c6210509f99a
- Chen and Robert, *Multivariate Realized Volatility Forecasting with Graph Neural Network*: https://dl.acm.org/doi/fullHtml/10.1145/3533271.3561663

### 4.4 Hybrid statistical–neural models

Residual hybrids use a statistical model for persistent/linear structure and a neural model for remaining nonlinear structure. A recent HAR–LSTM–GARCH study is directly related in spirit, although it focuses on energy-market volatility and is too context-specific to establish that the same result will hold for VN30 Parkinson variance.

- *A Hybrid HAR-LSTM-GARCH Model for Forecasting Volatility in Energy Markets*: https://www.mdpi.com/1911-8074/19/1/77

### 4.5 Why failure to beat HAR is unsurprising

Published comparisons repeatedly show that nonlinear models do not universally outperform HAR. One study across ten global indices finds no general statistical dominance of nonlinear ML over linear models, even though additional predictors can help at daily and weekly horizons. Another finds that neural networks and GARCH do not beat HAR because volatility persistence is crucial.

- Branco, Rubesam, and Zevallos, *Forecasting realized volatility: Does anything beat linear models?*: https://doi.org/10.1016/j.jempfin.2024.101524
- Vortelinos, *Forecasting realized volatility: HAR against Principal Components Combining, neural networks and GARCH*: https://doi.org/10.1016/j.ribaf.2015.01.004

## 5. Required baseline definitions

For ticker `i`, date `t`, and horizon `h`, define the HAR inputs using information available no later than the prediction cutoff:

$$
x^{d}_{i,t}=PK^2_{i,t},\qquad
x^{w}_{i,t}=\frac{1}{5}\sum_{k=0}^{4}PK^2_{i,t-k},\qquad
x^{m}_{i,t}=\frac{1}{22}\sum_{k=0}^{21}PK^2_{i,t-k}.
$$

The baseline forecast is:

$$
\hat y^{HAR}_{i,t+h}=\beta_{0,h}+\beta_{d,h}x^d_{i,t}+\beta_{w,h}x^w_{i,t}+\beta_{m,h}x^m_{i,t}.
$$

Use a separate HAR fit per horizon unless the existing implementation explicitly uses a valid multi-output specification. Fit HAR only on the training portion of each fold.

Persist the following for every evaluation row:

- ticker, forecast origin, horizon, target start/end dates;
- actual target;
- HAR prediction;
- every neural/hybrid prediction;
- learned alpha or gate value;
- regime/state inputs used by the gate;
- fold and model version.

## 6. Why concatenating HAR features may fail

Providing `daily`, `weekly`, and `monthly` HAR features to LSTM–GAT does not force the neural model to preserve the strong linear HAR relationship. The network must simultaneously learn persistence, temporal nonlinearity, cross-stock propagation, ticker heterogeneity, target scale, and optimization dynamics. A GAT can also dilute a node's strong local HAR signal by aggregating noisy neighbors. Therefore, the graph branch should first be tested as an incremental correction rather than as a replacement for the HAR forecast.

## 7. Experimental ladder

Use identical folds, input availability, seed budget, tuning budget, and reporting for all models.

| ID | Model | Formula/purpose |
|---|---|---|
| E0 | HAR | Locked benchmark |
| E1 | LSTM-HAR3 | LSTM using the three HAR features; tests nonlinear temporal mapping |
| E2 | LSTM–GAT-HAR3 | Existing full neural model; tests graph contribution when forecasting the entire target |
| E3 | Static convex blend | One global validation-fitted alpha |
| E4 | Horizon convex blend | One validation-fitted alpha per horizon |
| E5 | Additive HAR + LSTM residual | Tests temporal residual learnability |
| E6 | Additive HAR + GAT residual | Tests graph-only incremental value |
| E7 | Additive HAR + LSTM–GAT residual | Tests combined residual correction |
| E8 | Multiplicative/log residual | Positive, scale-aware HAR-anchored correction |
| E9 | Static gated residual by horizon | Learnable correction strength, but not time-varying |
| E10 | Dynamic regime-gated residual | Gate varies by ticker/date/horizon using observable state |

Do not skip E5 and E6. They identify whether temporal modeling or graph propagation creates incremental value. A complex gate cannot create useful information if neither residual expert predicts HAR's errors.

## 8. Experiment E3 — global static alpha

Freeze the already trained HAR and neural experts. Fit only:

$$
\alpha=\sigma(a),
$$

$$
\hat y=\alpha\hat y^{HAR}+(1-\alpha)\hat y^{NN}.
$$

Fit `alpha` on validation data only. Do not jointly retrain the experts for this first combination test.

For squared error, the unconstrained validation optimum is:

$$
\alpha^*=\operatorname{clip}_{[0,1]}
\left(
\frac{\sum (y-\hat y^{NN})(\hat y^{HAR}-\hat y^{NN})}
{\sum(\hat y^{HAR}-\hat y^{NN})^2}
\right).
$$

For QLIKE, minimize validation QLIKE by numerical optimization or a dense grid over `[0,1]`. Record both the selected alpha and the loss curve. If `alpha → 1`, report that the neural model provides no reliable incremental value under this combination; do not treat this as an implementation failure.

## 9. Experiment E4 — horizon-specific alpha

Fit:

$$
\alpha_h=\sigma(a_h),
$$

$$
\hat y_{i,t+h}=\alpha_h\hat y^{HAR}_{i,t+h}+(1-\alpha_h)\hat y^{NN}_{i,t+h}.
$$

This is preferred to a single alpha because HAR and graph effects may differ at T+1, T+5, and T+10. Report confidence intervals or bootstrap dispersion for each alpha.

An optional ticker-specific extension is:

$$
\alpha_{i,h}=\sigma(a_h+e_{i,h}),
$$

where ticker effects `e` receive strong L2 shrinkage or hierarchical partial pooling. Do not start with 30 fully independent unregularized weights.

## 10. Experiments E5–E7 — additive residual correction

Construct the training target from out-of-sample-style HAR predictions within training data. Do not calculate residual targets using a HAR model fitted on the same rows without cross-fitting, because overly optimistic in-sample HAR residuals differ from deployment residuals.

Define:

$$
r_{i,t+h}=y_{i,t+h}-\hat y^{HAR}_{i,t+h}.
$$

Then train:

$$
\hat r_{i,t+h}=f_{NN}(X_{i,t},G_t),
$$

$$
\hat y_{i,t+h}=\hat y^{HAR}_{i,t+h}+\lambda_{i,t,h}\hat r_{i,t+h}.
$$

Use the following experts separately:

- E5: ticker-local LSTM only;
- E6: GAT/cross-sectional encoder only;
- E7: LSTM plus GAT.

Initialize the final residual head at zero:

```python
nn.init.zeros_(residual_head.weight)
nn.init.zeros_(residual_head.bias)
```

At initialization, the hybrid prediction must equal HAR. Add a unit test for this invariant.

If direct addition can produce a negative variance prediction, apply a positive output mapping or prefer the multiplicative design below.

## 11. Experiment E8 — multiplicative/log residual correction

Recommended primary architecture:

$$
\log \hat y_{i,t+h}
=\log(\hat y^{HAR}_{i,t+h}+\epsilon)
+\lambda_{i,t,h}\delta^{NN}_{i,t+h},
$$

equivalently:

$$
\boxed{
\hat y_{i,t+h}
=(\hat y^{HAR}_{i,t+h}+\epsilon)
\exp\left(\lambda_{i,t,h}\delta^{NN}_{i,t+h}\right)
}.
$$

Advantages:

- positive variance forecast;
- exact HAR fallback when the correction is zero;
- correction is relative to HAR scale;
- LSTM–GAT does not need to relearn the dominant level and persistence.

Set `epsilon` once based on numerical precision and training-target scale, document it, and never tune it on test data. Clip log corrections only if necessary for numerical stability; determine bounds from training data.

## 12. Experiments E9–E10 — gated correction

### 12.1 Static correction strength

First fit one correction weight per horizon:

$$
\lambda_h=\sigma(b_h).
$$

This separates the question "is the residual model useful?" from "can a gate identify when it is useful?"

### 12.2 Dynamic gate

Fit:

$$
\lambda_{i,t,h}=\sigma(g_h(z_{i,t})).
$$

Use a small gate, initially `Linear` or `MLP(input → 8 → 1)`. Candidate state inputs must be observable at time `t`:

- current market Parkinson variance/volatility;
- volatility-of-volatility calculated from past observations;
- cross-sectional Parkinson dispersion;
- absolute market return known by cutoff;
- average pairwise correlation or graph density computed from eligible history;
- HAR forecast level;
- absolute disagreement between frozen HAR and neural forecasts;
- ticker embedding only if regularized and justified.

Do not use future target volatility, future regime labels, or a regime threshold fitted on validation/test. Do not initially feed a large LSTM hidden state into the gate; this makes overfitting and interpretation harder.

Regularize the gate toward HAR fallback. Options to test on validation:

- initialize gate bias so `lambda` is small, such as 0.05–0.15;
- L1/L2 penalty on `lambda` or on deviation from the fallback;
- entropy regularization only if gate collapse is empirically problematic;
- cap maximum correction strength during the first training stage.

Use soft routing first. Hard regime routing should be a later ablation because boundary errors make it unstable.

## 13. Suggested training stages

1. Fit HAR per fold and horizon.
2. Produce cross-fitted HAR predictions for residual-target construction inside training.
3. Train the neural residual expert with HAR frozen and gate fixed to `1` or a small constant.
4. Freeze the expert and fit static `lambda_h` on validation.
5. Only if the residual model adds value, fit the small dynamic gate.
6. Optionally fine-tune expert and gate jointly with a low learning rate, while keeping HAR frozen.
7. Lock every choice before evaluating the test set.

Use early stopping on validation QLIKE or the predeclared primary metric. Save best checkpoints by validation only. Run multiple seeds for neural models and report the distribution, not only the best seed.

## 14. Diagnostic tests before dynamic gating

### 14.1 Error complementarity

Compute on validation:

$$
e^{HAR}=y-\hat y^{HAR},\qquad e^{NN}=y-\hat y^{NN}.
$$

Report Pearson and Spearman correlations. If error correlation is near one, blending has little diversification potential.

### 14.2 Forecast disagreement

Define:

$$
d=\hat y^{NN}-\hat y^{HAR}.
$$

Bin rows by `|d|` and estimate:

$$
P(|e^{NN}|<|e^{HAR}|\mid |d|\text{ bin}).
$$

If NN is not increasingly likely to be correct when the models disagree, a gate has little signal.

### 14.3 Residual predictability

Report out-of-sample residual-prediction performance:

$$
R^2_{OOS}(r^{HAR},\hat r^{LSTM}),
\quad
R^2_{OOS}(r^{HAR},\hat r^{GAT}),
\quad
R^2_{OOS}(r^{HAR},\hat r^{LSTM+GAT}).
$$

If GAT residual `R²_OOS ≤ 0`, the current graph does not provide incremental predictive spillover.

### 14.4 Observable-state analysis

Compare HAR and neural errors across states defined only from information available at time `t`:

- low/medium/high current market volatility;
- low/high past correlation;
- low/high cross-sectional dispersion;
- small/large absolute market return;
- sparse/dense graph.

This is post-forecast analysis and may motivate a gate. Any thresholds subsequently used by the gate must be fitted within training data for every fold.

## 15. Graph design and ablations

Candidate dynamic edges already relevant to this project include:

- rolling return correlation;
- rolling Parkinson-volatility correlation;
- rolling volume-shock correlation;
- return lead–lag relationships;
- Parkinson-volatility lead–lag relationships.

Test rolling windows such as 20/60/120 trading days through validation only. For each graph version, record:

- whether it is directed or undirected;
- edge sign and whether negative edges are retained;
- threshold or top-k construction;
- edge normalization;
- self-loop policy;
- update frequency;
- exact historical window;
- information cutoff.

Required graph ablations:

- no graph / identity graph;
- static sector graph if sector metadata is historically available;
- correlation graph;
- volatility-correlation graph;
- directed lead–lag graph;
- learned attention with fixed eligible candidate edges;
- shuffled-edge placebo preserving approximate density/degree;
- stale graph versus dynamically updated graph.

A graph should be considered useful only if it beats both HAR and the same-capacity no-graph residual model, not merely the standalone LSTM–GAT baseline.

## 16. Losses and target-scale experiments

Predeclare one primary target and primary loss. Recommended candidates:

- QLIKE on positive variance forecasts;
- MSE on log Parkinson variance;
- a weighted combination of QLIKE and log-MSE selected on validation.

QLIKE for actual variance `y > 0` and forecast `ŷ > 0` can be written, up to an additive constant, as:

$$
L_{QLIKE}(y,\hat y)=\frac{y}{\hat y}+\log(\hat y).
$$

Use one consistent definition throughout code and reporting. Add epsilon only for numerical safety, not to hide invalid targets. Do not mix variance loss with volatility-scale evaluation without explicitly transforming predictions.

Run at least these controlled variants:

- raw variance + QLIKE;
- log variance + MSE, inverse-transformed for evaluation;
- multiplicative HAR residual + QLIKE.

## 17. Evaluation metrics

Report per horizon, per ticker, pooled across tickers, and across time:

- MAE;
- RMSE;
- QLIKE;
- out-of-sample `R²` relative to HAR;
- Pearson correlation between target and prediction;
- mean forecast bias;
- underprediction rate, particularly in high-volatility states;
- directional accuracy only if its definition is predeclared and economically meaningful.

Relative out-of-sample `R²`:

$$
R^2_{OOS}=1-\frac{\sum(y-\hat y^{model})^2}
{\sum(y-\hat y^{HAR})^2}.
$$

Also report relative loss improvement:

$$
\Delta L\%=100\times\frac{L_{HAR}-L_{model}}{L_{HAR}}.
$$

Never claim that a model beats HAR based solely on lower pooled point-estimate loss.

## 18. Statistical comparison

For each horizon, compare every candidate against HAR using:

- Diebold–Mariano test with dependence handling appropriate to overlapping multi-step targets;
- block bootstrap confidence intervals for loss differences;
- Model Confidence Set when many models are compared;
- multiple-comparison adjustment or an explicitly separated confirmatory test when many architectures/hyperparameters have been explored.

For panel data, naive row-level standard errors are invalid because rows are dependent across dates and tickers. Aggregate loss differentials by date or use an appropriate two-way/block procedure. Document the exact statistical implementation.

## 19. Temporal leakage controls

Treat leakage safety as a property of the entire pipeline.

### Mandatory controls

- Use chronological splits only.
- Purge training rows whose target interval overlaps validation; purge validation rows whose target interval overlaps test.
- Derive the purge gap from the exact target interval, not only from the numeric horizon label.
- Fit scaling, imputation, PCA, clustering, feature selection, graph thresholds, regime thresholds, and ticker groupings on training data only within each fold.
- Compute rolling/EWMA/correlation/lead–lag features only from data available by the prediction cutoff.
- Construct or update graph edges independently in each fold.
- For news features, use availability time with timezone, after-market handling, updates, and ingestion delay; publication date alone may be insufficient.
- Use validation for early stopping, hyperparameters, alpha, gate, graph window, and threshold selection.
- Use the locked test once for final evaluation.
- Check historical-universe/survivorship bias separately.

### Feature-availability manifest

Create a table or CSV containing:

| feature | event_time | available_time | lookback | fitted_on | used_for |
|---|---|---|---|---|---|

Include every rolling feature, graph, scaler, cluster, embedding, regime state, market feature, and news feature.

### Cross-fitted residuals

When generating HAR residual targets for neural training, use expanding-window or inner-fold HAR predictions. Do not use residuals produced by a HAR fit that has seen the row being predicted if deployment uses genuine future forecasts.

## 20. Hyperparameter discipline

Keep capacity and search budget controlled:

- same sequence length candidates for comparable models, e.g. 22 and 44;
- same hidden dimensions where applicable;
- same number of seeds;
- same training epochs/early-stopping policy;
- same validation metric;
- predeclared limited search space;
- record total trainable parameters and runtime.

Suggested initial search:

- LSTM hidden size: `{16, 32, 64}`;
- GAT hidden size: `{16, 32, 64}`;
- GAT layers: `{1, 2}`;
- attention heads: `{1, 2, 4}` subject to parameter matching;
- dropout: `{0.0, 0.1, 0.2}`;
- learning rate: `{1e-4, 3e-4, 1e-3}`;
- weight decay: `{0, 1e-5, 1e-4}`;
- graph windows: `{20, 60, 120}`;
- gate type: `{constant, linear, MLP-8}`.

Do not launch the Cartesian product. Use a staged search: establish residual value first, then graph design, then gate complexity.

## 21. Reproducibility requirements

- Central configuration file for dates, horizons, target, features, graph, loss, and seeds.
- Fix Python, NumPy, and PyTorch seeds and record deterministic settings.
- Save package/environment versions.
- Save every trained checkpoint and validation selection rationale.
- Save row-aligned predictions for all models.
- Save alpha/gate distributions by horizon, ticker, date, and state.
- Save training curves and best epoch.
- Save model parameter counts and wall-clock runtime.
- Hash or version the input dataset and feature-generation code.
- Produce one machine-readable summary CSV and one Markdown report.

## 22. Required result tables

### Overall performance

| Model | Horizon | MAE | RMSE | QLIKE | R² OOS vs HAR | ΔQLIKE % | DM p-value | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### Expert contribution

| Horizon | alpha HAR | lambda correction | HAR–NN error corr. | LSTM residual R² | GAT residual R² | LSTM–GAT residual R² |
|---|---:|---:|---:|---:|---:|---:|

### Regime/state performance

| Observable state | Count | HAR QLIKE | Hybrid QLIKE | ΔQLIKE % | Mean gate | Underprediction difference |
|---|---:|---:|---:|---:|---:|---:|

### Per-ticker robustness

| Ticker | Horizon | HAR QLIKE | Hybrid QLIKE | Difference | Winner |
|---|---:|---:|---:|---:|---|

## 23. Decision rules

Call a hybrid successful only when all applicable conditions hold:

1. It improves the predeclared primary validation metric and the locked test metric versus HAR.
2. The loss differential is statistically defensible, not merely numerically smaller.
3. Improvement appears across multiple seeds and is not driven by one ticker or a few dates.
4. It beats a same-capacity no-graph residual model if graph value is claimed.
5. No critical temporal leakage or target-overlap issue remains.
6. The gate does not merely collapse to HAR everywhere unless the conclusion explicitly states that neural correction adds no value.

Interpret outcomes as follows:

- `alpha ≈ 1`: retain HAR; neural full-target expert is not complementary.
- residual model beats full neural but not HAR: anchoring helps stability, but incremental signal is insufficient.
- LSTM residual beats HAR but GAT residual does not: keep local temporal correction; remove graph.
- GAT residual beats LSTM residual and no-graph control: graph contains incremental spillover information.
- dynamic gate beats static lambda robustly: correction value is state-dependent.
- dynamic gate wins only on validation: likely gate overfit; reject it.

## 24. Recommended implementation priority

1. Verify E0–E2 with a clean common evaluation pipeline.
2. Run static alpha E3 and horizon alpha E4 using frozen predictions.
3. Generate cross-fitted HAR residuals.
4. Run E5 and E6 to isolate temporal and graph value.
5. Run E7 and the recommended multiplicative E8.
6. Analyze error complementarity and observable states.
7. Only if residual signal exists, run static and dynamic gating E9–E10.
8. Run statistical tests and leakage audit.
9. Lock the chosen model, then evaluate once on test.

## 25. Recommended final architecture

The preferred model to test is:

$$
\boxed{
\hat{PK}^{2}_{i,t+h}
=(\hat{PK}^{2,HAR}_{i,t+h}+\epsilon)
\exp\left[
\lambda_{i,t,h}
f_{LSTM+GAT}(X_{i,t},G_t)
\right]
}
$$

with:

- HAR frozen as the anchor;
- a zero-initialized residual output head;
- LSTM encoding ticker-local temporal history;
- GAT encoding eligible cross-stock information;
- a small gate using only observable state variables;
- soft routing;
- horizon-specific output heads;
- validation-selected complexity;
- positive predictions and QLIKE-compatible training.

The first implementation must also include a `HAR + LSTM residual` and `HAR + GAT residual` ablation. Without these, it is impossible to attribute any improvement to the graph.

## 26. Deliverables expected from the implementing AI

1. A short data-contract document confirming target construction and information cutoff.
2. Leakage-safe fold generator with target-overlap purge tests.
3. HAR baseline and row-aligned prediction export.
4. E0–E10 implementations, with staged execution rather than an uncontrolled full grid.
5. Unit tests for HAR fallback, positive outputs, graph cutoff, residual cross-fitting, and tensor dimensions.
6. Prediction CSV/Parquet for every fold/model/seed.
7. Metrics and statistical-comparison scripts.
8. Graph and gate diagnostic plots/tables.
9. Final Markdown report containing all required tables and an explicit accept/reject decision for every hypothesis.
10. A reproducible command sequence or experiment runner.

Do not silently alter the volatility target, split, horizon aggregation, graph timing, loss definition, or HAR implementation. If any contract item is unknown, stop and record it as a blocking question before making performance claims.
