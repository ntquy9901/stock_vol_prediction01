# HAR–LSTM–GAT Result Diagnosis and Next-Experiment Handoff

## 1. Assignment

Use this document as an implementation and verification brief for the next iteration of the Parkinson-variance forecasting study.

The current study compares HAR with LSTM, LSTM–GAT, convex combinations, HAR-anchored residual models, and gated residual models on VN30, VN100, and S&P 500 panels. The immediate goals are:

1. Verify that the current conclusions are supported by the implementation and statistical design.
2. Identify why the graph branch fails to add reliable out-of-sample value.
3. Correct the graph representation and evaluation limitations before concluding that cross-stock spillover contains no signal.
4. Re-test the promising long-horizon VN100 combination under higher-power walk-forward evaluation.

Do not tune a larger GAT before completing the verification and model-free screening stages in this document.

## 2. Current prediction contract

Confirm all items from code and data rather than relying only on the report.

- Target: daily **Parkinson variance**, computed from daily High/Low:

  $$
  PK^2_{i,t}=\frac{1}{4\ln 2}\left[\ln\left(\frac{H_{i,t}}{L_{i,t}}\right)\right]^2.
  $$

- Source column is reportedly named `parkinson_volatility`, although its content is variance. Treat this as a semantic risk and verify the values and all transformations.
- No intraday data is available. Do not call the target realized volatility or realized variance.
- Horizons: `h ∈ {1,5,10,22}`.
- Current horizon target: terminal value `PK²[t+h]`, not the forward sum or average.
- Current lookback: 10 snapshot dates; HAR daily/weekly/monthly aggregates may be precomputed before the sequence, so verify their exact lineage.
- Current panels: VN30, VN100, and a long-history S&P 500 subset.
- Current split: chronological 80/10/10 with target-boundary purge.
- Current graph: train-fitted graphical-lasso, signed Top-5 partial-correlation adjacency, frozen across validation and test.
- Current panel representation: common-date snapshots.
- Current primary loss: QLIKE on variance:

  $$
  L_{QLIKE}(y,\hat y)=\frac{y}{\hat y}+\log(\hat y),\qquad y>0,\hat y>0.
  $$

- Current statistical comparison: loss differentials aggregated by date followed by HLN-adjusted Diebold–Mariano testing.
- Current seed set: `{42, 123, 2026, 7, 2024}`.

Document the prediction timestamp explicitly: is the forecast produced before or after the close of day `t`? Day-`t` High/Low is permitted only if it is known at that cutoff.

## 3. Current evidence and appropriate interpretation

### 3.1 VN30

The point estimates do not favour the hybrid:

| Horizon | HAR QLIKE | Best reported deep model | E3 improvement vs HAR | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 0.3946 | E1 0.4383 | −0.9% | HAR clearly preferred |
| 5 | 0.4531 | E1 0.4810 | −2.4% | HAR clearly preferred |
| 10 | 0.4849 | E1 0.5291 | −2.2% | HAR clearly preferred |
| 22 | 0.5025 | E1 0.5185 | −2.2% | HAR clearly preferred |

For the current configuration, VN30 is a relatively strong negative result. The failure is not explained only by low test power because the hybrid point estimates are also worse.

### 3.2 VN100

The combination displays a horizon-dependent pattern:

| Horizon | E3 QLIKE improvement vs HAR | Date-clustered DM p-value | Interpretation |
|---:|---:|---:|---|
| 1 | +0.5% | 0.24 | small, uncertain |
| 5 | +1.9% | 0.29 | small, uncertain |
| 10 | +4.8% | 0.20 | economically interesting, underpowered |
| 22 | +5.9% | 0.08 | most promising candidate, not confirmatory |

Do not interpret `p ≥ 0.05` as proof of no effect. With approximately 49–130 test dates, and overlapping multi-step forecasting dependence, power is limited. E3 at VN100 horizons 10 and 22 should be treated as candidates for confirmation on a longer locked walk-forward test.

### 3.3 Graph conclusion

The justified conclusion is currently:

> A static train-fitted Top-5 graphical-lasso graph, combined with HAR-only node features and the current common-date evaluation, does not add statistically reliable out-of-sample value.

Do **not** elevate this to:

> No cross-stock spillover exists or no graph definition can beat HAR.

The existing model-free neighbour-mean test rejects only a narrow class of equal-weight, contemporaneous, symmetric spillover.

## 4. Blocking verification items

Resolve these before running new models.

| ID | Severity | Verification | Why it matters | Required fix/evidence |
|---|---|---|---|---|
| V1 | HIGH | Confirm `parkinson_volatility` contains variance and is never square-rooted or squared again | A scale error invalidates HAR inputs and QLIKE | Add formula/scale assertions and metadata |
| V2 | HIGH | Determine whether signed edge values are used by GAT or only `adjacency != 0` masking | If only masking is used, the signed weighted graph is actually binary and unsigned | Trace adjacency through attention logits and messages; add unit tests |
| V3 | HIGH | Confirm exact prediction cutoff and availability of day-`t` OHLC | Day-`t` High/Low leaks if predicting before close | Document cutoff and enforce it in feature generation |
| V4 | HIGH | Verify target start/end dates and boundary purge | Numeric horizon alone is not proof that no label overlaps | Persist target interval per row and test partitions |
| V5 | HIGH | Verify test was not used to choose model, graph window, threshold, seed, floor, or gate | Test-driven selection invalidates performance claims | Separate exploratory and locked confirmatory test periods |
| V6 | MEDIUM | Verify DM HAC lag/bandwidth and HLN effective sample definition | Date aggregation handles cross-sectional dependence but not automatically serial dependence | Document code and add block-bootstrap CI |
| V7 | MEDIUM | Verify expanding cross-fitted HAR residuals use sufficient warm-up and the deployment HAR specification | Early weak anchors can create a residual-distribution mismatch | Add minimum warm-up and residual lineage |
| V8 | MEDIUM | Quantify common-date selection and survivorship bias | Complete-case snapshots shorten the test and favour long-history stocks | Report excluded dates/tickers and historical-universe policy |
| V9 | REVIEW | Reconcile claims that E2 “hurts” with VN100 h1, where E2 is better than E1 | The graph effect is panel-dependent, not uniformly harmful | Rewrite conclusion by panel and horizon |
| V10 | REVIEW | Reconcile test residual R² ≈ 0.039 with no QLIKE gain | Pooled residual R² may be driven by scale/regime and not translate to forecast loss | Add calibration and macro/date-level residual metrics |

The leakage verdict remains **PASS WITH CONDITIONS / NOT FULLY PROVEN** until V1–V8 are supported by file-and-line evidence.

## 5. Critical architecture issue: signed graph may be reduced to a binary mask

The report describes a signed Top-5 partial-correlation graph, but the verified graph path is summarized as:

```text
adjacency != 0 → -inf mask → softmax over neighbors
```

If the numerical value and sign of `A[i,j]` never enter the attention logit or the message, then:

$$
A^{model}_{ij}=\mathbb{1}(A_{ij}\neq0),
$$

and edges `+0.6`, `−0.6`, and `+0.1` are indistinguishable. This can cause neighbour averaging, uniform attention, and dilution of the node-local HAR signal.

### Required inspection

Trace and document:

1. The tensor returned by graphical lasso.
2. Top-k selection and preservation of sign/weight.
3. Batch broadcast from `[N,N]` to `[B,N,N]`.
4. Attention-logit calculation.
5. Whether edge sign/weight enters the logit.
6. Whether edge sign/weight enters the message.
7. Whether negative relations have separate parameters.

### Required unit tests

- Changing edge weight while keeping the mask constant must change the output of a weighted GAT.
- Flipping an edge sign must change the output of a signed GAT.
- Removing every non-self edge must match the no-graph/identity-graph result within tolerance.
- Shuffling edge values while preserving the mask must change the weighted model but not a binary-mask control.

### Architecture to implement

Test a dual-relation signed layer:

$$
m_i=
\sum_{j:A_{ij}>0}\alpha^+_{ij}W^+h_j
-
\sum_{j:A_{ij}<0}\alpha^-_{ij}W^-h_j.
$$

Alternatively, condition attention on edge attributes:

$$
e_{ij}=MLP\left(Wh_i\Vert Wh_j\Vert A_{ij}\Vert |A_{ij}|\Vert s_{ij}\right),
$$

where `s_ij` may include sign, edge stability, and edge age.

Do not claim that signed graphical-lasso edges were tested unless the sign or signed relation is consumed by the model.

## 6. Static graph instability

Reported train/test diagnostics show weak transfer:

- neighbour Jaccard: 0.17 VN30, 0.09 VN100;
- edge-set Jaccard: 0.16 VN30, 0.09 VN100;
- edge-weight correlation: 0.19 VN30, 0.15 VN100;
- negative-edge share changes from approximately 3% to approximately 19%.

This directly contradicts the assumption that one full-train graph should remain valid throughout validation and test.

### Required replacement

Construct leakage-safe rolling or expanding graphs:

$$
G_t=f(X_{t-W+1:t}),qquad W\in\{20,60,120,252\}.
$$

At forecast origin `t`, graph construction may use only observations available by `t`.

Test temporal shrinkage:

$$
\tilde A_t=\rho\tilde A_{t-1}+(1-\rho)A_t,
$$

with `rho` selected on validation only.

Also calculate edge stability across adjacent windows and optionally retain only edges appearing in a minimum proportion of recent windows.

For each graph version, save:

- direction;
- sign and weight;
- top-k/threshold;
- estimation window;
- minimum pairwise observations;
- update frequency;
- edge stability;
- graph density and degree distribution;
- exact information cutoff.

## 7. Model-free spillover screening before another GAT

The existing test adds the equal-weight mean of current Top-5 neighbours to HAR. This is useful but not decisive. Run the following inexpensive screening ladder using fold-safe features and validation selection.

### S0 — Existing equal-weight neighbour mean

$$
s^{mean}_{i,t}=\frac{1}{|N_i|}\sum_{j\in N_i}PK^2_{j,t}.
$$

Retain as the baseline diagnostic.

### S1 — Weighted neighbour variance

$$
s^{weighted}_{i,t}=\sum_j A_{ij,t}PK^2_{j,t}.
$$

### S2 — Separate positive and negative relations

$$
s^+_{i,t}=\sum_{A_{ij,t}>0}|A_{ij,t}|PK^2_{j,t},
$$

$$
s^-_{i,t}=\sum_{A_{ij,t}<0}|A_{ij,t}|PK^2_{j,t}.
$$

### S3 — HAR-residual neighbour signal

First calculate cross-fitted residual innovations:

$$
r^{HAR}_{j,t}=PK^2_{j,t}-\hat{PK}^{2,HAR}_{j,t}.
$$

Then:

$$
s^r_{i,t}=\sum_j A_{ij,t}r^{HAR}_{j,t}.
$$

This tests whether the graph predicts volatility innovations rather than the persistent level that HAR already captures.

### S4 — Directed lead–lag signal

Estimate only on training history:

$$
r^{HAR}_{i,t+h}=\gamma_{ij,h}r^{HAR}_{j,t}+\epsilon_{i,t+h}.
$$

Use directed edges `j → i` when the relationship is stable across inner training blocks. Do not select lead–lag edges using validation/test targets.

### S5 — Volume-shock graph

Construct historical volume shocks, for example from training-fitted or rolling standardized log volume. Test whether neighbour volume shocks incrementally forecast own future Parkinson variance.

### S6 — Static sector graph

Use a stable structural graph as a low-variance alternative. Sector classifications must be historically valid for the evaluated dates.

### S7 — Regime interaction

Test whether spillover is conditional on an observable state:

$$
r^{HAR}_{i,t+h}=\gamma_1s_{i,t}+\gamma_2s_{i,t}\times HighVol_t+\epsilon_{i,t+h}.
$$

`HighVol_t` must be constructed from information available at `t`, with thresholds fitted within training.

### Screening acceptance rule

Promote a graph to GAT only when it shows:

- positive incremental validation R² or QLIKE improvement;
- the same direction across multiple folds;
- reasonable edge transfer/stability;
- improvement not concentrated in one ticker or a few dates;
- better performance than a shuffled-edge/placebo signal.

## 8. Attention diagnosis

Attention entropy near 0.99 of the uniform maximum proves that the current layer acts approximately as neighbour averaging. It does not by itself prove that no predictive neighbour structure exists.

Instrument training to save by epoch, layer, head, date, and ticker:

- pre-softmax attention-logit variance;
- attention entropy;
- gradient norm reaching the GAT;
- gradient norm of the residual head;
- attention correlation with edge strength;
- correction mean, variance, and magnitude relative to HAR;
- proportion of nearly uniform rows;
- proportion of isolated/invalid nodes.

Only after verifying useful graph signal should the following be tested:

- GATv2;
- edge-conditioned attention;
- positive/negative relation channels;
- attention temperature;
- sparsemax/entmax as ablations.

Do not optimize for low attention entropy. Sharper attention is not evidence of better forecasts.

## 9. Target design experiment

The current terminal target is:

$$
y^{terminal}_{i,t,h}=PK^2_{i,t+h}.
$$

At h=22 this asks the model to forecast the variance of one specific day 22 sessions ahead. This is noisy and differs from a forward-risk target:

$$
y^{mean}_{i,t,h}=\frac1h\sum_{k=1}^{h}PK^2_{i,t+k}.
$$

Do not silently replace the current target. Run a separate experiment family:

- `terminal_h1`, `terminal_h5`, `terminal_h10`, `terminal_h22`;
- `forward_mean_h5`, `forward_mean_h10`, `forward_mean_h22`.

For forward means, purge rows based on the entire interval `[t+1,t+h]`. Refit HAR, scalers, residual targets, graphs, alpha, and gates independently. Never compare the two target families as if their QLIKE values measured the same task.

Hypothesis: cross-stock information may be more useful for forward-average risk because measurement noise is reduced and spillovers accumulate over the horizon.

## 10. Panel design and power

Strict common-date intersection greatly reduces the number of independent test dates and can create long-history/survivorship selection.

### Required reporting

- total dates before intersection;
- dates retained per split;
- ticker count per date;
- excluded tickers and reasons;
- historical index membership policy;
- number of effective date observations for inference.

### Preferred masked-panel alternative

Use the union of dates with:

- `node_mask` for valid input nodes;
- `target_mask` for valid labels;
- attention mask for unavailable nodes/edges;
- loss calculated only on valid targets;
- graph estimation using pairwise observations with minimum overlap.

If implementing a masked panel is too disruptive, retain a stable long-history subset but name it explicitly and do not claim that it represents the entire contemporary index.

Increasing test dates is more valuable for date-clustered inference than adding hundreds of tickers to the same small number of dates.

## 11. Residual calibration and positive reconstruction

The reported additive residual failures on S&P 500, where predictions approach zero and QLIKE explodes, indicate unstable reconstruction.

Prefer the multiplicative HAR anchor:

$$
\hat y_{i,t+h}=(\hat y^{HAR}_{i,t+h}+\epsilon)\exp(\lambda_{i,t,h}\delta_{i,t+h}).
$$

Bound the log correction if required:

$$
\delta'=c\tanh(\delta/c),
$$

where `c` is chosen from training/validation only.

Optionally regularize correction magnitude:

$$
L=L_{QLIKE}+\eta E[(\lambda\delta)^2].
$$

Save and inspect:

$$
R_{i,t,h}=\frac{\hat y^{hybrid}_{i,t+h}}{\hat y^{HAR}_{i,t+h}+\epsilon}.
$$

Report ratio quantiles, near-zero predictions, extreme predictions, and the fraction affected by any floor or clipping.

For residual predictions, report:

- pooled residual R²;
- macro per-ticker residual R²;
- date-level residual R²;
- correlation between residual target and prediction;
- `Var(predicted residual) / Var(actual residual)`;
- calibration regression slope in `r = a + b*r_hat + error`.

Train/test residual R² values are not directly comparable unless the same denominator and aggregation are used. Test R² above train R² is not sufficient evidence that overfitting is absent.

## 12. Statistical evaluation

For model `m`, calculate per-ticker/date loss differential against HAR:

$$
d_{i,t}=L(y_{i,t},\hat y^{HAR}_{i,t})-L(y_{i,t},\hat y^{m}_{i,t}).
$$

Aggregate by date:

$$
\bar d_t=\frac{1}{N_t}\sum_{i=1}^{N_t}d_{i,t}.
$$

Required inference:

1. Date-level DM/HLN-DM with documented HAC lag or bandwidth.
2. Moving-block bootstrap confidence interval over dates.
3. Effect size and confidence interval, not only p-value.
4. Model Confidence Set for locked candidate models.
5. Multiple-comparison adjustment or a separate confirmatory test after exploratory selection.
6. Per-ticker and per-state robustness tables.

Do not use row-level `N×T` DM. Do not select a statistical method because it produces a lower p-value. Driscoll–Kraay and two-way clustering are not automatically more powerful or more reliable when `T` is small.

For multi-step forecasts, explicitly handle serial dependence. Record the effective number of test dates and block length/HAC bandwidth for every horizon.

## 13. Next experimental ladder

Run in order. Stop escalation when the cheaper predecessor shows no validation signal.

| ID | Experiment | Purpose | Promotion condition |
|---|---|---|---|
| N0 | Locked HAR | Reproduce baseline | Exact row-aligned reproduction |
| N1 | Validation-fitted HAR/E1 or HAR/E2 convex alpha by horizon | Confirm combination value | Positive walk-forward validation effect |
| N2 | HAR + weighted neighbour residual | Test weighted contemporaneous signal | Better than S0 and shuffled control |
| N3 | HAR + signed positive/negative residual | Test sign information | Better than binary/unsigned version |
| N4 | HAR + directed rolling lead–lag residual | Test predictive spillover | Stable across folds |
| N5 | HAR + sector residual | Test stable structural relation | Positive incremental validation effect |
| N6 | HAR + edge-conditioned signed GAT residual | Neural graph after screening | Beats N0 and same-capacity N9 |
| N7 | N6 with shuffled edges | Placebo | Must be worse than N6 |
| N8 | N6 without edge sign/weights | Edge-attribute ablation | Must be worse than N6 to claim signed-edge value |
| N9 | N6 with identity/no graph | Same-capacity control | N6 must beat N9 |
| N10 | N0–N6 on forward-average target | Test target-noise hypothesis | Separate result family |

For each model use the same folds, observations, seeds, training budget, early-stopping rule, primary metric, and test isolation.

## 14. Recommended graph-residual architecture

Use HAR as a frozen anchor. The graph branch predicts only the innovation/correction:

$$
\hat{PK}^{2}_{i,t+h}
=(\hat{PK}^{2,HAR}_{i,t+h}+\epsilon)
\exp\left[
\lambda_{i,t,h}
f_{signed\text{-}GAT}(X_{1:N,t-W+1:t},G_t)
\right].
$$

Recommended properties:

- zero-initialized residual head;
- positive prediction by construction;
- rolling graph using eligible history only;
- edge value and sign consumed by the layer;
- separate positive and negative relations or edge-conditioned messages;
- ticker-local self-information retained through a residual/skip path;
- correction magnitude calibrated on validation;
- horizon-specific output heads;
- soft gate only after residual signal has been demonstrated.

## 15. Feature expansion after graph screening

Three HAR features mostly encode local volatility persistence. If graph screening on HAR-only features remains null, test additional fold-safe node features:

- return and absolute return;
- market return;
- market Parkinson variance;
- stock volatility relative to market volatility;
- volatility-of-volatility;
- cross-sectional volatility dispersion;
- volume shock and turnover;
- liquidity features;
- sector return and sector variance;
- historical beta or factor exposures.

All rolling estimates, factors, scalers, thresholds, and clusters must use data available by the forecast cutoff and must be fitted inside each training fold.

## 16. Execution phases

### Phase A — correctness audit

- Resolve V1–V10.
- Produce file-and-line evidence.
- Add target-scale and signed-edge unit tests.
- Reproduce existing metrics from saved row predictions.

### Phase B — cheap graph screening

- Implement S0–S7.
- Measure incremental validation QLIKE/R².
- Measure graph stability and placebo performance.
- Reject edge families with no stable signal.

### Phase C — corrected graph model

- Implement rolling signed/weighted graph.
- Implement edge-conditioned or dual-relation GAT.
- Use HAR-anchored multiplicative residual output.
- Run N6–N9 ablations.

### Phase D — higher-power confirmation

- Use expanding walk-forward evaluation.
- Extend the locked test period where possible.
- Evaluate VN100 E3 at horizons 10 and 22.
- Evaluate corrected S&P 500 panel.
- Run date-level bootstrap and confirmatory DM/MCS.

### Phase E — alternate target family

- Run terminal and forward-average targets as separate experiments.
- Compare qualitative conclusions, not raw loss levels across targets.

## 17. Acceptance criteria

### Claim: combination beats HAR

Require all of:

- lower locked-test primary loss;
- positive effect across multiple walk-forward folds;
- stable result across seeds;
- confidence interval supporting improvement or confirmatory test after model lock;
- no test-driven alpha/model selection;
- improvement not driven by a few dates/tickers.

### Claim: graph adds value

Require all of:

- corrected graph model beats HAR;
- corrected graph model beats same-capacity identity/no-graph model;
- corrected graph model beats shuffled-edge placebo;
- edge sign/weight ablation worsens performance if signed-edge value is claimed;
- model-free graph signal is positive or a justified explanation shows why only nonlinear GAT captures it;
- edge construction is leakage-safe and stable enough to transfer.

### Claim: graph contains no useful signal

Limit the claim to the graph family, features, target, panel, and period tested. A broad no-signal conclusion requires negative results across weighted, signed, directed, structural, and rolling graph definitions with adequate test power.

## 18. Required outputs

The implementing AI must create:

1. `reports/next_iteration_leakage_audit.md`
2. `reports/signed_graph_implementation_audit.md`
3. `reports/model_free_graph_screening.md`
4. `reports/walk_forward_results.md`
5. Machine-readable metrics by model/fold/horizon/seed.
6. Row-aligned predictions with ticker, forecast date, target interval, actual, HAR prediction, hybrid prediction, alpha/gate, fold, and seed.
7. Graph diagnostics by update date: density, degree, sign share, stability, weight distribution, and overlap.
8. Attention/gradient diagnostics.
9. Unit tests for target scale, cutoff, purge, edge sign/weight consumption, HAR fallback, and positive output.
10. A final hypothesis table with `SUPPORTED`, `NOT SUPPORTED`, or `INCONCLUSIVE` and evidence links.

## 19. Commands and reproducibility

Expose separate commands or configuration IDs for:

- audit only;
- model-free graph screening;
- one experiment/model/horizon/seed;
- full walk-forward suite;
- statistical comparison;
- report assembly.

Every output must record:

- dataset hash/version;
- code commit;
- target definition and scale;
- prediction cutoff;
- split/fold dates;
- target interval and purge rule;
- graph definition and update schedule;
- random seed;
- package versions;
- model parameter count;
- training runtime;
- validation selection rationale.

## 20. Final instruction to the implementing AI

Begin by inspecting the repository and writing a concise execution plan mapped to V1–V10. Do not assume the report is identical to the current code. Cite file paths and line numbers for every correctness conclusion.

Do not increase GAT depth, head count, or hidden size until signed-edge consumption and model-free spillover screening are verified. Do not use the locked test set to choose graph definitions or hyperparameters. Preserve existing results and unrelated user changes.

The most promising confirmatory candidate is currently the VN100 horizon-specific convex combination at horizons 10 and 22. The most important graph correction is to replace a potentially binary/unsigned static adjacency with a leakage-safe rolling graph whose weight and sign are actually consumed by the residual model.
