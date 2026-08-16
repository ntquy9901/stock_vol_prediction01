# Requirements — Pooled News and GNN Ablation Pilot

## 1. Objective

Determine, with a short horizon-5 pilot, whether a full-history pooled LSTM can outperform HAR on
the same pooled targets and whether news, a per-ticker news gate, and GNN message passing adds
measurable validation value. The pooled-versus-common-panel data effect is reported descriptively;
it is not treated as a causal architecture comparison in this pilot.

The pilot is a screening experiment. It does not establish a final paper result.

## 2. Experimental configurations

### Pooled protocol

- **P0 — HAR pooled:** non-neural HAR reference trained/evaluated on the same pooled sample IDs as
  P1-P3.
- **P1 — Price LSTM:** shared Price LSTM and shared prediction head.
- **P2 — Price + News:** P1 plus a shared News LSTM and fusion layer.
- **P3 — Price + News + Gate:** P2 plus a learned scalar gate selected by explicit `ticker_id`.

### Graph protocol

- **G0 — GNN OFF:** frozen P3 encoders with a trainable matched prediction head on
  graph-compatible samples, without message passing.
- **G1 — GNN ON:** identical to G0 except for GNN message passing.

P0-P3 form one valid ablation family. G0-G1 form a separate valid ablation pair. P3 and G1 may be
reported descriptively but must not be used as a causal GNN comparison because their sample sets
differ.

## 3. Data contract

- Forecast horizon: 5 trading days.
- Input sequence length: 22 observations.
- Universe: the same fixed project ticker vocabulary used for the pilot; the ordered
  `ticker -> ticker_id` mapping is resolved once from the sorted eligible price-file stems, saved
  as a versioned manifest, and persisted with every run and checkpoint.
- Split each ticker chronologically into 70% train, 15% validation, and 15% test before generating
  HAR features or sequence windows.
- Do not use random split.
- Generate HAR features independently inside each split, matching the current split-first project
  convention.
- Pool the completed per-ticker samples within each split and order them deterministically by
  `(target_date, ticker_id)`.
- Use `shuffle=False` for train, validation, and test loaders.
- P0-P3 must consume identical ordered sample IDs `(ticker_id, target_date)`.
- Eligibility is resolved once before model-specific tensors are built. A ticker is eligible only
  if each split can produce at least one 22-observation, horizon-5 window after split-local HAR
  generation. Excluded tickers and reasons are persisted and cannot change between P0-P3.
- The shared P0-P3 manifest stores hashes of sample IDs, raw price inputs, news inputs/masks, raw
  targets, and preprocessing versions; model-specific code may not remove samples.
- G0-G1 use their own common-date panel split 70/15/15 by a single global date axis. Every node in
  a graph snapshot therefore has the same split label; windows cannot cross graph split boundaries.
- G0-G1 must consume identical graph snapshot IDs, ordered node vocabulary, node masks, adjacency
  inputs, tensor hashes, and initial encoder/head checkpoint.
- A news feature for a sample may use only information available no later than that sample's
  forecast origin, defined as 15:00 Asia/Ho_Chi_Minh on the final input trading date. Records with
  unknown/unparseable timestamps or `published_at` after that instant are excluded. Missing news is
  a zero vector with `news_mask=0`; an all-missing sequence must produce a finite representation.
- `target_date` is the fifth subsequent trading observation after the final input observation.

## 4. Scaling contract

- Fit all price/HAR and target scaling parameters from the training partition only.
- Maintain separate price/target scaler parameters per ticker.
- Reuse the fitted training parameters unchanged for validation and test.
- Select the target inverse transform by explicit `ticker_id`, never by flattened position or
  `index % num_stocks`.
- Training loss uses the normalized target. Evaluation metrics use the original raw target and the
  prediction inverse-transformed with that ticker's training scaler.
- Persist scaler parameters, feature order, and ticker vocabulary with the run artifacts.
- News PCA or any learned news transformation must be fitted on training-period news only. A
  precomputed artifact may be reused only after its fit period is verified to be contained in the
  eligible training-news set under every per-ticker cutoff; otherwise it is refitted from the
  union of news records eligible for pooled training samples only.
- Outlier/winsorization statistics must not use validation or test data. If the pilot retains
  winsorization, its bounds are fitted per ticker from raw train, applied before HAR generation,
  and reused unchanged for all splits. Stored `y_raw` remains the unmodified evaluation target.
- Zero-variance training features/targets use the existing `VolatilityNormalizer` convention
  `std=1.0`; round-trip and finite-output behavior must be tested explicitly.

## 5. Training protocol

### Screening round

- Seed: 42.
- Epochs: exactly 5 unless a run fails fast.
- Architecture-independent settings such as loss, optimizer family, batch size, evaluation code,
  and early-stopping behavior must be held constant where the compared configurations permit.
- Model selection uses validation only. Test metrics are not used to select a configuration.

### Confirmation round

- Only configurations with a credible validation signal proceed.
- Epochs: up to 10.
- Seeds: 42, 123, and 2026.
- Validation results across the three seeds select exactly one final architecture. Test is then
  evaluated exactly once for each of that architecture's three predeclared seed checkpoints and
  aggregated; test results cannot change the architecture choice.

## 6. Metrics and artifacts

Every evaluated configuration reports all six mandatory metrics on raw volatility scale:

- MSE
- RMSE
- MAE
- R-squared
- QLIKE
- directional accuracy

Directional accuracy is calculated chronologically per ticker and then aggregated. Differences
must never cross ticker boundaries. For each ticker with at least two targets, it compares the
signs of successive target changes and successive prediction changes; ties retain sign zero. The
headline value is the unweighted macro mean across eligible tickers, with an observation-weighted
value reported as a secondary diagnostic.

MSE, RMSE, MAE, R-squared, and directional accuracy use unfloored denormalized predictions. Only
QLIKE applies the positive epsilon policy from `src.common.evaluation.qlike_loss`. The nonpositive
prediction rate is saved; a rate above 1% fails the screening gate rather than hiding an invalid
output distribution.

Every run saves:

- configuration and seed;
- ordered sample IDs or a reproducible manifest hash;
- ticker vocabulary and feature order;
- scaler and preprocessing parameters;
- epoch-level train/validation losses and validation metrics;
- a learning-curve image at epoch 5 and, when applicable, epoch 10; a failed run saves its partial
  diagnostic curve instead;
- best checkpoint and machine-readable results.

## 7. Acceptance criteria

### Data and implementation gate

- No sample ID overlaps across train, validation, and test.
- P0-P3 sample manifests are identical within each split.
- G0-G1 graph manifests are identical within each split.
- G0 and G1 start from a byte-identical graph-safe P3 checkpoint trained only on pooled samples
  whose target dates do not exceed the graph-training boundary. They use the same batches and
  optimizer settings and freeze all pretrained P3 components in evaluation mode; only G1's
  message-passing parameters are additional trainable parameters.
- Changing validation/test values cannot change train-fitted scaler or outlier parameters.
- Scale/inverse-transform round trips reconstruct unmodified values within numerical tolerance.
- Pooled inverse transformation and gate lookup select parameters by `ticker_id`.
- Raw evaluation targets remain unchanged even if normalized training targets are clipped.
- News alignment, timezone, publication cutoff, missing-news, and mapping tests pass, including a
  real-data sample. Zero matched news fails only when the independently parsed source contains an
  eligible ticker article in that period.
- Unit, integration, smoke, lint, changed-line coverage, and required adversarial review gates pass.

### Screening decision

A configuration is not promoted merely for winning one metric at one seed. Promotion to the
10-epoch confirmation round requires:

- finite losses and predictions;
- no data/scaler/news-coverage invariant failure;
- lower validation QLIKE than its direct control, RMSE degradation no greater than 1%, and
  directional-accuracy degradation no greater than one percentage point; and
- a learning curve that is stable or still improving at epoch 5.

### Architecture decision

- News is retained only if P2 beats P1 on QLIKE in at least two of three confirmation seeds and its
  median RMSE/DirAcc remain within the screening tolerances.
- The gate is retained only if P3 meets the same rule against P2.
- GNN is retained as an effective predictive component only if G1 improves consistently over G0
  on the identical graph protocol: lower paired QLIKE in at least two of three seeds, with median
  RMSE/DirAcc within the same tolerances. Otherwise it remains a documented research ablation.
- No claim that pooled training, news, gate, or GNN improves the model is made from the 5-epoch,
  single-seed screening result alone.

## 8. Go/no-go

- **Go:** all data/scaling gates pass and at least one learned configuration provides a credible
  validation signal under the rules above.
- **No-go:** any unresolved leakage or sample mismatch exists, or no learned configuration shows a
  credible validation signal. A no-go result stops longer training and is reported without adding
  architectural complexity.
