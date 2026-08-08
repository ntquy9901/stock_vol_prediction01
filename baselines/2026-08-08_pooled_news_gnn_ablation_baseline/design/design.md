# Design — Pooled News and GNN Ablation Pilot

## 1. Architecture

### Pooled model

Each sample represents one ticker and one forecast origin:

```text
(ticker_id, target_date, x_price[22,Fp], x_news[22,Fn], news_mask[22],
 y_normalized, y_raw)
```

The price and news encoders share weights across all tickers:

```text
x_price ----------------> Shared Price LSTM ----> h_price ----+
                                                               +--> fusion --> shared head --> y_hat_norm
x_news + news_mask -----> Shared News LSTM -----> h_news ------+
                                                   ^
ticker_id --> sigmoid(gate_logits[ticker_id]) ----+
```

P1 omits the news path. P2 uses `h_news` directly. P3 uses
`sigmoid(gate_logits[ticker_id]) * h_news`. No pooled configuration requires a stock dimension or
common trading dates.

### Graph ablation

G0 and G1 consume graph-compatible snapshots after per-stock encoding:

```text
per-stock Price+News representations --> stack available nodes --> H_base

G0: H_base -------------------------------> prediction head
G1: H_base --> GNN(H_base, adjacency) -----> prediction head
```

The first pilot reuses the established common-date graph subset to minimize implementation scope.
That subset has its own global-date 70/15/15 split, so a graph snapshot cannot mix per-ticker split
labels. Dynamic union-calendar graphs are out of scope. G0 and G1 start from a byte-identical
graph-safe P3 checkpoint trained only on pooled samples whose target dates do not exceed the graph
training boundary. They use identical batches and optimizer settings and freeze all pretrained P3
components in evaluation mode; only G1 adds trainable message-passing parameters.

## 2. Components

### `data.py`

- Load and validate per-ticker price data.
- Perform per-ticker chronological 70/15/15 split.
- Resolve one shared eligibility manifest before building model-specific tensors; reject any later
  asymmetric sample removal.
- Fit train-only preprocessing parameters.
- Generate HAR features and windows independently within each split.
- Align news causally to the ticker's own trading dates.
- Produce deterministic sample manifests sorted by `(target_date, ticker_id)`.
- Build the graph-compatible G0/G1 manifests without changing the pooled manifests.
- Persist tensor/preprocessing hashes, ordered node vocabulary, split labels, and exclusion reasons.

### `scaling.py`

- Store per-ticker train-fitted price and target parameters.
- Transform by ticker ID.
- Inverse-transform predictions by ticker ID.
- Serialize parameters and feature order.
- Expose no API that can fit on validation/test data.

This module is baseline-local because changing shared scaler code would expand the blast radius to
existing headline baselines.

### `models.py`

- `HARReference`: P0.
- `PooledPriceLSTM`: P1.
- `PooledPriceNewsLSTM`: P2/P3, with the gate enabled only for P3.
- `GraphAblationModel`: G0/G1 with an explicit message-passing switch.

Only the minimum interfaces required by the six configurations are implemented. There is no model
registry or general experiment framework.

### `train.py`

- Run one named configuration for an explicit seed and epoch count.
- Enforce the 5-10 epoch project policy.
- Save validation metrics and learning curves at epoch 5/10.
- Keep test evaluation disabled during screening unless the retained confirmation configuration is
  explicitly evaluated.
- Exempt the closed-form P0 HAR regression from epoch, optimizer, early-stopping, checkpoint, and
  learning-curve requirements while keeping its sample manifest and evaluation code identical.

### `run_pilot.py`

- Execute the approved screening matrix sequentially.
- Verify manifest equality before training each ablation family.
- Produce a comparison table without automatically promoting a configuration.

## 3. Data flow

```text
Raw price/news
  -> validate ticker/date schema
  -> per-ticker chronological split
  -> train-only preprocessing fit
  -> split-local HAR/news windows
  -> pooled deterministic manifests
       -> P0 / P1 / P2 / P3
  -> graph-compatible manifest
       -> identical pretrained state
       -> G0 / G1
  -> raw-scale validation metrics
  -> screening decision
```

## 4. Leakage defenses

- Split occurs before HAR generation, scaler fitting, outlier-bound fitting, and window generation.
- Every sample stores `ticker_id` and `target_date`; neither is inferred from batch position.
- News alignment enforces the forecast-origin cutoff.
- The forecast origin is 15:00 Asia/Ho_Chi_Minh on the last input trading date; unknown or later
  publication timestamps are excluded.
- Training loaders are deterministic and unshuffled.
- Validation/test transformations are read-only applications of train parameters.
- Evaluation receives stored raw targets rather than inverse-transforming a clipped normalized
  target.
- Directional accuracy groups by ticker and sorts by target date.
- Graph construction uses only data available inside each graph input window.
- Any fitted graph normalization, topology threshold, or learned construction parameter uses graph
  training data only; dynamic edges use observations available by that snapshot's forecast origin.

## 5. Testing strategy

Test-first order:

1. Synthetic split and manifest tests.
2. Scaler isolation, per-ticker selection, round-trip, and raw-target preservation tests.
3. News cutoff, mask, and coverage tests.
4. P1/P2/P3 shape and gate-selection tests.
5. G0/G1 identical-input and message-passing-difference tests.
6. One-batch train/evaluate integration tests.
7. Real price/news slice smoke test.
8. Five-epoch pilot only after all gates pass.

Required adversarial review covers blind defects, edge cases, and acceptance-criteria compliance.

## 6. Error handling

Fail before training on:

- invalid, duplicate, or unsorted ticker dates;
- empty split or insufficient history for a 22-step, horizon-5 window;
- sample overlap or manifest mismatch;
- missing ticker/scaler/gate mapping;
- news records after the forecast-origin cutoff;
- zero news coverage caused by a mapping error;
- non-finite features, targets, losses, predictions, or metrics;
- denormalized nonpositive-prediction rate above 1%;
- G0/G1 graph manifest mismatch.

Legitimate no-news days are represented by the mask and do not fail.

## 7. Simplicity and anti-abstraction gates

- **Simplicity Gate: pass.** The implementation is isolated to one baseline and supports only the
  approved horizon-5 pilot matrix.
- **Anti-Abstraction Gate: pass.** It directly uses PyTorch, pandas, NumPy, existing evaluation
  utilities, and the established GNN component. It does not introduce a generic training framework,
  registry, or configuration hierarchy.
- Existing baselines and shared `src/` behavior remain unchanged.

## 8. Planned file layout

```text
baselines/2026-08-08_pooled_news_gnn_ablation_baseline/
├── requirements/requirements.md
├── design/design.md
├── code/
│   ├── __init__.py
│   ├── data.py
│   ├── scaling.py
│   ├── models.py
│   ├── train.py
│   └── run_pilot.py
├── test/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_scaling.py
│   ├── test_models.py
│   └── test_train_smoke.py
└── code_review/code_review_2026-08-08.md
```

Implementation files are created only after the written specification is reviewed.

## 9. Addendum 2026-08-08: G1 positivity parameterization

Diagnostic evidence (`temp/agent_g1_positivity_diagnostic_output/positivity_diagnostic.json`) showed
G1's message-passing widens the normalized-prediction variance so 1.78% of validation predictions
denormalized to nonpositive volatility, tripping the <=1% safety guard. Decision: `GraphAblationModel`
applies a denormalized-scale positive floor `raw_pos = eps*softplus(raw/eps) + eps` (eps = 1e-6),
identity for the bulk (spread preserved, no collapse), renormalized so the evaluation/inverse-transform
path is unchanged. The floor uses the existing train-fitted per-ticker target mean/std (no new
statistic, scaler/manifest/provenance unchanged) and is applied identically to G0 and G1. A whole-output
softplus-at-head was rejected: Parkinson volatility (~1e-3) sits in softplus's constant-offset region and
would require abandoning normalized-MSE training (also changing G0), and a normalized-space soft-clamp
would distort near-threshold predictions and bias the G0/G1 comparison.
