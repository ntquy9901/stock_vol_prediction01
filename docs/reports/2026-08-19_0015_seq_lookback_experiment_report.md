# LSTM lookback (SEQ) experiment: seq 5 / 10 / 22 under the 90/10 retrain protocol

Date: 2026-08-19. Universe: VN30 (33 tickers). Target: Parkinson volatility (variance) at t+h,
point forecast. Protocol: retrain-on-(train+val) = ratios (0.80, 0.10, 0.10), test read once
(no selection on test). Loss families reported via Diebold–Mariano (HLN, HAC lag h−1) with the
shared QLIKE positivity floor. Rung order: HAR → FULL → minus_graph → minus_gate → minus_news →
lstm_only.

## 1. What was run

- **Exploratory (5 epochs, seed 42):** seq ∈ {5, 10, 22}, horizons {1, 5, 10, 22}, all 6 rungs.
  Runner: `scripts/seq_lookback/run_seq.py` (wraps the delivered `run_retrain_trainval`; overrides
  only `combo_ladder.SEQ`, the split ratio to 90/10, and adds a basis smoke-assert).
- **Confirmatory (15 epochs, seeds {42, 123, 2026}):** seq ∈ {5, 10}, horizons {1, 5, 10, 22},
  all 6 rungs. Driver: `scripts/seq_lookback/run_15ep_3seed.sh` (4 concurrent processes; the batch=1
  GPU is otherwise idle, so concurrency saturated it — observed 100% util at 4 processes).
- **DM:** `scripts/seq_lookback/dm_seq_compare.py` (cross-seq, same rung, seed-ensembled predictions,
  intersected on the common test observations — a shorter lookback yields extra early test windows,
  so keys are intersected, not required identical) and `dm_retrain.dm_pair` (deep-vs-HAR within a run).

## 2. Result — 15-epoch, 3-seed (authoritative)

### 2.1 Mean test QLIKE (3 seeds), lower is better

seq5:

| h | HAR | FULL | minus_graph | minus_gate | minus_news | lstm_only |
|---|---|---|---|---|---|---|
| 1 | 0.4642 | 0.4601 | 0.4571 | 0.4610 | 0.4578 | 0.4585 |
| 5 | 0.5531 | 0.5586 | 0.7092† | 0.5672 | 0.5458 | 0.5485 |
| 10 | 0.5924 | 0.6123 | 0.6082 | 0.6152 | 0.6005 | 0.5966 |
| 22 | 0.6406 | 0.6796 | 0.6927 | 0.6840 | 0.6607 | 0.6576 |

seq10:

| h | HAR | FULL | minus_graph | minus_gate | minus_news | lstm_only |
|---|---|---|---|---|---|---|
| 1 | 0.4647 | 0.4581 | 0.4555 | 0.4582 | 0.4581 | 0.4582 |
| 5 | 0.5493 | 0.5604 | 0.5682 | 0.5694 | 0.5442 | 0.5460 |
| 10 | 0.5944 | 0.6201 | 0.6134 | 0.6322 | 0.6028 | 0.5973 |
| 22 | 0.6403 | 0.7574† | 0.6963 | 0.7584† | 0.6747 | 0.6589 |

† Outlier from a diverged seed at the 15-epoch budget (see §4). RMSE/MAE are near-identical across
rungs at 4 decimals because volatility magnitudes are ~5e-4; QLIKE is the discriminating metric.

### 2.2 Deep vs HAR (DM QLIKE; negative = deep beats HAR; * p<0.05)

| h | seq5 FULL | seq5 lstm_only | seq10 FULL | seq10 lstm_only |
|---|---|---|---|---|
| 1 | −3.95* beat | −3.24* beat | −4.26* beat | −4.15* beat |
| 5 | −0.71 tie | −2.02* beat | +1.38 tie | −2.02* beat |
| 10 | +2.73* lose | +1.42 tie | +2.65* lose | +0.88 tie |
| 22 | +3.04* lose | +5.07* lose | +4.24* lose | +4.89* lose |

Both lookbacks beat HAR at h1 (FULL and lstm_only). `lstm_only` also beats HAR at h5 (both
lookbacks). Both lose to HAR at h10 and h22. Pattern is consistent across seq5 and seq10.

### 2.3 seq5 vs seq10 direct (DM QLIKE; negative = seq5 better; * p<0.05)

| h | FULL | lstm_only |
|---|---|---|
| 1 | +0.47 tie | +5.31* seq10 |
| 5 | −2.74* seq5 | −1.19 tie |
| 10 | +0.24 tie | +2.47* seq10 |
| 22 | −3.86* seq5 | −1.04 tie |

The winner is model-dependent: for **FULL**, seq5 is significantly better at h5 and h22 (tie
elsewhere); for **lstm_only**, seq10 is significantly better at h1 and h10 (tie elsewhere).

## 3. Answers to the driving questions

1. **Does a shorter lookback improve accuracy?** For this noisy anti-persistent target, shorter
   lookback (5, 10) is not worse than 22 and beats HAR at more horizons than seq22 does. seq22 beats
   HAR at zero horizons (best case: ties); seq5/seq10 beat HAR at h1 (and lstm_only at h5). Long
   horizons (h10, h22) remain HAR's — no lookback recovers them.
2. **seq5 vs seq10, which is better?** No overall dominance. FULL favors seq5 (h5, h22); pure
   lstm_only favors seq10 (h1, h10). Both are far ahead of seq22 at beating HAR.
3. **Why shorter does not lose information:** HAR features are per-day precomputed columns
   (har_weekly = 5d rolling, har_monthly = 22d rolling); SEQ only slices
   `feature_values[start:start+SEQ]` (data.py:437), so har_monthly is present in every timestep even
   at SEQ=5. Additionally the GAT branch consumes only the raw features at day t
   (`node_raw = price[:,:,-1,:]`, model.py:89) and is therefore invariant to SEQ; only the LSTM
   (price) and news-LSTM branches consume the SEQ window.

## 4. Caveats

- **15-epoch instability:** at the 15-epoch fixed budget with no early stopping, FULL/minus_gate
  produced diverged-seed outliers (seq10 h22 = 0.757 vs HAR 0.640). `lstm_only` and `minus_news`
  stayed robust. Longer training did not uniformly help the graph-carrying rungs.
- **Single market:** VN30 case study; no claim of cross-market generality.
- **Loss-family disagreement:** QLIKE (weights low-vol days) and SE/MAE can disagree in sign;
  conclusions above are on QLIKE, the volatility standard.

## 5. Literature (deep-research, adversarially verified)

There is **no universal proof that a shorter lookback is better** for point forecasts; the evidence
is mixed and task-dependent.

- **Shorter-can-win (landmark):** Zeng, Chen, Zhang & Xu, "Are Transformers Effective for Time
  Series Forecasting?" (DLinear/LTSF-Linear), AAAI 2023 (arXiv 2205.13504) — a one-layer linear model
  outperforms sophisticated long-context Transformers on nine datasets, often by a large margin.
- **Longer-helps (counter-evidence):** Nie, Nguyen, Sinthong & Kalagnanam, "A Time Series is Worth 64
  Words" (PatchTST), ICLR 2023 (arXiv 2211.14730) — patching lets a Transformer exploit longer
  history; MSE decreases as lookback grows 96→720 on standard LTSF benchmarks (not noisy RV series).
- **Volatility long memory:** Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility,"
  J. Financial Econometrics 7(2):174–196, 2009 — long-horizon lags matter, captured parsimoniously by
  a fixed 3-scale cascade (1/5/22 days), not an ever-longer raw window. Motivated by the Heterogeneous
  Market Hypothesis (Müller et al. 1993).
- **No universal optimum:** Abdelmalak et al., "Channel Dependence, Limited Lookback Windows, and the
  Simplicity of Datasets…," arXiv:2502.09683, 2025 — lookback is a critical hyperparameter often set
  arbitrarily; must be tuned per task, and failing to do so can invert model rankings.
- **LSTM finance example:** an S&P500 daily-close LSTM found a longer (100-day) window best among
  25/50/100 (medium confidence) — a direct counter-example to a universal "shorter is better" rule.

Recommended Related-Work framing: describe the result as *consistent with the observation that noisy,
anti-persistent financial series benefit little from a longer raw lookback*, NOT as evidence that
shorter is universally better.

## 6. Artifacts

- Runners: `scripts/seq_lookback/run_seq.py`, `run_15ep_3seed.sh`, `dm_seq_compare.py`.
- 5-epoch results: `results/volatility_retrain_h{1,5,10,22}_seed42_2026-08-18_2155_seq{5,10,22}/`.
- 15-epoch 3-seed results: `results/volatility_retrain_h{1,5,10,22}_seed{42,123,2026}_2026-08-18_2306_15ep_seq{5,10}/`.
- Architecture correction recorded in CLAUDE.md §Ablation and memory `project_gat_uses_raw_features`.

## 7a. Control — is beating HAR an artifact of the 90/10 split? (No)

Concern: the deep model might beat HAR only because of the 90/10 retrain protocol, not the lookback.
Two facts already argue against a data-advantage artifact: (i) HAR is refit on the SAME 90% train+val,
so both models see identical training data; (ii) under the same 90/10, seq22 beats HAR at zero
horizons while seq5/seq10 beat at h1 — so lookback is separable. To isolate the SPLIT itself, seq5 was
re-run under the STANDARD 70/15/15 split with early stopping (patience=3, min_epochs=6, ≤10 epochs,
seed 42) via `scripts/seq_lookback/run_seq_ablation.py` (wraps the delivered `run_ablation`).

seq5 @70/15/15 + early-stop, deep vs HAR (DM QLIKE; negative = beats HAR; * p<0.05):

| h | FULL vs HAR | lstm_only vs HAR | n |
|---|---|---|---|
| 1 | −0.77 (0.442) tie | −4.62* beat | 15169 |
| 5 | −2.23* beat | −2.02* beat | 15037 |
| 10 | +3.88* lose | +0.30 tie | 14872 |
| 22 | +3.57* lose | +3.85* lose | 14476 |

The beat-HAR-at-short-horizon reproduces under 70/15/15 (lstm_only beats at h1 AND h5), so it is NOT a
90/10 artifact — it is a genuine effect of the deep (short-lookback) model. `lstm_only` is the most
consistent HAR-beating rung across both protocols. HAR remains best at h10/h22 under both splits.

## 7. Code review / DoD status

- **Tests:** `scripts/seq_lookback/test/test_seq_lookback.py`, 12 tests, all pass under the GPU venv.
  Cover: DM intersection count, target-mismatch raise, no-overlap raise, seed-ensemble mean,
  seed-obs-mismatch raise, 90/10 ratio patch, and 5 basis smoke-assert modes (valid, wrong
  price_dim, wrong seq_dim, all-zero block, single-zeroed column).
- **Code review (3-layer adversarial):** Blind Hunter, Edge Case Hunter, Acceptance Auditor run in
  parallel. Verdict: no CRITICAL/MAJOR correctness defect; the paired-DM-on-intersection design is
  statistically sound, leakage-free, QLIKE floor identical across compared runs, seed-ensembling on
  predictions correct. Findings fixed:
  - Basis smoke-assert now inspects `graph.snapshots[0].x_price` (the tensor the model consumes) with
    a PER-COLUMN all-zero check (was a whole-block check on the windowed `x_price_raw`); the
    subset-of-tickers zero mode remains guarded upstream by `features._check_price_coverage`.
  - `run_15ep_3seed.sh` now propagates each job's real Python exit status (was masked by a trailing
    echo), collects per-job statuses, and exits non-zero on any failure (was an unconditional
    "ALL DONE"); it also validates MAX is a positive integer.
  - DM output columns relabeled "on common (intersected) observations"; a single-seed invocation now
    warns on stderr.
  - Left as documented low/info: NaN-target `allclose` and duplicate-key collapse are unreachable
    with same-harness dumps; the smoke inspects only the first snapshot.
- **Data-quality gate:** N/A (no data/manifest change; reused the tested processed data + pipeline).
- **diff-cover C0/C1:** Not run (tooling gap per CLAUDE.md).
- **Push:** the pre-push TDD gate previously blocked the earlier commit batch (implementation `.py`
  without tests). This change ships with its own tests, satisfying the TDD gate for the new scripts.
