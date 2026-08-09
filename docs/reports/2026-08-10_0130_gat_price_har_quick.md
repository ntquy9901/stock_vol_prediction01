# GAT on the price-only P1 backbone vs classical HAR (quick experiment, h=5)

Timestamp: 2026-08-10 01:30
Branch: `feature/gat-har-quick` (forked from `feature/masked-gnn` @ 2fdcd95)
Task type: code

## 1. Objective

Test whether a leaner graph model — a GAT on the **price-only** pooled LSTM backbone (news OFF,
gate OFF), i.e. `P1 + graph` — beats the classical HAR baseline (P0) on the consistent Track-B
ladder basis at horizon 5. This is leaner than G1, which puts the graph residual on the full
news+gate P3 backbone.

## 2. Model specification (as run)

- Per-node features: the 3 HAR scales `(parkinson_volatility, har_weekly, har_monthly)` — the same
  price features the ladder's P0/P1 use (`TickerPreprocessor.feature_order`, `price_dim = 3`). No
  news branch, no per-ticker gate.
- Temporal encoder: pooled price LSTM backbone (`PooledPriceLSTM`, the P1 configuration; hidden 64,
  2 layers, dropout 0.2). Trained 5 epochs on the leakage-safe graph-bound train set
  (`target_date <= graph.train_end_date`), then frozen.
- Graph: masked k-NN top-8 over Pearson volatility-correlation adjacency
  (`--graph knn --top-k 8`), present-node message passing (`apply_message_passing`), positivity
  floor. GAT head trained 20 epochs (user-approved for this run) with the backbone frozen.
- New model class `PriceGraphAblationModel` (price-only node embedding + the same
  `_ResidualMessagePassing` and positivity machinery as `GraphAblationModel`), driven by the
  existing pooled graph training/eval helpers.

## 3. Basis (identical to the ladder)

Reuses `ladder_consistent.build_basis`: masked manifest, snapshots = 6470, graph-bound
train_samples = 73026, val_obs = 14418, test_obs = 14464, horizon 5, shared per-ticker scalers,
positivity floor. The one-basis invariant (graph present-node obs == pooled val/test obs) is
asserted at build time.

Cross-check: the P0 (HAR) metrics recomputed here on this basis match
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json` P0 exactly (val RMSE 0.0014724, test
RMSE 0.0022893, val R² 0.739435, test R² 0.766788, val QLIKE 0.509637, test QLIKE 0.567625, etc.),
and HAR is bit-identical across all 3 seeds (deterministic) — confirming the comparison is
apples-to-apples with the ladder's P0.

## 4. Results — 6-metric table (n = 3 seeds: 42, 123, 2026)

GAT-price values are mean over seeds (seed std shown); HAR (P0) is deterministic.

### Validation

| Metric | GAT-price (mean) | seed std | HAR (P0) | Δ (GAT−HAR) | Verdict |
|---|---|---|---|---|---|
| MSE  | 0.00000212 | 4.0e-09 | 0.00000217 | −4.9e-08 | beats HAR |
| RMSE | 0.0014558 | 1.4e-06 | 0.0014724 | −0.0000166 | beats HAR |
| MAE  | 0.0004615 | 7.5e-07 | 0.0004737 | −0.0000121 | beats HAR |
| R²   | 0.745292 | 4.8e-04 | 0.739435 | +0.005857 | beats HAR |
| QLIKE| 0.505552 | 2.5e-04 | 0.509637 | −0.004084 | beats HAR |
| Dir Acc (%) | 48.7361 | 0.05 | 48.5192 | +0.2169 | beats HAR |

### Test (held-out)

| Metric | GAT-price (mean) | seed std | HAR (P0) | Δ (GAT−HAR) | Verdict |
|---|---|---|---|---|---|
| MSE  | 0.00000531 | 3.1e-09 | 0.00000524 | +6.5e-08 | worse than HAR |
| RMSE | 0.0023035 | 6.6e-07 | 0.0022893 | +0.0000142 | worse than HAR |
| MAE  | 0.0006000 | 3.3e-07 | 0.0006027 | −0.0000027 | beats HAR |
| R²   | 0.763877 | 1.4e-04 | 0.766788 | −0.002912 | worse than HAR |
| QLIKE| 0.572211 | 3.2e-06 | 0.567625 | +0.004586 | worse than HAR |
| Dir Acc (%) | 48.0968 | 0.05 | 48.5275 | −0.4306 | worse than HAR |

Per-metric verdict summary:
- Validation: GAT-price beats HAR on **all 6** metrics (small, consistent margins).
- Test: GAT-price beats HAR on **1/6** (MAE, marginal −0.0000027); HAR is better on the other 5.

The validation edge does not survive to the held-out test set. The seed std is tiny (e.g. test
RMSE std 6.6e-7 vs Δ +1.4e-5; test Dir Acc std 0.05pp vs Δ −0.43pp), so the test underperformance
is a robust sign across seeds, not seed noise.

## 5. Significance — Diebold-Mariano (test set, squared-error loss, h=5)

DM applied to the aligned per-observation test losses (A = GAT-price, B = HAR; negative favors GAT;
Bartlett/Newey-West HAC, lag = h−1 = 4; HLN small-sample correction; obs ordered chronologically —
same pooled application the ladder uses for G1 vs HAR):

| Seed | n | DM_hln | p-value | mean(SE_GAT − SE_HAR) | Direction |
|---|---|---|---|---|---|
| 42   | 14464 | +1.148 | 0.2509 | +6.31e-08 | GAT worse |
| 123  | 14464 | +1.307 | 0.1912 | +6.98e-08 | GAT worse |
| 2026 | 14464 | +1.162 | 0.2452 | +6.35e-08 | GAT worse |

GAT-price is directionally worse than HAR on test squared-error, but **not statistically
significant** at any seed (p = 0.19–0.25). Caveat: the test panel is pooled cross-section, so the
DM HAC assumptions are approximate; the result is reported for consistency with the ladder's
G1-vs-HAR DM (also null). A QLIKE-loss DM was not computed because some raw Parkinson targets are
zero (log(0)); SE-loss DM (the MSE/RMSE significance) is unambiguous.

## 6. Conclusion

The leaner price-only GAT does **not** beat HAR out-of-sample. It edges HAR on all 6 validation
metrics but only 1/6 on held-out test (MAE, marginally), and is directionally — though not
significantly (DM p ≈ 0.19–0.25) — worse on squared-error loss. This is consistent with the
project's established finding that the graph is null (G1 ties/does not beat P3/HAR under DM at all
horizons) and with the literature that daily/small-universe GNNs rarely beat HAR. Removing the
news+gate branch and running the GAT on the leaner price-only backbone does not change that verdict.

## 7. Files

| Path | Purpose |
|---|---|
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py` | added `PriceGraphAblationModel` (price-only node embedding + shared message passing + positivity) |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/gat_price_quick.py` | driver: build basis, train price backbone, train GAT head 20 epochs, eval vs HAR |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_price_graph_model.py` | 20 unit tests for the new model (RED→GREEN) |
| `docs/reports/gat_price_quick_2026-08-10_003120.json` | seed 42 results (per-obs metrics + comparison) |
| `docs/reports/gat_price_quick_2026-08-10_005412.json` | seeds 123 + 2026 results |
| `results/gat_price_quick_seed{42,123,2026}_*/h5/` | per-seed run dirs (P0, backbone, GAT_P1 predictions + learning curve) |

## 8. Quality gate

- Tests: `test_price_graph_model.py` 20/20 pass (test-first: RED confirmed before implementation,
  then GREEN); `test_models.py` 39/39 pass (no regression from the additive class). Full baseline
  suite run separately.
- Lint: `ruff check` clean on `models.py`, `gat_price_quick.py`, `test_price_graph_model.py`.
- diff-cover (C0): **models.py = 100%** on changed lines (measured with
  `pytest-cov --cov-branch` over `test_models.py + test_price_graph_model.py`, `diff-cover
  --compare-branch=2fdcd95`). `gat_price_quick.py` is a run entry point (like `ladder_consistent.py`),
  exercised end-to-end by the real GPU run (evidence: the two results JSON) rather than unit-covered —
  consistent with the repo's coverage-gate scope (`coverage_gate.sh` restricts C1 to `data.py` +
  `run_pilot.py`).
- Data-quality gate: N/A (no data change — this reuses the ladder's manifest/features/scalers;
  no change to `data/processed`, features, or manifests).
- Code review: focused adversarial self-review — new class kept self-contained so it never
  perturbs the news+gate G1 path; frozen price encoder asserted (no gradients) by the existing
  graph runner; positivity floor + leakage-safe graph-bound train + shared scalers preserved;
  HAR cross-check against the ladder JSON confirms identical basis.

## 9. Repro

```
# GPU venv (.venv_gpu_encode: py3.10, torch 2.6.0+cu124)
python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/gat_price_quick.py <TS> cuda 42
python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/gat_price_quick.py <TS> cuda 123,2026
```
