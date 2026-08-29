# Sector-GAT ablation scale-up on HNX (h1, 5 seeds, 15 epochs)

Date: 2026-08-29

## Objective

Test whether the sector-graph GAT edge is viable on HNX beyond the earlier 1-seed / 5-epoch
directional pilot. Compare three edge choices under the identical MaskedRichNet / HAR-X pipeline:

- `sector_GAT` — `use_graph=True`, static same-sector adjacency (VN ICB labels), the contribution under test.
- `stat_GAT_vol2pk` — `use_graph=True`, the shipped directed volume-shock -> Parkinson edge (`D.adj_vol2pk`).
- `no_graph_LSTM` — `use_graph=False`.

Question: does `sector_GAT` keep a significantly lower QLIKE than **both** `stat_GAT_vol2pk` and
`no_graph_LSTM`, with per-seed error bars that do not overlap, once run to convergence over 5 seeds?

## Run configuration

- Panel HNX, horizon 1. Universe screened to 154 nodes (162 kept by the liquidity/zero-frac screen,
  154 with enough history to form the panel); 60,028 test observations.
- 5 seeds `(42, 123, 2026, 7, 2024)`, 15 max epochs, early stopping on val MSE (patience 3, min 5).
- Sector coverage: 153/154 tickers mapped (99.4%), 23 sectors, avg off-diagonal degree 10.8, 4 singletons.
- GPU (`.venv_gpu_encode`, torch 2.6.0), uncontended. Wall time ~55 min.
- Command:
  `SECTOR_ABLATION_FORCE_CPU=0 .venv_gpu_encode/Scripts/python.exe baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py --panel hnx --horizon 1 --train-epochs 15`
  (default Config -> 5 seeds; `--seeds` left unset).
- Results: `results/sector_gat_ablation/sector_ablation_hnx_h1_15ep.json`
  (identical copy at `sector_ablation_hnx_h1.json`).

## Per-seed metrics (mean +/- std over 5 seeds)

| model | MSE | RMSE | MAE | QLIKE | R2 |
|---|---|---|---|---|---|
| sector_GAT | 1.3813e-06 +/- 1.31e-09 | 1.17527e-03 +/- 5.6e-07 | 6.3926e-04 +/- 3.7e-06 | 1.8277 +/- 0.0160 | 0.22722 +/- 7.3e-04 |
| stat_GAT_vol2pk | 1.3856e-06 +/- 1.37e-09 | 1.17712e-03 +/- 5.8e-07 | 6.4967e-04 +/- 6.0e-06 | 1.8219 +/- 0.0023 | 0.22480 +/- 7.7e-04 |
| no_graph_LSTM | 1.3852e-06 +/- 1.24e-09 | 1.17695e-03 +/- 5.3e-07 | 6.4747e-04 +/- 6.0e-06 | 1.8322 +/- 0.0138 | 0.22502 +/- 7.0e-04 |

Ensemble (seed-averaged prediction) QLIKE: sector 1.8181, stat 1.8202, no_graph 1.8245.

Note the basis dependence on QLIKE: on the **per-seed mean**, sector (1.8277) is marginally *higher*
(worse) than stat (1.8219); the ordering only flips to sector-lowest on the seed-ensembled prediction
(1.8181 vs 1.8202), because ensembling suppresses sector's larger seed variance (std 0.016 vs 0.002).

## Error-bar overlap (QLIKE, per-seed mean +/- std)

- sector_GAT: [1.812, 1.844]
- stat_GAT_vol2pk: [1.820, 1.824]
- no_graph_LSTM: [1.818, 1.846]

All three intervals overlap. On MSE and R2 the per-seed intervals are narrowly separated in sector's
favour (sector MSE max 1.3829e-06 < stat/no_graph min 1.383e-06; sector R2 min 0.2263 > stat/no_graph
max 0.2257), but the magnitude of that gap is ~0.3% of MSE.

## Diebold-Mariano (date-clustered, horizon 1)

| pair | QLIKE p | favours | SE (MSE) p | favours | AE (MAE) p | favours |
|---|---|---|---|---|---|---|
| sector vs stat | 0.500 | - | 0.0104 | sector | 2e-52 | sector |
| sector vs no_graph | 0.0112 | sector | 0.0070 | sector | 8e-42 | sector |
| stat vs no_graph | 0.0244 | stat | 0.908 | - | 7e-07 | no_graph |

## Over/under-fit evidence (fit verdict per model)

All three models: `status = ok` (no overfit, no underfit). The evidence is stamped into the result
JSON as `train_metrics`, `val_metrics`, `fit_diagnostics`, and per-seed `learning_curves`.

| model | train QLIKE / R2 | val QLIKE / R2 | test QLIKE / R2 | val->test QLIKE gap | train->test R2 drop |
|---|---|---|---|---|---|
| sector_GAT | 3.208 / 0.203 | 2.178 / 0.251 | 1.818 / 0.228 | -16.5% | -0.025 |
| stat_GAT_vol2pk | 3.199 / 0.204 | 2.170 / 0.250 | 1.820 / 0.225 | -16.1% | -0.022 |
| no_graph_LSTM | 3.211 / 0.200 | 2.183 / 0.250 | 1.825 / 0.225 | -16.4% | -0.025 |

Test QLIKE is *lower* than val for every model (negative gap) and test R2 is at or above train R2
(negative drop): the models generalize; there is no overfit signal. The QLIKE level difference across
train/val/test reflects the differing volatility distributions of the chronological splits, not a fit
pathology (the same pattern appears identically in all three arms).

**Convergence:** best-epoch per seed clusters at 13-15 (sector `[10,13,14,15,14]`, stat `[14,13,15,13,15]`,
no_graph `[15,12,15,15,14]`). Validation MSE curves plateau around epoch 9-13 (val ~1.87e-06). Several
seeds reach the 15-epoch cap, so a marginal further improvement past 15 is possible, but the plateau is
clear and the fit verdict is stable.

## Verdict: sector-GAT is NOT viable as a distinct improvement on HNX h1

The 1-seed / 5-epoch pilot reported sector-GAT beating both comparators on QLIKE (DM p=0.007 vs stat,
p=0.010 vs no_graph). **That directional finding does not hold at 15 epochs / 5 seeds.**

- **sector vs stat, QLIKE: not significant (DM p=0.50).** On the per-seed mean, sector's QLIKE is even
  marginally worse than stat's, and the per-seed error bars overlap heavily. Sector-GAT and the
  statistical volume-shock edge are statistically indistinguishable on the headline volatility loss.
- **sector vs no_graph, QLIKE: significant (DM p=0.011)** — but `stat_GAT_vol2pk` also beats no_graph on
  QLIKE (p=0.024). The QLIKE gain over no-graph is a "having a graph edge helps marginally" effect, not a
  property unique to sector structure.
- On MSE/SE and MAE/AE the sector edge is consistently but marginally best (SE p=0.01 vs stat,
  non-overlapping MSE/R2 bars), with an effect size around 0.3% of MSE. On MAE the stat edge is actually
  *worse* than no-graph (p=7e-7), while sector is best.

The acceptance bar stated for the task — sector-GAT significantly lower QLIKE than **both** comparators
with non-overlapping error bars — is not met (fails against stat-GAT on QLIKE, and QLIKE bars overlap).
The overfit question is answered: all three models are `ok` (no over/under-fit) and near-converged at 15
epochs.

## h5

Not run. h1 answers the viability question decisively (negative for QLIKE) and consumed ~55 min of the
single-GPU slot; leaving the GPU free for the queued MTGNN run was prioritised. h5 remains an available
follow-up (`--horizon 5`) if a horizon-dependent check is wanted.

## Code change (harness)

To satisfy the over/under-fit evidence mandate, `run_sector_ablation.py::run_training` now trains each
variant with `return_splits=True` and stamps `train_metrics`, `val_metrics`, `fit_diagnostics`
(per-model `classify_fit` verdict), and per-seed `learning_curves` into the result JSON. All additions
reuse the read-only `run_masked_rich` helpers (`_ens_split`, `_split_metrics`, `OF.classify_fit`); the
model, the sector-adjacency logic, and every live-training-path file are untouched.

- Files: `baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py` (evidence plumbing),
  `baselines/2026-08-29_sector_gat_ablation/test/test_runner_and_fetch.py`
  (new `test_run_training_emits_overfit_evidence`; existing stub updated for `return_splits`).
- Tests: 35 pass (`pytest baselines/2026-08-29_sector_gat_ablation/test/`). TDD: the new test was
  confirmed failing (missing `train_metrics`) before the plumbing was added.
- Diff coverage on changed source lines: C0 = 100%, C1 = 100% (13 changed lines, `diff-cover` vs
  `origin/master`).
- Data-quality gate (Pandera/Evidently): N/A — no data, feature, or manifest change (harness reporting
  only).

## Code review

Adversarial 3-lens review (Blind Hunter / Edge Case Hunter / Acceptance Auditor) run on the diff.
- Plumbing: correct and leakage-free; each of train/val/test metrics computed on its own split, no split
  feeds another; `learning_curves` structure matches the `run_masked_rich` convention; no mutation/aliasing.
- MAJOR (acceptance, dispositioned): the emitted evidence is not consumed by the pre-push overfit gate
  (`check_overfit_evidence.py`) — the gate globs `*result.json` under `masked_rich`, and this artifact
  lives in `results/sector_gat_ablation/` with model names `sector_GAT / stat_GAT_vol2pk / no_graph_LSTM`
  rather than the gate's `LEARNED = (LSTM, LSTM_wGAT_vol2pk)`. Disposition: this is a directional research
  artifact in a separate results tree; the mandate's intent (emit evidence so the overfit question can be
  answered) is satisfied and reported above. Machine-gate integration for this tree is out of scope for
  this task and noted as a follow-up.
- MINOR (fixed): the first version of the new test used a near-perfect stub that made the fit-verdict
  assertion tautological and would not catch a train/val/test split-swap. The stub was changed to
  split-distinguishable errors (train < val < test) and the test now asserts the MSE ordering
  `train < val < test`, pinning that each split's metrics are fed from the correct array.
- Performance: no anti-pattern introduced; `return_splits=True` adds only batched `infer()` passes.

## DoD checklist

- [x] Ablation run to 5 seeds / 15 epochs with early stopping, GPU, no OOM (uncontended).
- [x] All five metrics reported as per-seed mean +/- std for the three configs.
- [x] Date-clustered DM: sector vs stat, sector vs no_graph (and stat vs no_graph), all three families.
- [x] Over/under-fit evidence emitted and reported (train/val/test + verdict + learning curves).
- [x] Tests pass (35); TDD followed; diff-coverage C0=100% / C1=100% on changed lines.
- [x] Code review run; findings dispositioned.
- [x] Result JSON saved; report written.
- [ ] Push: intentionally NOT pushed (coordinator consolidates the three tangled edge branches). Committed
      locally only.
