# Summary — GBME + 2-layer GNN node-embedding baseline (2026-09-14)

## What changed
New baseline `baselines/2026-09-14_gbm_gnn_embed` implementing a pre-registered FALSIFICATION (5th distinct
graph attempt): does a **learned** 2-layer GNN node embedding `z` (penultimate hidden of the faithful
`GNNHAR`, dim `N_HID=9`), concatenated as features into the champion `GBM+earn` (GBME), beat GBME on
out-of-sample QLIKE? Expected NO-GO. Plus a disconnect-resilient SP500 Colab notebook.

## Files
| Path | Purpose |
|---|---|
| `baselines/2026-09-14_gbm_gnn_embed/requirements/requirements.md` | SDD spec: I/O, success/kill criterion, go/no-go |
| `baselines/2026-09-14_gbm_gnn_embed/design/design.md` | Design (updated): per-horizon-file deviation + §9 basis-drift validity caveat |
| `code/gnn_embed_config.py` | Single-source tunables (N_GCN, INNER_K, ARCH, epochs, kill criterion, spike windows) |
| `code/embed.py` | `GNNEmbedder` (subclass exposing `embed()`), `train_embedder`, `seed_embed`, OOF cross-fitting (`oof_train_z`, `test_z`, `_group_z`) |
| `code/run_gnn_embed.py` | Walk-forward driver: OOF z_train + z_test, GBME vs GBME+z, DM + verdict, per-horizon atomic checkpoint, HOSE spike-robustness |
| `test/test_gnn_embed.py` | 22 tests (embed shape, OOF coverage+causality, outer causality, z→GBM plumbing, real-embedder smoke, verdict/success, checkpoint, spike mask) |
| `code_review/code_review_2026-09-14.md` | 3-layer + leakage + performance review, findings + resolutions |
| `notebooks/gnn_embed_sp500_colab.ipynb` | SP500 Colab: per-horizon JSON + 180s git committer (disconnect-proof) |

## Critical correctness (no stacking leakage)
`z_train` is OUT-OF-FOLD: inner temporal K-fold (`INNER_K=3`) over each walk-forward fold's train window —
every train row embedded by a GNN that did NOT see it (`embed.oof_train_z`, `embed.py`). `z_test` from a GNN
trained on the full train window (`embed.test_z`). Graph, feature scaler, GNN and GBM all fit train/inner-train
only; outer target embargo at the train/test boundary. Asserted by `test_oof_coverage_and_causality`
(inner-train ∩ held = ∅, complement, full coverage) and `test_outer_fold_causality` (mutating genuine-future
rows leaves the fold result identical).

## Tests + coverage
22 tests pass (GPU venv). diff-cover **C0=100% line / C1=100% branch** on changed lines (`--cov-branch`).
ruff-F clean; config-hardcode clean (all tunables in `gnn_embed_config.py`, inline-commented to satisfy the
scanner). 1 `smoke`-tagged test (real embedder end-to-end).

## Code review
Adversarial 3-layer + leakage + performance review (subagent). **No leakage found**, performance PASS. Fixed
MAJOR-2 (in-sample val → fit GBM on `trf_e` for a true hold-out), MAJOR-1 partial (dropped embedding
seed-averaging; documented inherent inner-vs-test basis drift as a validity caveat — NO-GO is a lower bound),
MAJOR-3 (added outer-causality + real-embedder tests), MINOR-4/5 (`_safe_dm` hardened), MINOR-6/10. No
critical/major left open. Details in `code_review/code_review_2026-09-14.md`.

## Performance
GNN batched over dates on GPU (`G.BS`/step, no batch=1); one sync/epoch; GPU-resident tensors. Added OOF cost
= `INNER_K + 1 = 4` GNN trainings/fold/horizon (single embedding seed). GBM seed-ensembled; train/val/test
predicted in one pass (no 3× refit).

## Data-quality gate
N/A (no data change): the baseline only READS existing enriched frames (`data/processed_enriched/<market>/`)
and existing earnings/sector artifacts. No new raw ingested, no processed frames written.

## Output
`results/gamma_gbm/gnn_embed_<market>_h<h>.json`, one per horizon, atomic tmp+replace per fold. Carries
`metrics`/`train_metrics`/`val_metrics` (all 5) + `fit_diagnostics` + `learning_curves` for the learned model
`GBME+GNN-embed` (name contains `gnn` → overfit gate treats it as learned). HOSE adds `per_fold_qlike` +
`spike_robustness`.

## Status / follow-ups
- SP500: notebook-only (resilient, fresh-clone-import-clean verified via `git archive`).
- HOSE: full local run (8 folds × 3 seeds × 4 horizons) pending — per-horizon checkpoint so partial results
  survive; verdict to be appended when complete.
