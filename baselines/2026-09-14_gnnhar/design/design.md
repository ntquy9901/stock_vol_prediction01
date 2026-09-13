# GNNHAR baseline — design (Plan)

## 1. GNNHAR architecture (faithful to paper + official repo)
Source: arXiv:2308.01419 §3 (model) / §4 (training); official code
https://github.com/chaozhang-ox/GNNHAR (`GNNHAR.py`). The faithful core is already implemented and
**unit-tested** in this repo at `scripts/eda/gnnhar_sp500.py` (docstring cites the repo line numbers). This
baseline **reuses that core read-only** (per CLAUDE.md §3.F: read-only import of shared code is allowed;
re-deriving a large faithful implementation would violate §2 Simplicity and add a second bug surface).

Reused pieces (`import gnnhar_sp500 as G`):
- `G.GraphConvLayer`: `out = adj @ (X @ W) + b`, `xavier_uniform(W)` (relu gain), bias init ones
  (GNNHAR.py:129–147).
- `G.GNNHAR(in_f, n_hid, n_gcn)`: `H1 = Linear(F,1)` own-feature linear term (bias init 1.0);
  `H_gcn` = `n_gcn`-layer GCN (relu between) → `mlp(n_hid,1)`; output `relu(H1 + H_gcn)` (GNNHAR.py:150–277).
  `n_gcn=2` → **GNNHAR2L** (the paper's best); `A = I` → the per-node-MLP no-graph control.
- `G._qlike_loss`: masked paper QLIKE `tf = y/(f+1e-4); mean(tf − log tf)` (GNNHAR.py:317–328).
- `G.build_fold_tensors` / `G.row_preds` / `G._row_in_te`: pivot fold → `(D,N,F)` z-scored features
  (train stats), `(D,N)` target scaled by 1/train-mean, `(D,N)` mask; raw-scale floored row predictions.

This baseline's **own** code (`code/run_gnnhar.py`, `code/config.py`) contributes: the tunable-constant
config, a training loop that records **learning curves** (the reused `G._train_once` does not), the
walk-forward orchestration over the project's folds, the 4-model comparison, and the gate-aware single-file
incremental output.

## 2. Data flow
`FM.load(market)` → frames → for each h: `FM.panel(frames, {}, h)` → for each walk-forward fold k:
1. `tr = rows[TRAIN_START ≤ date < fold_start − embargo]`, `te = rows[fold_start ≤ date < fold_end]`;
   skip if `len(te)==0` or `len(tr) < min_train`.
2. Per-fold graph `Wc, _ = S1.build_graph(tr, tickers, rng)` — **train-only** correlation top-k
   (causal, no leakage). `adj_corr = tensor(Wc)`, `adj_none = eye(N)`.
3. Build the fold panel `fold = a[TRAIN_START ≤ date < fold_end]`; split into `trf` (fit window, full
   train incl. val dates for the deterministic baselines), `trf_e` (train minus last `VALID_LEN` dates),
   `vaf` (last `VALID_LEN` train dates = the GNN validation split), `tef` (test).
4. Baselines: `HAR = FM._har_ols(trf, ·)`, `GBM = mean_s FM.gbm(trf, ·, OWN8, s)` — scored on tef/trf_e/vaf.
5. GNNHAR: `G.build_fold_tensors(fold, tickers, HAR3, train_date_fn)` once (shared by both adj types);
   for each seed, `fit()` (train on `trf_e` dates, early-stop on `vaf` QLIKE, restart on dead-ReLU collapse),
   `G.row_preds` on tef/trf_e/vaf; seed-ensemble mean. Same for `GNNHAR-nograph` with `A=I`.
6. Accumulate pooled test/train/val preds + dates per model per fold.

After all folds of h: pool per-obs QLIKE (floor `FM.FL`), compute 5 metrics per split per model,
`classify_fit` verdict per learned model, date-clustered DM (GNNHAR vs HAR / GBM / nograph), merge into the
top-level output dicts under keys `<model>_h<h>`, **atomic checkpoint** (`_checkpoint`, tmp+replace).

## 3. Three gates (SDD §5)
- **Simplicity Gate — PASS.** No new abstraction/config beyond the one `config.py` of tunable constants; the
  faithful model is reused, not re-abstracted. 4 models (2 deterministic + full GNN + its 1 leave-one-out).
- **Anti-Abstraction Gate — PASS.** Uses the existing `FM`/`S1`/`metrics`/`stats`/`overfit_check`/`G`
  modules directly; no wrapper layers. Pure `torch` (no `torch_geometric`).
- **Performance / Batching Gate — PASS.** Training is **batched over dates** (`[B, N, F]` tensor, `B≤256`),
  full graph adjacency `(N,N)` applied to the whole batch in one matmul on **GPU**; all of X/Y/mask/adj are
  resident on the device for the fold (no host↔device copy in the hot loop, no batch=1, no per-item Python
  loop). Mask-aware QLIKE loss handles the varying cross-section. The only per-epoch host syncs are the two
  `.item()` calls that record the learning curve (2 syncs/epoch, not per-step). Seeds/configs are the outer
  loop (independent trainings); the inner training step is fully batched. Multi-seed ensemble averages
  predictions.

## 4. Constants (single source of truth — `code/config.py`)
`HAR3 = FM.HAR`; `OWN8 = [c for c in FM.OWN if c != 'rq']`; `SEEDS=(0,1,2)`; `N_HID=9`; `N_GCN=2`;
`VALID_LEN=22`; `LR=1e-3`; `WEIGHT_DECAY=1e-5`; `BATCH_DATES=256`; `MAX_EPOCHS=120`; `MIN_EPOCHS=20`;
`PATIENCE=15`; `GRAD_CLIP=1.0`; `COLLAPSE_VAL=1.6`; `MAX_RESTARTS=4`; `HORIZONS=(1,5,10,22)`;
`MIN_TRAIN_ROWS={'sp500':30000,'default':3000}`; `SMOKE_EPOCHS=40`.

Training policy: bounded `MAX_EPOCHS=120` with **early stopping** on val QLIKE (patience 15, min 20 epochs)
— this is the published GNNHAR protocol (best-validation checkpoint), not an open-ended long run. Multi-seed
(3) for the reported ensemble. Anti-overfit: early stopping + weight decay + grad clip + target/feature
scaling (per CLAUDE.md §3.E).

## 5. File list
- `code/config.py` — tunable constants (single source; avoids config-hardcode gate).
- `code/run_gnnhar.py` — `train_with_curves`, `fit`, `run(market, load_fn, out_path, ...)`, `_checkpoint`,
  `main()` (entry, `# pragma: no cover`).
- `code/__init__.py` — empty (package marker).
- `test/conftest.py` — sys.path bootstrap (dashes in folder name block `python -m`).
- `test/test_run_gnnhar.py` — model-forward shapes, causal (train-only) graph, `train_with_curves` curve
  capture, QLIKE/DM wiring, integration `run()` smoke on synthetic frames asserting no-exception + the full
  over/under-fit evidence schema (`OF.check_result_evidence` accepts it), both the SP500 branch and the HOSE
  `per_fold_qlike` branch.
- `test/__init__.py`.

## 6. Tasks (verifiable)
1. `config.py` constants → verify: `test_config` (rq dropped, HAR3⊂OWN8). 
2. `train_with_curves` → verify: returns model + non-empty curve with train/val per epoch.
3. `fit` restart-on-collapse → verify: returns finite-val model.
4. `run()` walk-forward + output → verify: integration smoke, structure + evidence keys + DM in [0,1].
5. causal graph → verify: `S1.build_graph` only ever receives train rows (date < fold_start − embargo).
6. HOSE local run → verify: `results/gamma_gbm/gnnhar_hose.json` exists, evidence present, gate accepts.
7. SP500 notebook → verify: valid JSON notebook, mirrors resilient pattern, runs the driver.
