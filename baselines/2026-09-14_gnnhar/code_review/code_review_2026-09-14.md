# GNNHAR baseline — adversarial code review (2026-09-14)

3-layer adversarial review (Blind Hunter + Edge Case Hunter + Acceptance Auditor) with a performance lens,
run on `code/run_gnnhar.py`, `code/gnnhar_config.py`, `test/test_run_gnnhar.py` against the spec
(`requirements/requirements.md`, `design/design.md`), the reused faithful core `scripts/eda/gnnhar_sp500.py`,
and the shared infra (`full_matrix`, `vn_gbm_graph_stage1`, `stats`, `overfit_check`). `archive/` out of scope.

## Verdict: no CRITICAL or MAJOR findings. 6 MINOR (perf/cleanliness). Acceptance: PASS.

### Verified correct (high-risk areas)
- **Leakage / causality — clean.** Per-fold graph `S1.build_graph(tr, …)` with `tr = date < ts − embargo`;
  feature z-scoring + target scale in `G.build_fold_tensors` use `train_cells = mask & (date < ts−embargo)`
  (train-period stats only). Target `parkinson_variance.shift(-h)` (future), trailing `h` dropped. Embargo
  `int(h*1.6)+5`. Train/val/test date sets mutually disjoint (`tr_idx = all_tr[:-VALID_LEN]`,
  `va_idx = all_tr[-VALID_LEN:]` both `< tcut`; `te_idx >= ts`). The `_fold_splits` (dataframe) and
  `_gnn_maps` (tensor) val boundaries resolve to the identical date set.
- **QLIKE floor consistency — clean.** Every model's pooled loss + `_metrics5` use `floor=FL` (`FM.FL`).
  Per-model prediction floors mirror `full_matrix.py` exactly (main-table comparability holds).
- **DM usage — clean.** `date_clustered_dm(e[x], e[b], dates, h)` aligned per observation; HLN lag `h−1`;
  `gain_pct` sign consistent with `mean_diff`. CMP = GNNHAR vs {HAR, GBM, no-graph} = leave-one-out.
- **Seed ensemble — correct.** Per-fold `np.mean(spreds, 0)` in raw prediction space, concatenated across
  folds; `per_seed_qlike` accumulated separately.
- **Gate schema — matches.** `_merge` writes flat `<model>_h<h>` keys for metrics/train/val for all models;
  `overfit_check.learned_models` detects `GNNHAR*`/`GNNHAR-nograph*` (the `"gnn"` fragment) and each carries
  train+val+test → `check_result_evidence` genuinely validates (never falsely skips); HAR/GBM exempt.
- **Helper reuse — correct argument order** for `row_preds` / `_row_in_te` / `build_fold_tensors`.
- **Performance — clean.** Batched over dates (`[B≤256, N, F]`), single `(N,N)` matmul over the batch,
  tensors resident on `DEVICE`, RNG on device, only 2 `.item()` syncs/epoch. No batch=1, no per-item loop,
  no per-step host↔device copy.

### MINOR findings + disposition
- **M1 (perf):** an extra full-train forward each epoch records the learning-curve train point (the reused
  `_train_once` did only the val forward), ~2× per-epoch forward cost. Correct + GPU-batched, just wasteful.
  **Disposition:** accepted as a follow-up (recording every K epochs would help); NOT changed now because the
  HOSE run is already in progress and the extra pass is correct — changing code mid-run would invalidate it.
- **M2 (documented/standard):** the GNN fits on `trf_e` (train − val, for early stopping) while HAR/GBM fit
  on `trf` (full train), so DM windows are not exactly equal. Intended, matches `full_matrix.py`, standard
  early-stopping practice. **Disposition:** noted in the summary report; no code change.
- **M3 (low risk):** `diebold_mariano` raises (not skips) if pooled unique test dates `≤ h` or `long_run ≤ 0`.
  Cannot trigger on the real multi-year pooled test spans. **Disposition:** accepted; follow-up guard optional.
- **M4 (cosmetic):** GBM uses `cfg.SEEDS` (3) even in smoke where the GNN uses `(0,)`; smoke-runtime only.
- **M5 (cleanliness):** the `n_gcn` column in `GCFG` is constant (`cfg.N_GCN`); leftover from the reused
  1L/2L sweep. **Disposition:** kept for parity with the reused config; harmless.
- **M6:** `best_val - 1e-6` carries `# config-ok` (numerical improvement guard, legitimate exception).

No critical/major issues to fix. MINOR items recorded as follow-ups per CLAUDE.md.

**M1 fixed after the review (perf).** The reviewer's M1 (an extra full-train forward every epoch, ~2× cost)
was ADDRESSED in the committed code: `train_with_curves` now records the learning-curve train point from the
already-computed per-batch training losses (`tl_sum += loss.detach()` accumulated over the epoch, one
`.item()` at epoch end) instead of a second full-train forward. Correctness unchanged (fit_diagnostics uses
the separately-pooled train predictions, not the curve); tests re-run green; the val forward (early-stop
signal) is unchanged. This also speeds the SP500 Colab run.

---

## SP500-path verification (correctness-before-commit, coordinator request 2026-09-14)

### Notebook 3-layer review — `notebooks/gnnhar_sp500_colab.ipynb`
Blind Hunter / Edge Case / Acceptance, focused on the fresh-clone Colab run.
- **SP500 argv correct.** Cell 7 runs `python baselines/2026-09-14_gnnhar/code/run_gnnhar.py sp500`
  (positional `sp500`), smoke-first (`assert run(smoke=True)==0`) THEN full — so a broken clone fails fast on
  the 1-fold smoke instead of wasting the overnight full run.
- **Every imported module + data dep TRACKED** (fresh-clone smoke proof below): the 7 `.py` deps
  (`full_matrix`, `vn_gbm_graph_stage1`, `gnnhar_sp500`, `overfit_check`, `metrics`, `pipeline_config`,
  `stats`) AND the 2 SP500 data files `results/gamma_gbm/sp500_sectors.json` +
  `results/gamma_gbm/sp500_earnings.parquet` (needed by `FM.load("sp500")`) are all `git ls-files`-tracked.
  Cell 2 prints HAS/MISSING for each so a missing file is visible before the run.
- **Deps cell (4)** installs `scikit-learn>=1.4 pandas numpy scipy pyarrow`; torch ships pre-installed with
  CUDA on Colab (cell 1 prints `torch.cuda.is_available()`). The driver is pure torch (no `torch_geometric`).
- **Bundle path matches the sibling SP500 notebooks:** `MyDrive/public_bk/luanvan_data/colab_bundle_sp500_clean.zip`
  (with `MyDrive/luanvan_data` fallback), unpacking `data/processed_enriched/sp500_clean/` into `/content/repo`.
- **Per-horizon incremental JSON + background committer.** The driver checkpoints
  `results/gamma_gbm/gnnhar_sp500.json` atomically after each horizon; cell 6 starts a daemon thread that
  `git add -f results/gamma_gbm/*.json run.log` + commit + `pull --rebase` + push `HEAD:master` every 180s —
  identical resilience to `notebooks/sp500_final_models_colab.ipynb`.
- **Over/under-fit evidence written.** The JSON carries top-level `metrics`/`train_metrics`/`val_metrics`
  keyed `<model>_h<h>` + `fit_diagnostics` + `learning_curves` for the learned models, so the pre-push
  overfit-evidence gate auto-detects `GNNHAR*`/`GNNHAR-nograph*` and validates them.
- **Minor notebook edge notes (non-blocking):** (E1) cell 8's display `open(gnnhar_sp500.json)` assumes ≥1
  full horizon completed; if the full run aborts before h1 it raises FileNotFoundError (the smoke already
  proved h1 works, and partial results are on git via the committer) — cosmetic display only. (E2) a
  disconnected full run is recomputed from h1 on re-run (completed horizons remain on git for analysis) —
  matches every sibling SP500 notebook.

### Fresh-clone import smoke (tracked files only — `git archive HEAD`, no untracked deps)
```
$ git archive -o gf.tar HEAD && tar -x -f gf.tar -C fresh/ baselines/2026-09-14_gnnhar scripts/eda \
      scripts/quality_gate submission/soict_lstm_gat baselines/2026-08-21_har_anchored_residual/code
$ python fresh/baselines/2026-09-14_gnnhar/code/run_gnnhar.py --help
usage: run_gnnhar.py [-h] [--smoke] [{hose,sp500}]
positional arguments:
  {hose,sp500}
options:
  -h, --help    show this help message and exit
  --smoke       1 horizon, 1 fold, 1 seed, few epochs
```
Zero `ModuleNotFoundError`: all module-level imports (`run_gnnhar` -> `full_matrix`/`vn_gbm_graph_stage1`/
`gnnhar_sp500`/`metrics`/`stats`/`overfit_check`/`gnnhar_config` -> torch/sklearn/scipy/pandas/numpy) resolve
from tracked files only. `gnnhar_config` uses a UNIQUE module name (not `config`) to avoid the known
`submission/soict_lstm_gat/config.py` sys.path collision.

### Tests + coverage (re-run post-M1-fix)
`11 passed`. diff-cover vs `origin/master`: **C0 line = 100%** on changed lines (gate `--fail-under=100`);
C1 branch = 96.9% (≥95%; the 2 partials are for-loop continuation edges). ruff `--select F`: clean.
config-hardcode scan: no BLOCK (2 non-blocking WARN on the `1e-3`/`1e-5` learning-rate/decay in the config
module, expected).

## Addendum 2026-09-14 — SP500 Colab notebook review (`notebooks/gnnhar_sp500_colab.ipynb`)

**CRITICAL bug found + fixed.** The notebook cells were written with `source` lists whose lines had **no
trailing newlines**, so Colab concatenated each cell into a single `#`-prefixed line — every statement was
commented out and nothing would run (user-reported "only comments, no code"). Root cause is the same
newline-loss class as the earlier notebook incident. Fix: re-added per-line newlines (nbformat) to all 8 code
cells and stripped a mojibake character. Notebooks are not executed by pytest, which is why the pre-push gate
did not catch it; added `test/test_notebook_valid.py` (4 tests, gate-enforced) asserting every code cell has
real newlines, compiles (magics stripped), is ASCII-clean, and drives `run_gnnhar.py sp500` -> reads
`gnnhar_sp500.json` with the background committer. This closes the gap for this notebook going forward.

**3-lens review of the fixed notebook (no critical/major remaining):**
- Blind Hunter: cell 8's JSON schema (`d['qlike'][f'{m}_h{h}']`, `d['dm'][...]['gain_pct'/'p_value']`,
  `d['fit_diagnostics'][...]['status']`) verified to match `run_gnnhar.py` output exactly against the real
  `gnnhar_hose.json` -> no KeyError on inspect. Unscored-horizon guard (`if f'GNNHAR_h{h}' not in metrics:
  continue`) protects the loop.
- Edge Case: bundle path tries both `public_bk/luanvan_data` and `luanvan_data` with an assert; smoke-first
  (`assert run(smoke=True)==0`) before the full run; committer uses `pull --rebase` then push (no force).
- Acceptance: clone master -> unpack enriched bundle -> deps -> token -> committer -> smoke+full ->
  inspect+final commit, mirroring the verified `sp500_final_models_colab.ipynb`; writes per-horizon JSON with
  train/val/test + fit_diagnostics + learning_curves (overfit-evidence schema).
- SDD: covered by this baseline's `requirements/` + `design/`; DoD: fixed, tested (4 notebook + 11 driver),
  reviewed, committed. Fresh-clone import smoke of `run_gnnhar.py` (tracked files only) already clean.
