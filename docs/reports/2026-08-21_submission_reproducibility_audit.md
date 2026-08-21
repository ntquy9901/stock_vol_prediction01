# SOICT submission reproducibility audit — `submission/soict_lstm_gat/`

Date: 2026-08-21. Scope: completeness and reviewer reproducibility of the self-contained SOICT
submission folder. Verdict summary: the folder is code-complete, self-contained, tests pass, and both
entrypoints run; the only material defect was that the reviewer-facing docs/scripts ran ONLY the graph
ablation and never the MAIN per-observation LSTM model — fixed here (docs/scripts only, no core logic).

## Inventory (verified, not assumed)
Core modules (14 .py): `config.py, metrics.py, data_utils.py, edges.py, model.py, snapshots.py,
baselines.py, train.py, evaluate.py, run_all.py` (graph/snapshot suite), `run_lstm.py` (MAIN =
per-observation LSTM), `conftest.py`.
Tests (7 files): `test_metrics.py, test_data_utils.py, test_edges.py, test_baselines.py, test_model.py,
test_snapshots.py, test_run_lstm.py`.
Docs/scripts: `README.md, REPRODUCE.md, EXTRACTION_LOG.md, TASKBOARD.md, reproduce.sh, requirements.txt`,
internal drivers `_run_suite.sh` (run_all), `_run_perobs.sh` (run_lstm).
Data: `data/vn30/` = 33 `*_processed.csv`, `data/vn100/` = 104 `*_processed.csv`. `data/sp500/` absent
(intentional). Columns: `date, parkinson_volatility`.

## Checks — PASS/FAIL with evidence

### 1. Main-model coverage — FAIL as shipped → FIXED
- The MAIN model per the latest requirements is the per-observation LSTM (`run_lstm.py`). The original
  `reproduce.sh` and `REPRODUCE.md` ran ONLY `run_all.py` (the HAR-LSTM-GAT graph snapshot suite) and
  never `run_lstm.py`. `README.md`'s "Run" example was also `run_all.py` only. A reviewer following the
  docs would NOT reproduce the headline LSTM-vs-HAR result — confirmed reproducibility gap.
- Evidence the LSTM is the actual headline driver: `logs_perobs/` contains completed run_lstm.py runs
  for vn30/vn100/sp500 × h1/h5/h10/h22; `_run_perobs.sh` drives `run_lstm.py`. The smoke run below
  reproduces LSTM beating HAR at vn30/h1 (QLIKE 0.4597 < 0.4675, DM favours LSTM p=0.0057).
- FIX APPLIED (reviewer-facing only): `reproduce.sh` now runs the MAIN LSTM suite (vn30/vn100 × h1/h5)
  first, then the graph ablation suite; `REPRODUCE.md` restructured into "Main model (run_lstm.py)" +
  "Graph ablation (run_all.py)" and adds run_lstm.py to the S&P500 section; `README.md` Run block shows
  `run_lstm.py` as MAIN with `run_all.py` labelled the graph-check ablation.

### 2. Runnable from the folder — PASS
- Bare-name imports resolve via `conftest.py`/`sys.path` (pytest) and via script-dir on sys.path[0]
  (direct `python run_*.py`). No import errors.
- CLIs parse; both smoke runs produced a `result.json`:
  - `run_lstm.py vn30 10 1 --smoke --data-root <folder>/data` → `[perobs] tickers=33 train_obs=84488
    test_obs=10577`; wrote `results/soict_perobs/vn30_lb10_h1/result.json` (QLIKE HAR 0.4675 / GARCH
    0.7434 / LSTM 0.4597; DM LSTM_vs_HAR p=0.0057 favours LSTM).
  - `run_all.py vn30 10 1 --smoke --data-root <folder>/data` → wrote
    `results/soict/vn30_lb10_h1/result.json` (QLIKE HAR 0.3946 / GARCH 0.65 / HAR-LSTM-GAT 0.5002 /
    LSTM w/o GAT 0.4286).
- Note (not a blocker): both scripts write results to `<two-levels-up-from-folder>/results/...`
  (`Path(__file__).parents[1]/results`). For a reviewer who unpacks only this folder, results land two
  directories above it, not inside the folder. Paths are relative (no absolute hardcoding), so it works;
  it is just an out-of-folder write location.

### 3. Tests — PASS
- `python -m pytest submission/soict_lstm_gat/tests -q` → **36 passed** (49s). (The docs previously said
  "34"; the count grew to 36 after `test_run_lstm.py` was added — REPRODUCE.md updated to 36.)
- Self-contained: all tests synthesize their own tmp CSVs (e.g. `test_run_lstm._write`,
  `test_data_utils`); none depend on the absent sp500 data. 3 non-fatal sklearn ConvergenceWarnings in
  `test_edges` (expected — the test deliberately exercises non-convergence).

### 4. Data shipping — PASS
- vn30 (33) + vn100 (104) processed CSVs present under `data/`. sp500 correctly NOT shipped (Yahoo
  license); regeneration documented in REPRODUCE.md via `src.data.download_sp500` +
  `src.common.process_parkinson_pipeline`, and (now) followed by both `run_lstm.py` and `run_all.py`.

### 5. Dependencies — was incomplete → FIXED
- `requirements.txt` had torch, numpy, pandas, scikit-learn, arch, matplotlib, pytest but was MISSING
  `scipy`, which is a real runtime dep (`metrics.py: from scipy import stats` for the DM Student-t
  p-value). `arch` (GARCH) is present. FIX: added `scipy` to `requirements.txt`. (statsmodels/patsy come
  in transitively via `arch`; scipy is also transitively pulled by scikit-learn/statsmodels but is now
  declared explicitly since it is imported directly.)

### 6. Self-containedness — PASS
- Grep of all import statements: every intra-folder import is by bare name (`import metrics`,
  `import data_utils`, `import baselines`, `from config import ...`, etc.). NO import reaches back into
  the main repo (`src.`, `scripts.`, `baselines/…`, or relative `..`). A reviewer with only this folder
  can import and run everything. Third-party imports are all standard PyPI (torch, numpy, pandas,
  sklearn, scipy, arch, matplotlib).

### 7. Determinism / absolute paths — PASS
- Seeds fixed in `config.py` (`seeds=(42,123,2026,7,2024)`; SMOKE `seeds=(42,)`); `train_lstm` and the
  graph trainer call `torch.manual_seed(seed)` + `np.random.seed(seed)` per seed; deep models are
  seed-ensembled. No hardcoded absolute paths in any core `.py` (grep for `C:\`, `/c/luanvan`, `/home/`,
  `/Users/` → none). All paths derive from `Path(__file__)` or `--data-root`.

## Fixes applied (reviewer-facing docs/scripts only — no core .py logic touched, no commit)
- `submission/soict_lstm_gat/reproduce.sh` — now runs MAIN LSTM suite (run_lstm.py, vn30/vn100 × h1/h5)
  then the graph ablation suite (run_all.py), with output-path guidance. `bash -n` syntax-clean.
- `submission/soict_lstm_gat/REPRODUCE.md` — split into Main model (run_lstm.py) vs Graph ablation
  (run_all.py); added run_lstm.py to the S&P500 regeneration block; corrected test count 34→36; added
  scipy to the deps line and a seeds/determinism note.
- `submission/soict_lstm_gat/README.md` — Run block shows run_lstm.py as MAIN + run_all.py as the graph
  ablation, with the two result-dir locations.
- `submission/soict_lstm_gat/requirements.txt` — added `scipy`.

## Answers to the three questions
1. Is any code missing? No. All modules referenced by the entrypoints, tests, and docs are present and
   import cleanly; nothing reaches into the main repo. (S&P500 *data* is intentionally omitted under
   Yahoo licensing and its regeneration is documented.)
2. Can a reviewer, from this folder alone, install/train/test/reproduce? Yes — after the fixes. Before
   them, only the graph ablation was reproducible from the docs; the headline LSTM-vs-HAR result was
   not, because no doc/script invoked `run_lstm.py`. requirements.txt also omitted scipy.
3. Is there a reviewer helper script, and does it work + cover the MAIN model? Yes — `reproduce.sh`
   (install → tests → suites). It worked but previously covered ONLY the graph ablation. It now covers
   the MAIN per-observation LSTM (`run_lstm.py`) and the graph ablation (`run_all.py`). Verified: tests
   36/36 pass, and both entrypoints produce `result.json` under `--smoke`.

## Final verdict
Reproducible from the folder alone: YES (post-fix). No remaining blockers. Residual minor notes (not
blockers): (a) results are written two directory levels above the folder rather than inside it;
(b) S&P500 requires the documented Yahoo regeneration; (c) full 5-seed × 20-epoch suite needs a CUDA GPU
for reasonable wall-clock (CPU works but is slow). Core `.py` logic was not modified and nothing was
committed.
