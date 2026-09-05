# Summary of update — contemp-edge data fix, baseline formalization, paper v3/v4 (2026-09-05)

## What changed

### 1. Data-integrity fix (contemporaneous-edge runs)
The VN100 contemp JSONs were config-inconsistent (h1/h5 = 7 folds/3 seeds, h10/h22 = 6 folds/5 seeds),
caused by multiple orphan run-chains racing to write the same JSON paths during a fold/seed config
switch. All orphan chains and GPU python were killed, the two inconsistent JSONs discarded, and a
single clean sequential chain re-ran VN100 h10/h22 and all four VN30 horizons at a uniform 7 folds /
3 seeds. All 8 contemp JSONs now verified `n_folds=7`, `seeds=[42,123,2026]`.

### 2. Contemporaneous-edge baseline formalized (§3.F)
`baselines/2026-09-04_contemp_edge/` completed to the mandated structure: `requirements/`, `design/`,
`code/`, `code_review/`, `test/`.
- `code/run_contemp.py` — probe reusing the delivered VolGA pipeline; only new logic is
  `build_contemp_adj` (train-only, self-loop, Top-K |corr|, NaN-safe) and `_fold_adj` (edge dispatch).
- `test/test_contemp_adj.py` — 7 unit tests (shape/self-loop, top-k count, picks correlated sources,
  train-only no-lookahead, NaN-safe, both dispatch branches). 7/7 pass under the GPU venv.
- Floor constants now sourced from `pipeline_config` (single source of truth) instead of hardcoded
  literals (numerically identical; results unchanged).

### 3. Paper v3 rebuilt on consistent data
`docs/paper/soict_harlstmgat_2026-09-05_v3.tex` — contemporaneous-edge section rebuilt with the uniform
7f/3s data and VN30 added (previously "in progress"). A prior claim that the contemp graph
"significantly hurts at h22 (p=0.001)" was an artifact of the discarded 6f/5s data; the clean data
shows h22 VolGA-vs-LSTM QLIKE p=0.849 (n.s.). Two new tables (VN30 metrics + VN30 DM). Compiles
(15 pages, no undefined refs).

### 4. Paper v4 = v3 + citation-review fixes + adversarial-review fixes
`docs/paper/soict_harlstmgat_2026-09-05_v4.tex` (16 pages). Applied the citation review
(`docs/paper/soict_harlstmgat_2026-09-04_citation_review_*.md`):
- New/placed citations (all verified real): `parkinson1980` at first use; `karpoff1987` for the
  volume-volatility motivation (framed predictive, not causal); `petersen2009` for cross-sectional
  dependence (the sqrt(N) statement qualified, no longer asserted as a rule); `bollerslev1986` for
  GARCH; `patton2011` on QLIKE (asymmetry softened).
- Wording: "famously hard to beat"/"close to optimal" -> strong-benchmark / limited-incremental-room;
  "spillover" (model mechanism) -> cross-sectional dependence, predictive not causal; split-invariance
  narrowed to a common multiplicative split factor with the corporate-action caveat retained.
- Data claim now sourced: VN30 mean pairwise sqrt-Parkinson correlation (0.345) > VN100 (0.270),
  computed from the data with the calculation window stated; interpretation reworded as an association,
  not an identified mechanism.
- Bibliography metadata completed and verified via arXiv/SSRN: `gnarharx2025` (O Nuallain, T.,
  arXiv:2510.24443), `clements2024` (Clements, Preve, Tee; SSRN 4733597, working paper).

### 5. Adversarial-review fixes (also applied to v3)
- VN30 "best QLIKE at every horizon" -> "three of the four" (LSTM edges ahead at h10).
- VN100 h10 graph favoured-model corrected (was mislabelled LSTM; VolGA favoured, n.s.).

## Files
- `baselines/2026-09-04_contemp_edge/{requirements,design,code,code_review,test}/*`
- `docs/paper/soict_harlstmgat_2026-09-05_v3.tex`, `..._v4.tex` (PDFs gitignored)
- `results/contemp_edge/contemp_contemp_{vn100,vn30}_h{1,5,10,22}.json` (8 clean JSONs)

## Tests + review
- `test_contemp_adj.py`: 7/7 pass (GPU venv).
- Code review: adversarial subagent; 1 MAJOR + 2 MINOR, all resolved; numbers cross-verified against
  the JSONs. See `baselines/2026-09-04_contemp_edge/code_review/code_review_2026-09-05.md`.

## Data-quality gate
Pandera/Evidently: N/A (no change to `data/processed`, features, or manifest; this change is code +
results JSON + paper). The contemp JSONs are model outputs, not raw/processed data.

## In progress (not part of this commit)
- Matched-protocol vol->PK run (7f/3s, both markets) to compare the two edges fairly; a side-by-side
  edge-comparison report is being generated to decide whether v5 should keep vol->PK as the main
  report (contemp as ablation) or promote contemp. Preliminary reading: contemp does not robustly beat
  the no-graph LSTM either, so the current vol->PK-as-main framing is expected to stand.

## Risks / follow-ups
- Estimator tables and HNX/HOSE/S&P 500 walk-forward remain [provisional] in v3/v4 (marked as such).
- Trim to <=12 pages excl. refs before submission (v4 is 16 pages).
