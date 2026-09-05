# Summary of update — overnight VN100+VN30 training + new VolGA paper draft (2026-09-06)

## Scope
Autonomous overnight run (plan agreed before user slept): train the final paper model set
(HAR-X, LSTM, VolGA on the fixed horizon-matched edge) on the two Vietnamese markets locally, verify
over/under-fit evidence, generate LaTeX tables, and draft the new paper's Results/Overfit/Discussion sections.
S&P 500 (primary market) is trained separately by the user on Colab (A100).

## Training runs (local RTX 4060, all data corruption=0)
| Market | Nodes | Config | Result |
|---|---|---|---|
| VN100 | 102 (of 104 tickers) | 4 horizons × 7 folds × 5 seeds × {LSTM,VolGA}+HAR/HAR-X, batch 64 | 4 result JSONs, fit=ok all |
| VN30 | 31 (of 33 tickers) | same | 4 result JSONs, fit=ok all |

Both markets verified clean before training: structural corruption = 0.0000% (high<low, nonpositive OHLC,
open/close outside range, NaN/inf), P3-cleaned with `dirty_flag`/`zero_range_flag` columns; residual markers are
real illiquidity (kept + flagged), not dirty data.

## Findings (honest, from result JSONs)
**VN100:** VolGA attains the lowest QLIKE, MSE, RMSE, R² at the daily horizon (QLIKE 0.4911), but the
Diebold–Mariano gap over LSTM (p=0.16) and HAR-X (p=0.14) is not significant. HAR-X is the strongest QLIKE
model at h5/h10/h22; VolGA holds the best MAE beyond h1.
**VN30:** VolGA leads MSE/MAE/R² at h1–h5; the linear models lead QLIKE at every horizon, and HAR-X's QLIKE
edge over VolGA at h1 is significant (DM p=0.009). At h22 HAR-X is strongest on every metric.
**Over/under-fit:** every learned model (LSTM, VolGA) is classified `ok` at all VN horizons — no over- or
under-fit — from the train/val/test QLIKE+R² gaps; learning curves recorded per fold×seed.

## Over/under-fit evidence guarantee (this session)
The pre-push gate now enforces fit evidence for ALL training result JSONs (broadened from masked_rich-only):
learned models auto-detected by name; the committed VN100/VN30 JSONs passed the gate (evidence complete).

## Artifacts (all committed + pushed to origin/master)
- `results/edge_hmatched/edgehm_{vn100,vn30}_h{1,5,10,22}.json` — metrics + train/val/test + fit_diagnostics + learning_curves.
- `docs/paper/tables/{vn100,vn30}_tables.tex` — per-metric booktabs tables (best per column bold) via `scripts/paper/build_edgehm_tables.py`.
- `scripts/paper/build_edgehm_tables.py` (+ test, 100%) — table/DM/fit builder from result JSONs.
- `docs/paper/volga_new_2026-09-06_skeleton.tex` — Method drafted; VN100 + VN30 Results, Overfit-evidence,
  Discussion, Limitations, Conclusion filled from real numbers; all prose passed the mechanical style gate
  (0 em-dash/filler/hedging/passive). S&P 500 subsection + final Introduction + Abstract remain placeholders.

## Tests + gate
Every push this session passed the pre-push gate (changed-scope tests, diff-cover C0/C1, ruff F, data-quality
329, delivered-baseline 69, config-hardcode, overfit-evidence, 7/7 checklist). No QG_SKIP used.

## Pending (needs user)
- **S&P 500 (primary market):** run `notebooks/train_volga_colab_a100.ipynb` on Colab A100 (cell 1 git clone
  gets the latest code; SP500 data from Drive `MyDrive/public_bk/luanvan_data/`). Results push to git from cell 6.
- After `git pull` of SP500 results: fill the S&P 500 Results subsection, then rewrite the final Introduction
  and Abstract from the complete evidence (introduction-twice), run the section checklists + red-team.
