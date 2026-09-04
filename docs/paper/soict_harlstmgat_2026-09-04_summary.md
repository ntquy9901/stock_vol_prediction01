# Changelog — soict_harlstmgat_2026-09-04.tex (draft version)

New draft built from the template `docs/paper/soict_harlstmgat_hnx.tex`, reusing its LaTeX class
(`llncs`), preamble, Method/Related-Work structure and bibliography, and updated with the project's
current confirmed walk-forward results. No existing file was modified; only this `.md` and the
sibling `.tex` were created.

## Numeric source of truth
- `results/walkforward_volga/walkforward_volga_{vn30,vn100}_h{1,5,10,22}.json`
  (keys `metrics[model].{qlike,mse,rmse,mae,r2}`, `metrics_per_seed[model].qlike_std`,
  `dm_date_clustered.{VolGA_vs_HARX,VolGA_vs_LSTM}`; VolGA is stored under key `LSTM_wGAT_vol2pk`).
- `results/pooled_transfer_vn30/pooled_vn30_h1.json` (arm0/arm1 metrics + `paired_dm`).
- Cross-checked against `deliverables/2026-09-04_paper_pipeline_review/RESULTS_SUMMARY.md` and
  `CLAIMS.md`.

## What changed vs the template
- **Primary market switched from HNX to VN30 + VN100.** The template's HNX-primary framing and its
  `\input{generated/*.tex}` tables (older HNX-run numbers) were dropped. The two main results tables
  (`tab:vn30`, `tab:vn100`) are now self-contained `tabular` blocks filled directly from the current
  JSONs, so no stale generated table is pulled in.
- **Title/abstract/intro reframed as a Vietnamese-market case study and a parsimony (partly-negative)
  result:** no deep model beats HAR-X on QLIKE at any horizon on either panel; deep models' measurable
  edge is significant MAE reduction at short horizons; graph adds significant QLIKE gain over the
  no-graph LSTM only on VN100 at h1/h5.
- **Protocol section** rewritten from single chronological split to the 22-fold walk-forward
  (lookback 22, 5 seeds, retrain cadence K) actually used.
- **Model naming** follows project convention: HAR, HAR-X, "no-graph LSTM", VolGA (no module paths).
  HAR-X documented as a published baseline (Corsi 2009 + Clements-Preve-Tee 2024) with the two
  disclosed deviations (range-based Parkinson target, direct t+h target); added two bib entries
  (`clements2024`, `gnarharx2025`).
- **DM tables** rebuilt: `tab:dm` (VolGA vs HAR-X, QLIKE + MAE) and `tab:abl-graph` (VolGA vs no-graph
  LSTM, QLIKE) with the exact date-clustered p-values and favoured model.
- **Volume z-score window corrected to 22 days** (project canonical), vs the template's "20-day".
  Yang-Zhang window also stated as n=22 to match the configured monthly window.
- **New Limitations section** added covering: VN prices not split/dividend-adjusted (overnight tail),
  variance (sigma^2) target, QLIKE positivity floor + floor sensitivity, VN30 small-N, scope.
- **GARCH rows removed** from the main tables (the current walk-forward JSONs contain only
  HAR/HAR-X/LSTM/VolGA — no GARCH numbers). GARCH is retained only as a related-work reference.
- Style: no "hurts/fails/never helps"; all four horizons reported; no directional-accuracy claim; no
  e-notation in prose; row order HAR -> HAR-X -> LSTM -> VolGA; best-per-column in bold.

## Current (confirmed, final) results in this draft
- VN30 and VN100 multi-horizon walk-forward tables (h=1,5,10,22): MSE/RMSE/MAE/QLIKE/R2 — all from the
  completed JSONs.
- Date-clustered DM: VolGA vs HAR-X (QLIKE + MAE) and VolGA vs no-graph LSTM (QLIKE) — all cells.
- Honest headline claims 1-5 and 7 from `CLAIMS.md`.

## Marked provisional / pending (NOT presented as final)
- **Pooled/transfer VN30 ablation** (Sec. "Widening the training universe"): only h1 present in the
  JSON, tagged `[provisional]`; h5/h10/h22 flagged with a `% TODO` to fill from
  `results/pooled_transfer_vn30/pooled_vn30_h{5,10,22}.json` when complete.
- **Alternative-estimator tables** (Rogers-Satchell / windowed Yang-Zhang across markets): stated as a
  narrative claim (Parkinson chosen; RS/YZ/close2close worse) but the tables are omitted with a
  `% TODO` — not regenerated on clean-data walk-forward yet.
- **Additional markets** (HNX / HOSE / S&P 500): section marked `[provisional]` with a `% TODO`; no
  numbers carried over from earlier runs.

## Open TODOs for the coordinator / user before submission
1. Fill pooled/transfer VN30 h5/h10/h22 once those runs finish; confirm the multi-horizon verdict.
2. Regenerate estimator ablation tables (RS / windowed YZ) on the clean-data walk-forward, or keep the
   claim narrative-only and cite the estimator report.
3. Add HNX / HOSE / S&P 500 clean-data walk-forward tables if those markets are to be in scope.
4. Decide whether to add a GARCH(1,1) walk-forward benchmark (not in current JSONs).
5. Fill author/affiliation placeholders; confirm the architecture figure
   (`diagrams/soict_harlstmgat.png`) is current for the VN30/VN100 framing.
6. Verify page count against the 12-page SOICT limit after tables/figure settle.

## Compile check
- CPU `pdflatex -draftmode` run once: **exit code 0** (llncs.cls resolved). Only the normal first-pass
  "undefined references / rerun to get cross-references right" warnings appeared; a second pass
  resolves them. Byproduct `.aux/.log/.pdf` were deleted so only the `.tex` and this `.md` remain.
  No packages were installed; GPU not used; no training run; no git operations performed.
