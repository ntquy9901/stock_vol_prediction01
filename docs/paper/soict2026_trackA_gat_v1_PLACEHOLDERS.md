# Review guide: `soict2026_trackA_gat_v1.tex` (Track-A-GAT leave-one-out draft)

Status: DRAFT for review. Format matches `soict2026_trackb_v1.tex` (llncs.cls, splncs04.bst, booktabs,
tikz). Architecture and ablation logic follow, without invention, the Track-A-GAT sources:
`.worktrees/trackA-gat/baselines/2026-08-15_trackA_gat_edge/design/ARCHITECTURE_DETAILED.md`,
`code/model.py`, `code/gat.py`, `code/run_ablation.py`.

All experimental numbers are UNMEASURED (training still running as of 2026-08-15) and are rendered as
placeholders: `\tbd` (a bare `{{TBD}}` cell) or `\phv{label}` (an inline `{{TBD:label}}` reminder). The
1-epoch smoke run is a diagnostic only and is intentionally omitted, per the task brief. Do not cite any
number from the `.tex` until the items below are filled from the canonical result JSONs.

## (a) Every placeholder to fill

Result source (per horizon/seed): `results/trackA_ablation_h{h}_seed{seed}_<TS>/ladder_metrics.json`
(keys: `rungs.{HAR,FULL,minus_graph,minus_gate,minus_news}.{validation_metrics,test_metrics}` and
`leave_one_out_effects.{graph,gate,news}`). DM statistics/p-values come from the DM/HLN routine on
seed-ensembled test predictions (FULL vs each variant and FULL vs HAR).

| ID | What | Count | LaTeX location |
|----|------|-------|----------------|
| P1 | Abstract quantitative claims: news/gate/graph effect sign+size, FULL-vs-HAR QLIKE verdict + DM p, DirAcc band | 5 `\phv` tokens | Abstract |
| P2 | Main held-out TEST metrics: rungs {HAR, FULL, minus_graph, minus_gate, minus_news} x horizons {1,5,10,22} x metrics {MSE,RMSE,MAE,R2,QLIKE,DirAcc} | 120 `\tbd` cells | Table 1 `tab:main` |
| P3 | VALIDATION QLIKE selection diagnostic: 5 rungs x 4 horizons | 20 `\tbd` cells | Table 2 `tab:val` |
| P4 | Leave-one-out effects effect(X)=QLIKE(FULL)-QLIKE(minus_X): X in {graph,gate,news} x 4 horizons | 12 `\tbd` cells | Table 3 `tab:effects` |
| P5 | Diebold-Mariano (HLN): FULL vs {HAR, minus_graph, minus_gate, minus_news} x 4 horizons, each cell = (DM stat; p) | 16 `\tbd` cells | Table 4 `tab:dm` |
| P6 | In-text numeric mentions / narration bracketed as `[To fill ...]` in Results and Discussion | ~4 bracketed notes + `\phv{DirAcc band}` | Sec. Results (main, direction), Sec. Discussion (effects reading), Conclusion |
| P7 | Author names / affiliation / e-mail | title block | `\author`, `\institute` |
| P8 | Compute environment (GPU / PyTorch / CUDA) | 1 `\phv` | Sec. Experimental Setup, "Compute" |

Bracketed `\emph{[To fill ...]}` / `\textit{[...]}` notes to replace with prose once numbers exist:
- Results/main: per-horizon statement of whether FULL beats HAR (which metrics) and whether any variant beats FULL.
- Discussion/effects: which components are DM-significant negative (helped) vs null, and sign stability across horizons.
- Conclusion: which components carry a DM-significant contribution and the FULL-vs-HAR out-of-sample verdict.

Grep aids: search the `.tex` for `\tbd`, `\phv`, `[To fill`, and `{{TBD`.

## (b) Section outline (where each placeholder group sits)

1. Abstract — P1 (claims), P7 (authors).
2. Introduction — 4 contributions; no numbers.
3. Background and Problem Setup — Parkinson variance (sigma^2, non-negative), HAR, target/losses (QLIKE, DirAcc per-ticker); no numbers.
4. Method — full model (Fig. 1), 5 node features + 22-day window, three branches, directed vol->PK Top-5 edge (Eq. 5), softplus floor (Eq. 4), leave-one-out variants (Table `tab:variants` = design, not results); no experimental numbers.
5. Experimental Setup — universe/split/leakage, objective/hyperparameters, reporting protocol, DM significance, P8 (compute).
6. Results — Table 1 (P2), Table 2 (P3), Table 3 (P4), Table 4 (P5), direction (P6).
7. Discussion — reading the effects (P6 bracket), prior EDA findings (labeled as prior, not new), direction rationale.
8. Related Work — econometric / deep-graph / lead-lag-spillover / news-augmented.
9. Limitations and Conclusion — P6 (conclusion bracket).

## (c) [NEEDS CLARIFICATION] items

1. Authors/affiliation: template default "Author One / Author Two, VNUHCM University of Science" is
   carried over verbatim (P7). Replace with the real author block before submission.
2. Number of seeds: the code header (`run_ablation.py`) states "1 seed first (extend later)"; `\nseed`
   is set to 3 (seeds 42/123/2026, matching the project convention and the abstract's "seed-ensembled").
   Confirm the final run uses 3 seeds; if it stays at 1, change `\nseed` and drop "seed-ensembled".
2b. DM on seed-ensembled predictions requires >1 seed; if only 1 seed is run, the DM basis reduces to a
   single prediction series (still valid) but "seed-ensembled" wording must be removed.
3. Universe size: fixed at `\ntick`=33 per the task brief and stated as a Limitation. The design doc does
   not itself pin the count; confirm 33 against the final basis (`build_trackA_basis`) before submission.
4. `market_pk` timing: the design doc marks it "contemporaneous (col t)" (cross-sectional median of
   sqrt(PK) over present tickers at t). It is a same-day cross-sectional aggregate, not a look-ahead into
   the target; the draft describes it as a market factor. Confirm this is acceptable as leakage-safe
   (it uses only date-t information available at the forecast origin).
5. `volume_zscore` window: draft uses 22 (per ARCHITECTURE_DETAILED "22-trading-day unified window");
   note the underlying column key retains the legacy name `volume_zscore_20`. No paper text depends on
   the legacy name; confirm the final basis uses window 22.
6. Compute environment (P8): not specified in the sources read; fill GPU/PyTorch/CUDA from the actual run.
7. Venue target: filename/format assume SOICT 2026 (LNCS), matching the newest template. Confirm the
   target venue; an IEEE two-column reformat would be a separate compression pass (per the note in
   `soict2026_draft_v3.tex`).
