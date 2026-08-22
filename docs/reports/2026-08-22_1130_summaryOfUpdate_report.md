# Summary of update — SOICT paper reframed to honest-nuanced framing (md + tex)

Date: 2026-08-22

## Scope
Finalize the SOICT submission in both markdown and LaTeX, reconciling the older per-observation
"deep beats HAR on VN" headline into a single coherent, honest-nuanced framing. Docs-only change.

## Files changed (path → purpose)
- `docs/paper/soict_paper_complete.md` → markdown source; abstract, introduction/contributions,
  method protocol, results reading, new S&P 500 date-clustered table, new model-free graph-screening
  subsection (§6.7), findings, discussion, limitations, conclusion rewritten.
- `docs/paper/soict_harlstmgat.tex` → LaTeX synced to the same content; new S&P 500 HAR-anchored table
  (`tab:haranchor-sp500`), model-free screening paragraph, graph-attribution extended with S&P 500 rows,
  narrative sections rewritten. Structure verified (5 tables/5 tabulars balanced, 1 figure, 1 document).
- `docs/reports/2026-08-22_1130_summaryOfUpdate_report.md` → this report.

## Framing (now coherent across both files)
Three contributions: (1) panel-correct multi-horizon benchmark with date-clustered Diebold–Mariano
inference (naive per-observation DM overstates significance by ~sqrt(N)); (2) HAR not significantly beaten
on the small VN panels, while on the large S&P 500 subset the deep temporal LSTM significantly beats HAR
and the convex combination beats HAR at all horizons; (3) the cross-sectional graph adds no OOS value on
any panel (model-free evidence; genuine null, not bug/overfit); (4) per-metric fairness (all five metrics).
The former per-observation VN significance is demoted to descriptive; the surviving beat-HAR result is on
the large S&P 500 panel under date-clustered inference.

## Headline numbers placed in the abstract (spot-check vs `reports/experiment_results.md`)
- S&P 500, deep temporal LSTM (E1) vs HAR, QLIKE, date-clustered DM: h1 +3.01% (p<0.001), h5 +4.09%
  (p<0.001), h10 +4.18% (p=0.001), h22 +7.10% (p<0.001).
- S&P 500 convex combination (E3) vs HAR: +3.16%/+3.58%/+3.62%/+5.39% at h1/h5/h10/h22, all p<0.001.
- VN30/VN100: no model significantly beats HAR at any horizon under date-clustered DM; closest is the
  convex combination at VN100 h22 (p=0.078).
- Graph: no graph residual (E6/E7) beats the no-graph residual (E5) anywhere; S&P 500 favors no-graph at
  h1 (p=0.0003) and h10 (p=0.015). Model-free neighbour-signal incremental OOS R² ≤ ~1%, ≈0 for
  innovation/lead-lag screens.

## Verification
- All quoted numbers cross-checked against `reports/experiment_results.md` (E0–E10 date-clustered ladder),
  `reports/model_free_graph_screening.md`, and `docs/reports/2026-08-22_graph_no_value_analysis.md`.
- Corrected stale S&P 500 fragility numbers (previously E5=165.6/E7=16.4 from a ~35-test-date run) to the
  current 457-node / ~300-test-date run (E5 QLIKE 2.9978 at h22, E7 5.7090 at h10).
- Fixed a sign error (VN100 h22 convex dQLIKE is +5.85%, improvement, not −5.85%).
- Horizon scope corrected: primary study covers h∈{1,5,10,22}; descriptive per-observation/snapshot
  studies use h∈{1,5}.

## PDF compile
Not produced — no LaTeX toolchain installed (pdflatex/latexmk/xelatex/tectonic/lualatex all absent).
`llncs.cls` and `splncs04.bst` are present in `docs/paper/`, and the figure PDF
(`diagrams/soict_harlstmgat.pdf`) is referenced via `\includegraphics`, so the `.tex` should compile once
a TeX distribution is available: `latexmk -pdf soict_harlstmgat.tex` from `docs/paper/` (run twice for
refs). The `.md` and `.tex` are the deliverables produced here.

## Data-quality gate
N/A (no data change).

## Objective wording
No personal address, no self-praise/"honest" declarations in the paper prose; VN30/VN100 stated as case
study, S&P 500 as cross-market check; no directional accuracy; p-values written as p<0.001 (no e-notation
in prose).

## Follow-ups / not included
- LaTeX→PDF compile pending a TeX toolchain.
- Masked-panel robustness check is ongoing (mentioned in one sentence in Limitations; results not included).
- Code review (`/code-review`) not run on this prose change given the deadline; recommended before final.
