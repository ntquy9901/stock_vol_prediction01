# Val/test reporting reorganization: lead with TEST, demote validation to selection-only

Presentation-only edit to the Track-B paper (both the Markdown draft and the LNCS `.tex`). No metric
value was changed; validation is demoted from a co-equal results block to a labelled model-selection
diagnostic, TEST becomes the headline in every main-results table and passage, and the val-vs-test
divergence is reframed as a robustness finding. Source recommendation:
`docs/reports/2026-08-11_1458_val_test_reporting_and_gnn_gap.md`.

## Files changed

| File | Purpose of change |
|---|---|
| `docs/paper/track_b_paper_draft.md` | Reorder ladder/graph tables to TEST-first, relabel VAL rows selection-only, add reporting-protocol sentence, reframe graph/gate/h22 divergence as robustness, add EDA-grounded parsimony sentence |
| `docs/paper/soict2026_trackb_v1.tex` | Same edits mirrored in LNCS structure (Table `tab:ladder` reordered, `tab:pairedt` caption relabelled, `\subsubsection{Reporting protocol.}` added, §Results/§Discussion reframed) |

`submission_track_b/PAPER_MAP.md` and `submission_track_b/reproduce.py` were NOT changed: they are
already test-oriented, and `python reproduce.py view` reproduces every headline number unchanged.

## What moved / got relabelled

1. **Ladder table (Table 2 / `tab:ladder`).** The TEST block now appears first as "TEST (reported)";
   the VALIDATION block follows, labelled "VAL (selection only)" / "VAL (selection)". Caption states
   the TEST block is the reported result and the VALIDATION block is shown for the model-selection
   period only, not a co-equal result. Validation numbers are kept (demoted, not deleted). No move to
   an appendix — the subordinate+relabel option from the recommendation was taken to preserve LNCS
   compile structure and avoid new cross-refs.
2. **Paired-t table (Table 3 / `tab:pairedt`).** Caption now states the TEST rows are the reported
   significance and the VAL rows are a selection-period diagnostic, with the primary graph significance
   being the Diebold-Mariano test on the held-out test set (Table 4 / `tab:graph`). Rows and numbers
   unchanged.
3. **Graph ablation table (Table 4, `.md`).** TEST row moved above the VAL row; TEST labelled
   "reported", VAL labelled "selection only". The `.tex` graph table (`tab:graph`) was already
   TEST-only (multi-horizon), so only its surrounding prose was reframed.
4. **Multi-horizon table (Table 5, `.md`).** Caption now states the held-out test columns are the
   reported result and set the verdict; the companion validation QLIKE is a selection-period diagnostic
   only.

## Protocol sentence added (stated once, in Experimental Setup)

> Model selection — early-stopping and best-checkpoint — uses the validation split only; all reported
> results and significance tests are computed on the held-out test set, which is consulted once.
> Validation metrics appear in this paper solely to expose the selection-period behaviour of each
> component; the held-out test column is the reported result, and where the two diverge (the gate and
> the graph) the test column governs the verdict.

`.md` places it as a "Reporting protocol" paragraph in §5; `.tex` as `\subsubsection{Reporting
protocol.}` in the Experimental Setup section. A one-line version was also added to the abstract of
both files.

## Reframing added (val-vs-test divergence as robustness, not a second result)

- **Graph (§6.3 / §sec:graph):** prose now leads with the held-out-test null (paired-t p=0.79, DM not
  significant, verdict B), then reads the validation QLIKE gain (all-3-seed, p=0.0096) as
  "selection-period optimism rather than a second result."
- **Gate (§6.2 / Table 3 prose):** the gate's squared-error improvement is now stated as "confined to
  validation and does not carry to test, a selection-period gain rather than a generalizing one."
- **Multi-horizon h22 (§6.4 / §sec:multihorizon):** the h22 val-improves/test-reverses case is now
  framed as "the same selection-period optimism seen at h5, here an outright validation-to-test
  reversal, so the test column governs and the verdict stays null."
- **Parsimony/robustness sentence (§7 / §Discussion, recommendation point 5):** added, citing the
  project graph EDA — about 77% of cross-stock Parkinson correlation is a single market factor HAR
  already captures (mean R^2 on the market factor 0.4241; only 23.1% survives market adjustment) and
  the k-NN neighbourhoods reshuffle out of sample (consecutive-snapshot Top-5 neighbour Jaccard 0.3900,
  edge turnover 0.5982), so a correlation edge selected on the train/validation window need not persist
  into the later test regime. Numbers verified against `docs/eda/reports/EDA_GRAPH_REPORT.md`.

## Verification

- **No metric value changed.** Numeric-token diff vs HEAD: the `.tex` removed zero numeric tokens
  (pure reorder). The `.md`'s only removals are punctuation-variants of `0.5599`/`0.559854` and one
  prose occurrence of `0.513001` that still appears in Tables 2 and 4. Added numeric tokens are the new
  EDA figures (77 / 0.4241 / 23.1 / 0.3900 / 0.5982) plus section/horizon references in new prose.
- **LaTeX structure intact.** Environments balanced (18 begin/end pairs), all `\ref`/`\cite`/macro
  references resolve to a defined `\label`/`\bibitem`/`\newcommand`; no macro left unused.
- **Cross-file consistency.** `python reproduce.py view` runs clean and reproduces every headline number
  unchanged (G1 test QLIKE 0.5759, P2 0.5031 val / 0.5599 test, HAR 0.5793, HARQ 0.5737, multi-horizon
  deltas and paired-t p-values 0.530/0.791/0.067/0.143). PAPER_MAP and view remain test-oriented and
  consistent with the paper.
- **Lead-with-test confirmed.** Grep shows both main-results tables begin with "TEST (reported)" rows
  and validation rows are labelled selection-only in both files.

## Commands run

- `python temp/texlint.py` (balanced envs, resolved refs/cites/macros; temp script removed after use)
- numeric-token diff vs `git show HEAD:<file>` for both files (no metric change)
- `python reproduce.py view` (bundle smoke; numbers consistent)

## Code review

Self-review only for this presentation-only docs change: grep-verified test-first ordering and
selection-only labels in both files, numeric-token diff confirming no metric value changed, EDA figures
verified against the source report, and LaTeX structural lint (balanced envs, resolved refs). No
`/code-review` 3-layer run was performed for this docs-only reorganization.

## DoD

- Code/docs match the request (presentation reorg, no number change): yes.
- Tests/lint/diff-cover/data-quality: N/A — docs-only change, no code or data/pipeline touched
  (`reproduce.py view` run as a consistency smoke only).
- Summary report: this file.
- Ledger + dashboard: entry `val-test-reporting-fix-2026-08-11` added, dashboard regenerated (0-red).
- Push: after `git pull --rebase origin master`.
