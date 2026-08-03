# SOICT 2026 draft v1 — review companion

Companion to `soict2026_draft_v1.tex`. Lists length estimate, completeness, the content decisions
worth your scrutiny, and the quality process that was run. Written 2026-08-04.

## Files in `docs/paper/`

| File | Purpose |
|---|---|
| `soict2026_draft_v1.tex` | The draft. Springer LNCS (`\documentclass{llncs}`), single-column. |
| `soict2026_draft_v1_summary.md` | This file. |
| `project_context.md` | Skill's Stage-1 input (identity, contributions, term/number map). Gitignored. |
| `llncs.cls`, `splncs04.bst` | Springer class + bib style, copied from the SOICT template zip so the draft compiles as-is. |
| `.gitignore` | Ignores `project_context.md` only. |

## Length estimate (compile to confirm — no LaTeX toolchain was available here)

- Body prose ~3,200 words; 1 architecture figure (TikZ, self-contained), 2 result tables, 3
  equations, 12 references.
- Estimated **9–11 pages total, ~8–9 excluding references** in LNCS single-column. Comfortably
  under the 12-page (excl. refs) limit. It was **not padded** to fill pages (the skill forbids
  padding). There is room to add a learning-curve figure (see below) if you want it.
- **Action:** compile with `pdflatex` + `bibtex` and check the real page count
  (`pdfinfo paper.pdf | grep Pages`). No compiler was installed in the drafting environment.

## Section completeness

| Section | Status |
|---|---|
| Abstract, Introduction | Complete, real headline numbers. |
| Background (Parkinson, HAR, QLIKE, DirAcc definition) | Complete with equations. |
| Method (dataset+news pipeline, architecture, per-ticker gate, fusion) | Complete; exact tensor shapes/dims translated from `BAO_CAO_TONG_HOP.md` §2. |
| Experimental setup | Complete (universe, split, training, what the comparison controls). |
| Results (3-seed head-to-head, DirAcc, correction note, model-free references) | Complete; Tables 1–2 populated with real numbers. |
| Discussion (why direction is near-random; gate non-interpretability; horizon) | Complete. |
| Related work, Limitations, Conclusion | Complete. |
| **Author block / affiliations** | **PLACEHOLDER** (`Author One/Two`, generic HCMUS affiliation). Single-blind venue — fill in real names before submission. |
| **Learning-curve data figure** | **NOT INCLUDED.** Would show convergence at epoch ~20 from `results/.../loss_history.json`; route through a plotting step. Optional; the argument stands on the tables. |

## Content decisions to scrutinize before trusting this draft

1. **The horizon-reversal finding is included only as one clearly-caveated Discussion paragraph, no
   table.** The docs' own `2026-08-04_horizon_news_usefulness_analysis.md` flags that the 1/10/22-day
   numbers predate the P1.1/P1.2 fixes and were never re-run. The draft states this staleness
   directly and presents only the *structural* evidence (raw-data autocorrelations, unaffected by
   the fixes) plus a "needs corrected multi-seed re-run" caveat. **Decision to confirm:** keep it as
   a hedged observation, or cut it entirely. It is currently kept, hedged.

2. **Citations — 8 of 12 are now verified against the PDFs already in `docs/paper/`; 4 are
   standard-canon and left unverified-but-high-confidence.**
   - Verified against the actual source PDFs in this folder: `sonani2025` (arXiv 2502.15813),
     `korkusuz2023` (Finance Research Letters 55, 103992), `ouyang2021` (North American J. of
     Economics and Finance 56, 101383), `das2024` (Decision Analytics Journal 10, 100417).
   - The project docs' originally-named `Zhang, Pu, Cucuringu & Dong (IJF 2025)` and
     `Chi et al. (J. Forecasting 2026)` were **removed**: no matching PDF is present here and the
     details could not be verified, so they were replaced by the verified `ouyang2021`/`das2024`
     rather than cite something unconfirmed. If you specifically want Zhang/Chi cited, supply the
     exact references.
   - Left as high-confidence standard references (widely known; verify final volume/pages/DOI
     against the publisher before camera-ready): `corsi2009` (HAR), `parkinson1980` (range
     estimator), `patton2011` (QLIKE robust loss), `hochreiter1997` (LSTM), `velickovic2018` (GAT),
     `devlin2019` (BERT), `phobert2020` (PhoBERT), `kingma2015` (Adam). No fabricated DOIs or page
     numbers; a couple carry an inline `% [VERIFY]` comment on the exact pages.

3. **DirAcc numbers appear at two precisions and this is intentional.** Table 1 (headline) uses the
   3-seed means (48.47% / 47.77%); Table 2 (model-free reference comparison) uses the single seed-42
   run (48.1% / 47.6%) because the persistence/trailing-mean baselines were computed against that one
   run. Table 2's caption states this and gives the 3-seed means for reconciliation. Confirm you are
   comfortable with the two-precision presentation.

4. **"VN30" is qualified, not asserted as the live index.** The universe is 33 fixed tickers
   (28 of 30 official constituents + 5 long-history names that left the index; BSR/VPL excluded for
   short history). The Limitations section states this point-in-time caveat explicitly, per the data
   audit. Confirm the framing matches how you want to name the dataset.

5. **Core claim scoping.** The paper claims news improves QLIKE/RMSE (paired t-test, n=3) and
   explicitly does **not** claim a directional-accuracy improvement; it reframes the ~48% DirAcc as a
   data property (anti-persistent daily volatility changes) proven with model-free baselines. This
   matches the readiness report exactly. n=3 seeds is stated as a limitation (≥5 recommended).

## Quality process run (per the installed skill)

- **Skill pipeline followed:** Stage 1 project_context built from the project's own docs (the
  interactive 34-question brainstorm was skipped as instructed, no human to answer). Stages 2–5
  (architecture → drafts in the enforced order → integration → light compression) applied.
- **Mechanical gate (`gate_mechanical.md` Part C) run with greps; rendered prose is clean:**
  em-dashes M1 = 0 (table "n/a" cells and comments fixed/justified), passive voice M11 = 0 in prose
  (2 hits remain inside LaTeX comments only), banned adjectives M5 = 0 in prose ("significant" →
  "5% level", "robust" → "tolerates"), fancy verbs M17 = 0, exclamations M15 = 0, content-free
  openers M18 = 0.
- **Semantic gate (`gate_semantic.md`) fixes applied:** acronyms expanded at first use (HAR, LSTM,
  GAT, PCA, OHLCV, HOSE); Table 2/Table 1 numbers reconciled (S9/S10/S15); every `\cite` resolves to
  a `\bibitem` (12/12) and every `\ref` to a `\label` (no broken references); numbers traced to the
  source reports via `project_context.md`'s number-map (S15).
- **Not run:** actual `pdflatex` compile + `pdffonts`/`pdfinfo` pre-submission mechanical checks (no
  LaTeX toolchain in the environment). Run these before submission.

## Before submission checklist (carry-over)

- [ ] Fill real author names/affiliations (single-blind).
- [ ] Compile; confirm page count ≤ 12 excl. refs; check `pdffonts` all embedded.
- [ ] Verify exact volume/pages/DOI for the 8 canon references.
- [ ] Decide: keep or cut the horizon Discussion paragraph.
- [ ] Optional: add a learning-curve figure from `results/.../loss_history.json`.
- [ ] Optional: run ≥5 seeds to strengthen the QLIKE/RMSE paired-t claim.
- [ ] Later: separate ~40–50% compression pass + IEEE reformat for RIVF 2026 (note left in the .tex).
