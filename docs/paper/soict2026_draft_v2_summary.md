# SOICT 2026 draft v2 — review companion

Companion to `soict2026_draft_v2.tex`. Written 2026-08-05. v2 supersedes v1; `soict2026_draft_v1.tex`
and `soict2026_draft_v1_summary.md` are kept as the frozen prior version.

## What changed from v1 (the reframing)

v1 headlined the comparison as **price-only LSTM--GAT backbone (no news) vs. gated news fusion**.
v2 reframes the baseline structure:

- **Primary baseline is now the classical HAR econometric model** (Corsi 2009), an ordinary linear
  regression on the three HAR volatility scales. This is the standard benchmark any daily-volatility
  forecaster must beat. Numbers come from the rerun of `src/har_baseline/train.py` after a
  per-ticker temporal-split fix (see below), stored in `results/har_baseline_2026-08-05_224208/`.
- **The deep no-news comparison became an ablation.** The price-only LSTM--GAT backbone (identical
  to the full model minus the news branch) is now presented as an ablation that isolates the news
  branch's marginal contribution, not as the external baseline. Its 3-seed multi-seed statistics and
  paired t-tests are unchanged from v1; only the framing moved.
- **Terminology:** v1's "HAR-only backbone" (a deep model that uses HAR *features*) was renamed
  **"price-only backbone"** throughout, to remove collision with the **"classical HAR"** linear
  baseline. Both terms are defined before first use (classical HAR in Background §2; price-only
  backbone in Method §3.1 and Setup §4).

## The split-bug fix behind the rerun

The classical HAR numbers in v1's lineage were not usable as a primary baseline because
`train_har_baseline()` concatenated every ticker's full series end-to-end (arbitrary `os.listdir`
order) and applied a single global 80/20 cut. That put some tickers entirely in train and others
entirely in test, not a temporal split (violated CLAUDE.md §3.A). The fix
(`load_har_train_test_split`) cuts each ticker at 80% of its own chronological rows, mirroring
`HARVolatilityDataset._load_all_data`. Regression tests in `tests/har_baseline/test_train_split_fix.py`
assert every ticker appears in both splits with no per-ticker date leakage, plus a real-data smoke
test. Rerun (33 tickers, 84,549 train / 21,154 test).

## The honest result (mixed, reported as such)

Primary comparison (classical HAR vs. full gated news model, Table 1):

| Metric | Classical HAR | Gated news fusion | Winner |
|---|---|---|---|
| QLIKE | 0.5493 | 0.4430 | news |
| RMSE | 0.002182 | 0.002734 | HAR |
| MAE | 0.000575 | 0.0007930 | HAR |
| R² | 0.7419 | 0.8031 | news |
| DirAcc (%) | 48.65 | 47.77 | ~tie (near-random) |

The news model wins on QLIKE and R²; the classical HAR keeps lower RMSE and MAE. The paper reports
this split plainly (abstract, Table 1 with bold-on-winner per row, §5.1 takeaway, §6.2 explanation,
Related Work, Limitations) rather than narrating a clean win. The causal claim about the news branch
rests on the controlled ablation (Table 2), where the protocol is held fixed across the two models.

## Protocol caveat (stated in the paper)

The classical HAR uses a point-wise 80/20 per-ticker split; the deep models use a windowed 70/15/15
split. Table 1 is therefore a reference comparison against the field-standard model, and Section §5.2
(ablation) carries the protocol-matched causal statement. This caveat appears in Setup §4,
Discussion §6.2, and Limitations.

## Section structure (v2)

| Section | Change from v1 |
|---|---|
| Abstract | Rewritten: classical HAR primary + ablation for the news branch; HAR-lower-on-RMSE stated. |
| Introduction | Contribution 2 reframed (beats classical HAR on QLIKE/R²; ablation isolates news); results preview updated. |
| Background §2 | Added "and baseline" to the HAR subsection: HAR used both as features and as the classical linear baseline. |
| Method §3 | §3.1 relabels the no-news backbone as the "price-only backbone" (ablation control). |
| Setup §4 | Rewrote "what the comparison controls" into "Baselines and what each comparison controls" (two references + protocol caveat). |
| Results §5 | §5.1 new primary table (classical HAR vs news); §5.2 ablation (former Table 1); §5.3 direction; §5.4 correction; §5.5 references (added a classical-HAR row). |
| Discussion §6 | Added §6.2 "Why the classical HAR keeps lower RMSE". |
| Related Work §7 | Econometric paragraph now states HAR keeps lower RMSE/MAE in our own evaluation. |
| Limitations §8 | Added the protocol-mismatch limitation (now four limitations). |

## Number-map additions (classical HAR macros in the .tex)

| Macro | Value | Source |
|---|---|---|
| `\qlikeHARc` | 0.5493 | results/har_baseline_2026-08-05_224208/test_metrics.csv |
| `\rmseHARc` | 0.002182 | same |
| `\maeHARc` | 0.000575 | same |
| `\rsqHARc` | 0.7419 | same |
| `\diraccHARc` | 48.65 | same |

Backbone macros were renamed `\qlike/rmse/rsq/diracc HAR` → `\qlike/rmse/mae/rsq/diracc BB`
(price-only backbone); news and ablation-paired-t macros are unchanged from v1.

## Quality process run on v2

- **Mechanical gate (`gate_mechanical.md`) re-run with greps on the v2 .tex:** em-dashes M1 = 0 in
  prose (`---` hits are in comments; ` -- ` hits are TikZ arrow syntax; unicode dash = 0), passive
  voice M11 = 0 in prose (3 introduced hits fixed: "is tuned for exactly" → "optimizes", two "protocol
  is held/is fixed" → "which holds the protocol fixed"; 1 remaining is in a LaTeX comment), banned
  adjectives M5 = 0 in prose (the "robust"/"comprehensive" hits are a verify-comment and the das2024
  paper title), intensifiers M4 = 0, wordiness M12 = 0, throat-clearing M6 = 0, exclamations M15 = 0,
  content-free openers M18 = 0. The 11 "rather than" hits pass the M2 keep-test (factual contrasts).
- **Semantic gate (`gate_semantic.md`):** define-before-use holds for both renamed terms; no leftover
  "HAR-only backbone"; term counts consistent (price-only backbone ×10, classical HAR ×27); all 9
  `\ref` resolve to labels; all 12 `\cite` resolve to 12 `\bibitem`; Table 1/2/3 numbers trace to the
  macros and the rerun `test_metrics.csv`; honest-positioning verified (the RMSE/MAE loss to HAR is
  stated in five places).
- **Not run:** actual `pdflatex` compile + `pdffonts`/`pdfinfo` (no LaTeX toolchain in the
  environment). Run before submission. Adding the classical-HAR primary table plus the ablation table
  adds one table over v1 (three tables total); confirm page count stays ≤ 12 excl. refs on compile.

## Before-submission checklist (carry-over from v1, still open)

- [ ] Fill real author names/affiliations (single-blind).
- [ ] Compile; confirm page count ≤ 12 excl. refs; check `pdffonts` all embedded.
- [ ] Verify exact volume/pages/DOI for the 8 canon references.
- [ ] Decide: keep or cut the horizon Discussion paragraph (kept, hedged).
- [ ] Optional: add a learning-curve figure from `results/.../loss_history.json`.
- [ ] Optional: run ≥5 seeds to strengthen the QLIKE/RMSE ablation paired-t claim.
- [ ] Optional: re-run the classical HAR under the deep models' 70/15/15 windowed protocol for a
      protocol-matched primary comparison (removes the caveat currently stated in §4/§6.2/§8).
