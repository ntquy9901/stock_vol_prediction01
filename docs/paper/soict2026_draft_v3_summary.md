# SOICT 2026 draft v3 — review companion

Companion to `soict2026_draft_v3.tex`. Written 2026-08-06. v3 supersedes v2; `soict2026_draft_v2.tex`
/`_v2_summary.md` and `soict2026_draft_v1.tex`/`_v1_summary.md` are kept as frozen prior versions.

## What v3 does

v3 consolidates three ablation studies into one coherent paper. v2 had a single ablation (news
branch vs price-only backbone). v3 adds two ablations run overnight and reframes the paper's
contribution around what the three ablations jointly show.

- **Ablation 1 — news branch** (carried from v2): full model vs price-only backbone. News lowers
  QLIKE and RMSE across three seeds with a paired t-test (t = -6.22, -9.38; |t| > 4.30). Significant.
- **Ablation 2 — cross-stock graph** (NEW): real k-NN adjacency vs identity adjacency (no
  message passing, same parameter count), on the price-only backbone. No metric passes the 5%
  threshold (all |t| < 1.9). Source: `docs/reports/2026-08-05_graph_ablation_results.md`.
- **Ablation 3 — per-ticker gate** (NEW): learned gate vs always-on news fusion (no gate). No metric
  passes the threshold (all |t| < 2.3), and the simpler no-gate variant is marginally better on every
  metric's mean. Source: `docs/reports/2026-08-05_gate_ablation_results.md`.

## The reframed contribution (the single biggest interpretive judgment — scrutinize this first)

**v2 framing:** "our per-ticker gated graph architecture improves VN30 volatility forecasts."
**v3 framing:** "news *features* improve forecasts over classical HAR on QLIKE/R² (not RMSE/MAE, not
DirAcc); this holds *independently* of whether the architecture uses graph-based cross-stock sharing
or per-ticker gating — the two mechanisms built around the news features add no measurable value, so
a simpler architecture matches the full one."

Concrete changes that carry this reframe:

| Location | v2 | v3 |
|---|---|---|
| Title | "A Per-Ticker News Gate Improves Volatility Magnitude Forecasts for VN30 Without Improving Direction" | "News Features Improve VN30 Volatility Magnitude Forecasts Independently of Graph and Gate Mechanisms" |
| Abstract | gate is the headline mechanism | three ablations; only news features carry a paired-t gain; graph and gate add no measurable value |
| Intro key-abstraction | "We close both gaps with a per-ticker news gate" (gate = the fix) | "We build a news-fusion forecaster and ask which of its added components improves the forecast"; concatenation framed as a hypothesis to test, not a proven flaw |
| Intro contributions | 3 (architecture / news beats HAR / direction near-random) | 4 (adds "Only the news features earn their place" — the parsimony finding) |
| §5 Results | 1 ablation | 3-part ablation study (§5.2) with an integrative parsimony takeaway |
| §6 Discussion | §6.3 "The gate improves accuracy but resists interpretation" | §6.3 rewritten to "Parsimony: the news features, not the mechanisms around them" — the gate improves neither aggregate accuracy (Ablation 3) nor interpretability (4-method disagreement) |
| Related Work | graph "models cross-asset coupling"; gate "replaces uniform admission" (motivated as necessary) | softened: our ablation finds the cross-stock graph adds no measurable value; the gate does not beat uniform concatenation; contribution located in the news channel, not the mechanisms |
| Conclusion | gate lets the model admit different news per stock (mechanism as contribution) | news helps independently of graph/gate; simpler architecture matches the full one |

**Honesty guardrails applied (per task and CLAUDE.md objective-tone rule):**
- "No significant effect" is stated as "no metric passes the 5% threshold," never softened into an
  implied benefit.
- The n=3 limitation is stated explicitly: the two null ablations *bound* the graph and gate effects
  rather than *prove* their absence. The paper does not claim "proven equivalent."
- The consistent direction and small magnitude across all three ablations is offered as *qualitative*
  support for parsimony, not as a significance claim.
- Scope caveat added: each ablation removes one component; the graph and gate are not removed jointly.

## Section-by-section status

| Section | Status in v3 |
|---|---|
| Abstract | Rewritten. Three ablations + parsimony + direction finding. ~250 words (LNCS upper end). |
| §1 Introduction | Problem-gap + key-abstraction paragraphs reframed (hypothesis, not settled premise); 4th contribution added; results preview adds the parsimony sentence. |
| §2 Background | Unchanged from v2. |
| §3 Method | Unchanged except one sentence noting each component is ablated in §5.2. Full architecture still described as built (the gate and graph exist in the trained model). |
| §4 Setup | "Training" subsection retitled "Training and ablation protocol"; one-component-at-a-time ablation protocol stated. |
| §5 Results | §5.1 headline (Table 1, unchanged). §5.2 three-part ablation study: Ablation 1 (Table 2, ex-v2), Ablation 2 graph (Table 3, NEW), Ablation 3 gate (Table 4, NEW), integrative parsimony takeaway. §5.3 direction (compressed). §5.4 measurement correction. §5.5 model-free references (Table 5). |
| §6 Discussion | §6.1 direction, §6.2 HAR-lower-RMSE unchanged; §6.3 rewritten to parsimony; §6.4 horizon (hedged, unchanged). |
| §7 Related Work | Graph and news paragraphs softened to match the null ablations. |
| §8 Limitations + Conclusion | Limitation 1 now covers both the positive and the two null results at n=3; limitation 4 adds the "not jointly ablated" caveat; conclusion reframed to parsimony. |

## Page / word estimate

- **Prose word count (approx, excluding tables/TikZ/equations):** ~5,240 words.
- **Structure:** 1 TikZ architecture figure, 5 tables (was 3 in v2), 2 display equations, 12 references.
- **Estimated length:** ~11–12 pages excluding references (LNCS single-column). This sits at the
  budget edge. Two compression trims were applied (§5.3 direction sentence-shortened; Related Work
  news paragraph tightened) to hold margin, per the paper-writing skill's compression guidance
  (`compression_patterns.md` patterns 1 and 6). **Not compiled** — no LaTeX toolchain in the
  environment (same caveat as v2). Run `pdflatex` + `pdfinfo | grep Pages` before submission and, if
  over 12pp, apply a further compression pass (candidates: merge Tables 3+4 into one two-panel
  ablation table; trim §5.4 and §6.2).

## Quality gates run on v3

- **Mechanical gate (`gate_mechanical.md`), greps run on the v3 .tex:**
  - M1 em-dashes: 0 in prose (no `---`, no unicode `—`/`–`, no ` -- ` dash outside TikZ/tabular).
  - M11 passive voice: 0 in prose (2 introduced hits "no value is marked" fixed to "we mark no
    value").
  - M5/M16 banned adjectives: 0 in prose. Three grep hits are all out-of-scope: a header comment
    ("significant"), a `[VERIFY]` bibliography comment ("robust"), and the verbatim das2024 title
    ("comprehensive survey").
  - M6 throat-clearing, M15 exclamations, M18 content-free openers, M4 intensifiers, M12 wordiness: 0.
  - M2 antithesis: 11 "rather than" hits, all pass the keep-test (factual contrasts), consistent with
    v2's disposition. The one rhetorical "rather than a claim that the gate is the better design" was
    removed during compression.
- **Semantic gate (`gate_semantic.md`):**
  - S1/S3 define-before-use: new terms glossed at first use ("identity adjacency", "always-on news
    fusion"); acronyms carried from v2.
  - S8/S9 lexical + decomposition consistency: "price-only backbone", "cross-stock graph",
    "per-ticker gate" used consistently; cardinality agrees (three components / three ablations / four
    contributions / four limitations).
  - S10 non-duplication: each number lives in its home table/section.
  - S15 mappability: every new number traces to `2026-08-05_graph_ablation_results.md` §3 or
    `2026-08-05_gate_ablation_results.md`; t-stats rounded to 2 dp (report values -6.22/-9.38 exact;
    graph +0.30/-1.64/-1.89/+1.63/-0.98; gate +1.24/+0.37/+1.01/-0.39/-2.28 from +1.238/.../-2.283).
  - S20 positives-first, S21 honest-positioning: contributions lead the abstract/intro/conclusion;
    graph/gate null results framed as building-upon, not trashing prior work.
  - All 12 `\cite` resolve to 12 `\bibitem`; all 12 `\ref` resolve to labels; no duplicate labels.
- **Not run:** `pdflatex` compile + `pdffonts`/`pdfinfo` (no LaTeX toolchain). Run before submission.

## Number-map additions (new macros in the .tex)

| Macro | Value | Source |
|---|---|---|
| `\qlikeNoGraph` `\rmseNoGraph` `\maeNoGraph` `\rsqNoGraph` `\diraccNoGraph` | 0.4657 / 0.002788 / 0.0007877 / 0.7953 / 48.29 | graph_ablation §3 (identity adjacency, 3-seed mean) |
| `\qlikeNoGate` `\rmseNoGate` `\maeNoGate` `\rsqNoGate` `\diraccNoGate` | 0.4366 / 0.002723 / 0.0007873 / 0.8047 / 48.22 | gate_ablation (no-gate, 3-seed mean) |
| `\tcrit` | 4.30 | paired-t critical, two-sided 5%, df=2 |

The graph ablation's "k-NN graph" reference column reuses the `*BB` (price-only backbone) macros; the
gate ablation's "gated" column reuses the `*News` macros. Both equalities are verified against the
report files.

## Most important things to scrutinize (5–8)

1. **The contribution reframe itself** (see the table above). v3 changes the paper's claim from "our
   gated graph architecture wins" to "news features help; the graph and gate around them do not
   measurably help further." This is the single biggest interpretive call in the consolidation. If
   you disagree with demoting the gate/graph from selling points to null results, this is the place
   to push back — every downstream edit (title, abstract, contributions, §6.3, Related Work,
   conclusion) follows from it.
2. **The title change.** New title foregrounds "News Features … Independently of Graph and Gate
   Mechanisms." Confirm this is the framing you want on the first page; it commits the paper to the
   parsimony story.
3. **Keeping the full architecture in §3 while the ablations show two of its parts do not help.** The
   Method still describes the gate and graph as built. Decide whether that is honest-and-useful (we
   present what we ran, then ablate it) or whether a reviewer will ask "why ship the gate at all."
   §6.3 answers this (interpretability), but the answer is weak given the 4-method disagreement.
4. **n=3 statistics.** All three ablations rest on three seeds (paired-t, df=2, t_crit=4.30). The
   positive news result clears it (|t| = 6.22, 9.38); the two null results are "fail to reject," not
   "proven equal." The paper says so, but confirm you are comfortable publishing two null ablations
   at n=3. ≥5 seeds would strengthen all three (noted in Limitations).
5. **Page budget.** Estimated ~11–12pp excl. refs, uncompiled. If a compile comes back over 12,
   Tables 3 and 4 can merge into one two-panel ablation table. Confirm before submission.
6. **Classical HAR protocol caveat** (carried from v2, unchanged): point-wise 80/20 for HAR vs
   windowed 70/15/15 for deep models. Still a reference comparison, not protocol-matched.
7. **Related Work softening.** Confirm the softened graph/gate language still positions the paper
   fairly against Sonani et al. (backbone source) and the sentiment+GNN line — it now says the graph
   adds no measurable value *on this panel*, scoped to VN30.

## Before-submission checklist (carry-over, still open)

- [ ] Fill real author names/affiliations (single-blind).
- [ ] Compile; confirm page count ≤ 12 excl. refs; check `pdffonts` all embedded.
- [ ] Verify exact volume/pages/DOI for the `[VERIFY]` references.
- [ ] Optional: re-run all three ablations at ≥5 seeds to strengthen the paired-t claims.
- [ ] Optional: merge Tables 3+4 if the compile is over budget.
- [ ] Optional: add a learning-curve figure from `results/.../loss_history.json`.
