# Unified Roadmap — Track A (paper) + Track B (pooled news-GNN), architecture based on Track B

Date: 2026-08-08. This roadmap unifies all remaining Track A and Track B work into one
dependency-ordered plan. Per the standing directive, the ARCHITECTURE FOUNDATION is Track B (the
pooled LSTM/News/GNN), and the paper (Track A) is to be rebuilt on it; Track A's old common-date
panel architecture is superseded and, where useful, is reframed as the pooled-vs-common-date
ablation (A1).

## Decisions and constraints in force
- Architecture basis = Track B (pooled). Classical HAR remains the external baseline.
- Paper rebuilt as v4 (v3 kept frozen).
- Autonomous execution authorized; each task under CLAUDE.md DoD (tests, ruff, 3-layer code-review,
  summary report, commit+push) and superpowers skills (systematic-debugging, TDD, verification-
  before-completion with captured evidence, requesting-code-review). Leakage/scaler/provenance
  preserved. Training budget 5-epoch screening / ≤10-epoch confirm; not beyond 10.
- PARKED for user sign-off (do NOT auto-execute): R10 graph-training-loop batching (changes
  gradient/update semantics + provenance). Only safe optimizations auto-run.

## Validated current state (evidence-backed)
- Track B Task 7 DONE + pushed on `feature/pooled-news-gnn-pilot` (`702289f`): G1 positivity fixed
  via a denormalized floor (verified: pytest 92 passed, ruff clean, floor at models.py:206). Result
  (seed 42, 5 epochs, 33 tickers): **graph message-passing HURTS** — G1 worse than its no-graph
  control G0 on all 6 metrics (RMSE 0.002850 vs 0.002611, R² 0.642 vs 0.700); neither graph variant
  beats HAR/P1. GNN is confirmed (indicatively) an ablation, not the main architecture.
- Track B P0-P3 5-epoch screening done (P1 pooled Price LSTM best; P2/P3 not promotion-eligible).
- Track A paper v3: parsimony story (news helps QLIKE/RMSE vs backbone; graph/gate null), ~12pp,
  SOICT deadline 2026-09-16.

## Consolidated finding shaping the roadmap
Track B's data-regime change does not overturn the parsimony story; it reframes it. The pooled
architecture's value (if any) is (1) more data and (2) the news channel — not the graph or gate.
So the headline experiments are A1 (does pooling give the data advantage) and the pooled P1/P2/P3
parsimony core; the graph work (A2/A7) is confirmatory of a likely null.

---

## Phase 0 — Foundations (Track B, low-risk, start now)
- **T0.1 Multi-horizon `--horizon` knob** (multi-horizon plan M0-M6). Plumb one `horizon` into the
  three manifest builders in `run_pilot.py`, add `--horizon {1,5,10,22}`, namespace outputs, add the
  h=22 leakage regression test (single shared horizon across pooled+graph manifests; graph-safe
  boundary). NO model/scaler change (target scaler is horizon-independent). Enables the paper's
  horizon table (G6). Verify: `pytest test/` green incl new h cases; smoke each horizon.
- **T0.2 Graph-path optimization**: (i) batch the forward-only warm-start / graph-safe P3 builders IF
  confirmed forward-only (cuts the ~21-min warm-start bottleneck); (ii) **R10 — batch the graph
  TRAINING loop (APPROVED by user 2026-08-08).** R10 changes SGD-batch-1 update semantics, so it
  re-baselines graph results: all graph comparisons (A2/A7, and a re-run of G0/G1) must use the
  batched trainer for both arms; Task 7's single-seed batch-1 G0/G1 numbers are then superseded, not
  compared against. Verify training still converges + positivity gate holds after batching.

## Phase 1 — Headline Track B experiments (validate the architecture; n≥5 for claim-defining)
- **T1.1 A1 pooled-vs-common-date** (HEADLINE): identical P1/P2 trained on pooled vs common-date-only
  manifest; proves pooling is the data advantage that lets deep models reach/beat HAR. New data code
  (common-date per-sample manifest from the existing graph common-date axis). n≥5 seeds.
- **T1.2 Pooled P1/P2/P3 multi-seed** (news on/off, gate) at proper seeds — the parsimony core on the
  pooled regime; replicate/contrast Track A's news-helps + gate-null on more data.
- **T1.3 A2 news-on-graph vs price-only-graph** (3-way: G0 / G1-price-only / G1-news+price): isolate
  news propagation (the novelty Track A never tested). Likely null given Task 7 — show it cleanly.

## Phase 2 — Rigor ablations (pre-empt reviewer objections)
- **T2.1 A6** shared vs per-ticker gate (config knob).
- **T2.2 A7/A8** edge construction (k-NN/corr/identity) + k on the news-graph (cite Track A method).
- **T2.3 A9** shared vs per-ticker LSTM — tests the pooling rationale (per-ticker loses on small data).

## Phase 3 — Multi-horizon runs (1/5/10/22) on the validated pooled architecture (uses T0.1)

## Phase 4 — Paper rebuild (Track A on Track B) = v4
- Rewrite Method (§3) to the pooled news-GNN; regenerate all comparison tables from Track B results;
  HAR external baseline; old common-date models become the A1 ablation.
- Fold paper-audit gaps, in value order: **G2 Diebold-Mariano** (reuse saved predictions — highest
  value/lowest cost), **G4** repro/compute + public-repo statement, **G6** horizon table (from
  Phase 3), **G7** k-sensitivity sentence, **G1 GARCH(1,1)** baseline (needs a real run; Bollerslev
  1986), **G3** news-corpus stats, **G5** news-weighting related work (verify Rahimikia/MANA-Net),
  **G8** calm-vs-turbulent regime robustness. Keep ≤12pp (compress §5.4, merge ablation tables).

## Cross-cutting guardrails per task
Leakage: split-before-HAR/scaler/window; train-only fit; ticker-ID inverse; causal news cutoff;
graph-safe P3 boundary; manifest content-hash equality; shuffle=False. Evidence-capture
(verification-before-completion) before any "done" claim; 3-layer code-review before commit;
commit+push per task.

## Needs user sign-off (surfaced, not auto-done)
1. ~~R10 graph-training-loop batching~~ — APPROVED by user 2026-08-08; now in T0.2 (re-baselines
   graph results, see T0.2 note).
2. Final acceptance of the paper v4 rebuild direction once Track B numbers land.

## Execution order (autonomous)
T0.1 → T1.1 → T1.2 → T1.3 → T2.* → Phase 3 → Phase 4. Graph work (T0.2/T1.3/T2.2) deprioritized
behind the pooled headline given Task 7's graph-null. A consolidated final report will summarize
every task's evidence for review.
