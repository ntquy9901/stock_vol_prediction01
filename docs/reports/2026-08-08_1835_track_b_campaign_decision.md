# Track B campaign — consolidated findings and strategic decision

Date: 2026-08-08. This report consolidates the autonomous Track B campaign (Task 7 → T0.1 → T0.2 →
T1.1/A1 → T1.2) and the paper-audit, and states the decision now required before the paper is rebuilt.
Live evidence dashboard: `docs/reports/task_dashboard.html`. All numbers below were independently
re-run and captured, not copied from agent reports.

## 1. What was completed (all verified, committed, pushed)

| Task | Result | Evidence | Commits |
|---|---|---|---|
| Task 7 | G1 positivity fixed (denormalized floor) | pytest 92→ verified | `8055413`,`702289f` (branch) |
| T0.1 | `--horizon {1,5,10,22}` knob; scaler/model unchanged; h5 byte-identical; leakage guard | 110 tests | `7346c33` |
| T0.2 | Batch graph path (R10 approved): ~10× speedup; G0/G1 re-baseline; graph_hash byte-identical | 117 tests | `29babb2`,`b96e4d2`,`eae9403` |
| T1.1 / A1 | Pooled vs common-date (headline) | 125 tests | `a14482d`..`bdceadb` |
| T1.2 | Parsimony on pooled | 6 tests | `5e9ba5a` |
| quality-gate, roadmap, dashboard | tooling/infra | 13/9 tests | master `48a14bb`,`127dace`,`e6c8d0d` |

Quality process per task: TDD (RED→GREEN), verification-before-completion (captured evidence),
adversarial/focused code-review, ledger+dashboard, commit+push. `diff-cover` gate not run
(repo tooling gap, per CLAUDE.md).

## 2. Consolidated findings (5-epoch, 3-seed, horizon-5 screening; df=2 low power)

Of Track B's two ideas that distinguish it from Track A, **neither is supported**:

- **Pooling data advantage — NOT supported (A1).** ~7.6× more train data (73,026 vs 9,606) moves
  validation <1% with mixed sign (P1: common-date marginally better; P2: pooled marginally better).
  Classical HAR is on par with or ahead of the deep models on RMSE/R² under BOTH regimes. Pooling does
  not create a regime where the deep model beats HAR while common-date could not.
- **News-on-graph propagation — NULL (T0.2).** Batched G0/G1 re-baseline: G1 (message-passing on) ≈ G0
  (off) on all metrics (paired val-loss +0.00198); the cross-stock graph does not help.

The **robust positive that DOES survive** — and holds across BOTH architectures (Track A common-date
and Track B pooled):

- **News features help the error metrics (T1.2, pooled).** P2 vs P1: RMSE t=−4.81, QLIKE t=−6.94,
  3/3 seeds — significant. News recovers the RMSE the price-only deep model loses vs HAR (P1 worse
  than HAR t=7.5 → P2 ties HAR t=1.1) and wins QLIKE decisively (P2 vs HAR QLIKE t=−59.5).
- **Gate inert (T1.2), graph inert (T0.2), pooling inert (A1)** — the "clever mechanisms" add nothing.
- **Direction near-random everywhere** (~48.5% DirAcc, anti-persistence ceiling).

So the consolidated story across the whole project is a **parsimony / robustness** result: news features
give a modest, robust magnitude-forecast gain (large on QLIKE, tie-to-recover on RMSE); graph, gate,
and the pooled data regime do not add measurable value.

## 3. Decision required (paper rebuild is PAUSED)

The standing directive was to base the paper on Track B. Because Track B's two distinguishing ideas are
unsupported at this screening budget, rebuilding the paper's architecture on Track B is not obviously
correct. Three options:

1. **Strengthen A1 first (n≥5 seeds, longer epochs) before concluding.** A1's own caveat asks for this;
   df=2 is low-powered. Needs approval for >10 epochs. Resolves the headline decisively either way.
2. **Keep the paper on the Track A common-date architecture (v3).** Track B becomes a "data-regime does
   not help" ablation. Lowest deadline risk (v3 already ~12pp, parsimony story intact).
3. **Reframe as a robustness/parsimony negative-result paper spanning BOTH architectures** (recommended):
   "news features improve VN30 volatility magnitude forecasts robustly across data regimes and
   architectures; graph, gate, and pooling do not." Uses everything already run on both tracks; honest;
   turns the negatives into a genuine contribution. Then fold the paper-audit must-haves (Diebold-Mariano
   on saved predictions, GARCH baseline, corpus stats, repro statement).

**Recommendation: Option 3.** It is the most defensible use of the evidence and the most honest framing.

## 4. Not done / paused (awaiting the decision)
- Phase 4 paper rebuild (v4) — PAUSED (depends on the option chosen).
- T1.3 (A2 news-on-graph 3-way), T2.* rigor ablations, Phase 3 multi-horizon runs — NOT started; they
  presuppose continuing to invest in Track B and should follow the direction decision.
- A1 at n≥5 / >10 epochs — needs approval.

Caveats throughout: n=3 seeds, 5 epochs, horizon 5, validation-split metrics — screening signals, not
final paper claims.
