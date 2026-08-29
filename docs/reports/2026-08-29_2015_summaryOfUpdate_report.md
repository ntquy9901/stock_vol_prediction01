# Summary of update — DY (2014) volatility-spillover graph edge (HNX ablation)

## What changed
New isolated baseline `baselines/2026-08-29_dy_spillover_ablation/` implementing the Diebold & Yilmaz
(2014) directed variance-decomposition connectedness network as a FIXED (frozen-on-train) GAT graph edge,
plus a checkpointed HNX-h1 ablation comparing it against the no-graph LSTM and the shipped statistical
vol->PK edge under the identical MaskedRichNet / HAR-X / masked-panel / QLIKE pipeline (only the EDGE
differs). No live-training-path file was edited (all imported read-only).

## Files (path -> purpose)
- `code/dy_connectedness.py` -> DY 2014 generalized-FEVD builder: elastic-net VAR(1) (Demirer et al. 2018
  high-dim fix) -> VMA(inf) -> generalized FEVD (Pesaran-Shin) -> row-normalise -> Top-K + self-loop.
- `code/run_dy_ablation.py` -> panel build + DY-adjacency + dry/CPU-smoke + one-shot comparison.
- `code/run_dy_incremental.py` -> checkpointed, resumable driver; writes train/val/test metrics +
  fit verdict + learning curves per variant (CLAUDE.md over/under-fit-evidence mandate).
- `test/*.py` -> 23 tests (independent formula recompute + real-HNX-slice smoke + resume/evidence checks).
- `requirements/`, `design/`, `code_review/` -> SDD docs + adversarial self-review.
- `docs/reports/2026-08-29_dy_spillover.md` -> full results report.
- `results/dy_spillover_ablation/dy_ablation_hnx_h1.json` + `ckpt/` -> result JSON + per-(variant,seed)
  checkpoints + full-N=154 connectedness stats.

## Result (HNX h1, 10 epochs, seeds {42,123,2026}, GPU; all fit verdicts "ok")
| model | QLIKE (seed mean+/-std) | RMSE | R2 |
|---|---|---|---|
| dy_GAT | 2.203 +/- 0.424 | 0.001181 | 0.220 |
| stat_GAT_vol2pk | 1.832 +/- 0.007 | 0.001180 | 0.221 |
| no_graph_LSTM | 1.835 +/- 0.008 | 0.001179 | 0.222 |
| sector_GAT (context) | 1.818 | 0.001175 | 0.228 |

DM (date-clustered): dy vs no_graph QLIKE p<0.001 favors no_graph; dy vs stat QLIKE p<0.001 favors stat;
stat vs no_graph QLIKE p=0.068 (ns). On squared-error (MSE) all edges are indistinguishable (DM-SE p>0.18).

**Verdict:** the DY spillover edge does NOT help — it significantly WORSENS QLIKE vs both no-graph and the
statistical edge and is seed-unstable (one seed's QLIKE=2.80). Fourth edge (after statistical, sector,
MTGNN-learned) to fail to beat the no-graph LSTM on HNX h1: the graph/spillover structure adds no OOS
value; a parsimonious no-graph LSTM (and HAR) is preferred. Connectedness matrix itself is well-formed
(total connectedness 44.7%, directed, rows sum to 1.0), so this is a clean negative robustness finding.

## Tests + coverage
`python -m pytest baselines/2026-08-29_dy_spillover_ablation/test/` -> 23 passed. Diff-scope coverage on
the three new modules: **C0 line = 100%, C1 branch = 100%** (import-time GPU guard + `main()` carry
`# pragma: no cover`). Connectedness tests also pass under the gpu venv (pandas 2.3). ruff-F clean.

## Code review
`code_review/code_review_2026-08-29.md`: no critical/major bug outstanding; M1 (train-window boundary)
resolved as leakage-safe (train-only, frozen); minors are documented modeling choices.

## Performance
DY-matrix build is a one-off CPU/VAR step. Training reuses the delivered batched `[B,N,...]` pipeline
(batched block adjacency, mask-aware loss) on GPU — no batch=1 anti-pattern. ~25 min for 9 trainings.

## Data-quality gate
N/A (no data change — reuses the existing screened HNX processed panel; no raw ingestion).

## Risks / follow-ups
- QLIKE seed-instability for dy_GAT (one seed 2.80) — the dense-derived edge occasionally destabilises the
  positivity-sensitive QLIKE; MSE/RMSE unaffected. Not pursued further (edge is a confirmed no-lift).
- Not pushed per coordinator (they consolidate all four edge results to master); committed locally.

## DoD checklist
- [x] Code satisfies request (DY 2014 edge built exactly; harness wired to the shared pipeline).
- [x] Tests + coverage (23 tests; C0=100/C1=100 on changed lines); ruff-F clean.
- [x] Code review run + findings addressed.
- [x] Over/under-fit evidence in result.json (train/val/test + verdict + curves; all "ok").
- [x] Summary report (this file) + results report filled.
- [x] Commit locally (push deferred to coordinator per their instruction).
