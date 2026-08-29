# Summary of update — Graph WaveNet ablation on HNX (2026-08-29)

## What changed
Added a new baseline `baselines/2026-08-29_graphwavenet_ablation/` that reproduces Graph WaveNet (Wu et al.,
IJCAI 2019, arXiv:1906.00121) and runs it as an ablation on HNX daily volatility (h1) to test whether a
TCN backbone + self-adaptive adjacency changes the "graph does not help HNX" picture from the prior 4-edge
sweep. Extends that sweep with a fifth, architecturally-distinct probe.

## Files (path → purpose)
- `baselines/2026-08-29_graphwavenet_ablation/requirements/requirements.md` — spec, variants, go/no-go.
- `.../design/design.md` — architecture + paper→code mapping, gates, review caveats.
- `.../code/gwn_model.py` — faithful Graph WaveNet (`NConv`, `Linear1x1`, `GCN`, `GraphWaveNet`) + w/o-graph ablation.
- `.../code/run_gwn_ablation.py` — panel build, `train_gwn` (mirrors delivered `train_masked_rich`), `run_training`, DM, dry/CLI.
- `.../test/test_gwn_graph.py|test_gwn_train.py|test_gwn_runner.py|test_gwn_smoke.py` — unique basenames (no duplicate-basename shadowing).
- `.../code_review/code_review_2026-08-29.md` — 3-lens adversarial review + resolutions.
- `results/graphwavenet_ablation/graphwavenet_ablation_hnx_h1.json` — metrics + per-seed + fit evidence + DM.
- `docs/reports/2026-08-29_graphwavenet.md` — metric table, DM, fit verdicts, paper→code mapping, conclusion.

## Result (HNX h1, 3 seeds, 60,028 test obs)
QLIKE (ensemble): LSTM 1.8063 < LSTM+wGAT 1.8091 < GWN_adaptive 1.8128 < GWN_no_adaptive 1.8139 < HAR 1.8284
< HAR-X 1.8615. Date-clustered DM: the self-adaptive graph adds no robust QLIKE gain within GWN
(GWN_adaptive vs GWN_no_adaptive p=0.355, per-seed mean sign-reverses); the GWN backbone forecasts slightly
worse than the no-graph LSTM (p=0.0044 / 6.9e-5 in the LSTM's favour); all deep models beat HAR/HAR-X
(p≤0.004). Conclusion: a fifth robustness result that the graph mechanism adds no OOS value on this cell,
and the TCN backbone does not beat the LSTM. Reported straight (no inflation).

## Tests + coverage
25 tests pass under the GPU venv. Diff-coverage on the two changed code modules: **C0 line 100%, C1 branch
100%** (`--cov-branch`, term-missing verified locally). Independent adaptive-adjacency formula test present
(recomputes `softmax(relu(E1@E2), dim=1)` in numpy, not reusing the module). `run_training` covered as a
stubbed integration test with a `train<val<test` split-wiring assertion. Smoke test asserts the result JSON
passes `overfit_check.check_result_evidence`.

## Code review (3-lens adversarial)
No CRITICAL. One MAJOR (BatchNorm pools stats over zero-padded invalid nodes) — resolved by documentation +
reporting `valid_node_fraction_test=0.817`; it is common-mode so it cancels in the headline in-family
ablation. Minors (paper-wording overclaim, per-epoch re-inference perf, cross-model batch size, NaN-guard
pragma) resolved by rewording / accepted as inherited-from-delivered-path. Two test gaps closed
(partial-invalid nmask forward; adaptive-vs-no-adaptive differ + param presence). Details in
`code_review/code_review_2026-08-29.md`.

## Performance
Training hot loop is fully batched (`[B, …]` tensors), data preloaded to device, graph conv batched over B,
inference batched — no batch=1 / per-step host-device sync anti-pattern. GPU run peaked at ~0.8 GB VRAM
(`--batch 16` for LSTM/GAT to avoid the batch-512 GAT VRAM thrash observed initially; `--gwn-batch 64`).
Over/under-fit gate passes: all six models `ok` (no over/under-fit).

## Data-quality gate
N/A (no data change) — this baseline imports the delivered pipeline and data read-only; no `data/` files
touched, no crawl/append.

## DoD checklist
- [x] 5 baseline sub-folders present with spec/design/code/review/test.
- [x] Faithful named-method reproduction + independent formula test + paper→code mapping documented.
- [x] Tests pass; C0=100% / C1=100% on changed lines; unique test basenames.
- [x] Over/under-fit evidence in result.json; overfit gate passes.
- [x] 3-lens adversarial review done; critical/major addressed.
- [x] Report with metric table + DM + fit verdicts + honest conclusion.
- [x] Push after gate passes (pre-push hook enforces locally).
