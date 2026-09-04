# Adversarial code review — PatchTST + GAT baseline (2026-09-04)

Self-review conducted cynically (assume bugs exist), 3 lenses: correctness/leakage, performance,
and the project-specific gates (config-hardcode, silent-degradation, named-estimator, batching).
`archive/` out of scope (N/A — no archive touched).

## Scope reviewed
`code/patchtst_config.py`, `code/patchtst_net.py`, `code/run_patchtst.py`,
`code/run_patchtst_walkforward.py`, `code/tests/*`. Shared modules are imported read-only and NOT
reviewed here (out of scope; unchanged).

## Findings

### C-1 (CRITICAL, FIXED) — empty `CUDA_VISIBLE_DEVICES` does not hide the GPU on this box
The task said set `CUDA_VISIBLE_DEVICES=` (empty). Empirically on this Windows + torch build,
`torch.cuda.is_available()` stays **True** with an empty value, so `train_patchtst`
(`cuda if is_available else cpu`) would run on the **busy** GPU. The first walk-forward smoke DID run
on the GPU before this was caught by the `test_cpu_only` guard.
**Fix:** `conftest.py` sets `CUDA_VISIBLE_DEVICES="-1"` (verified → `is_available()` False) before
importing torch, and `test_cpu_only` asserts CUDA is off. All subsequent smokes ran on CPU.
**Residual:** one ~15 s GPU co-tenancy already happened (cannot undo) — flagged to the coordinator.

### C-2 (CRITICAL, FIXED) — CPU einsum access violation (MKL/OpenMP)
On CPU, the existing `WeightedGATLayer` `einsum` triggers a "Windows fatal exception: access
violation" under multi-threaded MKL/OpenMP. This is an **environment** issue in the reused GAT layer
(the same einsum runs fine on GPU), not a defect in the new code, but it crashed the CPU smokes.
**Fix:** `conftest.py` caps native thread pools at runtime via `threadpoolctl.threadpool_limits(1)`
(plus `torch.set_num_threads(1)` and env pins). With this, all 14 tests pass CPU-only with no shell
env. **Note for the coordinator:** ad-hoc CPU runs of the CLI (outside pytest) should export
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; the GPU sweep is unaffected (GPU path avoids the crash).

### M-1 (MAJOR, ACCEPTED w/ mitigation) — over/under-fit gate model-name mismatch
`scripts/quality_gate/check_overfit_evidence.py` defaults to `LEARNED=("LSTM","LSTM_wGAT_vol2pk")`.
A PatchTST result names its learned models `PatchTST` / `PatchTST_wGAT_vol2pk` and carries
`metrics_per_seed`, so it is recognised as a training result but the default gate cannot find its
learned keys → it would **BLOCK** the coordinator's push. Shared code must not be edited from a
baseline.
**Mitigation:** `run_walkforward` self-runs `OF.check_result_evidence(res, learned=LEARNED)` with the
correct tuple and stamps `evidence_self_check` into the result, so evidence IS enforced in-code.
Flagged in design §8 + final report for the coordinator to either invoke the gate with the correct
`learned=` or extend the shared `LEARNED` tuple. **Not silently evaded.**

### M-2 (MAJOR, VERIFIED OK) — leakage / no-lookahead
The temporal branch consumes only `X[:, :, :lookback, :]` ending at anchor `t`; the GAT reads
`x[:, :, -1, :]` (day t). Targets are `pk[t+h]`. Scalers and the vol→PK graph come from
`pack_fold`/`_fit_scalers`/`_directed_vol2pk` estimated on the fold's **TRAIN** rows only
(`last_tr_row`). `assert_no_leakage` (date-space purge) is run every walk-forward. Tests
`test_leakage_free_on_real_slice` + `test_pack_fold_scalers_and_graph_are_train_only` cover it.
No new leakage surface was introduced (all splitting logic is reused unchanged).

### M-3 (MAJOR, VERIFIED OK) — performance / batching
The trainer is fully batched (`[B,N,...]`), GPU-first, mask-aware pooled MSE, mini-batch loop, no
batch=1 / per-item path. PatchTST folds all `B·N·D` (node×channel) series into the batch dim, so the
transformer processes them in one call — high arithmetic density, not a Python loop. Verified by the
CPU smoke (43 s for a full-OOS single fold, 8 nodes, 2 epochs). Meets §Performance & batching.

### m-1 (MINOR) — inherited magic numbers in the trainer
`run_patchtst.train_patchtst` copies `1e-12`, `15.0` (exp clamp), and `tiny` verbatim from the
reference `train_masked_rich` to keep numerical behaviour identical (§3 Surgical: match existing
style). These are not new tunables. All genuinely-new PatchTST knobs live in `patchtst_config.py`
(single source of truth) and are CLI-exposed. No hardcoded window/threshold in the new pipeline path.

### m-2 (MINOR) — channel-mixing head deviates from vanilla PatchTST
The final projection mixes the 5 channel embeddings into one per-node vector (the LSTM branch's
role). This is a deliberate, documented deviation (design §3); the result is labelled
`backbone="patchtst"` with the caveat recorded — it is NOT claimed to be vanilla PatchTST. Consistent
with the "named methods use the published formula / label variants clearly" rule.

### m-3 (MINOR, OK) — recency preservation
Non-padded patching with `floor` could drop the most recent day for arbitrary `patch_len/stride`.
Defaults `patch_len=6, stride=4` tile `lookback=22` so the last patch covers index 21;
`test_default_patch_stride_preserve_last_day_at_lookback22` pins this. Documented as a constraint on
the default; `assert seq>=patch_len` guards degenerate configs.

## Tests / evidence
- `pytest baselines/2026-09-04_patchtst_gat/code/tests/` → **14 passed** (CPU, conftest-forced).
- CLI `--smoke --no-gpu-wait` → full result.json schema written; `evidence_self_check` present;
  DM + fit_diagnostics populated. (Smoke numbers are wiring-only, not a result.)
- `ruff check --select F` → clean (pyflakes F-codes = the hard block). E/W are house-style WARN-only.

## Coverage
Diff-coverage gate (C0=100% / C1≥95% on changed lines) is the coordinator's pre-push responsibility.
The entry driver `main()` and the trivial glob wrapper carry `# pragma: no cover` (matching the
reference walk-forward runners); all library logic (`run_fold`, `run_walkforward`, net, encoder,
trainer, config) is exercised by the CPU tests. Not independently measured here (task scope = CPU
smokes only).

## Verdict
No open CRITICAL/MAJOR defects in the new code (C-1/C-2 fixed; M-1 mitigated + flagged; M-2/M-3
verified). Minors documented. Ready for coordinator review + GPU sweep, subject to the M-1 gate note.
