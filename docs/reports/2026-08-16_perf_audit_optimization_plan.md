# Track A training/eval performance audit and optimization plan

Date: 2026-08-16
Scope: performance audit only. No production code changed, nothing committed. `archive/` out of scope.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM), 32 GB host RAM, Windows (WDDM).
GPU venv: `.venv_gpu_encode\Scripts\python.exe` (torch 2.6.0+cu124, CUDA available).

## 1. Summary

- Batch size is confirmed **1 graph snapshot per gradient step**. The training loop iterates
  snapshots one at a time in Python and calls `.to(device)` on every tensor of every snapshot on
  every step.
- The pipeline is **overhead / occupancy bound, not compute bound**. Measured GPU utilization
  during the batch=1 path reads ~88-91%, but this is misleading: throughput is only ~142
  graphs/s while VRAM use is 366 MiB of 8188 MiB. Batching the identical model to B=128 raises
  throughput to ~2680 graphs/s (~19x) on the same GPU. The high util% at B=1 reflects a stream of
  tiny kernels keeping the counter busy at low occupancy, not useful saturation.
- Top 3 optimizations by expected impact:
  1. **Batch B snapshots per step** (`train_resume.py`): measured ~17-20x throughput. The model
     already accepts a batch dimension, so this is the single biggest win.
  2. **Keep snapshot tensors resident on GPU / use pinned non-blocking transfers**: removes the
     per-step CPU->GPU churn that dominates the B=1 path.
  3. **Remove per-step CPU syncs** (`torch.isfinite(loss)` in train, `.item()` in val) and wrap
     evaluation in batched no-grad passes.
- **Does batching unlock a bigger A100 speedup?** Yes, and it is the prerequisite. At B=1 an A100
  would not be faster (it is launch/latency limited, and the 33-node ops never fill the SMs). Once
  batched and compute-bound, an A100 (80 GB, ~7x memory bandwidth, far more SMs) both removes the
  8 GB VRAM cliff seen here at B=256 and speeds up the batched GEMM/LSTM/GAT kernels; it also lets
  all seeds x horizons run concurrently.

Report path: `docs/reports/2026-08-16_perf_audit_optimization_plan.md`.

## 2. Measured evidence

Benchmark harness: `temp/perf_bench.py` (synthetic snapshots at the real shapes N=33, seq=22,
price_dim=5, news_dim=146, num_tickers=33; the real `TrackAGatModel` with Adam + grad-clip). This
mirrors the standard 33-ticker Track A graph exactly. Contention: GPU was idle (0%, 0 MiB) before
the run; no other compute apps present.

Throughput vs batch size (same model, same GPU):

| Batch B | Path                | graphs/s | Speedup vs B=1 |
|--------:|---------------------|---------:|---------------:|
| 1       | CPU->GPU each step  | ~134     | 1.0x           |
| 32      | tensors on GPU      | ~2331    | ~17x           |
| 128     | tensors on GPU      | ~2680    | ~20x           |
| 256     | tensors on GPU      | ~149     | ~1.1x (VRAM cliff) |

B=256 collapses because the dense GAT attention tensor `[B,N,N,H]` plus news `[B,N,seq,146]`
exceeds the 8 GB laptop VRAM and spills to host memory under WDDM. The usable sweet spot on this
GPU is roughly B=64-128.

GPU utilization during a sustained B=1 run (4000 steps, 28.2 s, ~142 graphs/s): 86-91% reported,
366 MiB VRAM. High util% with 19x idle throughput headroom and <5% VRAM used = overhead-bound.

Workload size (from a recent basis log, `results/trackA_ablation_h1_seed42_2026-08-16_vn100quick.progress`):
`snapshots=5799` total across the three splits (~4060 train snapshots per epoch at a 70/15/15
split). Each run trains 5 ablation rungs x up to 4 horizons, each up to ~9-15 epochs, so snapshot
count per epoch is the dominant multiplier. The ~6500/epoch figure in the task is the same order
of magnitude; the measured value depends on ticker universe and horizon.

## 3. Bottleneck findings (code locations)

Hot path files under `baselines/2026-08-15_trackA_gat_edge/code/`.

- **Batch = 1, Python loop** — `train_resume.py:63-71` (`train_with_resume`) iterates
  `rng.permutation(len(train_snaps))` and processes one snapshot per `optimizer.step()`.
  `_forward_snap` (`train_resume.py:12-17`) does `snap[...].unsqueeze(0).to(device)` for all five
  tensors on every call, so every step pays CPU->GPU transfer + kernel-launch overhead for a tiny
  `[1,33,...]` workload.
- **Per-step CPU<->GPU transfers** — `train_resume.py:13-16`, `train_resume.py:66`
  (`s["target"].to(device)`), and the eval mirror `run_trackA.py:136-140`. Snapshots live on CPU
  as numpy-backed tensors (`run_trackA.py:64-75`, `run_retrain_trainval.py:72-83`) and are moved
  each step; nothing is cached on device. No `pin_memory`, no `non_blocking=True`.
- **Per-step synchronization points** — `train_resume.py:67` `if not torch.isfinite(loss)` forces
  a device->host sync every training step (Python reads the bool); `train_resume.py:27`
  `.item()` in `_val_loss`; `run_trackA.py:140` `.cpu().numpy()` per snapshot in eval.
- **No DataLoader / workers / prefetch** — snapshot assembly and iteration are single-threaded on
  the main thread; there is no overlap of host-side batch prep with GPU compute.
- **No AMP / mixed precision** anywhere in the model or loop.
- **GAT dense O(N^2) attention** — `gat.py:26-31` materializes `e` of shape `[B,N,N,H]` and does a
  `masked_fill` + full softmax over all N sources even for the sparse directed vol->PK Top-5
  adjacency (only ~520 real edges over N nodes). This is fine at N=33 but is the term that blows up
  VRAM as B grows and is the first thing to sparsify if N (ticker universe) grows.
- **Model already batches** (important, lowers batching effort) — `model.py:44-50` `_encode_seq`
  reshapes `[B,N,seq,d] -> [B*N,seq,d]` through the LSTM, and `gat.py` operates on `[B,N,N]`
  adjacency with a batched einsum. `forward` therefore already accepts B>1; only the training/eval
  loops force B=1 via `unsqueeze(0)`.

Snapshot construction (not per-step hot, but relevant to residency and RAM): masked snapshots are
built once as CPU float32 arrays in `data.py:build_masked_graph_manifest` (`data.py:982-1009`) and
wrapped to torch in `run_trackA.py:60-75`. Total resident size is modest (see item 5).

## 4. Prioritized optimization plan

Effect estimates are for this 8 GB laptop GPU unless noted. Effort: S<half day, M~1-2 days,
L>2 days.

| # | Optimization | Where (file:func) | Why it helps | Expected gain | Effort | Risk |
|---|--------------|-------------------|--------------|---------------|--------|------|
| 1 | **Batch B snapshots/step** | `train_resume.py:train_with_resume`, `_forward_snap`, `_val_loss` | Amortizes kernel-launch + transfer overhead over B graphs; fills SMs. Model already supports `[B,N,...]` (`model.py:_encode_seq`, `gat.py`). Stack B snapshots into `[B,N,seq,P]`, `[B,N,seq,news]`, `[B,N,N]` adjacency, `[B,N]` targets/presence. | ~17-20x throughput (measured 134->2680 graphs/s at B=128) | M | Presence-masked loss must be added (see note below) or results shift; keep B<=~128 to avoid the VRAM cliff |
| 2 | **GPU-resident snapshots / pinned non-blocking transfers** | `run_trackA.py:build_trackA_basis` (build tensors), `train_resume.py` (consume) | Removes per-step CPU->GPU churn that dominates B=1. Preload all snapshot tensors to `device` once (fits, see item 5) or `.pin_memory()` + `.to(device, non_blocking=True)`. | Large at B=1; folded into item 1 once batched. Removes the "CPU->GPU each step" penalty (B=1 CPU-snap path measured ~134 vs on-GPU ~2331 at B=32) | S | Preloading assumes VRAM budget; fall back to pinned host buffers if OOM |
| 3 | **Drop per-step syncs** | `train_resume.py:67` (`torch.isfinite`), `:27` (`.item()`), `run_trackA.py:140` (`.cpu().numpy()`) | Each is a device->host stall serializing the stream. Check finiteness every K steps or accumulate on-device; batch eval readout into one `.cpu()` per batch. | Removes ~one sync/step; a few % once batched, more at B=1 | S | Non-finite loss detected slightly later (bounded by K) |
| 4 | **DataLoader with `num_workers>0` + prefetch** | new `Dataset` wrapping `basis["snaps"]`, used in `train_with_resume` | Overlaps batch stacking/collation with GPU compute. On Windows must use spawn and guard entrypoints with `if __name__=='__main__'` (the runners already do). | Marginal once items 1-2 make data GPU-resident; useful only if collation stays on CPU or N grows | M | Windows spawn re-imports; worker startup cost; low payoff if tensors already on GPU |
| 5 | **Exploit 32 GB RAM / VRAM headroom** | `run_trackA.py:build_trackA_basis`, `run_retrain_trainval.py:_build_deep_basis` | At B=1 only 366 MiB VRAM is used. Resident cost of all 5799 snapshots is small: price `5799*33*22*5*4B ~= 84 MB`, news `~2.5 GB` at 146 dims (largest term), adjacency `5799*33*33*4B ~= 25 MB`. News dominates; the full set may not fit 8 GB alongside activations, so preload price/adjacency/targets to GPU and keep news pinned on host, or shard by split. All fits comfortably in 32 GB host RAM. | Enables items 1-2; allows larger B | S/M | News tensor is the VRAM constraint on 8 GB; needs split-wise residency or on 80 GB fits entirely |
| 6 | **AMP / mixed precision** | `train_with_resume` (autocast + GradScaler) | fp16/bf16 GEMMs on tensor cores; halves activation memory so B can grow. | ~1.2-1.6x on top of batching; more headroom for B | S | Softplus positivity floor + QLIKE are sensitive; validate metrics parity vs fp32 |
| 7 | **Sparsify GAT for large N** | `gat.py:forward` | Dense `[B,N,N,H]` is wasteful for Top-5 directed edges; a gather/scatter over the ~520 edges avoids the N^2 softmax. Only matters if the ticker universe grows (e.g. VN100). | Neutral at N=33; large at N>=100 and removes the B=256 VRAM cliff | L | Correctness of masked softmax over gathered edges; more code |

Note on the presence-masked loss (item 1 dependency): the current B=1 training loss
`torch.mean((pred - target)**2)` at `train_resume.py:66` runs over all N nodes, including absent
nodes whose `y_norm` is 0 (`data.py:991`, `run_trackA.py:70`). Evaluation already skips absent
nodes via `presence_mask` (`run_trackA.py:145-147`). A batched implementation should apply the
presence mask in the loss (sum over present nodes / count of present nodes). This will change
training behavior slightly versus today (absent nodes currently contribute a regress-to-zero
signal); the change is a correctness improvement but must be validated for metric parity before it
replaces the recorded results, per the project's audit-record rules.

## 5. Re-answering the A100 question

Batching is the gate. Concretely:

- At B=1 the workload is launch/latency-bound; an A100 would not beat the 4060 here (its
  advantage is throughput per kernel and memory bandwidth, neither of which is exercised by
  `[1,33,...]` tensors) and could even be marginally slower per step.
- Once item 1 lands, the pipeline becomes compute/bandwidth-bound. Then an A100 (80 GB, ~2 TB/s
  vs the 4060 laptop's ~256 GB/s, and far more SMs) helps in three compounding ways:
  1. **No VRAM cliff**: the B=256 collapse observed here (dense GAT + 146-dim news exceeding 8 GB)
     disappears; batches of thousands of snapshots and the entire snapshot set become resident.
  2. **Faster batched kernels**: the LSTM `[B*N,seq,d]` GEMMs and GAT attention scale with memory
     bandwidth and SM count, where the A100 has a large multiple over the laptop 4060.
  3. **Concurrency**: seeds x horizons (already separate processes) can run simultaneously within
     80 GB, collapsing wall-clock for the full ablation suite.
- Recommended cloud sequencing: implement items 1-3 first (validate on the 4060 at B<=128), then
  move to A100 and raise B and concurrency. Porting to A100 before batching would waste the GPU.

## 6. Notes

- Benchmark kept short (a few thousand steps, seconds each) on an otherwise idle GPU; numbers are
  representative of the standard 33-ticker Track A shape. Larger universes (VN100) increase N and
  shift the GAT term, making item 7 more valuable.
- No production code was modified. The only artifact created is the throwaway benchmark
  `temp/perf_bench.py`.
