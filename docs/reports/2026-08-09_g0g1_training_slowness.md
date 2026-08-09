# G0/G1 Masked-Graph Training Slowness — Diagnosis and Speedup Recommendations

Date: 2026-08-09
Scope: Read-only diagnosis of why a masked-path G0/G1 graph ablation run takes ~15 min per seed at 15 epochs. No code changed, no training launched. Analysis is on committed code at the idle detached worktree `.worktrees/graph-viz` (commit 2a0bed6) — `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/{run_pilot.py, models.py, data.py}` — plus the existing masked result artifacts in that worktree's `results/`.

## 0. Inputs and measured constants (from artifacts, not assumptions)

Source: `results/pooled_news_gnn_masked_knn8_seed42_2026-08-08_230837/h5/**` and `..._g0g1_2026-08-08_212959_seed42/**` (graph-viz worktree).

- Graph mode: `masked`; horizon 5; device selected = `cuda` (RTX 4060 Laptop, torch 2.6.0+cu124).
- Nodes per snapshot: **33** (padded to full ticker vocabulary; `data.py:958,982`).
- Sequence window: **22** steps (`data.py:330,386`), price + news features.
- Snapshots: total 6470, distinct dates 4941, split = train **4523** / val **1237** / test 710.
- Present nodes per snapshot ≈ **15.75 overall** (`edge_density.present_row_count` 101908 / 6470), ≈ 11.66 on the val split (14418 present val nodes / 1237). So ~**48%** of the 33 padded node slots are actually present; the rest are zero-padded absent tickers.
- Runtime metadata stored in `results.json` records only device/library versions — **no wall-clock seconds are persisted**, so stage timings below are derived from forward/backward-pass counts, not logged durations.
- `nvidia-smi` sampled once during this investigation: GPU util **0%**, 0 MiB in use (the DM-confirmation agent was not in a GPU-compute phase at that instant; single sample, weak evidence, but consistent with a host-bound rather than compute-saturated workload).

## 1. Pipeline stages of a `--phase graph` run and the dominant cost

`run_graph_screening` (`run_pilot.py:364`) executes, per invocation (i.e. per seed, since each seed is a separate process):

1. Load + split price data, select tickers, fit graph preprocessors, build pooled manifest, attach news (`run_pilot.py:373-396`). Seed-independent.
2. Build the masked graph manifest — `build_masked_graph_manifest` (`data.py:939`) groups 6470 snapshots and computes a per-snapshot correlation/knn adjacency (`_masked_correlation_adjacency`, `data.py:1007`). Seed-independent.
3. `build_graph_bound_p3_warm_start` (`run_pilot.py:122`) — 1 epoch over pooled train samples at/before the graph boundary. **Seed-dependent** (`_seed_graph_device`, Adam init).
4. `build_graph_safe_p3_checkpoint` (`run_pilot.py:166`) — loads the warm-start, 1 more epoch over the same samples. **Seed-dependent.**
5. Train G0 then G1 — `_run_one_graph_model` (`run_pilot.py:537`), `epochs` each (15 in the slow run), with per-epoch validation.

### Forward/backward accounting (the evidence)

Let a "node-forward" = one 22-step sequence through the frozen 2-layer price LSTM + 2-layer news LSTM (hidden 64 each), which is the unit of encoder work.

Stage 5 (graph training), per model per epoch:
- train: 4523 snapshots × 33 padded nodes = **149,259** node-forwards (142 optimizer steps at `--graph-train-batch-size 32`)
- val: 1237 × 33 = **40,821** node-forwards (39 batches at `--graph-batch-size 32`)
- = **190,080** node-forwards / model / epoch

At 15 epochs × 2 models (G0+G1): **5,702,400** node-forwards.

Stages 3+4 (the two P3 builders), once per seed:
- allowed pooled train samples ≈ present train nodes ≈ 4523 × 15.75 ≈ **71k** (matches the ~73k figure in the brief)
- warm-start 1 epoch + graph-safe 1 epoch = ~**142k** single-node sequence forward+backward, batched at `--batch-size 256` (~570 optimizer steps total)

Ratio: graph training issues **~40×** more encoder forward passes than the two builders (5.70M vs 0.142M). The builders do a full backward through both LSTMs (~2–3× a forward), while graph training backprops only through the tiny message-passing + head (encoders are frozen). Even after that correction, **stage 5 (G0+G1 training) is the dominant wall-clock cost by roughly an order of magnitude at 15 epochs.** At 5 epochs (the `_212959_` run) training is 1.90M forwards, still ~13× the builders — training dominates at any epoch count in the experimentation range.

## 2. Is any batch-1 loop still present post-T0.2? — No active one.

T0.2's two claims verify true at commit 2a0bed6:

- **Both P3 builders are mini-batched.** `_pooled_training_batches` (`run_pilot.py:91`) stacks `batch_size` samples per `optimizer.step`; called with `train_batch_size=args.batch_size` (default 256) from both `build_graph_bound_p3_warm_start` (`run_pilot.py:147`) and `build_graph_safe_p3_checkpoint` (`run_pilot.py:218`). No per-sample `optimizer.step`.
- **The graph train/val loops are mini-batched.** Training steps over `train[start:start+train_batch_size]` with one `optimizer.step` per batch (`run_pilot.py:570-578`); validation likewise (`run_pilot.py:476`, `:593`). `_graph_prediction_batch` (`run_pilot.py:653`) runs the whole snapshot batch through the model in one call via `np.stack` + a single `model(...)`.

The only per-snapshot Python loops that remain are **dead on the masked and intersection paths**:
- `_graph_prediction_batch` has a fallback (`run_pilot.py:666-674`) that loops snapshot-by-snapshot **only when snapshot tensor shapes differ**. In masked mode every snapshot is padded to 33 nodes (`data.py:982-985`), and in intersection mode the node set is fixed, so all shapes match and the batched branch (`run_pilot.py:675+`) is always taken. The single-graph `_graph_prediction` helper (`run_pilot.py:635`) and the `adjacency.unsqueeze(0)`/`base.unsqueeze(0)` single-graph branch in `GraphAblationModel.forward` (`models.py:303-304`) are on that same unused fallback path — not batch-1 training loops.

There is, however, a **per-sample Python loop inside each batch** in `PooledPriceNewsLSTM._encode_news` (`models.py:79`): `sequences = [x_news[index, news_mask[index]] for index in valid_indices]` followed by `pad_sequence` + `pack_padded_sequence`. This is not a batch-1 optimizer loop, but it is host-side per-row work executed on up to 33×`batch` rows every batch of every epoch — a launch-bound cost (see §3).

## 3. Compute-bound vs launch-bound

Indicators point to **launch/host-bound**, not GPU-compute-saturated:

- The frozen encoders run under `torch.no_grad()` on batches of 32×33 = 1056 sequences, but each batch first does host-side `np.stack` of `[32,33,22,feat]` arrays + `.copy()` + host→device transfer (`run_pilot.py:682-689`), then the Python list-comprehension news packing (`models.py:79`). Tensors are re-created and re-transferred **from numpy every batch, every epoch** — nothing is pre-stacked or cached on device.
- `--graph-train-batch-size` and `--graph-batch-size` default to **32** (`run_pilot.py:915-918`); `num_workers=0` on the P3 loaders (`run_pilot.py:858`, Windows). Small batches + single-process collate maximize per-batch Python/dispatch overhead relative to GPU math.
- Observed GPU util 0% at sample time (single sample; weak but consistent).

So the slowness is a combination of **(a) redundant recomputation** — the frozen encoders (dropout=0, `requires_grad_(False)`, `models.py:176-180`) produce **bit-identical** `base = cat(price_hidden, gated_news)` embeddings on every epoch and identical between G0 and G1, yet are recomputed 15×/model and again for the second model — and **(b) host-bound dispatch** from per-batch numpy stacking, H2D copies, and the Python news-packing loop, all repeated for that redundant work.

Quantifying the redundancy: the unique frozen-encoder work actually needed is one pass over train+val snapshots = (4523+1237)×33 = **190,080** node-forwards. The 15-epoch × 2-model run performs 5,702,400. **~96.7% of the encoder forward work (5.51M of 5.70M passes) is redundant recomputation of frozen, unchanging outputs.**

## 4. The masked 3.8× multiplier, quantified

- Distinct dates: masked 4941 vs intersection ~1296 = **3.8×** (matches the brief).
- Train snapshots: masked 4523 vs intersection ~900 (≈ intersection train from ~1296 dates) = **~5.0×**.
- Per-epoch encoder forwards: masked 149,259 (4523×33 padded) vs intersection 29,700 (900×33, all present) = **~5.0×**.
- Of the masked per-epoch cost, only ~48% (≈71k) is on present nodes; the other ~52% is spent encoding **zero-padded absent tickers** whose outputs are then zeroed in message passing (`models.py:146`). So masked's 5× per-epoch encoder cost decomposes as **~2.4× genuinely more data** (71k present vs 29.7k) **× ~2.1× wasted work on absent padded nodes**.

Builders vs training under masked: the builders run once over ~71k present train samples (~142k forward+backward total for the two of them); training runs 5.70M encoder forwards over 15 epochs × 2 models. **The builders are not the bottleneck — the 15-epoch G0+G1 training loop is**, and the masked path amplifies exactly that dominant stage 5× relative to intersection while adding only a fixed one-time cost to the seed-independent manifest build.

Convergence check (supports §5 rec 7): masked knn8 seed42 val loss is essentially flat after ~epoch 5 — G0 val 0.8399 (ep5) → 0.8392 (ep15), a 0.0007 gain over 10 extra epochs; G1 0.8406 → 0.8367. The last ~8–10 epochs buy almost nothing.

## 5. Ranked speedup recommendations

| # | Change | Expected impact | Effort | Leakage / provenance risk |
|---|--------|-----------------|--------|---------------------------|
| 1 | **Cache the frozen-encoder `base` embeddings once per seed and reuse across all epochs and across G0+G1.** Compute `cat(price_hidden, gated_news)` for every snapshot once (over present nodes), store `[snapshots, nodes, hidden]`; the training loop then runs only message-passing + head on the cache. | Removes 96.7% of the dominant stage's forward work. Roughly **~15× faster training at 15 epochs** from de-duplicating epochs, plus **~2×** from sharing the cache between G0 and G1. Largest single win. | Moderate | **None.** Encoders are already frozen + dropout=0, so cached outputs are bit-identical to recomputed ones; the graph-safe P3 provenance gate (`GraphAblationModel.from_p3_checkpoint`) still governs which weights produce the cache. |
| 2 | **Encode only present nodes** (gather present indices, encode, scatter) instead of running the LSTM on zero-padded absent nodes. | ~**2.1×** on encoder cost (present ≈48% of padded). Subsumed by #1 if #1 caches over present nodes only. | Low–Moderate | None (absent outputs are already zeroed downstream). |
| 3 | **Pre-stack snapshot tensors once and keep them resident** (device or pinned host) instead of `np.stack`+`.copy()`+H2D every batch every epoch (`run_pilot.py:682-689`). | Removes the repeated host-side stacking + H2D that dominates the launch-bound portion; complements #1. If #1 is done the resident object is the small `base` cache, so this largely folds into #1. | Moderate | None; watch device memory if stacking raw windows (4523×33×22×feat) — prefer caching `base` (far smaller) per #1. |
| 4 | **Reuse seed-independent artifacts across seeds.** The data load, pooled manifest, news attachment, and `build_masked_graph_manifest` (incl. per-snapshot adjacency over 6470 snapshots) are seed-independent but rebuilt in every per-seed process; serialize once, load per seed. Keep the two P3 builders per-seed (they are seed-dependent). | Removes the full fixed setup cost from 2 of every 3 seed runs (manifest build is non-trivial CPU). | Low–Moderate | Low — must preserve the manifest content hashes / provenance guards on load. |
| 5 | **Raise `--graph-train-batch-size` / `--graph-batch-size`** from 32 (e.g. 128–256). Amortizes per-batch Python news-packing + dispatch over more graphs; the 4060 is far from saturated at 1056 sequences/batch. | Moderate on the launch-bound overhead. | Trivial (flag only) | Low — larger batch changes gradient averaging (the approved R10 SGD re-baseline). G0 and G1 change identically and the result is a null, but note the optimization trajectory differs from the 32-batch runs; keep batch fixed within a comparison set. |
| 6 | **Vectorize `_encode_news`** (`models.py:79`) to avoid the per-row Python list-comprehension + pad/pack each batch. | Moderate launch-bound win. Subsumed by #1 (news encoding becomes part of the one-time cache); standalone value only if #1 is not adopted. | Moderate | None if numerical parity is verified. |
| 7 | **Cut epochs to ~7** given convergence by epoch ~5 (val loss flat, §4). | ~**2×** on the dominant stage (15→7). | Trivial | None — confirm plateau from the existing `learning_curve.png` / `validation_losses` first (cheap). |
| 8 | Builders on CPU vs GPU — **not recommended.** Builders are batched (256) and do real LSTM backprop; they are not the bottleneck. Since the workload is host/launch-bound, batch size (#5) and caching (#1) matter far more than device choice. | Negligible/negative | — | — |

### Recommended order
Do **#1** first (it removes the redundant-recompute root cause and dwarfs everything else), fold **#2/#3/#6** into its implementation, then **#7** (free 2×) and **#4** (per-seed setup), with **#5** as a trivial complementary flag. #1 + #7 alone would be expected to take the 15-epoch, 2-model masked run from ~15 min/seed to low single-digit minutes, dominated afterward by the one-time seed-independent manifest build (addressable by #4).

## Provenance note
All figures are derived from committed code at 2a0bed6 and the persisted result artifacts listed in §0. Wall-clock per-stage seconds are not logged in the current runs, so §1–§3 stage attribution is by forward/backward-pass counting; adding a lightweight per-stage `time.perf_counter` log would let a future run confirm the predicted breakdown empirically.
