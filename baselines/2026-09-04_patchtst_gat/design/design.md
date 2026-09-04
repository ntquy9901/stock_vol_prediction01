# Design / Plan — PatchTST + GAT (2026-09-04)

## 0. SDD lifecycle + gates (per CLAUDE.md §5)

- **Constitution**: `CLAUDE.md`. This baseline obeys it (single-source config, no silent
  degradation, over/under-fit evidence, leave-one-out graph contrast, batched GPU-first trainer).
- **Specify**: `requirements/requirements.md`.
- **Clarify**: no `[NEEDS CLARIFICATION]` remained — the task fully pinned the architecture, the
  interfaces to mirror, and the constraints. One documented adaptation (see §3, channel-mixing head).
- **Plan**: this document.
- **Tasks**: §6.
- **Implement**: Test-First (tests in `code/tests/` written first; see `code_review`).
- **Validate**: acceptance 1–6 in requirements, verified on CPU (see `code_review` + final report).

### Three plan gates

- **Simplicity Gate — PASS.** No new project/app; one net + one trainer + one CLI + one config,
  all inside this baseline. PatchTST is ~120 lines of standard torch (no wrapper library).
- **Anti-Abstraction Gate — PASS.** Uses `nn.TransformerEncoder`/`nn.TransformerEncoderLayer`
  directly. Reuses the existing `WeightedGATLayer`, `train_masked_rich` helpers (`_pred_dict`,
  `_ens`, `_metrics`, `_split_metrics`, `_ens_split`, `seed_metric_stats`, `_dm_all`), the enriched
  panel/packer, folds + leakage guard, and HAR-X OLS — all imported read-only, none re-implemented.
- **Performance / Batching Gate — PASS.** The trainer mirrors `train_masked_rich`: fully batched
  `[B,N,...]` tensors, GPU-first (`cuda` when available), mask-aware pooled MSE, mini-batch loop,
  early stop; **no** per-item / batch=1 path. PatchTST processes all `B·N` nodes and all 5 channels
  in one batched forward (`z.reshape(B*N*D, seq)`), so the transformer sees `B·N·D` patch-token
  sequences per step. CPU is used **only** for the tiny smokes here because the GPU is busy; the GPU
  sweep runs the same code path with `cuda`.

## 1. Data flow

```
enriched CSVs ─ build_enriched_panel ─► EnrichedPanel (feats [T,N,5], anchors, masks)
                       │ frozen_universe (train-row screen, once)
   make_folds ────────┤
 assert_no_leakage ────┘
        per fold ► pack_fold ► MaskedRichData D (TRAIN-only scalers + TRAIN-only vol→PK graph)
                        │
       ┌────────────────┼───────────────────────────────┐
       ▼                ▼                                 ▼
   HAR / HAR-X     PatchTST (no-graph)            PatchTST + wGAT(vol→PK)   = "VolGA with PatchTST"
    (OLS)          train_patchtst(use_graph=F)     train_patchtst(use_graph=T)
       └────────────────┴───────────────────────────────┘
                        ▼
     pooled OOS preds ► metrics + metrics_per_seed + date-clustered DM
                       + train/val/test fit evidence + learning curves ► result.json
```

## 2. PatchTST encoder (from scratch, channel-independent)

Input `x [B, N, seq, D=5]`. Steps (all batched):

1. `z = x.permute(0,1,3,2).reshape(B*N*D, seq)` — one univariate series per (node, channel).
   **Channel-independent**: the same patch-embed + transformer weights process every channel and
   every node (weight sharing), and channels never attend to each other in the backbone.
2. **Patching**: `patches = z.unfold(-1, patch_len, stride)` → `[B*N*D, P, patch_len]`,
   `P = floor((seq - patch_len)/stride) + 1`.
3. **Patch embedding**: `Linear(patch_len → d_model)` → `[B*N*D, P, d_model]`; add a **learnable**
   positional embedding `[1, P, d_model]`; dropout.
4. **Transformer encoder**: `nn.TransformerEncoder(TransformerEncoderLayer(d_model, n_heads,
   dim_feedforward=ff_dim, dropout, activation="gelu", batch_first=True), num_layers=depth)` over
   the `P` patch tokens → `[B*N*D, P, d_model]`.
5. **Pool**: `flatten` (`P*d_model`, faithful to PatchTST) or `mean` over patches (`d_model`,
   parsimony lever). Default `flatten`.
6. **Per-node projection**: reshape channels back together → `[B*N, D*pool_dim]` →
   `Linear(D*pool_dim → hidden=64)` → reshape `[B, N, 64]`.

**Recency preservation**: defaults `patch_len=6, stride=4` are chosen so that, at the experiment
`lookback=22`, patches start at {0,4,8,12,16} and the last patch covers indices 16..21 — i.e. the
**most recent day (index 21) is inside the last patch**, no trailing days dropped. `P=5`. (At
`lookback=10`: `P=2`, last patch covers 4..9, again including the last day.) An `assert seq>=patch_len`
guards degenerate configs. `d_model % n_heads == 0` is asserted.

## 3. Documented adaptation (honesty)

Vanilla PatchTST keeps channels independent **through the head** (each channel forecasts itself).
Here the temporal branch must emit **one** per-node embedding that fuses all 5 features (exactly the
role the LSTM branch played: `nn.LSTM(in_dim=5, …)` mixed the 5 features into `h_lstm`). So the
**backbone is channel-independent** (shared weights, no cross-channel attention), but the **final
linear projection mixes the 5 channel embeddings** into the node embedding. This is a deliberate,
documented deviation from the pure PatchTST forecasting head; it is the minimal change needed to
slot PatchTST into the parallel-branch architecture. It is *not* labelled "vanilla PatchTST" without
this caveat.

## 4. Net = `PatchTSTRichNet` (mirror of `MaskedRichNet`)

Identical to `MaskedRichNet` except the temporal submodule:
- temporal: `PatchTSTEncoder(seq_len, in_dim=5, …, out_dim=hidden)` → `[B,N,hidden]`.
- spatial (optional): the **same** 2-hop `WeightedGATLayer` stack, reused by import; reads RAW node
  features at day t (`x[:, :, -1, :]`) — unchanged from VolGA.
- head: `Linear(hidden + gdim, hidden) → ReLU → Dropout → Linear(hidden, 1)` — unchanged.
- `forward(x, adj_b)` returns `[B,N]`.

Two variants (leave-one-out graph contrast, like VolGA): `use_graph=False` (PatchTST only) and
`use_graph=True` (PatchTST + vol→PK wGAT).

## 5. Trainer + walk-forward

- `train_patchtst(D, cfg, seed, use_graph, adj, output_param, return_splits)` — a line-for-line
  mirror of `train_masked_rich` (same scaling, masked pooled MSE, `ReduceLROnPlateau`, early stop,
  train/val learning curves, positivity floor, `zscore_floor`/`ratio_exp` output params). Only the
  instantiated net differs (`PatchTSTRichNet` with `seq_len=D.X_tr.shape[2]`).
- `run_patchtst_walkforward.py` — a mirror of `run_volga_walkforward.py`: reuses
  `build_enriched_panel` / `frozen_universe` / `pack_fold`, `make_folds` / `assert_no_leakage`,
  `_har_ols_preds` / `training_config` / `wait_for_gpu`, and all RMR metric/DM/evidence helpers.
  Learned model keys: `PatchTST`, `PatchTST_wGAT_vol2pk`. Output tree:
  `results/walkforward_patchtst/`.

## 6. Tasks (each verifiable)

1. `patchtst_config.py`: NEW tunables (patch_len, stride, d_model, n_heads, depth, ff_dim, pool);
   shared constants imported from `pipeline_config`. → verify: import + values sane.
2. `patchtst_net.py`: `PatchTSTEncoder` + `PatchTSTRichNet`. → verify: shape/patch/forward tests.
3. `run_patchtst.py`: `train_patchtst`. → verify: returns finite preds on a tiny synthetic `D`.
4. `run_patchtst_walkforward.py`: `run_fold` / `run_walkforward` / CLI. → verify: real-data CPU
   smoke (1 fold, 2 epochs, ≤8 tickers) returns finite QLIKE; leakage guard passes.
5. `tests/`: (a) encoder shape, (b) patch math, (c) full-net forward both variants, (d) leakage
   sanity on real slice, (e) real-data CPU smoke. → verify: `pytest` green on CPU.
6. `code_review/`: adversarial self-review; fix criticals. → verify: recorded.

## 7. Config / single-source-of-truth

Shared tunables (`hidden`, `heads`, `dropout`, `lr`, `weight_decay`, `grad_clip`, epochs, patience,
seeds, floors, windows, lookback, edge params) come from `submission/soict_lstm_gat/pipeline_config.py`
via `Config` / `training_config`, imported — **not** duplicated. PatchTST-specific constants (patch
size/stride, d_model, n_heads, depth, ff_dim, pool) are NEW tunables that do not exist in the shared
config; per the task they live in this baseline's own `patchtst_config.py` (the shared
`pipeline_config` must not be edited) and are exposed as CLI args with documented defaults. Editing a
PatchTST value = editing exactly one place (`patchtst_config.py`).

## 8. Known integration point — over/under-fit gate model names (FLAG for coordinator)

`scripts/quality_gate/check_overfit_evidence.py` → `overfit_check.check_result_evidence` uses the
**default** `LEARNED = ("LSTM", "LSTM_wGAT_vol2pk")`. A PatchTST result names its learned models
`PatchTST` / `PatchTST_wGAT_vol2pk`, and (because it carries `metrics_per_seed`) is recognised as a
training result. The pre-push gate would therefore look for `LSTM` keys, not find them, and **BLOCK**
the push. This baseline does **not** edit the shared gate. Mitigations (coordinator's choice):
(a) invoke the gate with `learned=("PatchTST","PatchTST_wGAT_vol2pk")`, or (b) extend the shared
`LEARNED` tuple. `run_patchtst_walkforward.run_walkforward` **self-runs** `check_result_evidence`
with the correct learned tuple and prints the verdict, so the evidence IS enforced in-code regardless.
