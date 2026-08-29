# Design — Graph WaveNet ablation on HNX

## Files (this baseline only; live-training-path files IMPORTED read-only)
- `code/gwn_model.py` — faithful Graph WaveNet (`NConv`, `Linear1x1`, `GCN`, `GraphWaveNet`) + paper→code map.
- `code/run_gwn_ablation.py` — panel build, `train_gwn`, `run_training`, `run_dry`, `main` (CLI).
- `test/test_gwn_graph.py` — independent adaptive-adjacency recompute + gcn/nconv shape/propagation tests.
- `test/test_gwn_runner.py` — `run_training` integration with a tiny real HNX slice + stubbed training (no GPU),
  overfit-evidence structure, DM plumbing, `main` branches.
- `test/test_gwn_smoke.py` — `@pytest.mark.smoke` one real CPU forward pass of both GWN variants.

UNIQUE basenames (`test_gwn_*`) avoid the pytest prepend-import duplicate-basename shadowing that silently
skips tests sharing a basename with a sibling baseline (`test_runner.py`, `test_smoke_forward.py`).

## Architecture — paper→code mapping (arXiv:1906.00121 + nnzhan/Graph-WaveNet `model.py`)
Input panel snapshot `x [B, N, seq, 5]` → permute to GWN layout `[B, in_dim=5, N, seq]`.

- `NConv.forward(x, A)` — graph propagation. Official: `einsum('ncvl,vw->ncwl', x, A)` for a fixed `[N,N]`
  A. We ALSO accept a batched `[B,N,N]` A (`einsum('ncvl,nvw->ncwl')`) so the per-sample node-validity mask
  can zero invalid SOURCE nodes — the same masked-union-panel convention the sibling ablations apply
  (`base * nmask`). `out[w] = Σ_v x[v]·A[v,w]`.
- `Linear1x1` — `Conv2d(c_in, c_out, 1×1)` (official `linear`).
- `GCN.forward(x, supports)` — order-K diffusion: `out=[x]; for a in supports: x1=nconv(x,a); out+=x1;
  for k in 2..order: x2=nconv(x1,a); out+=x2; x1=x2`; concat over channels → `Linear1x1` → dropout.
  `c_in = (order·support_len + 1)·c_in`. order=2, support_len=1 (adaptive only).
- **Self-adaptive adjacency** (`GraphWaveNet.adaptive_adjacency()`): `nodevec1 [N, node_dim]`,
  `nodevec2 [node_dim, N]`; `A_adp = softmax(relu(nodevec1 @ nodevec2), dim=1)`. Equals paper
  `SoftMax(ReLU(E1·E2ᵀ))` (E1=nodevec1, E2ᵀ=nodevec2). Masked per batch:
  `A_b = A_adp[None] · nmask[:, :, None]` (zero invalid source rows), fed to the GCN batched.
- **Temporal block** (`blocks=4 × layers=2`, `kernel=2`, dilation resets to 1 each block, doubles per
  layer → 1,2,1,2,…): per layer `filter=tanh(FilterConv(x))`, `gate=sigmoid(GateConv(x))`, `x=filter⊙gate`;
  `skip = SkipConv(x) + skip[..., -T:]`; then `x = GCN(x, [A_b])` (adaptive) OR `x = ResidualConv(x)`
  (no-adaptive, pure TCN); `x = x + residual[..., -T:]`; `BatchNorm2d`. Receptive field
  `1 + blocks·(1+2) = 13`; when `seq(10) < 13` the input is left-padded (official behaviour).
- **Head**: `relu(skip)` → `relu(EndConv1)` → `EndConv2` → `[B, out_dim=1, N, 1]` → squeeze → `[B, N]`
  (one horizon-1 value per node, in the standardized target space).

## Ablation = the paper's own w/o-adaptive
- `GWN_adaptive` = `adaptive=True`: the only graph is the self-adaptive adjacency (no predefined support),
  matching the paper's "forecasting with only the self-adaptive adjacency matrix" configuration.
- `GWN_no_adaptive` = `adaptive=False`: NO graph conv — each layer uses a 1×1 `ResidualConv` instead of the
  GCN (the paper's "w/o adaptive" = adaptive adj removed, identity/no graph conv). The TCN backbone,
  gating, skip/residual, channels and training loop are IDENTICAL → the adaptive graph is the only
  variable between the two.

## Training (`train_gwn`) — mirrors delivered `train_masked_rich` zscore_floor path
Same optimizer (Adam, `lr`, `weight_decay`), `ReduceLROnPlateau(factor=0.5, patience=2)`, masked-MSE loss
on the standardized target `(y − t_mean)/t_std`, grad-clip, early stop on val masked-MSE with `min_epochs`/
`patience`, best-state restore, and split/curve capture (train_curve/val_curve/best_epoch + train/val/test
predictions). Denorm: `max(pred·t_std + t_mean, 1e-2·t_mean + 1e-12)`. Only the network differs (GWN vs LSTM).

## Isolation of the variable + honesty
GWN replaces the whole temporal+spatial stack, so `GWN_adaptive vs LSTM/HAR` is a **backbone** comparison
(reported as such), while `GWN_adaptive vs GWN_no_adaptive` is the **clean in-family adaptive-graph
ablation**. Both are reported with date-clustered DM; a positive result is claimed only with DM p-values +
seed stability (per CLAUDE.md paper-writing-style, objective wording).

## Gate compliance
`design` string contains "masked" so `check_overfit_evidence` recognizes the result as a training result;
it validates `LEARNED=(LSTM, LSTM_wGAT_vol2pk)` — both present with real evidence (no-graph LSTM + stat
GAT trained in-run on the same folds). GWN variants also carry train/val/test + verdict + curves.

## Gates (§5 SDD)
- **Simplicity Gate**: one model file + one runner; reuse the delivered pipeline/DM/fit helpers unchanged.
- **Anti-Abstraction Gate**: use PyTorch conv/BN/einsum directly; no framework wrappers.
- **Performance/Batching Gate**: batched `[B,·]` tensors throughout; GPU-allowed by default
  (`GWN_FORCE_CPU=1` forces CPU). GWN uses a smaller batch (VRAM: skip/end channels) than the LSTM — a
  training-only optimization detail that does not change the evaluation basis. No batch=1 main-thread loop.
  Channel widths reduced from the paper's traffic defaults (256/512 → 64/128) for 8 GB VRAM and the smaller
  daily-vol panel; the faithful part (blocks/layers/dilation/gating/gcn/adaptive) is unchanged — documented.
  LSTM/GAT use the delivered batch scale (`--batch 16`) while GWN uses `--gwn-batch 64` (VRAM: skip/end
  channels); batch is a per-architecture optimization hyperparameter recorded in the result JSON.

## Known caveats (from 3-lens adversarial review 2026-08-29)
- **BatchNorm over zero-padded invalid nodes (MAJOR, documented):** the official `gwnet` uses `BatchNorm2d`
  whose statistics pool over `[B, N, T]`; the masked panel zero-fills invalid nodes, so BN normalizes valid
  nodes using batch stats that include the zeros. Kept for fidelity (not swapped for a mask-aware norm). It
  is COMMON-MODE across both GWN variants → cancels in the headline in-family adaptive ablation
  (GWN_adaptive vs GWN_no_adaptive); it does not cancel vs LSTM/HAR (no BN), so that comparison is framed as
  a backbone comparison and `valid_node_fraction_test` is reported to bound it.
- Minor (accepted, inherited from the delivered path): per-epoch full-train re-inference rebuilds train
  batches from numpy for the learning curve (perf, not correctness); the `# pragma: no branch` best-state
  guard over-asserts under a NaN epoch-0 loss (guarded in practice by the positivity floors).
