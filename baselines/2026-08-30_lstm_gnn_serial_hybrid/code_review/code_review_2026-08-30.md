# Code review — SERIAL LSTM→GNN hybrid (2026-08-30)

3-lens adversarial review (Blind Hunter · Edge Case Hunter · Acceptance Auditor), incl. a performance lens
per CLAUDE.md (train/inference code). Scope: `serial_hybrid_net.py`, `run_serial_hybrid.py`, both test files.
`archive/` and reused read-only upstream modules (`run_masked_rich`, `masked_rich`, `corrlift_edge`, …) are
out of scope (not modified here).

## Lens 1 — Blind Hunter (hidden bugs / logic errors)

| # | Finding | Severity | Resolution |
|---|---|---|---|
| B1 | Would the "serial" claim silently regress to parallel if someone wired `gat(x[...,-1,:])`? | — | Guarded by `test_gnn_input_is_the_lstm_embedding` (forward hook asserts the GAT input equals `lstm_embed(x)`, shape `[B,N,hidden]`) + `test_seq_history_changes_the_gnn_input`. A regression to raw-feature input fails both. |
| B2 | `train_serial` early-stop could keep the FIRST (untrained) weights if `best_state` never updated. | Minor | `best_state` is initialised to the pre-loop state and reassigned whenever val improves; epoch 0 always improves vs `inf`, so the trained state is kept. No dead `if best_state` branch. |
| B3 | Per-node denorm `pn*t_std + t_mean` broadcast correctness. | — | `pn [n,N]`, `t_std/t_mean [N]` → NumPy broadcasts over the node axis (verified: `test_train_serial_real_net_finite` asserts positive finite `[n,N]`). Same floor `1e-2*mean` as the delivered trainer → fair QLIKE. |
| B4 | DM pair keys must intersect (serial vs no_graph vs delivered share the SAME test obs). | — | All three variants train on the SAME `D` (same `tmask_te`), so `_pred_dict` keys coincide; `_dm_all` intersects keys defensively anyway. |
| B5 | Leakage: is the corr+lift graph train-only? | — | `serial_adj_for` cuts at `D.d_va[0]` (first val target); `build_corrlift_adjacency` uses only `date < cutoff` then freezes — identical discipline to the corrlift baseline it reuses. |

## Lens 2 — Edge Case Hunter

| # | Finding | Severity | Resolution |
|---|---|---|---|
| E1 | Invalid node (node-mask column zeroed, incl. its self-loop) → all-`-inf` attention row → NaN. | Major-if-unhandled | `WeightedGATLayer` applies `nan_to_num(alpha, 0.0)`; `test_mask_awareness_isolated_node_finite` pins a finite output for a masked node. |
| E2 | Empty val split → cannot locate the frozen-graph boundary. | Minor | `serial_adj_for` raises `RuntimeError` (`test_serial_adj_for_empty_val_raises`). |
| E3 | Empty test split / non-finite forward in the dry smoke. | Minor | `forward_pass_smoke` raises on both (`test_forward_pass_smoke_empty_test_raises`, `test_forward_pass_non_finite_raises`). |
| E4 | Panel build yields < 2 tickers. | Minor | `build_panel_masked` raises (`test_build_panel_masked_too_few_raises`). |
| E5 | All-invalid-target batch → divide-by-zero in masked MSE. | — | `tmb.sum().clamp(min=1)` guards the denominator. |

## Lens 3 — Acceptance Auditor (vs requirements.md)

| Criterion | Status |
|---|---|
| SERIAL: GNN node features = LSTM temporal embedding | MET (wiring + 2 pinning tests) |
| Combined corr+lift graph, TRAIN-only frozen, reused read-only | MET (`serial_adj_for` → `build_corrlift_adjacency`) |
| Dense thresholds ρ>0.25/lift>1.2 + paper 0.7/1.7 density recorded | MET (`DENSE_*`/`PAPER_*`, both in `edge_density*`) |
| 3 variants (serial / no_graph / delivered parallel), same folds/seeds | MET |
| DM serial-vs-no_graph, serial-vs-delivered | MET |
| Over/under-fit evidence (train/val/test + verdict + curves) | MET (`run_training` emits all 4 blocks) |
| Unique test basenames (no duplicate-basename collision) | MET (`test_serialhybrid_model.py`, `test_serialhybrid_runner.py`) |
| No edit to any live-training-path file | MET (all reused modules imported read-only) |

## Performance lens (CLAUDE.md ENFORCED)
- **Batched:** LSTM runs one call on `[B·N, SEQ, 5]`; GAT is an einsum over `[B,N,N,heads]`. No per-item
  Python loop in the hot path. PASS (not a batch=1 anti-pattern).
- **GPU-first:** `resolve_device()` uses `torch.cuda.is_available()`; train tensors (`X_tr`, masks, adj base,
  scalers) are **preloaded to the device once**, so the training loop has no per-step host→device copy. AMP not
  used (tiny model; not the bottleneck). PASS.
- **Single process / VRAM:** batch 16 over a 154-node GAT stays well under 8 GB; single process; no concurrent
  GPU training (guarded poller waits for `util<15 && VRAM<1200 MiB`). PASS.
- Minor (accepted): the `no_graph` variant still computes `adj_batch` though the net ignores it (a few cheap
  broadcasts); and `edge_density_paper_thresholds` rebuilds `close_wide` once (CPU, negligible). Not worth
  extra branching.

## Verdict
No critical/major findings open. All edge cases are guarded and pinned by tests (19 tests, C0 line=100% /
C1 branch≥95% on changed lines, ruff `--select F` clean). Ready for the empirical run.
