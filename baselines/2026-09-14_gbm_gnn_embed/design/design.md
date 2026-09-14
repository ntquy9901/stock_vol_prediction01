# Design — GBME ⊕ 2-layer GNN node-embedding (GBM consumes learned GNN features)

> Status: DESIGN + sample code (scaffold). Build/TDD/gate = follow-up. Pre-registered as a falsification
> (5th distinct graph attempt). Expected NO-GO given the prior four; the point is to close the
> "GNN-embedding-as-feature" door with a controlled, leakage-safe test.

## 1. Idea and how it differs
Extract a **learned** node embedding from a 2-layer GNN (GCN or GAT) and **concatenate it as features** into
the champion `GBM+earn` tree, instead of a hand-crafted scalar aggregate.

```
per fold k (train-only graph Wc):
   2-layer GNN (GCN/GAT) trained on the vol target  ─►  node embedding  z_{i,t} ∈ R^d
   GBM(gamma tree)  ⟵  x_{i,t} = [ OWN-8 | EARN-4 | z_{i,t} (d dims) ]
```

| baseline | graph part | learned? | who predicts |
|---|---|---|---|
| GBME+graph | 1 scalar `g_corr` | no | GBM tree |
| concat-7 (NO-GO) | 7 scalar aggregates | no | GBM tree |
| GNNHAR | 2-layer GCN | yes | the GNN (end-to-end) |
| residual (NO-GO) | ridge on neighbour aggregates of GBM residual | linear | GBM + ridge |
| **this** | **2-layer GNN embedding** | **yes** | **GBM tree** (GNN = feature extractor) |

## 2. The embedding `z`
Reuse the faithful GNNHAR model `scripts/eda/gnnhar_sp500.py::GNNHAR` read-only. Its forward is
`hg = x; for gcn in gcns: hg = relu(gcn(hg, A)); out = relu(linear1(x) + mlp1(hg))`. The **embedding is
`hg` after the two GCN layers, BEFORE `mlp1`** — an `R^{n_hid}` (=9) learned representation per node-day.
We expose it with a thin `embed()` wrapper (subclass, no edit to the shared model). GAT variant swaps the
`GraphConvLayer` (fixed `A`) for an attention layer; GCN first (reuse), GAT as a config flag.

## 3. THE critical decision — avoid stacking leakage (out-of-fold embeddings)
If the GNN is trained on the same TRAIN rows the GBM is then trained on, the GBM sees **in-sample** GNN
representations `z_train` that are over-optimistic (the GNN fit those exact rows). The tree overfits to `z`,
and `z_test` does not generalise → a spurious in-sample "gain" that dies OOS. This is the same trap the
residual baseline avoided. **Fix: cross-fitting.**

Within each walk-forward fold's train window:
1. **Inner K-fold (temporal, K=3) over train rows.** For each inner fold: train the GNN on the other K−1
   inner folds, embed the held-out inner fold → **out-of-fold (OOF) `z_train`**. Every train row gets an
   embedding produced by a GNN that did NOT see it.
2. **Test embedding:** train the GNN once on the FULL train window, embed the test rows → `z_test`.
3. The GBM is fit on `[OWN-8|EARN|z_train^{OOF}]` and predicts `[OWN-8|EARN|z_test]`.

The graph `Wc`, the feature scaler, the GNN, and the GBM all fit on train/inner-train rows only; embargo
between train and test as elsewhere. A causality test asserts fold-k output is invariant to permuting later
folds, and that `z_test` depends only on train-window rows.

## 4. Files
- `code/gnn_embed_config.py` — tunables: `N_GCN=2`, `N_HID=9`, `INNER_K=3`, `ARCH∈{gcn,gat}`, epochs, lr,
  early-stop patience, seeds (reuse GNNHAR's where possible; single-source, no literals in pipeline code).
- `code/embed.py` — `GNNEmbedder` (subclass of `G.GNNHAR` exposing `embed(x,adj)->R^{n_hid}`), plus
  `oof_train_embeddings()` (inner-K OOF) and `full_train_test_embeddings()`.
- `code/run_gnn_embed.py` — walk-forward driver: per fold build Wc, compute OOF `z_train` + `z_test`, fit
  `GBM(own8+earn+z)` vs `GBM(own8+earn)` (=GBME baseline), pool, QLIKE + date-clustered DM, per-horizon
  atomic checkpoint (Colab-resilient), overfit evidence. Output `results/gamma_gbm/gnn_embed_<market>.json`.
- `test/` — TDD: embed shape (`R^{n_hid}`), OOF coverage (every train row embedded by a GNN that didn't see
  it), causality (no future-fold leak), signal-recovery on synthetic where z predicts residual, run-structure
  smoke, verdict logic, checkpoint. Real-data driver `# pragma: no cover`, logic unit-tested.

## 5. Gates (SDD)
- **Simplicity:** one embedder + one driver; reuse GNNHAR + FM.gbm + M/ST. No new abstraction.
- **Anti-Abstraction:** subclass the existing GNNHAR; use `sklearn` GBM + project metrics/stats directly.
- **Performance/Batching:** GNN trained batched over dates on GPU (reuse GNNHAR's batched loop); GBM is the
  existing batched seed-averaged fit; the only extra cost is the inner-K OOF (K× GNN trainings/fold). No
  batch=1. Report the added GPU cost.

## 6. Success / kill criterion (pre-registered)
`success=True` only if `GBME+z` beats `GBME` (gain>0 AND DM p<0.05) at **both h1 and h5**. Otherwise NO-GO.
Do NOT chase an in-sample or MAE-only win. Expected outcome, given GNNHAR (its 2-layer graph HURTS) and the
concat err_corr 0.98: the OOF embedding adds no orthogonal OOS QLIKE value. HOSE spike-robustness applies
before any positive HOSE claim.

## 7. Overfit evidence (gate-required — learned model)
The result JSON carries per horizon/model `train`/`val`/`test` QLIKE + `fit_diagnostics` + the GNN
`learning_curves`; name contains no `gnn` token by accident — set the JSON model key to `GNN-embed` so the
pre-push overfit gate treats it as learned and enforces the evidence.
