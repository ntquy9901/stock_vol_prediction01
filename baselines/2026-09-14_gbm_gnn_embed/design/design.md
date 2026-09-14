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
`learning_curves`; the learned model key is `GBME+GNN-embed` (contains `gnn`) so `overfit_check.looks_learned`
treats it as learned and the pre-push overfit gate enforces the evidence.

**Deviation from §4 (output file layout) — ONE JSON PER HORIZON.** §4 named a single
`gnn_embed_<market>.json`. Changed to **`results/gamma_gbm/gnn_embed_<market>_h<h>.json`** (one per horizon,
mirroring `scripts/eda/gnnhar_sp500.py`). Reasons: (a) the pre-push overfit gate
(`check_overfit_evidence.py::_is_masked_rich_result`) only recognises a training result when the TOP-LEVEL
`metrics` dict holds a learned-model key — a single multi-horizon file nests metrics under `h1/h5/...`, so it
would be silently SKIPPED and the evidence never enforced; a per-horizon file has top-level
`metrics={GBME, GBME+GNN-embed}` and IS checked. (b) Atomic per-horizon files give the Colab-resilient
"each completed horizon is durable + committable" property the run needs. `learning_curves` are captured
from the full-train (z_test) embedding GNN per seed (representative; the inner-OOF GNNs are not recorded to
keep the 4× OOF cost down).

## 9. Validity caveat — embedding basis non-identifiability (biases toward NO-GO)
Neural hidden units are defined only up to a permutation/rotation/sign. The leakage-safe OOF design stitches
`z_train` from `INNER_K` separately-trained inner GNNs, and `z_test` from a distinct full-train GNN, so
different train rows — and `z_train` vs `z_test` — can live in **different latent coordinate systems**. A GBM
split "`z3 > 0.5`" learned on `z_train` need not carry the same meaning on `z_test`. Mitigations applied: a
**single** embedding seed (no averaging embeddings across differently-initialised seeds, which would shrink
them in an ill-defined basis); the GBM is still seed-ensembled. The inner-vs-test basis drift is **inherent**
to leakage-safe cross-fitting of neural features (the alternative — one GNN embedding both its own train rows
and the test rows — reintroduces the stacking leakage this design exists to prevent). Consequence: a NO-GO
here is a **lower bound** on graph value (the test could mask a real signal), not proof of none. This is
stated in the report and must be stated in any paper use.

## 8. Node features + GNN target (decisions)
- **GNN node features = OWN-8** (same as the GBM input). The embedding therefore adds only the *graph
  mixing* of own-history features; that marginal graph value is exactly the falsification target.
- **GNN target = the `pk` variance `shift(-h)`** (same target, QLIKE loss), so the embedding is trained to be
  predictive of the very quantity the GBM forecasts.
- **`ARCH`** config flag reserved for `{gcn,gat}`; only `gcn` (reuse of the faithful `GNNHAR` GraphConvLayer)
  is implemented — `gat` raises `NotImplementedError` (fail loud, no silent fallback). GCN-first per §5.

## Why `log pk` for the graph correlation (not raw variance)
The graph adjacency `Wc` keeps each node's top-k neighbours by **Pearson correlation of `log pk`**, not of raw
`pk`. Definitions:
- **pk** (Parkinson variance) `= ln(H/L)^2 / (4 ln 2)` — a **variance** (sigma^2): always >= 0, heavy
  right-skew (log-normal), spans orders of magnitude (1e-6 to 1e-2).
- **log pk** `= ell_t = ln(max(pk_t, eps))`, eps = 1e-8 — the natural log of that variance.

Rationale: Pearson correlation measures **linear** dependence and is meaningful only when the inputs are
roughly symmetric/Gaussian and on a comparable scale. Realized variance is well modelled as **log-normal**
(Andersen-Bollerslev-Diebold-Labys 2001), so `log pk` is approximately Gaussian and symmetric, and its
correlation captures **persistent co-movement**. Correlation on **raw variance** is dominated by a few
storm days and by high-variance stocks (heteroskedasticity inflates covariance), producing an
outlier-driven, unstable graph. The spillover literature (Diebold-Yilmaz, GNNHAR, MTGNN) builds graphs on
**log-RV / log-vol** for the same reason.

**log pk vs pk for the graph:** `log pk` is the principled choice (log-normality of RV). Empirically it is
close to moot here: the graph adds no OOS value under DM regardless of how the adjacency is built, so
`corr(log pk)` vs `corr(pk)` does not change the (null) result. **Role split (do not conflate):** `log pk`
is used only for (a) the graph correlation and (b) the `mr_*` log-volatility momentum features; the
**forecast target is `pk` (variance)**, and the gamma-loss GBM/GNN predict `pk` directly.

## Choosing node features (what to include besides own-history + earnings)
Principle: node features must be **stationary and comparable across the ~500 pooled stocks** (the model
pools all tickers of a market). Raw levels break this.

| Feature | Include? | Why |
|---|---|---|
| Raw prices O/H/L/C (level) | **No** | Non-stationary (trending), incomparable scale across stocks, not predictive of variance. |
| Range estimators from OHLC (Garman-Klass, Rogers-Satchell, Yang-Zhang; overnight gap `ln(O_t/C_{t-1})`, intraday `ln(C_t/O_t)`) | **Maybe** | Stationary, same family as pk. Already tested: only +0.4% (DM-sig h1/h10), small. |
| Raw volume (level) | **No** | Non-stationary, cross-stock scale differences. |
| Volume z-score (standardised, 22-day) | **Maybe** | Scale-free; volume-volatility relation; already used in the neighbour block. |
| Downside semivariance / leverage (`semi_neg`) | **Maybe** | Stationary; own+semi_neg gave +0.42% at h1 (DM-sig), small. |

Net: do **not** add raw price/volume levels. The stationary, scale-free candidates (range estimators,
volume z-score, downside semivariance) were tried and give only marginal gains -- the own-8 own-history set
is near-saturated. Any node feature added here should be per-stock standardised or a scale-free ratio.
