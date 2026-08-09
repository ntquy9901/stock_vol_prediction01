# Track B Pooled Convergence — Root-Cause Investigation

Date: 2026-08-09
Scope: read-only. Why do Track B pooled models P1/P2/P3 reach best validation at
epoch 5-6 (then, for P2/P3, overfit) while Track A took ~40 epochs? Is the Track B
result trustworthy or an under-regularization artifact?

Sources (committed code, worktree `.worktrees/pooled-news-gnn-pilot`, branch
`feature/pooled-news-gnn-pilot`):
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/{train.py,run_pilot.py,models.py}`
- Results: `results/pooled_20ep_seed{42,123,2026}/h5/{P1,P2,P3}/results.json`,
  `results/a1_{pooled,commondate}_seed{42,123,2026}/h5/validation_comparison.json`
- Track A: `src/lstm_gat_hybrid/config.py`, `src/lstm_gat_hybrid/train_parallel_enhanced.py`

---

## 0. Correction to the preliminary grep

The preliminary finding stated the pooled P1/P2/P3 models use `dropout=0.0`
(run_pilot.py:143,211). That is **incorrect for the P1/P2/P3 screening path**.

- The P1/P2/P3 screening models are built in `train.py:146-148`:
  `PooledPriceLSTM(price_dim)` and
  `PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=...)` — **no dropout
  argument passed**, so they inherit the class default `dropout=0.2` (models.py:25,51).
  That 0.2 is applied both between the 2 stacked LSTM layers and in the head
  (`nn.Dropout(dropout)`, models.py:31,60).
- The `dropout=0.0` at run_pilot.py:143 and :211 belongs only to
  `build_graph_bound_p3_warm_start` / `build_graph_safe_p3_checkpoint` — the **G0/G1
  graph-ablation** warm-start checkpoints, a different code path, not the P1/P2/P3
  pooled screening under investigation.

So the pooled P1/P2/P3 models **are** regularized with dropout 0.2 + weight_decay 1e-5
+ grad-clip 1.0. The genuine gaps are: no LR scheduler, no early-stopping break, no
layernorm, no augmentation, and a higher (default) learning rate than Track A.

---

## 1. Regularization inventory — Track B pooled P1/P2/P3 vs CLAUDE.md §E vs Track A

| Technique | CLAUDE.md §E mandate | Track B pooled P1/P2/P3 | Track A (parallel LSTM-GNN) |
|---|---|---|---|
| LSTM/head dropout | 0.2 | **USED 0.2** (models.py:25,51; default, applied in LSTM + head) | USED (lstm 0.2 / gat 0.1 / fusion 0.3 — config.py:32,44,50) |
| Weight decay (L2) | 1e-5 | **USED 1e-5** (train.py:151) | USED 1e-5 (config.py:60) |
| Gradient clipping | max_norm 1.0 | **USED 1.0** (train.py:185) | USED 1.0 (config.py:68) |
| Early stopping / patience | patience 15 | **MISSING** — fixed `for _epoch in range(...)`, no break (train.py:172); best-val checkpoint is saved (train.py:196-198) but training does not stop | USED — `EarlyStopping(patience=15, min_epochs=20)` with `break` (train_parallel_enhanced.py:41,601-603,672-677) |
| LR scheduler | ReduceLROnPlateau | **MISSING** — no scheduler (train.py has none) | USED — `ReduceLROnPlateau(factor=0.5, patience=5)` (train_parallel_enhanced.py:594-598,652) |
| Learning rate | (unspecified) | 1e-3 (Adam default; no `lr=` at train.py:151) | 5e-4 (config.py:59) |
| Layer normalization | listed | **MISSING** | (not confirmed here) |
| Data augmentation | listed (if <5000) | **MISSING** (n/a; pooled train = 73,026) | (not confirmed here) |
| Best-checkpoint selection | (implied) | USED — min-val checkpoint (train.py:196-198) | USED |
| Batch size | — | 256 (run_pilot.py default, batch_size branch) | 32 (config.py:63) |
| Max epochs | — | 5 (A1 runs) / 20 (pooled_20ep runs) | 70 (config.py:57) |

Net: Track B P1/P2/P3 satisfy **3 of the 5** §E model-centric techniques (dropout,
weight_decay, grad-clip) and miss **early-stopping-break, LR-schedule** (and
layernorm/augmentation). It selects the best-val checkpoint, so the missing
early-stopping *break* does not corrupt the reported metric — it only wastes compute
past the optimum. The missing LR schedule is the one that could plausibly change the
*achievable* optimum.

---

## 2. Steps-per-epoch / total-updates quantification

Sample counts are read from the committed `sample_manifest.json` (P1 split counts;
identical across P1/P2/P3 by manifest-hash equality check).

| Run | Train samples | Batch | Optimizer steps/epoch | "Convergence" epoch | Updates to best |
|---|---|---|---|---|---|
| Track B pooled (20ep) | 73,026 | 256 | ⌈73026/256⌉ = **286** | 5-6 | ~1,430-1,716 |
| Track B common-date (A1) | 9,606 | 256 | ⌈9606/256⌉ = **38** | ~5 | ~190 |
| Track A (parallel LSTM-GNN) | common-date panel (~300 snapshots) | 32 | order 10-300 depending on per-snapshot vs mini-batch stepping | ~40 | order 400-12,000 |

Key quantitative point: the pooled regime does **286 optimizer updates per epoch**,
roughly **7.5× more than the common-date pooled regime (38)** and one to two orders of
magnitude more than a per-panel Track A epoch. By its convergence point (epoch 5-6) the
pooled model has already taken **~1,430-1,716 gradient steps** — comparable in order of
magnitude to a ~40-epoch Track A run. So "converges in 5-6 epochs" is largely a
**relabelling of the x-axis**: an epoch over 73k pooled ticker-days is not the same unit
of work as an epoch over ~300 common-date panel snapshots. This is the data-volume
explanation, and it is the dominant factor.

Secondary accelerant: Track B uses Adam LR **1e-3** (default) vs Track A's **5e-4** —
2× larger steps, which also compresses the epoch count to reach the basin.

(Track A's exact snapshot count and per-epoch step count were not confirmed from
committed artifacts in this read-only pass; its column is an order-of-magnitude estimate.
The pooled-side numbers are measured.)

---

## 3. Per-config converge-vs-overfit classification (20-epoch trajectories)

Validation MSE (normalized) by epoch, from `pooled_20ep_seed*/h5/*/results.json`:

- **P1 (price only) — CONVERGED + PLATEAU (no meaningful overfit).**
  seed42: best 0.8456 @ep5, ep20 = 0.8474 (span across ep3-ep20 ≈ 0.0018).
  seed123: best 0.8462 @ep11. seed2026: best 0.8459 @ep14.
  The curve drops to ~0.846 by epoch 3-5 and then wanders inside a flat noise band; the
  argmin epoch (5 / 11 / 14) moves with the seed, confirming a flat basin rather than a
  sharp peak. Even with dropout=0.2 and no early stopping, P1 does not degrade.

- **P2 (price + news, no gate) — PEAKED + OVERFIT.**
  best ~0.846 @ep5-7, then **monotonic rise** to 0.92-0.95 by ep20
  (seed42: 0.8456→0.9454; seed123: 0.8464→0.9195; seed2026: 0.8459→0.9373).

- **P3 (price + news + gate) — PEAKED + OVERFIT (milder than P2).**
  best ~0.846 @ep5-7, then rise to 0.88-0.95 by ep20
  (seed42: 0.8455→0.9092; seed2026: 0.8461→0.8809). The sigmoid news gate provides a
  small amount of extra regularization, so P3's post-peak climb is consistently gentler
  than P2's.

Two facts stand out:
1. **All three configs reach essentially the same best val loss (~0.846).** Adding the
   news branch does not lower the achievable optimum below price-only; it only adds an
   extra 2-layer news LSTM whose capacity overfits after epoch ~6.
2. The "best at epoch 5-6" is therefore **clean convergence for P1** and a
   **peak-then-overfit for P2/P3** — but in both cases the *reported* number is the
   min-val checkpoint (train.py:196-198), which is taken at the basin, not on the
   overfit tail.

---

## 4. Impact on the reported conclusions

The conclusion-bearing runs (`a1_*`) were executed at **epochs=5** (screening_metadata),
i.e. right at the shared basin and **before** the P2/P3 overfit onset (which starts at
epoch 6-7). The A1/news/gate/HAR numbers are thus **not drawn from the overfit tail**:

- **A1 (pooled vs common-date), seed42:** P1 qlike 0.5109 (pooled) vs 0.5101
  (common-date); P2 0.5082 vs 0.5124; P3 0.5087 vs 0.5123. Differences are in the 3rd-4th
  decimal; RMSE identical at 0.0015; DirAcc ~48.5% in both (below 50%).
- **news (P2 vs P1)** and **gate (P3 vs P2):** sub-0.005 qlike deltas, DirAcc essentially
  flat — consistent with the epoch-20 finding that news does not lower the achievable
  optimum.
- **HAR (P0):** fit by a single `LinearRegression` (run_pilot.py:257), no epochs, no
  regularization knobs — its comparison is unaffected by any of this.

Risk assessment:
- The missing **early-stopping break** does **not** bias the reported metrics (best-val
  checkpoint is used); it only wastes compute. Low risk.
- The missing **LR scheduler** is the one gap that could move the optimum: ReduceLROnPlateau
  would lower the LR once val plateaus (epoch ~5), potentially squeezing val loss slightly
  below 0.846 and/or delaying the P2/P3 overfit. However, because **all three configs
  already converge to the same ~0.846 basin**, a scheduler would most plausibly deepen that
  shared basin roughly equally for P1/P2/P3 — the *relative* A1/news/gate ordering (the
  actual conclusion) is unlikely to flip, since news adds no separation even at the optimum.
- The pooled models are **not meaningfully under-regularized** — dropout, weight decay, and
  grad clipping are all present at §E-mandated values. The epoch-count difference from
  Track A is explained by **data volume per epoch (7.5×-plus more steps/epoch) plus a 2× LR**,
  not by absent dropout.

---

## 5. Root-cause conclusion

The fast (epoch 5-6) convergence of Track B pooled models is **primarily a data-volume /
optimizer-step effect, secondarily a learning-rate effect — not an under-regularization
artifact.** Concretely:

1. Pooled training runs **286 optimizer updates/epoch** (73,026 samples / batch 256) vs a
   Track A panel epoch of order 10s of steps; ~1,430-1,716 updates by epoch 5-6 is
   comparable in magnitude to a full ~40-epoch Track A run. The epoch axis measures
   different amounts of work.
2. Adam LR 1e-3 (default) vs Track A's 5e-4 further compresses the epoch count.
3. Dropout (0.2), weight decay (1e-5) and grad-clip (1.0) **are present** in P1/P2/P3, so
   the convergence is genuine, not the collapse of an unregularized model. P1 converges to
   a stable plateau; P2/P3 reach the same optimum then overfit their extra news-LSTM
   capacity — but the reported metric is the best-val checkpoint at the basin.

---

## 6. Recommendation

A CLAUDE.md-§E-compliant re-run (add `ReduceLROnPlateau` + a real early-stopping patience;
Track A already has both) is **advisable for paper-grade defensibility but is unlikely to
change the P0-P3 / A1 conclusions.** Rationale:

- The conclusion runs use epochs=5, at the shared basin, so they are **not currently
  contaminated** by the P2/P3 overfit tail. The numbers as reported are defensible against
  the "you trained into overfitting" objection.
- The one substantive gap (no LR scheduler) would most likely deepen the common ~0.846
  basin for all configs together; because news adds **no separation even at the optimum**
  (all configs ≈ 0.846), the relative news/gate/A1 ordering should be robust.

Suggested action, in priority order:
1. **Low-cost robustness check (recommended before the paper cites these numbers):** re-run
   P1/P2/P3 pooled at the current 5-epoch setting **with ReduceLROnPlateau added**, across
   the existing 3 seeds, and confirm the A1/news/gate deltas stay within noise. This
   directly closes the only conclusion-relevant gap.
2. **Optional (defensiveness only):** add the early-stopping `break` so the 20-epoch
   diagnostic runs stop at patience rather than burning through the overfit tail. This does
   not change any reported metric.
3. A full Track-A-style 70-epoch re-run is **not warranted** — pooled convergence is
   reached in ~1,500 updates and the extra epochs only overfit P2/P3.

Bottom line: the Track B pooled result is **trustworthy as reported** (best-val checkpoint,
regularized, converged), and the "epoch 5-6 vs epoch 40" gap is a units-of-work artifact,
not an under-regularization artifact. A one-line LR-scheduler robustness re-run is the
proportionate way to remove residual doubt before the paper relies on these numbers.
