# Beat-HAR Solution Plan — Ranked Pooled-LSTM + Graph Configurations (20-epoch sweep)

Date: 2026-08-10. Scope: DESIGN / PLANNING only. No training performed, no code modified, no
`.worktrees/` or `archive/` touched. This document specifies a ranked, sequentially-runnable set of
trainable configurations whose goal is to beat the classical HAR baseline for VN30 daily 5-day-ahead
Parkinson-variance forecasting, on the consistent Track-B evaluation basis.

Hard constraint honored: every configuration is **Pooled LSTM + graph (GNN/GAT/message-passing)**, or
**Pooled LSTM + news + graph**. No non-graph model, no configuration drops the graph.

---

## 0. What "beat HAR" means here (the fair-basis bar)

All numbers below are on the **consistent Track-B basis**: leakage-safe chronological 70/15/15 split,
Parkinson-**variance** target `shift(-h)`, and the EXACT pooled val/test observations (same keys + raw
targets) scored by `train.evaluate_records` — n_val = 14418, n_test = 14464, 33 tickers. Sources:
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`,
`docs/reports/classical_baselines_h5_2026-08-09_182129.json`.

TEST-set reference rows (mean over 3 seeds where applicable), the bar to beat:

| Model | QLIKE | RMSE | R² | MAE | DirAcc |
|---|---|---|---|---|---|
| **P0 pooled HAR anchor** (deep-pipeline HAR) | **0.5676** | 0.0022893 | 0.76679 | 0.0006027 | 48.53 |
| Classical per-ticker HAR (OLS) | 0.5793 | 0.0022897 | 0.76670 | 0.0006312 | 48.40 |
| **HARQ** (daily RQ proxy) — strongest classical on RMSE/R² | 0.5737 | **0.0022891** | **0.76682** | 0.0006289 | 48.38 |
| EWMA | 0.6006 | 0.0023107 | 0.76241 | 0.0006106 | 48.03 |
| **G1 current** (masked kNN-8 corr GAT + news, MSE loss) — ties HAR | 0.5759 | 0.0023053 | 0.76352 | 0.0005996 | 48.22 |
| P2 (price+news, graph OFF, MSE loss) — best QLIKE already in ladder | 0.5599 | 0.0022703 | 0.77064 | 0.0006016 | 48.04 |

Key facts that shape the plan:

1. **The effective QLIKE bar is P0 = 0.5676**, not classical HAR 0.5793. P0 (pooled HAR on
   standardized features) is already the strongest HAR variant on the shared basis, so a credible
   "beat HAR" must clear ~0.5676 DM-significantly, not merely 0.5793.
2. **News alone already crosses that bar on QLIKE**: P2 (price+news, no graph, MSE) = 0.5599 test
   QLIKE < P0 0.5676. This is the single most encouraging pre-existing signal in the ladder, and it is
   why NEWS is retained in the front-running configs. (It has not yet been DM-tested vs HAR — that test
   is part of every success check below.)
3. **The graph adds no significant level-metric value over HAR** on this basis (graph-effect verdict
   "B", test paired-p 0.79; `ladder_consistent…json` `graph_effect_verdict`). The graph is kept
   because it is the mandated architecture and a research target, not because it is currently the
   winning lever.
4. **RMSE / R² are the hardest wall**: HARQ RMSE 0.0022891 / R² 0.76682. No current deep config beats
   these. A QLIKE-only win is therefore the realistic target; an RMSE win would be a strong result.

**Success definition (per config):**
- **Partial win (legitimate, primary target):** beat P0 on QLIKE (and ideally classical HAR + HARQ),
  DM-significant p<0.05, consistent sign across all 3 seeds, paired-t over seed means. Per the
  research synthesis (`…_2209_gnn_volatility_beat_har_research.md` §1–3;
  `…_2214_gnn_hybrid_combinations_research.md` §3), QLIKE and short-horizon are the likeliest wins.
- **Full win (stretch):** beat P0 AND HARQ on QLIKE **and** RMSE **and** R², all DM-significant.
- **Honest framing:** the located literature has **no precedent for a GNN beating HAR on daily
  range-based variance at ~30-asset scale** (closest rigorous analogue GNNHAR/DJIA-30 found the graph
  component null and the win in QLIKE-loss + nonlinearity). A QLIKE-only, DM-significant partial win is
  the plausible ceiling and is ranked as a real success.

**Units trap (applies to every config with a volatility-native feature):** the target
`parkinson_volatility` is the Parkinson **variance** σ² = (ln(H/L))² / (4 ln2), not σ (verified
corr=1.0 vs OHLCV, median ~1.3e-4; `classical_baselines…json` `target_units`). Any added range /
overnight / semivariance feature MUST be expressed in variance (σ²) units to match the target and the
existing 3 HAR features.

**Leakage-safety invariants (mandatory in every design):** graph structure (adjacency / VAR /
co-mention / learned-A) estimated on the **train window only** and frozen; per-ticker feature+target
scalers fit on train only (already enforced in `scaling.py`); temporal 70/15/15 with no window
crossing a split boundary (already enforced in `build_masked_graph_manifest`); positivity floor on
denormalized predictions (`POSITIVITY_EPSILON`, already in `models.py`); present-node masking so absent
tickers never leak into present outputs (already enforced). A leakage-driven win is disqualified.

---

## 1. Pipeline facts each config builds on (pilot code, `feature/masked-gnn` branch)

`baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/` (read via `git show`, not modified):

- **Node price features = exactly 3 HAR**, variance units: `[parkinson σ², rolling(5).mean,
  rolling(22).mean]`, clipped ±3σ and standardized per ticker (`scaling.py:_har_features`,
  `TickerPreprocessor.fit`). `price_dim = 3`, `seq_length = 22`, `horizon = 5`.
- **News branch**: precomputed per-(ticker,date) PhoBERT feature parquet, encoded by a second LSTM and
  concatenated; per-ticker input-independent gate (`models.py:PooledPriceNewsLSTM`). The panel is
  per-ticker-aligned — it carries **no article-level multi-ticker structure** (relevant to C7).
- **Graph**: masked kNN-8 mutual-correlation adjacency, symmetric, **self-loops kept** (diagonal=1),
  signed weights, row-softmax attention in `_ResidualMessagePassing`. Present-node masked, graph-bound
  to the train window. `min_offdiag_nonzeros_per_present_row = 0` in the current run — some present
  nodes have zero neighbors and rely on the self-loop to satisfy the message-passing invariant "each
  present node requires a self-loop OR a neighbor" (`models.py`). Adjacency modes already supported:
  `dense`, `knn`, `threshold` (`data.py:_validate_adjacency_config`).
- **Loss = `nn.MSELoss()`** on **normalized** predictions (`train.py:152,181`). A numpy `qlike_loss`
  exists for evaluation only (`src/common/evaluation.py:42`) — it is **not** differentiable and is
  **not** used in training.
- **Epoch cap**: `run_training` raises if `epochs < 1 or epochs > 10` (`train.py:116`). The current
  ladder trains BACKBONE_EPOCHS=5 + GRAPH_EPOCHS=15 (graph head only; backbone frozen after P3).
- **Encoder cache**: `GraphAblationModel.encode_base` produces frozen, dropout-free, deterministic node
  embeddings reused across epochs/seeds; `apply_graph_head` is the only trainable+graph path. Every
  config below reuses this cache and the consistent-ladder basis unless stated.

**Two cross-cutting pipeline changes shared by most configs (do once, up front):**

- **(A) Differentiable QLIKE loss.** Add a ~15-line torch QLIKE module operating on **denormalized,
  positivity-floored** predictions (inverse per-ticker target scaler + existing `_apply_positivity`),
  because QLIKE is level-sensitive and undefined for non-positive ŷ. Swap `criterion` in
  `train.py:run_training` (flag `--loss {mse,qlike}`). Apply QLIKE at BOTH the backbone-training and
  graph-head-training stages, otherwise the loss only reshapes the message-passing residual.
- **(B) Raise the epoch cap** `train.py:116` from 10 to 20 (or parametrize), and set the 20-epoch
  budget split (e.g. backbone 12–15, graph head 20) for the sweep.

These two are prerequisites of C1 and are reused by C2–C6.

---

## 2. Ranked configuration table (ranked by expected P(beat HAR) × low effort)

Effort: **S** ≈ a few hours + run; **M** ≈ 1–3 days code + run; **L** ≈ >3 days / larger plumbing.
Every config: pooled 2-layer LSTM price encoder (hidden 64, dropout 0.2) + graph message-passing
residual + positivity floor; 3 seeds (42, 123, 2026); 20 epochs; same fair basis + encoder cache.

### Rank 1 — C1: QLIKE-loss GAT + news  *(cheapest, highest P(beat))*
- **Hypothesis.** Switching the training loss MSE→QLIKE is the single decisive, architecture-agnostic
  HAR-beating lever in the most rigorous GNN-vs-HAR study, and it targets the exact headline metric;
  news already pushes QLIKE below the HAR bar (P2 0.5599 < P0 0.5676). [Zhang, Pu, Cucuringu & Dong,
  IJF 2025 (GNNHAR); ladder P2 row]
- **Architecture.** Exactly current G1 — 3 HAR node features + PhoBERT news LSTM + per-ticker gate +
  masked kNN-8 mutual-correlation GAT residual (symmetric, self-loops kept) + standardized linear head
  + positivity floor. News **ON**. Loss **QLIKE** (change A). No HAR-residual decomposition (monolithic).
- **Code change.** Cross-cutting (A)+(B) only; no model/graph change. Flag `--loss qlike` in
  `train.py`; apply at backbone + graph-head stages. Reuses ladder basis + encoder cache verbatim.
- **Evaluation.** Same basis; all 6 metrics val+test; paired-t + Diebold–Mariano of QLIKE vs P0 and vs
  classical HAR. **Most plausibly wins: QLIKE** (possibly MAE). RMSE/R² likely still tie HARQ.
- **Effort.** **S.** ~2 h code (shared (A)/(B)) + 3 seeds × 20 ep; backbone-frozen graph head is fast.
- **Risk / leakage.** QLIKE requires strictly-positive ŷ — the existing denormalized floor covers it;
  verify the floor is applied *inside* the loss path. No new leakage surface.

### Rank 2 — C2: HAR + graph-residual additive decomposition + news  *(downside-protected)*
- **Hypothesis.** Let HAR own own-persistence and the GNN model ONLY the cross-sectional spillover
  residual (ŷ = ŷ_HAR + g(graph)). This DCRNN-HAR / GNHAR additive form is where the graph becomes
  "essential, not marginal", and it structurally floors at HAR (residual→0 ⇒ ŷ=HAR), so it cannot do
  much worse than HAR. [Chi et al., J. Forecasting 2026 (DCRNN-HAR); Boetti & Nunes, arXiv:2606.03828]
- **Architecture.** Per-node HAR linear term = the frozen train-fit per-ticker HAR OLS (reuse the
  classical HAR already computed) producing ŷ_HAR from the 3 HAR features; the pooled-LSTM+news graph
  branch predicts a residual added to ŷ_HAR; final = ŷ_HAR + g(graph), positivity floor. Node feats 3
  HAR + news; edges masked kNN-8 corr. News **ON**. Loss **QLIKE** (fall back to MSE if QLIKE unstable).
- **Code change.** New head wiring in `models.py` (`GraphResidualHARModel`): compute the fixed HAR term
  from node HAR features, add `message_passing(...)`-driven residual instead of the current monolithic
  `head(base + mp)`. Small residual-scale / weight-decay regularizer so the residual stays a correction.
  Reuses `encode_base` cache; HAR coefficients fit train-only.
- **Evaluation.** Same basis; DM of QLIKE vs P0/HAR. **Most plausibly wins: QLIKE, then RMSE** (by
  construction the level is anchored to HAR, so RMSE regression is bounded).
- **Effort.** **M.** New head module + train-only HAR term; 3 seeds × 20 ep.
- **Risk / leakage.** Residual can overfit → regularize + early-stop on val QLIKE. HAR term must be
  train-fit only. Ranked high because it is the safest route to a non-negative result.

### Rank 3 — C3: Directed spillover-edge GAT + news + QLIKE  *(composes the two best levers)*
- **Hypothesis.** Directed Diebold–Yilmaz / Granger spillover edges are the one edge construction with
  a published head-to-head win over correlation (~25–40% lower error); combined with QLIKE loss this is
  the "minimal change most likely to move the needle" from both research tracks. [Kumar et al.,
  arXiv:2410.16858 2024; Diebold & Yilmaz 2012/2014; both research reports §4]
- **Architecture.** As C1 (pooled LSTM + news + GAT + QLIKE) but replace the correlation adjacency with
  a **directed** generalized-FEVD Diebold–Yilmaz spillover matrix from a train-window VAR on the
  per-ticker volatility panel (or pairwise Granger F-stats), row-normalized, present-masked. Self-loops
  kept (swept off in C5). `_ResidualMessagePassing` already handles asymmetric adjacency (row softmax).
- **Code change.** New `spillover.py` (`_spillover_adjacency(train_vol_panel)` → [33,33] directed) + a
  new adjacency mode `"spillover"` in `_validate_adjacency_config` and `build_masked_graph_manifest`.
  VAR fit on train dates only, frozen (single static spillover graph, per GHAR's stable-graph finding).
  Reuses ladder basis + encoder cache.
- **Evaluation.** Same basis; DM of QLIKE **and RMSE** vs P0/HAR/HARQ. **Most plausibly wins: QLIKE +
  RMSE** (spillover is the RMSE-moving lever in the literature).
- **Effort.** **M.** Spillover module + tests; 3 seeds × 20 ep.
- **Risk / leakage.** DY/VAR estimation on 33 daily series is noisy (the wins are on intraday
  multi-index panels) — treat as an experiment, not a guaranteed lift. **Leakage flag:** the VAR /
  connectedness matrix MUST be fit on the train window only and frozen; never refit per-snapshot using
  future dates.

### Rank 4 — C4: HAR-RV-X richer node features (range + overnight) + GAT + news + QLIKE
- **Hypothesis.** Range estimators (Garman–Klass, Rogers–Satchell, Yang–Zhang) + an overnight
  (close-to-open) variance term improve daily RV forecasts in the HAR-RV-X framework on exactly this
  daily-OHLC input set; overnight is the most consistent, and simpler estimators win. This enriches the
  node with more genuine signal than 3 HAR. [Korkusuz, Kambouroudis & McMillan, FRL 2023 — local PDF in
  `docs/paper/`]
- **Architecture.** Pooled LSTM consumes an expanded node feature set: 3 HAR **+ GK variance + RS
  variance + overnight variance** (all σ², from daily OHLC), `price_dim` 3→6. Edges masked kNN-8 corr.
  News **ON** (also a news-OFF variant to isolate). Loss **QLIKE**.
- **Code change.** Extend `scaling.py:_har_features` to compute the extra columns from OHLC, and thread
  O/H/L/C through `data.py:build_ticker_samples` → `TickerPreprocessor` (currently only the Parkinson
  column flows through). New per-ticker train-only scalers for the added features. Biggest data-plumbing
  change of the set. Encoder cache still valid (recompute once).
- **Evaluation.** Same basis; DM of QLIKE + RMSE vs P0/HARQ. **Most plausibly wins: RMSE + QLIKE.**
- **Effort.** **M–L.** Feature plumbing + scaler wiring; 3 seeds × 20 ep.
- **Risk / units.** **Units trap:** GK/RS/YZ and overnight must be **variance σ²**, not σ, to match the
  Parkinson-variance target and the existing HAR features — a σ/σ² mix silently corrupts the node.
  Leakage: new-feature scalers train-only.

### Rank 5 — C5: Spillover edges + omit self-loops + k-sweep + QLIKE  *(structure sensitivity)*
- **Hypothesis.** Omit self-loops so the backbone/HAR owns own-history and the GNN carries pure
  cross-asset spillover (crypto GNN-HAR decomposition); and k is unpinned for ~30-node universes
  (generic sweet spot 30–80 is non-financial), so sweep it. [research report §2.3, §4 rec 6–7; GARNET
  arXiv:2201.12741]
- **Architecture.** As C3 (directed spillover + news + QLIKE) but adjacency **diagonal zeroed**
  (self-loops removed) and **k swept** over {4, 8, 12, 16, all-but-self}. Report the best-k with DM.
- **Code change.** Small flags in the adjacency builder: `omit_self_loop` (zero diagonal after
  sparsify) and reuse the existing `top_k` parameter (sweep = a few extra runs). **Invariant flag:**
  with self-loops removed, mutual-kNN can produce isolated present nodes (current run already shows
  min-degree 0), which would trip the message-passing "self-loop or neighbor" guard — use **directed
  top-k** (every node keeps exactly k out-edges) or add an isolated-node self-loop fallback.
- **Evaluation.** Same basis; DM per k-setting vs P0. Wins: QLIKE (sensitivity/robustness result).
- **Effort.** **S–M.** Mostly extra runs on C3's plumbing.
- **Risk.** Lower standalone P(beat) than C1–C3 — this is primarily a robustness/ablation config that
  strengthens the paper regardless of outcome, and is cheap once C3 exists.

### Rank 6 — C6: Dynamic / learned adjacency (MTGNN-style) + news + QLIKE  *(speculative novelty)*
- **Hypothesis.** A learned adjacency (graph structure learning) lets the forecast loss shape the
  graph; "dynamic beats static" is sometimes reported. No daily-equity-volatility-vs-HAR precedent, so
  this is a speculative payoff / novelty angle. [Wu et al., MTGNN KDD 2020; Kumar 2024; research report
  §2.2 rank 6, §4 rec 4 — flagged "mixed, often no advantage" in equity vol]
- **Architecture.** Replace the fixed adjacency with a learned one: A = ReLU(tanh(α(E₁E₂ᵀ − E₂E₁ᵀ))),
  top-k sparsified, from per-ticker learnable embeddings (33×d, **input-independent** so no cross-time
  leakage), jointly trained with the message-passing head under QLIKE, present-masked. News **ON**,
  node feats 3 HAR + news.
- **Code change.** New `LearnedAdjacency` module in `models.py`; thread a learned-A path through
  `apply_graph_head` (currently adjacency arrives from the manifest). Moderate new module; encoder cache
  still supplies base embeddings.
- **Evaluation.** Same basis; DM of QLIKE vs P0. Wins: QLIKE (speculative).
- **Effort.** **M–L.** Highest-effort, most speculative → ranked below the evidence-backed levers.
- **Risk / leakage.** Learned A must not encode the target: keep embeddings input-independent (learnable
  per-ticker only). 33×d extra params on a small universe → overfit risk; regularize + val early-stop.

### Rank 7 (contingent) — C7: News-as-EDGE co-mention graph + backbone + QLIKE  *(feasibility-gated)*
- **Hypothesis.** News co-mention edges test the news-as-edge-vs-node ablation that, per the hybrid
  survey, **nobody has published** — a legitimate contribution regardless of outcome. [NIST-GNN,
  Quantitative Finance 2025; Chen et al., arXiv:2306.03763; research report §4 rec 5]
- **Architecture.** Build edges from ticker co-mention counts in the **raw** Vietnamese news corpus over
  the train window (co-mention → edge weight), combined with or replacing the correlation kNN. Pooled
  LSTM + news node features retained. QLIKE loss. Compare news-as-edge vs news-as-node (P2) to isolate.
- **Feasibility gate (do FIRST).** The precomputed news panel is per-(ticker,date) PhoBERT vectors with
  **no article-level multi-ticker structure** (§1). Co-mention edges require the **upstream raw crawl**
  to retain per-article ticker tags. If articles are single-ticker (likely for a per-ticker crawl),
  **this config is INFEASIBLE** and is dropped. Verify before scheduling.
- **Code change.** Conditional on the gate: `_comention_adjacency` from the raw corpus (train-window
  only) + adjacency mode `"comention"`. Moderate.
- **Evaluation.** Same basis; DM of QLIKE vs P0; plus news-edge-vs-node isolation.
- **Effort.** **M**, gated. **Risk:** likely infeasible on the current per-ticker corpus → ranked last.

---

## 3. Recommended execution order (sequential 20-epoch GPU sweep)

Front-loads the cheapest, highest-probability, most-reused levers (QLIKE loss, HAR-residual, spillover
edges), consistent with both research tracks' "do these first" verdicts. The QLIKE-loss plumbing (A)+(B)
is a dependency for steps 1–7, so it is built once at step 1.

1. **C1 — QLIKE-loss GAT + news.** Also delivers the shared (A) differentiable-QLIKE loss and (B)
   epoch-cap change that C2–C6 reuse. Establishes whether the QLIKE lever alone clears the HAR bar.
2. **C2 — HAR + graph-residual + news (QLIKE).** Downside-protected; quick check for a floored win.
3. **C3 — Directed spillover edges + news + QLIKE.** Reuses C1's QLIKE plumbing; the one published
   edge-win lever.
4. **C4 — HAR-RV-X range/overnight node features + news + QLIKE.** Biggest feature plumbing; reuses
   QLIKE. (Watch the σ² units trap.)
5. **C5 — Spillover + omit self-loops + k-sweep + QLIKE.** Cheap variations layered on C3's spillover
   module; robustness/ablation.
6. **C6 — Learned/dynamic adjacency + news + QLIKE.** Most speculative and expensive; run only after the
   evidence-backed levers are exhausted.
7. **C7 — News-as-EDGE co-mention** — ONLY if the raw-corpus multi-ticker feasibility gate passes;
   otherwise dropped and reported as infeasible.

Early-stop rule for the sweep: if C1–C3 all fail to beat P0 on QLIKE DM-significantly, the null result
(GNN ties HAR on daily range-variance) is confirmed and C4–C6 become optional / lower priority, matching
the literature's expectation. A QLIKE-only DM-significant partial win from any of C1–C4 is a legitimate,
reportable success.

---

## 4. Per-config success check (one line each, metric vs HAR + significance)

Every check is on the fair basis (n_test 14464, 3 seeds), all 6 metrics reported val+test; "beat" =
lower QLIKE/RMSE/MAE/MSE or higher R²/DirAcc than the named HAR reference, DM p<0.05 + consistent sign
across seeds + paired-t on seed means.

| Config | Primary success check | Full-win stretch |
|---|---|---|
| C1 QLIKE-loss GAT+news | test QLIKE < P0 0.5676, DM p<0.05 vs P0 & classical HAR | + RMSE < HARQ 0.0022891 |
| C2 HAR+graph-residual | test QLIKE < P0 0.5676, DM p<0.05; RMSE not worse than HAR | QLIKE + RMSE both DM-sig |
| C3 spillover+news+QLIKE | test QLIKE **and** RMSE < P0, DM p<0.05 vs P0/HARQ | + R² > HARQ 0.76682 |
| C4 HAR-RV-X features | test RMSE < HARQ 0.0022891 **or** QLIKE < P0, DM p<0.05 | RMSE + QLIKE both DM-sig |
| C5 self-loop/k-sweep | best-k test QLIKE < P0, DM p<0.05; report k-sensitivity curve | any k beats on QLIKE+RMSE |
| C6 learned adjacency | test QLIKE < P0, DM p<0.05 (else report speculative null) | QLIKE + RMSE DM-sig |
| C7 news-as-edge | feasibility gate passes AND test QLIKE < P0 DM p<0.05; edge-vs-node isolated | QLIKE beat + edge>node |

DM is run per seed on the identical observation vectors (as in the existing `graph_effect_dm` block);
report the across-seed paired-t on QLIKE-deltas plus the per-seed DM stats/p, exactly as
`ladder_consistent…json` already does for the graph effect. An MCS across {HAR, HARQ, all winning
configs} is the recommended final arbiter but is optional for the 20-epoch screening sweep.

---

## 5. Honest expectation

The located literature offers no precedent for a GNN beating a well-specified HAR on daily range-based
variance at ~30-asset scale; the closest rigorous analogue (GNNHAR/DJIA-30) found the graph component
null and located the win in QLIKE loss + nonlinearity. The realistic, defensible outcome of this sweep
is a **QLIKE-only, DM-significant partial win** — most likely from C1 (QLIKE loss) or C2/C3 (residual /
spillover), plausibly helped by the fact that news alone already crosses the QLIKE bar (P2 0.5599 <
P0 0.5676) — with RMSE/R² still tying HARQ. A clean QLIKE partial win, or a well-documented null, are
both reportable; the sweep is ordered so the cheapest evidence-backed levers decide the outcome first.

### Sources
- `docs/reports/2026-08-09_2209_gnn_volatility_beat_har_research.md` (levers §1–4; QLIKE, range/overnight, spillover)
- `docs/reports/2026-08-09_2214_gnn_hybrid_combinations_research.md` (hybrid families §3–4; HAR-residual, spillover, news-as-edge)
- `docs/reports/2026-08-09_2148_old_gat_vs_new_g1_diagnosis.md` (why the old GAT HAR-win was an eval artifact; fair basis)
- `docs/reports/ladder_consistent_h5_2026-08-09_154402.json`; `docs/reports/classical_baselines_h5_2026-08-09_182129.json` (fair-basis numbers)
- Pilot code (`git show feature/masked-gnn:baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/`): `models.py`, `data.py`, `scaling.py`, `train.py`, `ladder_consistent.py`; `src/common/evaluation.py:42` (numpy qlike_loss)
