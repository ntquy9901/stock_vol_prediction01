# Why LSTM beats HAR in the earlier experiments but loses in the SOICT suite — comparison + fixes

Date: 2026-08-21. Question: the seq-lookback (2026-08-19) and cross-market (2026-08-20) studies found
`lstm_only` **beats HAR at h1/h5**; the SOICT suite (2026-08-21) finds `LSTM (w/o GAT)` **loses to HAR**
on VN30/VN100. Same 3 HAR features, same MSE loss, same per-ticker scaling, same QLIKE+DM. What changed?

## 1. Side-by-side setup

| Aspect | seq-lookback (LSTM WINS h1/h5) | sp500 cross-market (LSTM WINS) | **SOICT (LSTM LOSES on VN)** |
|---|---|---|---|
| Data structure | **per-observation pooling** (every ticker×window, ~84k train windows for VN30) | per-observation pooling (~3M windows) | **common-date SNAPSHOTS** (only dates where ALL nodes present → VN30 ~1050 train dates) |
| Split | per-ticker 80/10/10 (retrain on train+val) | per-ticker 70/15/15 | **global-date 80/10/10** |
| Early-stop criterion | fixed budget / QLIKE | **val QLIKE** | **val MSE** |
| Test set | last 10% of each ticker's own history | last 15% per ticker | last 10% of the COMMON recent date window |
| VN30 h1: HAR QLIKE | 0.4642 | — | **0.3946** |
| VN30 h1: lstm QLIKE | 0.4585 (**beats**) | — | 0.4120 (**loses**) |

Note the HAR baseline QLIKE itself changed (0.4642 → 0.3946): the two runs test on **different periods**,
so absolute numbers are not comparable — but the lstm-vs-HAR RELATIONSHIP flipped.

## 2. Root causes (ranked)

1. **Common-date intersection starves the deep model of data + shifts the regime.** SOICT snapshots keep
   only dates where all 33 (VN30) / 104 (VN100) tickers are present, discarding ~60% of ticker-days and
   restricting to a shorter, more recent common window. The earlier runs pooled EVERY window (~84k vs
   ~1050 dates). Less data + a narrower recent test regime → the deep model overfits and regresses to the
   training-mean; HAR (current-feature-driven) is unaffected. Training logs confirm **val MSE ≈ 1.2 > 1.0**
   (worse than the standardized mean) — the deep model is not learning generalizable structure here.
2. **Early-stop on val MSE, not val QLIKE.** The decision metric is QLIKE, but SOICT selects the checkpoint
   by val MSE. Under the regime shift these disagree, so SOICT often keeps a checkpoint that is poor on
   QLIKE. The winning experiments early-stopped on val QLIKE.
3. **Global-date snapshot split vs per-ticker split.** The graph forces a per-date snapshot structure; the
   per-ticker per-observation split (proven) interleaves each ticker's regimes and yields far more, more
   diverse training examples.
4. **The GAT graph is pure overhead** — it hurts in all 8 SOICT configs (independent of the above).

So the SOICT "LSTM loses" is largely an artifact of the **snapshot/common-date/global-split data design
required by the graph**, NOT a property of the LSTM. On the large S&P500 (500 nodes, more common dates)
the SOICT LSTM already recovers and beats HAR at h5 — because more data offsets cause #1.

## 3. Improvement directions (ranked, actionable)

1. **Decouple the deep-vs-HAR test from the graph.** Report `lstm_only` in the PROVEN per-observation,
   per-ticker split (where it beats HAR at h1/h5) as the main "can deep beat HAR" result; report the graph
   as a separate leave-one-out ablation on snapshots (graph hurts). Two clean, honest comparisons instead
   of one confounded one. **Highest value, lowest effort** — the per-observation runner already exists
   (`scripts/sp500_crossmarket` / `run_retrain_trainval`).
2. **Early-stop on val QLIKE** (keep MSE as the training loss). Cheap; aligns selection with the decision
   metric; likely recovers a chunk of the gap.
3. **Drop the common-date intersection: variable-N snapshots.** Build a snapshot per date from whatever
   nodes are present (pad + presence-mask), instead of requiring all nodes → recovers the discarded ~60%
   of data while keeping the graph. Medium effort; the VN Track-A pipeline already does masked snapshots.
4. **Stronger regularization for the small VN data** (higher dropout/weight-decay, smaller hidden) — the
   deep model overfits 33–104 nodes; the effect is small but free.
5. **Drop the graph** for the headline model. Every experiment (this suite + all prior) says the graph
   adds no OOS value and usually hurts; a HAR-LSTM (no graph) is the honest best deep candidate.
6. **Graph reformulation, if a graph must stay:** the glasso partial-correlation edge hurt most; the
   directed vol→PK lead-lag edge had (weak) prior value — but expectations should be low.

## 4. Recommendation

For the SOICT paper: (a) run `lstm_only` under per-ticker per-observation 80/10/10 with **val-QLIKE
early-stop** to get the fair "deep beats HAR at short h" headline (matches the two prior studies), (b)
keep the snapshot-based graph ablation to show the graph hurts, (c) report the data-scaling story
(HAR wins small VN, LSTM wins large S&P500). This turns the confounded negative into two clean, honest,
publishable findings. Directions 1+2 are the immediate next actions.
