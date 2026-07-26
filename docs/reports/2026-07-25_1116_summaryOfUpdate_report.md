# Summary Report — News Usefulness Ablation (2026-07-25)

**Baseline:** `baselines/2026-07-25_news_usefulness_ablation/`
**Type:** Analysis/measurement baseline (not itself a comparison baseline) — derives a ticker
ON/OFF classification from the project's own LSTM-GNN architecture, replacing the
HGB/XGBoost-derived EDA that failed to transfer in the previous two baselines today.

## What changed

Trained a fresh HAR-only reference (`ParallelLSTMGNN`, own fusion, not frozen) on the exact same
data pipeline as `2026-07-25_dual_group_news_embedding_baseline` (same 32 stocks, same windows —
guaranteed by calling `create_dual_news_dataloaders(news_panel_path=None)`, since x_har/adj/y
don't depend on the news panel). Compared its per-ticker QLIKE/MSE/DirAcc against the existing
all-32-stocks-ON dual-group checkpoint, epoch-matched at 10 epochs each, to derive a per-ticker
"does news help" delta measured WITHIN this architecture — no more borrowing signal from a
different model family.

### Files

| Path | Purpose |
|---|---|
| `code/train_har_only_reference.py` | Trains the epoch-matched HAR-only reference, saves per-ticker QLIKE/MSE/DirAcc |
| `code/eval_checkpoint_per_ticker.py` | Loads an existing dual-group checkpoint, computes per-ticker metrics (no training) |
| `code/compute_ablation_deltas.py` | Computes per-ticker delta, ranks, classifies ON/OFF (QLIKE primary) |
| `code_review/code_review_2026-07-25.md` | 1 HIGH finding (epoch-mismatch confound), fixed |

## Result — derived ticker classification

**NEWS-ON (11):** HDB, HPG, MWG, NVL, PDR, PLX, SSI, VHM, VJC, VPB, VRE
**NEWS-OFF (21):** ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, MBB, MSN, POW, SAB, SHB, SSB, STB, TCB,
TPB, VCB, VIB, VIC, VNM

11/11 ON tickers agree on both QLIKE and MSE improvement (high internal consistency); only 4/11
also show DirAcc improvement (DirAcc confirmed noisy at this sample size, consistent with
observations in the two prior baselines today).

**Notable divergence from the HGB/XGBoost EDA:** ACB and VIB — 2 of the EDA's "strongest 3"
signals (used in `2026-07-25_top3_news_gate_baseline`) — land in NEWS-OFF here. PLX (flagged
clearly negative in the EDA) lands in NEWS-ON here. The two measurement approaches disagree
substantially on WHICH tickers benefit, though both agree the overall effect is modest. This is
expected (different model families capture different signal) but worth flagging plainly:
neither list should be treated as ground truth without further validation.

## Bug found and fixed (methodology, not code correctness)

First delta computation compared a 40-epoch (converged) all-ON checkpoint against a 10-epoch
HAR-only reference — confounded by training budget (26/32 tickers looked "ON", implausibly
high). Fixed by re-evaluating against the epoch-matched 10-epoch all-ON checkpoint
(`models/dual_group_news_2026-07-25_011719/best.pt`), which changed the result to a much more
selective 11 ON / 21 OFF split. See `code_review/code_review_2026-07-25.md`.

## Honest caveat

Single-seed (1 training run per model). Per-ticker deltas can still be noisy given the small test
set (~163 windows/ticker) — this is an improvement in METHODOLOGY (measuring within the actual
architecture) over the EDA approach, not a guarantee of a stable signal. If the downstream gated
baseline's result is ambiguous, multi-seed averaging would be the next rigor step.

## Commands run

```
python train_har_only_reference.py --epochs 10
python eval_checkpoint_per_ticker.py --checkpoint models/dual_group_news_2026-07-25_011719/best.pt
python compute_ablation_deltas.py
```

## Definition of Done checklist

- [x] Requirements + design docs
- [x] Scripts run successfully (fixed 1 JSON-serialization bug + 1 epoch-mismatch methodology bug)
- [x] Code review (self-directed) — 1 HIGH found and fixed before downstream use
- [x] Summary report (this file)
- [ ] diff-cover — Not run (pre-existing tooling gap)
- [x] Impact analysis — no edits to sibling baselines, shared `src/`, or existing checkpoints; new HAR-only reference trained fresh, doesn't overwrite anything
