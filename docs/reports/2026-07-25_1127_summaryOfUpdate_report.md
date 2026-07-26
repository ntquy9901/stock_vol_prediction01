# Summary Report — Ablation-Derived Gate Baseline (2026-07-25)

**Baseline:** `baselines/2026-07-25_ablation_derived_gate_baseline/`
**Type:** New baseline, 3rd and final ticker-gating iteration today, using a ticker list derived
from the project's OWN LSTM-GNN architecture (`2026-07-25_news_usefulness_ablation`) instead of
the HGB/XGBoost EDA used by the previous two.

## What changed

Applied the fixed-mask mechanism (from `2026-07-25_selective_news_gate_baseline`) using the
11-ticker ON list derived from an epoch-matched (10-vs-10), QLIKE-based ablation within this
project's actual LSTM-GNN architecture: HDB, HPG, MWG, NVL, PDR, PLX, SSI, VHM, VJC, VPB, VRE.
All other 21 tickers get bias=0.

### Files

| Path | Purpose |
|---|---|
| `code/model_ablation_gate.py` | `AblationDerivedGateBaseline(SelectiveGateNewsBaseline)` — 11-ticker allowlist from the ablation |
| `code/train_ablation_gate.py` | Train loop, ON(11)/OFF(21) group DirAcc breakdown |
| `test/test_mask_correctness.py` | 4 tests, same exact-zero-contribution pattern as siblings |
| `code_review/code_review_2026-07-25.md` | 0 code findings; result discussed honestly |

## Tests

`pytest baselines/2026-07-25_ablation_derived_gate_baseline/test/ -v` → **4/4 passed**.

## Results — 10 epochs

| Split | MSE | RMSE | MAE | R² | QLIKE | DirAcc (overall) | DirAcc (ON, 11) | DirAcc (OFF, 21) |
|---|---|---|---|---|---|---|---|---|
| Val | 0.000006 | 0.002442 | 0.000721 | 0.662 | 0.702 | 69.14% | 47.35% | 47.68% |
| Test | 0.000007 | 0.002657 | 0.000718 | 0.711 | 0.562 | **68.23%** | **50.47%** | **47.33%** |

## Interpretation — the most encouraging of 3 attempts, still not conclusive

**DirAcc direction is correct this time:** NEWS_ON tickers (50.47%) beat NEWS_OFF (47.33%) by
+3.1pp on test — reversing the 22-ticker EDA gate's clear contradiction (OFF beat ON by 5.3pp)
and improving on the 3-ticker EDA gate's tie. This is consistent with the idea that measuring
"does news help ticker X" WITHIN the actual architecture (not borrowed from HGB/XGBoost) gives a
more trustworthy signal.

**But QLIKE — the metric this list was actually selected on — is NOT a clear win:** 0.5623,
essentially tied with the epoch-matched HAR-only reference (0.5623) and only marginally better
than the all-ON model (0.5652 @10ep). The list was chosen by delta_QLIKE > 0 per ticker, but the
resulting model's AGGREGATE QLIKE doesn't clearly beat either comparison point. Overall DirAcc
(68.23%) also remains below HAR-only (69.98%) and in the same narrow band as every other
dual-group variant tried today (68.23-68.71%).

**Per-ticker noise is still substantial:** within the 11 ON tickers, individual test DirAcc spans
35.58% (HPG) to 77.91% (VHM) — a >40pp range. The group-average signal (+3.1pp) is a real
improvement in direction over the two EDA-based attempts, but is still built from a highly
variable per-ticker foundation at this sample size (~163 windows/ticker).

## Overall conclusion across all 4 baselines built today (selective/ablation gating)

| Baseline | ON tickers | Test DirAcc (ON avg) | Test DirAcc (OFF avg) | Overall Test DirAcc |
|---|---|---|---|---|
| All-ON (no gate) | 32 (all) | — | — | 68.50% |
| 22-ticker EDA gate | 22 | 46.29% | 51.60% | 67.56% (contradicted) |
| 3-ticker EDA gate | 3 | 48.67% | 48.89% | 68.23% (tie) |
| 11-ticker ablation gate | 11 | 50.47% | 47.33% | 68.23% (right direction) |

No gating variant beats the unmasked all-ON model's overall DirAcc (68.50%), and none beat
HAR-only (69.98%). The internal-ablation-derived list is directionally the most promising signal
of the three, but the overall-metric improvement is within noise. **Recommendation: this line of
investigation (selective per-ticker gating) has not yet produced a baseline worth promoting over
the plain all-ON dual-group model.** If pursued further, multi-seed ablation (not single-seed)
would be the necessary next step before trusting any per-ticker list enough to report as a
finding rather than an exploratory signal.

## Commands run

```
python -m pytest baselines/2026-07-25_ablation_derived_gate_baseline/test/ -v   # 4/4 passed
python train_ablation_gate.py --epochs 10                                       # real run
```

## Risks / follow-ups

- Not extending to 20/40 epochs without user approval (CLAUDE.md policy) — the qualitative
  picture (modest, noisy, non-decisive) is unlikely to change materially.
- Multi-seed ablation (train HAR-only + all-ON multiple times with different seeds, average the
  per-ticker deltas) is the concrete next step if this direction is worth more investment.

## Definition of Done checklist

- [x] Code satisfies the request (ablation-derived 11-ticker gate, bias=0 elsewhere)
- [x] Tests written + run (4/4 pass)
- [ ] diff-cover — Not run (pre-existing gap)
- [x] Code review (self-directed) — 0 findings, result reported honestly (encouraging but inconclusive)
- [x] Summary report (this file)
- [x] Impact analysis — no edits to sibling baselines or shared `src/`
