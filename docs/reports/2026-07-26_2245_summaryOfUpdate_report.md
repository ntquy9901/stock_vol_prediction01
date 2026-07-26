# Summary — Per-Ticker News Gate: Resumed Training to Epoch 20 (2026-07-26)

User request: "train tiếp 10 epoches nữa" (continue training 10 more epochs) after reviewing the
epoch-10 result of `2026-07-26_per_ticker_news_gate_baseline`.

## What changed in code

Added resume support to `train_per_ticker_gate.py` (not present before — the previous run could
only start fresh):
- `--resume_checkpoint` (previous `best.pt`) + `--resume_results_dir` (previous run's
  `results/` dir) — loads model weights and continues epoch numbering (11→20, not reset to 1).
- New `load_resume_state()` helper reads the previous run's `gate_history.json` to compute the
  correct start epoch, and a previous `loss_history.json` if present (added `loss_history.json`
  saving now too, so any FUTURE resume has a fully continuous loss curve — this first resume's
  loss curve only covers epochs 11-20 since epochs 1-10's raw losses weren't persisted before
  this fix).
- 4 new tests for `load_resume_state` (no-resume default, missing-file error, start-epoch
  computed as MAX epoch key not `len()`, loss-history loaded when present) — **16/16 total pass**.

## Real result: epoch 10 vs. epoch 20 (same run, continued)

| Metric | Epoch 10 | Epoch 20 | Diff |
|---|---|---|---|
| Test DirAcc | 68.76% | 68.90% | +0.14pp |
| Test R² | 0.7159 | 0.7154 | -0.0005 (flat) |
| Test QLIKE | 0.5497 | **0.5473** | -0.0024 (new best, still improving) |
| Test RMSE | 0.002635 | 0.002637 | +0.000002 (flat) |

Aggregate metrics are essentially **plateaued** — QLIKE ticked down slightly further (new
project-best), everything else is flat within noise. 10 more epochs did not produce a large
further gain, but also did not regress.

## The per-ticker gate values are NOT stable — this is the important part

Comparing each ticker's gate value at epoch 10 vs. epoch 20 (same continued run):

| Ticker | Epoch 10 | Epoch 20 | Change |
|---|---|---|---|
| BID | 0.26 | 0.65 | **flipped low→high** |
| HDB | 0.77 | 0.43 | **flipped high→low** |
| VNM | 0.35 | 0.63 | flipped low→high |
| TPB | 0.51 | 0.76 | shifted strongly up |
| SHB | 0.66 | 0.86 | shifted up |
| PLX | 0.20 | 0.05 | shifted further down |

Overall epoch-10-vs-epoch-20 gate correlation across all 32 tickers: **r=0.79** (moderate-high —
the general ranking is roughly preserved) but individual tickers like BID/HDB/VNM swing by
0.3-0.4, enough to flip which "half" (news-helps vs. news-hurts) they'd be classified into. **This
means a single-epoch snapshot of gate values should not be read as a settled, trustworthy
per-ticker signal** — which ticker looks "high gate" depends materially on which epoch you stop
at, even while the aggregate loss stays roughly flat.

**Correlation with the independent ablation's `delta_qlike` also re-checked at epoch 20:**
Pearson r=0.28 (p=0.12), Spearman ρ=0.25 (p=0.17) — higher than epoch 10's r=0.14, trending in the
"more consistent" direction, but still not statistically significant (p>0.1) and still far from a
reliable signal.

## Interpretation

- The **architecture** (per-ticker isolated gate) continues to be the project's best QLIKE/R²
  news-fusion result — that finding from epoch 10 holds and improved slightly further.
- The **specific per-ticker gate values are still in flux** even after 20 epochs — not a reason to
  distrust the aggregate result, but a strong reason NOT to read individual ticker gate values
  (at epoch 10 OR 20) as "the model has decided ticker X needs news." More epochs, multiple
  seeds, or an explicit convergence criterion (e.g., stop when gate deltas fall below a threshold
  for N consecutive epochs) would be needed before individual gate values could be trusted as a
  per-ticker usefulness signal — none of that was implemented here (out of scope for this
  session, flagged as a follow-up below).

## Files

- `results/per_ticker_gate_2026-07-26_223428/` — epoch 11-20 continuation: `results.json`,
  `gate_history.json` (full 1-20, continuous), `loss_history.json` (11-20 only, see above),
  learning-curve + gate-evolution PNGs.
- `models/per_ticker_gate_2026-07-26_223428/best.pt` — new checkpoint (best across epoch 11-20).

## Tests

`pytest baselines/2026-07-26_per_ticker_news_gate_baseline/test/` → **16/16 pass** (12 previous +
4 new for the resume helper).

## Risks / follow-ups

1. **Gate instability is itself a finding worth investigating further** — if you want to pursue
   per-ticker news usefulness via this mechanism seriously, the next step should be a stability
   check (train from 2-3 different random seeds, see if the SAME tickers end up high/low, or if
   it's arbitrary per run) before trusting any specific ticker's gate value.
2. `loss_history.json` is now saved going forward — any further "train N more epochs" request
   will have a fully continuous loss curve across the resume boundary (this one only has epochs
   11-20 for the loss plot specifically; gate history is fully continuous 1-20).
3. Per Training policy, further epochs beyond 20 would again need your explicit go-ahead.
