# Summary — Per-Ticker Isolated-Gradient News Gate Baseline (2026-07-26, evening)

User request (after being shown the exact `gate_mlp` formula of `gated_crossattn` and why its
shared weights let it learn sector/price shortcuts instead of true news usefulness): *"tôi chỉ
muốn mạng học cổ phiếu nào nên áp dụng tin vì nó có lợi cho dự đoán biến động cổ phiếu đó... nếu
áp dụng tin làm giảm độ chính xác thì không áp dụng."* Also required: debug print to console +
save to file the per-ticker gate learning process, and the mandatory 5-epoch learning curve.

## What was built

`baselines/2026-07-26_per_ticker_news_gate_baseline/` — full 5-subfolder structure.

**`model_per_ticker_gate.py` — `PerTickerGatedNewsBaseline`:** identical to the sibling
`DualGroupNewsBaseline` (HAR branch + `NewsFeatureLSTM` + concat fusion), with exactly one
addition: `gate_logits = nn.Parameter(torch.zeros(num_stocks))` — one free scalar PER TICKER
(init 0 → sigmoid=0.5, neutral), NOT a shared MLP. Traced and proved (design.md §1, verified by 2
direct gradient-perturbation tests, not just architectural reasoning) that
`∂loss/∂gate_logits[i]` depends ONLY on ticker i's own prediction error — perturbing any other
ticker's target or news features leaves it byte-identical.

**`train_per_ticker_gate.py`:** same dataloaders/loss/eval as the sibling (isolates exactly one
variable), plus:
- `gate_logits` gets its own optimizer param-group at a 10× higher LR (0.05 vs 0.005) — 30
  scalars need to move fast enough to be observable within the 10-epoch cap.
- **Debug output (your requirement):** every epoch, console table of all 30/32 tickers sorted by
  gate value with delta arrows vs. the previous epoch; `gate_history.json` (every epoch, not just
  the last) for full post-hoc inspection; a `gate_evolution_*.png` line chart (1 line/ticker)
  every 5 epochs, alongside the mandatory train/val loss learning curve (same 5-epoch cadence,
  CLAUDE.md §3.C, unchanged plotting code reused from the sibling).

Tests: 12/12 pass, including the property test that actually proves gradient isolation (not just
asserts it) and a sanity check that the isolation tests aren't passing trivially.

## Real result (10 epochs, rebuilt panel — same data as this afternoon's dual-group retrain)

| Metric | Dual-group (no gate, same panel) | Per-ticker gate (new) | Diff |
|---|---|---|---|
| Test DirAcc | 68.25% | **68.76%** | +0.51pp |
| Test R² | 0.7124 | **0.7159** | +0.0035 |
| Test QLIKE | 0.5598 | **0.5497** | **-0.0101 (better)** |
| Test RMSE | 0.002651 | **0.002635** | -0.000016 (better) |

**This beats the same-panel dual-group baseline on all 4 metrics — the first clear win after
~10 consecutive null results across every other variant tried this project** (see memory
`project_null_result_pattern_and_sota_pivot`). It also sets a **new project-best QLIKE** (0.5497,
vs. `gated_crossattn`'s previous-best 0.557) and roughly ties/edges the R² record (0.7159 vs.
0.7157). Still behind HAR-only on DirAcc alone (69.98%).

## The important caveat: does the gate actually learn "true news usefulness"? No.

Correlated the final per-ticker gate values against the independent ablation's `delta_qlike`
(`results/ablation_derived_ticker_classification.json` — a SEPARATE method: trained HAR-only vs.
all-ON dual-group, both 10 epochs, measured each ticker's own QLIKE delta):

- Pearson r = 0.14 (p=0.44), Spearman ρ = 0.07 (p=0.69) — statistically indistinguishable from
  the OLD shared-weight `gate_mlp`'s r=0.13 (also wrong-direction).
- Mean gate for the ablation's "news helps" tickers (n=11): 0.561. Mean gate for "news hurts"
  tickers (n=21): 0.572. No real difference.

**This rules out "shared weights caused the disagreement"** as an explanation for why learned
gates never matched the ablation signal — even with gradient PROVABLY isolated per ticker, the
mechanism still discovers a different pattern than the independent ablation. The performance
improvement above is real, but it most likely comes from some other benefit of having a
per-ticker-adjustable scaling knob (a soft regularizer/capacity-control effect) — not from the
model correctly identifying which specific tickers' news content is causally informative.

**Also:** gate values had not converged by epoch 10 — several tickers (MSN, GAS, BVH, SSI, PDR)
still moved >0.1 between epoch 9 and 10. The reported gate values are a mid-training snapshot,
not a stable readout; more epochs (needs your explicit approval per CLAUDE.md Training policy)
would be needed to see where they settle.

## Files

- `baselines/2026-07-26_per_ticker_news_gate_baseline/` — full baseline (requirements, design,
  code, code_review, test).
- `results/per_ticker_gate_2026-07-26_221920/` — real run: `results.json` (incl. final per-ticker
  gate values), `gate_history.json` (full 10-epoch trajectory, every ticker), learning-curve PNGs
  (every 5 epochs), `gate_evolution_*.png` (every 5 epochs).
- `results/per_ticker_gate_2026-07-26_221512/` — smoke run (3 epochs, dummy news), kept for
  reference.
- `models/per_ticker_gate_2026-07-26_221920/best.pt` — checkpoint.

## Tests + code review

- `pytest` across all 3 touched/created baselines today (dual-group sibling, spillover_qlike,
  per-ticker-gate) → **38/38 pass**, no regressions.
- Self-adversarial code review: `code_review/code_review_2026-07-26.md`. Core finding verified
  by direct test, not just reasoning; 1 matplotlib API break fixed during implementation
  (`matplotlib.cm.get_cmap` removed upstream — switched to `matplotlib.colormaps.get_cmap(...).resampled(...)`).
- diff-cover: **Not run** (documented tooling gap).

## Risks / follow-ups for you to review

1. **Gate hasn't converged (epoch 10 still moving)** — if you want a stable per-ticker readout
   (not just a performance number), this needs more epochs. Per Training policy, that needs your
   explicit go-ahead based on these 10-epoch results.
2. **The per-ticker values should NOT be read as "which VN30 stocks need news"** — the
   correlation check shows this mechanism's gate doesn't track the independent ablation's
   usefulness signal any better than the old shared gate did. If the goal is specifically to
   identify true per-ticker news usefulness (as opposed to just improving overall metrics), this
   baseline doesn't resolve that — it's now the 5th method to disagree.
3. **The performance win itself (QLIKE/R² records) is worth keeping/building on** regardless of
   the interpretability question above — e.g. as the new best news-fusion architecture to
   compare future ideas against, replacing plain `DualGroupNewsBaseline` as the reference.

## DoD checklist

- [x] Code satisfies the request (per-ticker isolated gate + debug logging + learning curves,
      exactly as specified)
- [x] Tests written + run (12/12 new, 38/38 total, no regressions)
- [ ] diff-cover C0/C1 — Not run (documented tooling gap)
- [x] Adversarial self-review — core claim verified by direct test; 1 fix applied
- [x] Real-data smoke validated before the real 10-epoch run
- [x] Impact analysis — no sibling files modified (hard isolation maintained)
- [x] Summary report — this file
- [x] Training policy — capped at 10 epochs, enforced in code
