# Summary Report — Selective News Gate Baseline (2026-07-25)

**Baseline:** `baselines/2026-07-25_selective_news_gate_baseline/`
**Type:** New baseline, autonomous session (continuation of the dual-group news embedding work
from earlier today).

## What changed

Built a new baseline that applies the news branch (from
`2026-07-25_dual_group_news_embedding_baseline`) **selectively per ticker**, based on a
per-ticker EDA (`docs/suggestion/2026-07-25_professor_report.md`, HGB/XGBoost, delta-R^2 at the
t+5 horizon specifically). A fixed (non-learnable) 0/1 mask zeroes `news_rep` after the LSTM,
before the fusion concat, for tickers with no positive news evidence.

**Ticker classification (22 ON / 10 OFF, out of 32 in the actual training universe):**
- NEWS_ON: ACB, MWG, VIB, TPB, SAB, VJC, MBB, POW, TCB, MSN, HPG, VIC, SSB, BID, SSI, STB, FPT,
  VCB, CTG, GVR, VNM, HDB
- NEWS_OFF: SHB (user-excluded, suspected time-proxy artifact), GAS, PLX, NVL, BVH (negative
  delta-R^2), VHM, BCM, PDR (negligible delta-R^2), VPB, VRE (not covered by the EDA — the
  actual pipeline's 32-stock universe has 2 more tickers than the EDA's 30; defaulted OFF,
  conservative direction, found via a fail-loud check on the first real run).

### Files (path → purpose)

| Path | Purpose |
|---|---|
| `requirements/requirements.md` | Ticker list + rationale, success criteria (per-group DirAcc comparison) |
| `design/design.md` | Mask-after-LSTM design, file list, isolation |
| `code/model_selective_gate.py` | `SelectiveGateNewsBaseline` (subclass of sibling's `DualGroupNewsBaseline`) + `build_stock_mask` |
| `code/train_selective_gate.py` | Train loop, reuses sibling's dataloaders, adds per-ticker + NEWS_ON/OFF group DirAcc breakdown |
| `test/test_mask_correctness.py` | Proves exact-zero news contribution for NEWS_OFF stocks (not approximate) |
| `test/test_model_smoke.py` | Forward/backward, shape, no NaN, mask is a non-learnable buffer |
| `code_review/code_review_2026-07-25.md` | 1 MEDIUM finding (ticker-universe mismatch), fixed |

## Tests + coverage

`pytest baselines/2026-07-25_selective_news_gate_baseline/test/ -v` → **6/6 passed**.
diff-cover: Not run (tooling gap, pre-existing).

## Code review result + actions

Self-directed adversarial review (same rationale as the sibling baseline: `/code-review` expects
a GitHub PR). 1 MEDIUM finding: the EDA's 30-ticker classification didn't cover 2 tickers (VPB,
VRE) that the actual training pipeline uses — caught by the mask builder's own fail-loud check on
first run, fixed by defaulting them OFF (conservative). Full details in
`code_review/code_review_2026-07-25.md`.

## Results — 6 mandatory metrics + per-group breakdown (10 epochs)

| Split | MSE | RMSE | MAE | R² | QLIKE | DirAcc (overall) | DirAcc (NEWS_ON avg) | DirAcc (NEWS_OFF avg) |
|---|---|---|---|---|---|---|---|---|
| Val | 0.000006 | 0.002440 | 0.000719 | 0.663 | 0.700 | 69.64% | 47.66% | 45.03% |
| Test | 0.000007 | 0.002656 | 0.000715 | 0.711 | 0.561 | **67.56%** | **46.29%** | **51.60%** |

## Interpretation — the hypothesis is NOT confirmed

**The core premise is contradicted by the result.** The hypothesis (turning news OFF for
EDA-flagged low-benefit tickers should help, or at least not hurt) predicted NEWS_ON tickers
would show higher DirAcc than NEWS_OFF. The opposite happened: **NEWS_OFF tickers averaged
51.60% test DirAcc vs. NEWS_ON's 46.29%** — a 5.3pp gap in the wrong direction. Overall DirAcc
(67.56%) is also lower than both the unmasked dual-group baseline (68.50% @10ep) and HAR-only
(69.98%).

**Most likely explanation:** the EDA's per-ticker ΔR² was computed with a completely different
model family (HGB/XGBoost, per-ticker independent regression) and feature set (price+news_adv_full,
~500 cols) than this baseline's shared multi-stock LSTM-GNN (32 stocks jointly, GAT-mixed HAR
embeddings, only the 146-col dual-group+EWMA subset). "Usefulness of news for ticker X" measured
in one model family does not necessarily transfer to a different, jointly-trained architecture —
a plausible, if disappointing, negative result. A secondary factor: per-ticker test DirAcc is
computed over only ~163 diffs (164 test sequences per ticker), so individual ticker numbers (see
`results/selective_gate_2026-07-25_102926/results.json`'s `per_ticker_test_dir_acc`) are noisy;
the 22-vs-10 group averages are more stable than any single ticker's number, but still a small
sample for a 5.3pp gap to be fully conclusive.

## Commands run

```
python -m pytest baselines/2026-07-25_selective_news_gate_baseline/test/ -v   # 6/6 passed
python train_selective_gate.py --epochs 1 --smoke                             # wiring check
python train_selective_gate.py --epochs 10                                    # real run
```

## Risks / follow-ups

- **Do not extend to 20/40 epochs by default** — CLAUDE.md training policy requires explicit
  user approval beyond 10, and given the hypothesis is already contradicted at 10 epochs, more
  training is unlikely to change the qualitative conclusion (per-group ranking, not magnitude, is
  the finding). Asking the user before any further training.
- If this line of investigation continues, the next scientifically sounder step would be deriving
  the ON/OFF split from THIS architecture's own per-ticker ablation (train with vs. without news
  per ticker, using this LSTM-GNN, not the HGB/XGBoost EDA) rather than porting another model
  family's feature-importance signal.
- VPB/VRE's OFF default is untested against evidence (no EDA data either way) — flagged, not
  a blocker.

## Definition of Done checklist

- [x] Code satisfies the request (fixed per-stock news mask from EDA, bias=0 for excluded stocks)
- [x] Tests written + run (6/6 pass), including a numerical-exactness test for the mask
- [ ] diff-cover C0=100%/C1≥80% — Not run (tooling gap, pre-existing)
- [x] Lint — Not run (ruff not installed, pre-existing gap); no obvious style issues on manual read
- [x] Code review (self-directed adversarial) — 1 MEDIUM found and fixed
- [x] Summary report (this file)
- [x] Smoke test(s) pass (tagged `smoke`)
- [x] Impact analysis — no edits to sibling baseline or shared `src/`; read-only imports only
- [x] Similar-pattern check — n/a (first "selective per-ticker gate" baseline in the project)
