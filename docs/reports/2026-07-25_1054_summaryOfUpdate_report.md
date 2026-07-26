# Summary Report — Top-3 News Gate Baseline (2026-07-25)

**Baseline:** `baselines/2026-07-25_top3_news_gate_baseline/`
**Type:** New baseline, autonomous session (2nd iteration of the selective-news-gate idea today).

## What changed

Narrowed the selective news gate (`2026-07-25_selective_news_gate_baseline`, 22 ON tickers,
which contradicted its own hypothesis) down to just **3 tickers with the strongest, most
consistent 4-horizon average delta-R^2** in the EDA report (§4 "Nhóm 1"): VIB (+0.914), ACB
(+0.707), MWG (+0.560). SHB (+3.124, highest) excluded per user decision — suspected time-proxy
artifact. All other 29 tickers (of the actual 32-ticker training universe) get bias=0.

### Files

| Path | Purpose |
|---|---|
| `code/model_top3_gate.py` | `Top3NewsGateBaseline(SelectiveGateNewsBaseline)` — overrides the mask to a strict 3-ticker allowlist |
| `code/train_top3_gate.py` | Train loop, ON(3)/OFF(29) group DirAcc breakdown |
| `test/test_mask_correctness.py` | 4 tests: allowlist correctness + exact-zero contribution for non-top-3 stocks |
| `code_review/code_review_2026-07-25.md` | 0 code findings; the negative/neutral result is discussed as an empirical finding, not a bug |

## Tests + coverage

`pytest baselines/2026-07-25_top3_news_gate_baseline/test/ -v` → **4/4 passed**. diff-cover: Not run (pre-existing gap).

## Results — 10 epochs

| Split | MSE | RMSE | MAE | R² | QLIKE | DirAcc (overall) | DirAcc (ON, 3 tickers) | DirAcc (OFF, 29 tickers) |
|---|---|---|---|---|---|---|---|---|
| Val | 0.000006 | 0.002446 | 0.000724 | 0.661 | 0.697 | 69.72% | 52.35% | 46.75% |
| Test | 0.000007 | 0.002643 | 0.000720 | 0.714 | 0.559 | **68.23%** | **48.67%** | **48.89%** |

Per-ticker test DirAcc for the 3 ON tickers: VIB 53.37%, MWG 47.24%, ACB 45.40%.

## Interpretation — inconclusive, not a repeat of the prior contradiction

Unlike the 22-ticker version (clear contradiction: OFF beat ON by 5.3pp on test), this narrower
3-ticker version is **essentially a tie on test** (ON 48.67% vs OFF 48.89%, -0.22pp — noise-level).
The val set showed a promising +5.6pp gap in the hypothesized direction, but this did **not
replicate on test** — a classic small-sample instability signal: with only 3 tickers in the ON
group, the group average is dominated by per-ticker variance (VIB/MWG/ACB individually span an
8pp range on test). Overall DirAcc (68.23%) is in the same narrow band as every other
dual-group-news variant tried today (68.25-68.71%), and still below HAR-only (69.98%).

**Bottom line across both selective-gating experiments today:** neither the broad (22-ticker,
ΔR²@t+5≥0.01) nor the narrow (3-ticker, highest 4-horizon avg ΔR²) selection from the HGB/XGBoost
EDA produced a convincing positive signal when transplanted into this shared LSTM-GNN
architecture. The 22-ticker version was actively contradicted; the 3-ticker version is a coin
flip. This strengthens (doesn't just repeat) the conclusion already recorded in memory: per-ticker
news usefulness from a different model family does not reliably transfer here — regardless of
how the evidence threshold is set.

## Commands run

```
python -m pytest baselines/2026-07-25_top3_news_gate_baseline/test/ -v   # 4/4 passed
python train_top3_gate.py --epochs 10                                    # real run
```

## Risks / follow-ups

- **Not extending to 20/40 epochs** — same reasoning as the sibling baseline: the qualitative
  conclusion (no reliable signal from this ticker-selection approach) is unlikely to flip with
  more epochs; asking the user before further training.
- Given 2 different evidence thresholds (broad and narrow) both failed to produce a clear win,
  the most promising remaining direction (already suggested in the prior report) is deriving
  ON/OFF from THIS architecture's own ablation rather than porting the HGB/XGBoost signal again.

## Definition of Done checklist

- [x] Code satisfies the request (news gate narrowed to VIB/ACB/MWG only, bias=0 elsewhere)
- [x] Tests written + run (4/4 pass)
- [ ] diff-cover — Not run (pre-existing gap)
- [x] Lint — Not run (pre-existing gap); no obvious style issues on manual read
- [x] Code review (self-directed adversarial) — 0 findings, result honestly reported as inconclusive
- [x] Summary report (this file)
- [x] Smoke test(s) pass (tagged `smoke`)
- [x] Impact analysis — no edits to sibling baselines or shared `src/`; read-only imports only
- [x] Similar-pattern check — mirrors `2026-07-25_selective_news_gate_baseline` exactly except the ticker list
