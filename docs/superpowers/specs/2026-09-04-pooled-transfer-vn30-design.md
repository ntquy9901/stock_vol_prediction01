# Design spec — Pooled/transfer ablation for VN30 (train-universe 31 vs 102)

Date: 2026-09-04. Status: approved for planning. Type: new baseline (SDD §3.F).
Baseline dir: `baselines/2026-09-04_pooled_transfer_vn30/`.

## 1. Objective & hypotheses

Scientific ablation: does widening the training universe from 31 stocks (VN30) to 102 stocks
(VN100) overcome VN30's data-scarcity barrier for the deep volatility forecasters? Report the
answer honestly regardless of sign — a null is a valid contribution.

- **H0 (null; prior A1 2026-08-08 found pooling did not help deep beat HAR):** VN30 forecast
  metrics of the deep model do not change materially when 71 extra stocks are added to training.
- **H1:** training on the wider universe improves the deep model's VN30 forecasts.

Motivating evidence (clean-data walk-forward, this project): on VN30 (31 nodes) HAR/HAR-X win
pooled QLIKE at h1/h5/h22; VolGA−LSTM graph value is not significant at any horizon
(p = 0.179 / 0.112 / 0.265 / 0.928). On VN100 (102 nodes) the graph is significant at h1/h5.
The gap is attributed to node breadth × liquidity, not correlation magnitude — so more training
breadth is the variable under test here.

Prior caveat that must be stated in the report: Track B ablation A1 (2026-08-08) already tested
"more data via pooling" and found <1% metric movement with mixed sign at a 5-epoch / 3-seed /
dirty-data / fixed-split screening budget. This ablation differs: clean enriched data,
walk-forward, all horizons, 5 seeds, VolGA graph, and cross-universe (31→102) rather than
async-date pooling within one universe.

## 2. Experimental design — exactly one independent variable

- **Independent variable:** training universe ∈ {VN30 (31 nodes), VN100 (102 nodes)}.
- **Held fixed:** score set = the 31 VN30 tickers, the OOS date grid, the fold structure,
  lookback = 22, 5 seeds, the four models (HAR / HAR-X / LSTM / VolGA), training config
  (epochs 16, patience 5, min-epochs 5, batch 32, qlike_floor from config).

| | Arm 0 (baseline) | Arm 1 (pooled) |
|---|---|---|
| Train nodes | 31 (VN30) | 102 (VN100) |
| VolGA graph (per-fold, train-only) | 31-node vol→PK | 102-node vol→PK |
| Score nodes | 31 VN30 | 31 VN30 (subset) |
| OOS (ticker, date) | identical | identical |

VN30 ⊂ VN100, so Arm 1 trains on VN30 plus 71 additional stocks whose only role is training
context; they never enter any test metric.

## 3. Data flow — shared fold calendar (the crux)

1. Build the fold structure **once** from the VN100 panel, with boundaries expressed by **date**
   (test-start date + retrain dates, K = 21 trading days, 22 folds), not by integer index.
2. **Arm 1:** VN100 panel (102 nodes) → train all, 102-node graph → **score only the 31 VN30
   tickers.**
3. **Arm 0:** VN30 panel (31 nodes) with the **same date boundaries imposed** → train 31, 31-node
   graph → score the 31 tickers.
4. Because both panels share the trading calendar and boundaries are date-based, the 31 VN30 OOS
   `(ticker, date)` points align 1-to-1 across arms → clean paired Diebold–Mariano.

Cleanliness property: each VN30 ticker's per-node feature/target scaler is fit on that ticker's
own training history, so it is **identical across arms**. The only differences between arms are
(a) the shared model weights (trained on 31 vs 102 nodes) and (b) the graph breadth. This is the
intended single-variable contrast ("combined" data+graph effect, per the 2-arm choice; isolating
data vs graph would need a third arm and is deferred).

## 4. Leakage controls (reuse from delivered VolGA)

- vol→PK graph and every scaler fit **train-only per fold**, frozen for val/test
  (`assert_no_leakage` reused).
- Arm 1's 71 non-VN30 stocks appear only as training context; targets and scoring are restricted
  to VN30, so no non-VN30 node ever enters a test metric and no cross-stock target leaks.

## 5. Metrics & decision rule (fixed before running)

- **Headline (A) — does the deep model self-improve:** paired date-clustered DM, **Arm 1 vs
  Arm 0**, computed separately for VolGA and for LSTM, on the identical VN30 OOS points, on all
  **three loss bases (QLIKE, squared error, absolute error)**, per horizon. "Pooling helps the
  deep model" ⟺ Arm 1 is significantly better (p < 0.05).
- **Secondary (B) — position vs HAR:** difference-in-differences of gap(deep − HAR) in Arm 0 vs
  Arm 1, plus an absolute-QLIKE table (4 models × 2 arms × 4 horizons). Indicates whether pooling
  narrows or closes the deep-vs-HAR gap. HAR/HAR-X are pooled-OLS over the training universe, so
  the HAR bar itself shifts between arms; this is reported, not hidden.
- **Mandatory evidence:** per-arm train/val/test metrics + `fit_diagnostics` + learning curves
  (reuse the overfit-evidence gate).
- **Honest verdict:** state H0 vs H1 explicitly; do not select the result on the test set (rule
  fixed here, a priori).

## 6. Code changes (hard-isolated new baseline, §3.F)

New baseline `baselines/2026-09-04_pooled_transfer_vn30/` with the five subfolders
(requirements / design / code / code_review / test). Reuse **read-only**: `wf_enriched_panel`,
`run_masked_rich` (VolGA trainer + metric/DM/evidence helpers), `run_volga_walkforward` fold/leakage
machinery. New code only:

- **Date-based fold boundaries:** helper that derives boundary dates from the VN100 panel and maps
  them onto the VN30 panel by date (not index).
- **Train-universe vs score-universe separation:** driver runs one arm per invocation, with
  `--train-universe {vn30,vn100}` and `--score-universe vn30`.
- **Score restriction:** pool prediction dicts for the 31 VN30 tickers only at test time.
- No edits to other baselines' files (import only).

## 7. Testing (C0 = 100%, C1 ≥ 95% on changed lines)

- Reuse: `assert_no_leakage`, no-lookahead perturbation test.
- New:
  - **Alignment:** Arm 0 and Arm 1 produce the identical set of VN30 `(ticker, date)` OOS keys.
  - **Train-universe mask:** Arm 0 excludes the 71 non-VN30 stocks from both training loss and the
    graph adjacency.
  - **Score subset:** only the 31 VN30 tickers are scored in both arms.
  - **Real-data smoke:** a small ticker/fold/epoch slice runs without exception and returns sane
    pooled metrics.

## 8. Compute & execution (queued)

- GPU; **queued after the sp500/hose walk-forward runs finish** (GPU-contention rule). Launch
  detached to escape the harness reaper; poll result JSONs.
- Estimate: Arm 1 ≈ one VN100 run (~4 h/horizon), Arm 0 ≈ one VN30 run (~1.2 h/horizon) → both
  arms × 4 horizons ≈ ~21 h at 5 seeds.
- **Go/no-go:** run **h1 first** (Arm 0 + Arm 1, ~5 h) and inspect the headline DM; extend to
  h5/h10/h22 if a clear signal (either sign) warrants, else stop with the h1 conclusion.

## 9. Deliverables

- New baseline folder (five subfolders) + per-arm, per-horizon result JSONs under
  `results/pooled_transfer_vn30/`.
- **Two-arm HTML dashboard:** Section 1 data-organisation (walk-forward schematic) + Arm0-vs-Arm1
  metric tables + paired DM (headline A) + difference-in-differences (secondary B) + fit evidence.
- Objective report in `docs/reports/` (neutral technical style).
- Quality gate + push in a GPU-free window.

## 10. Out of scope (YAGNI)

- Ticker-embedding pooled model (Approach 3) — deferred; shared weights suffice per Track B.
- Isolating data-effect vs graph-effect (third arm) — deferred.
- All-VN training universe (VN100+HOSE+HNX, Approach B 3-arm) — deferred.
- Applying the same ablation to other markets — deferred.
