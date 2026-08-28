# S&P 500 expanding walk-forward: robustness of the deep-model beat-HAR result

Purpose: test whether the single-split S&P 500 result (deep temporal model significantly beats HAR on
QLIKE under date-clustered Diebold-Mariano) reproduces across several distinct, later test windows, or
is confined to one favourable split.

## Setup

- Data: corrected S&P 500 snapshot panel, common-date node set with `min_common=3000` -> **457 nodes**,
  T=3029 common dates, 2977 valid anchors (anchor calendar fixed to the study's max horizon h=22 so all
  horizons share the same test windows).
- Design: **expanding-window walk-forward**, 4 folds. Each fold trains on `anchors[:tr_end_k]` (train
  window grows with k), validates on the immediately-preceding block (`val_len=150`), and is scored on a
  **locked, non-overlapping test block** (`test_len=180` anchors ≈ 180 trading dates). Test blocks tile
  the tail of the calendar so each fold's test is a distinct, strictly-later window; earlier test blocks
  re-enter later training windows. Target-overlap purge (drop last `h` train and val snapshots) applied
  at both split boundaries (reused, tested `experts._purge_snapshots`).
- Models per fold: **E0** pooled HAR anchor; **E1** LSTM, no graph (`use_graph=False`); **E2** LSTM+GAT
  (`use_graph=True`); **E3** = val-fit convex blend HAR+E2; **E3b** = val-fit convex blend HAR+E1. All
  neural experts trained with **3 seeds** (42, 123, 2026) and seed-ensembled (reduced from 5 for runtime;
  see Runtime note). Framing = `full` (predict normalized target), identical QLIKE floor 1e-8 across all
  compared models.
- Metrics on each locked test block: QLIKE, dQLIKE% vs HAR (`+` = model beats HAR), relative
  R²_OOS vs HAR, date-clustered Diebold-Mariano vs HAR (p-value), and a circular block-bootstrap CI of
  the QLIKE loss differential (`mean_diff = loss_model − loss_HAR`; a fully-negative interval = a
  significant beat, fully-positive = significantly worse, straddling 0 = not significant).
- Test windows (target dates), identical across horizons up to the horizon shift:
  fold0 ≈ 2023-09 → 2024-05/06, fold1 ≈ 2024-05/06 → 2025-02/03,
  fold2 ≈ 2025-02/03 → 2025-10/11, fold3 ≈ 2025-10/11 → 2026-07/08. Each fold = 180 test dates.

Machine-readable results: `results/walkforward/sp500/walkforward.json` (merged) and per-cell
`results/walkforward/sp500/wf_h{H}_f{K}.json`.

## Per-fold results

dQ% = QLIKE improvement over HAR (%). p = date-clustered DM p-value vs HAR. CI = 95% block-bootstrap
interval of `loss_model − loss_HAR` (negative ⇒ beats HAR).

### Horizon h1

| fold | window (target) | HAR QLIKE | E1 dQ% (p) | E2 dQ% (p) | E3=HAR+E2 dQ% (p) | E3b=HAR+E1 dQ% (p) |
|---|---|---|---|---|---|---|
| 0 | 2023-09→2024-05 | 0.3889 | −0.36 (0.757) | −19.26 (0.022) | +1.63 (6.4e-4) | +1.53 (7.3e-3) |
| 1 | 2024-05→2025-02 | 0.3857 | +2.72 (4.7e-4) | −14.80 (0.028) | +3.24 (7.5e-16) | +3.10 (1.6e-10) |
| 2 | 2025-02→2025-10 | 0.4069 | +2.85 (0.012) | +3.85 (0.011) | +4.53 (1.3e-4) | +3.23 (4.2e-4) |
| 3 | 2025-10→2026-07 | 0.3648 | +1.53 (6.6e-3) | −1.73 (0.386) | +2.62 (2.5e-5) | +1.57 (4.7e-3) |

### Horizon h5

| fold | window (target) | HAR QLIKE | E1 dQ% (p) | E2 dQ% (p) | E3=HAR+E2 dQ% (p) | E3b=HAR+E1 dQ% (p) |
|---|---|---|---|---|---|---|
| 0 | 2023-09→2024-05 | 0.4485 | +4.22 (1.7e-5) | +1.46 (0.262) | +2.79 (3.5e-3) | +4.52 (6.2e-8) |
| 1 | 2024-05→2025-02 | 0.4328 | **−23.79 (0.279)** | +2.32 (8.1e-3) | +3.86 (2.9e-12) | **−23.79 (0.279)** |
| 2 | 2025-02→2025-10 | 0.5746 | +1.67 (0.309) | +2.08 (7.5e-3) | +3.20 (9.0e-11) | +2.29 (0.051) |
| 3 | 2025-11→2026-07 | 0.4010 | +3.03 (0.019) | +0.95 (0.408) | +2.86 (4.1e-4) | +3.90 (2.0e-4) |

### Horizon h10

| fold | window (target) | HAR QLIKE | E1 dQ% (p) | E2 dQ% (p) | E3=HAR+E2 dQ% (p) | E3b=HAR+E1 dQ% (p) |
|---|---|---|---|---|---|---|
| 0 | 2023-09→2024-06 | 0.4797 | +9.03 (9.7e-7) | +5.32 (0.013) | +6.34 (1.4e-4) | +9.03 (7.5e-7) |
| 1 | 2024-06→2025-02 | 0.4517 | **−310.15 (0.270)** | **−223.60 (0.264)** | **−223.60 (0.264)** | **−310.15 (0.270)** |
| 2 | 2025-02→2025-11 | 0.6132 | +3.92 (4.5e-4) | +2.56 (0.044) | +3.73 (4.1e-5) | +4.36 (2.2e-6) |
| 3 | 2025-11→2026-07 | 0.4139 | +5.39 (1.3e-5) | +0.97 (0.322) | +2.43 (2.0e-3) | +5.84 (2.8e-8) |

### Horizon h22

| fold | window (target) | HAR QLIKE | E1 dQ% (p) | E2 dQ% (p) | E3=HAR+E2 dQ% (p) | E3b=HAR+E1 dQ% (p) |
|---|---|---|---|---|---|---|
| 0 | 2023-10→2024-06 | 0.5127 | +12.94 (2.2e-8) | +9.89 (9.4e-5) | +9.89 (9.4e-5) | +12.94 (2.2e-8) |
| 1 | 2024-06→2025-03 | 0.4709 | +8.32 (0.019) | +4.82 (0.166) | +4.82 (0.166) | +8.32 (0.019) |
| 2 | 2025-03→2025-11 | 0.6084 | +6.94 (8.3e-4) | +8.48 (2.9e-3) | +8.63 (2.1e-3) | +7.10 (2.4e-4) |
| 3 | 2025-11→2026-08 | 0.4494 | −1.54 (0.602) | −3.75 (0.257) | −3.75 (0.257) | −1.54 (0.602) |

## Pooled summary (robust)

Fold-median dQ% (mean is not reported as the pooled point estimate because two folds contain
QLIKE blow-ups — see below — that make the arithmetic mean uninformative). `beat` = folds with dQ%>0;
`sig-beat` = folds with dQ%>0 and date-clustered DM p<0.05.

| model | h1 | h5 | h10 | h22 |
|---|---|---|---|---|
| E1 (LSTM, no graph) | +2.12 (beat 3/4, sig 3/4) | +2.35 (3/4, 2/4) | +4.65 (3/4, 3/4) | +7.63 (3/4, 3/4) |
| E2 (LSTM+GAT) | −8.26 (1/4, 1/4) | +1.77 (4/4, 2/4) | +1.76 (3/4, 2/4) | +6.65 (3/4, 2/4) |
| E3 (HAR+E2 blend) | +2.93 (4/4, 4/4) | +3.03 (4/4, 4/4) | +3.08 (3/4, 3/4) | +6.72 (3/4, 2/4) |
| E3b (HAR+E1 blend) | +2.34 (4/4, 4/4) | +3.10 (3/4, 2/4) | +5.10 (3/4, 3/4) | +7.71 (3/4, 3/4) |

## Two QLIKE blow-up folds (fold1 at h5 and h10)

The fold1 test window (target dates 2024-06 → 2025-02) produces catastrophic QLIKE for the raw neural
level forecast at h5 and h10: E1 = −23.79% (h5) and −310.15% (h10). Characteristics:

- The date-clustered DM p-values for these folds are **not significant** (p ≈ 0.27–0.28), i.e. the
  blow-up is concentrated in a few test dates rather than a broad shift; the block-bootstrap CIs for
  these cells straddle 0 with very large upper bounds (e.g. h10 fold1 E1 CI [−0.026, +3.95]).
- The failure is reproducible across the 3 seeds and appears at h5 and h10 but **not** at h1 or h22 for
  the same window, and it originates in the `full`-framing level reconstruction
  (`pred = c·std + mean`) extrapolating to extreme values on a few dates in that regime.
- The convex blends do not reliably neutralise it: at h5 fold1 the val-fit HAR+E1 weight put ≈all mass on
  E1 (E3b = −23.79%, tracking E1), while HAR+E2 was protected only because **E2** did not blow up in
  that fold (E3 = +3.86%). At h10 fold1 **both** E1 and E2 blew up, so both blends inherited the failure
  (E3 = E3b-analogue ≈ −223.60%). The val-fit alpha trusted the neural expert on validation and was not
  robust to its out-of-sample failure.

## Does the graph (E2) ever beat no-graph (E1)?

E2 (LSTM+GAT) beats E1 (LSTM only) in a minority of folds: h1 fold2, h22 fold2, and marginally h5 fold2
— all in the 2025 test window. (E2 also shows a lower loss than E1 at h5 fold1 and h10 fold1, but only
because E1 blew up in those folds, not because the graph added skill.) In the remaining 11 of 16 folds
E1 ≥ E2. The graph does not add robust out-of-sample value over the no-graph LSTM; this is consistent
with the project's prior single-split and screening findings.

## Verdict

**Partially stable — a real, recurring beat that is neither one lucky window nor uniformly robust.**

- The deep model beats HAR in **3 of 4 folds at every horizon** on the point estimate, and the beat is
  statistically significant (date-clustered DM p<0.05) in most of those folds, so it is **not** an
  artifact of a single test window.
- The convex blend E3/E3b is the most consistent variant at short horizons: HAR+E2 beats HAR in **4/4
  folds with significance at h1 and h5**, and HAR+E1 in 4/4 (all sig) at h1. This confirms the
  single-split finding that the convex combination beats HAR at short horizons across time.
- Robustness fails in two identifiable ways: (i) a **concentrated QLIKE blow-up** in the 2024-06→2025-02
  window at h5/h10 (not DM-significant, but large in magnitude, and not reliably absorbed by the blend);
  and (ii) at h22 the beat is significant in the three earlier folds but **disappears in the most recent
  fold** (fold3, 2025-11→2026-08: E1 −1.54%, p=0.60; E3 −3.75%, p=0.26).
- Reconciliation with the single split: the original 80/10/10 test window overlaps folds 2–3. At h22 the
  single-split +7.1% is consistent with the strong beats in folds 0–2 (+12.9/+8.3/+6.9%); the walk-forward
  additionally reveals that the most recent window (fold3) shows **no** beat, which the single split did
  not surface. The short-horizon convex-blend beat (single split +3.2/+3.6% at h1/h5) is confirmed as
  stable across all four folds.

Overall: promote the short-horizon convex-blend beat-HAR result (h1, h5) to confirmatory — it holds in
every walk-forward fold with significance. Treat the long-horizon (h10, h22) beat as **time-varying**:
present in most folds but with a catastrophic-failure fold at h10 and a no-beat most-recent fold at h22.

## Runtime note

Reduced to **3 seeds** (42, 123, 2026) from the 5-seed single-split protocol to keep each horizon's
4-fold run within the environment's background-task wall-clock limit; the priority horizons requested
(h1, h22) plus h5 and h10 were all completed with the 3-seed ensemble. Each horizon ran in a separate
process (≈29–35 min on the RTX 4060, batch 32, `min_common=3000`), writing kill-robust per-(horizon,fold)
JSON cells that are merged into `walkforward.json`. All four horizons × four folds completed. Results are
deterministic across re-runs (verified: h1/h5/h10/h22 fold values reproduced bit-for-bit).
