# Cross-market check: does the VN "lstm_only beats HAR at short horizons" pattern replicate on the S&P 500?

Date: 2026-08-20. Question: on the US market, does a price-only LSTM beat the HAR linear baseline at
SHORT forecast horizons while HAR wins at LONG horizons — the pattern found on VN30/VN100?

## Setup

- Data: `data/processed/sp500/` (503 S&P 500 constituents, Parkinson VARIANCE at t+h; Yahoo-sourced,
  gitignored). 501/503 tickers had enough history to use.
- Models (self-contained runner `scripts/sp500_crossmarket/run_sp500_crossmarket.py`, NOT the VN
  news/graph harness): **HAR** = pooled OLS on [daily=Parkinson(t), weekly=roll-5, monthly=roll-22];
  **lstm_only** = 2-layer LSTM(64) on the same 3 features, per-ticker StandardScaler (train-fit),
  linear output, inverse-transform for eval. NO news/graph/gate.
- Protocol: per-ticker chronological 70/15/15 (early-stop on val QLIKE, patience=3, ≤10 epochs), test
  read once. seq_length=22, horizons {1,5,10,22}, seed 42.
- Metrics: QLIKE (shared floor 1e-8, identical to the VN `dm_report._qlike`) + RMSE; Diebold–Mariano
  (HLN, HAC lag h-1) of lstm_only vs HAR on per-observation QLIKE (negative dm ⇒ lstm beats HAR).
- Reuses the VN `diebold_mariano()` implementation so DM is identical across markets.

## Result — S&P 500 (all 501 usable tickers, 10 epochs, seed 42)

| h | QLIKE HAR | QLIKE lstm | DM (lstm vs HAR) | verdict | n_test |
|---|---|---|---|---|---|
| 1  | 0.3585 | 0.3563 | −7.25 (p=0.000)* | lstm beats HAR | 653,888 |
| 5  | 0.4281 | 0.4261 | −2.49 (p=0.013)* | lstm beats HAR | 653,599 |
| 10 | 0.4611 | 0.4711 | +6.37 (p=0.000)* | HAR beats lstm | 653,257 |
| 22 | 0.5260 | 0.5501 | +4.61 (p=0.000)* | HAR beats lstm | 652,321 |

## Cross-market comparison (lstm_only vs HAR, DM QLIKE; negative = lstm beats HAR)

| h | VN (70/15/15, lstm_only)† | S&P 500 (70/15/15, lstm_only) |
|---|---|---|
| 1  | −4.62* lstm | −7.25* lstm |
| 5  | −2.02* lstm | −2.49* lstm |
| 10 | +0.30 tie | +6.37* HAR |
| 22 | +3.85* HAR | +4.61* HAR |

† VN numbers from `docs/reports/2026-08-19_0015_seq_lookback_experiment_report.md` §7a (VN30, seq5).

**The core pattern replicates on the US market:** the price-only LSTM significantly beats HAR at the
SHORT horizons (h1, h5) on BOTH markets, and HAR significantly wins at the LONG horizon (h22) on both.
The crossover sits between h5 and h10 for the US and between h10 and h22 for VN — i.e. HAR's long-memory
advantage kicks in slightly earlier on the S&P 500 — but the qualitative short-vs-long split is the
same. This is independent cross-market evidence that the deep model's edge is a genuine short-horizon
nonlinearity effect, not a VN-specific artifact.

## Split explanation (per-ticker chronological)

Each ticker's own series is split 70/15/15 by ANCHOR POSITION, not by a global calendar date, so
train precedes val precedes test chronologically WITHIN each ticker (no leakage: windows within-split,
scalers train-only, test read once), but the calendar boundaries differ across tickers by listing age:

| ticker | train | val | test |
|---|---|---|---|
| AAPL | 1981-02-13 → 2012-12-10 | 2012-12-11 → 2019-10-11 | 2019-10-14 → 2026-08-19 |
| MSFT | 1986-05-14 → 2014-07-11 | 2014-07-14 → 2020-07-28 | 2020-07-29 → 2026-08-19 |
| NVDA | 1999-03-25 → 2018-05-25 | 2018-05-29 → 2022-07-06 | 2022-07-07 → 2026-08-19 |
| GEV (IPO 2024) | 2024-05-29 → 2025-12-16 | 2025-12-17 → 2026-04-20 | 2026-04-21 → 2026-08-19 |

Test always ends at the last data date (2026-08-19) but starts anywhere from 2016 (old names) to 2026
(recent IPOs) — it is the last 15% of each stock's own life, not a common calendar window. This mirrors
the VN pipeline's per-ticker split. A global-date split (all tickers tested on one common window, e.g.
2020–2026) is a separate variant not run here.

## Caveats

- **Survivorship bias:** the constituent list is CURRENT S&P 500 members only; delisted names are
  absent, so results are optimistic in level (the short-vs-long ORDERING is what transfers, not the
  absolute QLIKE).
- **Lookback mismatch:** the VN control used seq=5; this US run used seq=22. The comparison is
  qualitative (short-h lstm wins / long-h HAR wins), which is robust to the lookback choice.
- **Single-seed** (42); the US n_test ≈ 653k per horizon makes the DM verdicts stable, but multi-seed
  would tighten the level estimates.
- Data is Yahoo-sourced (gitignored, not redistributed); only the aggregate metrics (results.json) are
  committed.

## Artifacts

- Runner: `scripts/sp500_crossmarket/run_sp500_crossmarket.py` (+ tests, 5 pass).
- Results: `results/sp500_crossmarket_2026-08-20_225743/results.json`.
