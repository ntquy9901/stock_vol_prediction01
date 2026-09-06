# S&P 500 — where the QLIKE goes bad, all horizons (which dates, which stocks, how badly)

Source: `--dump-cells` test-split logs `results/edge_hmatched/cells/cells_sp500_clean_h{1,5,10,22}.parquet`
(from Drive; gitignored). Per-cell QLIKE $=y/f-\log(y/f)-1$, floor $10^{-8}$. Ticker-date observations
(480 tickers × ~665 unique test dates ≈ 320k per horizon). Aggregate QLIKE per horizon:

| horizon | HAR | HAR-X | LSTM | VolGA |
|---|---|---|---|---|
| h1 | 0.3895 | 0.4061 | 0.6141 | 0.5492 |
| h5 | 0.4605 | 0.4550 | 0.5640 | 0.5531 |
| h10 | 0.4781 | 0.4797 | 0.6821 | 0.6375 |
| h22 | 0.4993 | 0.4959 | 0.6066 | 0.6026 |

The bad QLIKE comes from three distinct sources, in order of impact.

## 1. One idiosyncratic outlier hurts every model: GL, 2024-04-11
Globe Life (GL) realized Parkinson variance $=3.1\times10^{-1}$ on 2024-04-11 (σ ≈ 56% intraday — a real
short-seller-driven crash, not a data error). It is the single worst cell at **every** horizon and it inflates
**both** models: per-cell QLIKE ≈ 1187–1253 (HAR-X) and 1552–2270 (LSTM). No causal model forecasts this from
price history; it is an unforecastable event that alone adds ~0.003–0.007 to every model's mean QLIKE. It is the
top candidate for winsorization / a robustness caveat.

## 2. The LSTM-specific failure: forecast collapses on variance spikes (HAR-X tracks them)
On the next tier of worst cells the pattern is identical across horizons: the realized variance spikes to
$\sim10^{-3}$, **HAR-X forecasts $\sim10^{-4}$ (tracks it), the LSTM collapses to $\sim2$–$6\times10^{-6}$**
(near the floor), so LSTM QLIKE is 250–2400 while HAR-X QLIKE is 3–50. Representative worst LSTM cells:

| horizon | ticker | date | y_true | HAR-X f | LSTM f | QLIKE LSTM | QLIKE HAR-X |
|---|---|---|---|---|---|---|---|
| h1 | HCA | 2024-06-28 | 2.0e-3 | 1.3e-4 | 4.0e-6 | 502 | 12 |
| h1 | KEY | 2024-11-06 | 2.2e-3 | 1.1e-4 | 6.7e-6 | 326 | 16 |
| h5 | EIX | 2025-01-08 | 5.9e-3 | 1.3e-4 | 3.5e-6 | 1669 | 40 |
| h5 | UAL | 2024-10-16 | 5.5e-3 | 3.7e-4 | 1.4e-5 | 398 | 11 |
| h10 | EIX | 2025-01-13 | 8.3e-3 | 1.5e-4 | 3.5e-6 | 2379 | 51 |
| h10 | F | 2025-04-08 | 4.2e-3 | 3.2e-4 | 6.2e-6 | 669 | 10 |
| h22 | STT | 2024-07-16 | 2.4e-3 | 2.4e-4 | 4.9e-6 | 487 | 7 |
| h22 | UAL | 2024-10-16 | 5.5e-3 | 3.9e-4 | 1.4e-5 | 396 | 11 |

Mechanism (confirmed in `2026-09-06_sp500_lstm_vs_harx_cell_analysis.md`): the z-scored LSTM mean-reverts and
under-reacts to the current elevated level, while HAR-X's daily-lag term carries it forward. This tail is the
entire LSTM−HAR-X gap: dropping the worst 1% of cells flips the ranking (LSTM below HAR-X at h1 and h10).

## 3. Real market-event dates dominate the aggregate (top dates by summed QLIKE)
The worst dates are recognizable macro/idiosyncratic events; on them both models are stressed and the LSTM more so.

- **h1:** 2024-11-06 (US election, LSTM sum 2140 vs HAR-X 597), 2024-04-11 (GL crash), 2024-12-02, 2025-01-27
  (DeepSeek selloff), 2025-06-09, 2024-06-28, 2024-03-20.
- **h5 / h10:** the **April 2025 tariff crash cluster** dominates — 2025-04-09 (LSTM sum 5457 at h5, **9555 at
  h10**), 2025-04-07, 2025-04-08, 2025-04-04; plus 2025-01-08/13 (EIX, LA wildfires → utility crash) and
  2026-01-05.
- **h22:** 2025-04-09 (tariff), 2024-04-11 (GL), 2024-08-01/02/05 (yen carry-trade unwind), 2026-02-03.

## Which stocks are worst (total QLIKE across the test set)
Consistent offenders across horizons: **CNP** (CenterPoint), **GL** (Globe Life), **PG**, **UAL** (United
Airlines), **STT** (State Street), **ADP**, **F** (Ford), **AIG**, **KKR**, **EIX** (Edison), **HCA**. These are
names that had large idiosyncratic or sector spikes (utilities EIX/CNP, airlines UAL, financials STT/AIG/KKR) —
exactly where a spike arrives and the LSTM under-forecasts it. HAR-X absorbs the same names far better.

## Takeaways
- The aggregate QLIKE is driven by a small set of **real spike ticker-days**, not by broad error and not by
  dirty data (S&P 500 has no limit-lock days; the extreme values are genuine crashes/events).
- **HAR-X wins QLIKE because it tracks the level into a spike; the LSTM collapses.** The fix levers are a
  QLIKE-loss deep model or a HAR-X-anchored residual (documented in the companion analysis).
- For evaluation robustness, report QLIKE with the GL 2024-04-11 outlier and the April-2025 cluster excluded (a
  top-1%-date / winsorized view); the model ranking on the calm 99% already favors the deep models.
