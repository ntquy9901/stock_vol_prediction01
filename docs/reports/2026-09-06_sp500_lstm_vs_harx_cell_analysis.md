# Why the no-graph LSTM has much higher QLIKE than HAR-X on the S&P 500 — cell-level analysis

Source: per-ticker-per-day prediction logs from the canonical lb10 run (`--dump-cells`), test split only,
`results/edge_hmatched/cells/cells_sp500_clean_h{1,10}.parquet` (produced on Colab, retrieved from Drive; the
parquet files are gitignored — ~1.3 GB each). QLIKE per cell = $y/f - \log(y/f) - 1$ with the shared floor
$10^{-8}$. Ticker-date observations, not calendar days.

## Headline
On the S&P 500 the LSTM's large aggregate QLIKE disadvantage vs HAR-X is **not a broad weakness**. It is
produced almost entirely by a **~1% tail of extreme variance-spike ticker-days on which the LSTM
under-forecasts catastrophically**, amplified by QLIKE's asymmetric penalty on under-forecasting. On the other
~99% of cells the LSTM is as good as or better than HAR-X.

## Evidence (h1; h10 identical pattern)

| quantity | h1 | h10 |
|---|---|---|
| QLIKE HAR-X (all test cells) | 0.4061 | 0.4797 |
| QLIKE LSTM (all test cells) | 0.6141 | 0.6821 |
| **QLIKE after dropping the worst 1% of (LSTM−HAR-X) gap cells** | LSTM **0.3745** < HAR-X 0.3965 | LSTM **0.4365** < HAR-X 0.4508 |
| per-cell median (qlike_LSTM − qlike_HARX) | **−0.0045** (LSTM better) | (same sign) |
| top-1% of cells share of the total gap | **110.5%** | dominant |

- **Direction:** every one of the top-1% gap cells is an LSTM **under-forecast**. On them the realized variance
  is a spike (median $y\approx1.3\times10^{-4}$); HAR-X forecasts $\approx0.87\times$ the realized level, while
  the **LSTM forecasts $\approx0.05\times$** (median $6.3\times10^{-6}$). QLIKE's $y/f$ term explodes when the
  forecast collapses below a spike, so these few cells carry the entire aggregate gap.
- **Not one event:** the gap is spread over 515 of 667 test dates (top-10 dates only 14% of the gap) and across
  many tickers (CNP, AIG, STT, UAL, F, GL, PG, ADP, …). It is a systematic response to spikes, not a single
  shock cluster.
- **On the median cell and even the median top-5% spike, the two are comparable** (spike-cell median $f/y$:
  LSTM 0.388 vs HAR-X 0.369 at h1) — the divergence is confined to the extreme upper tail of spikes.

## Mechanism
HAR-X regresses the target on the stock's own recent Parkinson variance; its daily-lag coefficient carries the
current (elevated) level forward, so on a spike day — which follows elevated variance — its forecast stays high.
The LSTM is trained on a per-ticker z-scored target with regularization and a linear output; on an extreme day
it mean-reverts and its inverse-transformed forecast lands far below the realized spike. QLIKE punishes that
collapse far more than it rewards the LSTM's better fit on the other 99% of cells, so HAR-X wins the aggregate
QLIKE while losing on 99% of cells (and, consistently, the LSTM/VolGA lead squared- and absolute-error at the
short horizons).

## Implications
- The result is a property of **QLIKE + spike under-forecasting**, not of dirty data (the S&P 500 has no
  limit-lock days; `n_limitlock=0`).
- The concrete lever to make a deep model beat HAR-X on QLIKE is to **stop the spike collapse**, e.g. train the
  deep branch on a QLIKE (not MSE) loss so under-forecasting is penalized in training, or **anchor the deep
  forecast to HAR-X** (a residual/multiplicative correction, $\hat y = \text{HAR-X}\times e^{z}$) so it inherits
  HAR-X's level-tracking on spikes. The delivered XGBoost residual-ratio study already anchors to HAR-X and does
  not collapse, but its correction did not add QLIKE value; a QLIKE-loss deep model is the untested next step.
- VolGA (graph) partially closes the gap (h1 0.549 vs LSTM 0.614) because the cross-sectional edge injects
  neighbour information that dampens some of the collapses.

## Reproduce
`results/edge_hmatched/cells/cells_sp500_clean_h*.parquet` (Drive: `gdrive:luanvan_data/cells/`), read the
`split=='test'` rows, pivot on (ticker, date) × model, compute the QLIKE above, and rank cells by
`qlike_LSTM − qlike_HARX`.
