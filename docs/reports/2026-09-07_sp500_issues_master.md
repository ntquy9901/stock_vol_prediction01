# S&P 500 forecast issues — master analysis + fix roadmap

Consolidated record of the cell-level issue analysis (why the deep models trail HAR-X on QLIKE, which
dates/stocks, whether outlier removal helps, and the concrete fix options). Detailed companions:
`2026-09-06_sp500_lstm_vs_harx_cell_analysis.md` (mechanism), `2026-09-07_sp500_worst_cells_all_horizons.md`
(per-horizon worst cells/dates/tickers). Data: `--dump-cells` test-split logs
`results/edge_hmatched/cells/cells_sp500_clean_h{1,5,10,22}.parquet` (from Drive; gitignored). Per-cell
QLIKE $=y/f-\log(y/f)-1$, floor $10^{-8}$; ticker-date observations (480 tickers × ~665 test dates).

## 1. The issue
Aggregate QLIKE, S&P 500 (lb10 canonical): HAR-X is best at every horizon; the deep models trail.

| horizon | HAR | HAR-X | LSTM | VolGA |
|---|---|---|---|---|
| h1 | 0.3895 | 0.4061 | 0.6141 | 0.5492 |
| h5 | 0.4605 | 0.4550 | 0.5640 | 0.5531 |
| h10 | 0.4781 | 0.4797 | 0.6821 | 0.6375 |
| h22 | 0.4993 | 0.4959 | 0.6066 | 0.6026 |

## 2. Mechanism (confirmed)
The gap is a **tail phenomenon, not a broad weakness**. On the median cell the LSTM is as good or better
(median per-cell LSTM−HAR-X = −0.0045 at h1); the worst 1% of cells by gap carry 110% of the total gap. Every
one of those cells is an **LSTM under-forecast of a variance spike**: realized $\sim10^{-3}$, HAR-X forecasts
$\sim10^{-4}$ (tracks it), the LSTM collapses to $\sim2$–$6\times10^{-6}$. QLIKE's $y/f$ term explodes on
under-forecasts, so a few spike ticker-days dominate the aggregate. Cause: HAR-X's daily-lag carries the current
elevated level into a spike; the z-scored LSTM mean-reverts and collapses.

## 3. Where it goes bad (which dates, which stocks), all horizons
- **One idiosyncratic outlier hurts every model:** GL (Globe Life) 2024-04-11, realized variance 0.31 (σ≈56%, a
  real −53% short-report crash). Worst cell at every horizon; QLIKE ≈ 1187–1253 (HAR-X), 1552–2270 (LSTM).
- **LSTM spike-collapse cells** (LSTM QLIKE 250–2400 vs HAR-X 3–50): HCA 2024-06-28, KEY 2024-11-06 (h1);
  EIX 2025-01-08, UAL 2024-10-16 (h5); EIX 2025-01-13, F 2025-04-08 (h10); STT 2024-07-16, UAL 2024-10-16 (h22).
- **Real market-event dates dominate the aggregate:** the April-2025 tariff cluster (2025-04-07/08/09; LSTM
  date-sum 9555 at h10), 2024-11-06 (US election), 2025-01-27 (DeepSeek selloff), 2025-01-08/13 (EIX / LA
  wildfires), 2024-08-01/02/05 (yen carry-trade unwind).
- **Worst stocks (total QLIKE):** CNP, GL, PG, UAL, STT, ADP, F, AIG, KKR, EIX, HCA — utilities/airlines/
  financials that had large idiosyncratic or sector spikes.
- Not dirty data: the S&P 500 has no limit-lock days; the extremes are genuine events.

## 4. Can removing outliers make the deep model beat HAR-X? No (fairly).
Model-agnostic removal (exclude the top $p$% by realized variance $y$, the SAME cells for every model) keeps
HAR-X best at every level and horizon:

| SP500 h1, exclude top | HAR-X | LSTM | VolGA |
|---|---|---|---|
| 0% | 0.406 | 0.614 | 0.549 |
| 1% | 0.366 | 0.575 | 0.510 |
| 5% | 0.326 | 0.523 | 0.462 |

The earlier "drop worst 1% by the LSTM−HAR-X gap → LSTM wins" is **data snooping** (it removes exactly the cells
where the LSTM loses) and is not a valid claim. The LSTM's spike collapse is graded across all elevated-variance
days, so fair outlier removal cannot rescue it. Outlier exclusion is a legitimate robustness caveat only.

## 5. Fix roadmap (model-side, not eval-side)
**A. Anchor the deep forecast to HAR-X (residual):** predict a bounded multiplicative correction instead of the
level. $\hat y = f_{\text{HAR-X}}\cdot \exp(\mathrm{clip}(z,-c,c))$, with $z$ from the deep model trained on
$z_{\text{true}}=\log(y/f_{\text{HAR-X}}^{\text{OOF}})$. If the model outputs $z\approx0$ it falls back to HAR-X
(no collapse); the clip bounds the worst case. Concrete cell HCA 2024-06-28 h1 ($y=2.03\times10^{-3}$):

| forecast | QLIKE |
|---|---|
| HAR-X standalone (1.27e-4) | 12.2 |
| LSTM standalone collapse (4.0e-6) | 502.8 |
| anchored, z=−0.5 (worst under clip) | 22.1 |
| anchored, z=0 (no correction) | 12.2 (=HAR-X) |
| anchored, z=+0.3 (correct direction) | 8.4 (beats HAR-X) |

Honest caveat: the delivered **XGBoost residual-ratio** baseline already uses this exact anchoring and is
NO-GO — it prevents the collapse but the learned correction adds no QLIKE value (validation drives $\alpha\to0$
on several cells), i.e. it lands at $\approx$HAR-X, not better. A deep (LSTM/VolGA) anchored residual is untested
but, given the tree version with rich + graph features did not help, is likely also $\approx$HAR-X.

**B. Train the deep model on a QLIKE loss (not MSE):** MSE-trained models mean-revert and under-forecast spikes;
a QLIKE training loss penalizes under-forecasting directly, so the model learns not to collapse. This is the
more promising untested lever to actually BEAT HAR-X (changes what the model optimizes rather than only bounding
its downside).

**Not a fix:** removing outliers at evaluation (§4). Winsorizing the target lowers every model's QLIKE equally
and preserves HAR-X's lead.

## 6. Training cost (VN100, from the lb10 re-run log; LSTM+VolGA, 5 seeds, 7 folds, batch 32, RTX 4060)
Per horizon: h1 ≈ 60 min, h5 ≈ 58, h10 ≈ 52, h22 ≈ 20 → **all four horizons ≈ 3–3.5 h**; h1+h5 only ≈ 2 h. An
anchored-residual or QLIKE-loss variant uses the same architecture/folds/seeds, so the cost is the same order
(the extra OOF HAR-X fit is negligible CPU).
