# Stock-Universe Choice for Daily Volatility Forecasting (VN market)

Generated: 2026-08-16
Scope: which Vietnamese stock universe (VN30 / VN100 / HOSE / HNX / custom-liquid) to use for a
DAILY (not intraday) Parkinson-variance forecasting problem, for a POOLED cross-sectional
LSTM + cross-stock GAT + gated-news model at horizons 1/5/10/22.

This report is a decision aid grounded in (a) international literature/GitHub practice, (b) Vietnamese
practice, and (c) this project's own measured data-quality facts. All external claims are cited with
URLs. Where web access was limited, the limitation is stated inline.

---

## 1. International practice — how many stocks is "enough"?

### 1.1 Per-series models (HAR / GARCH): universe size is not the constraint

The workhorse daily-volatility model is the HAR-RV of Corsi (2009) — a constrained AR(22) regressing
next-period realized volatility on daily, weekly (5-day) and monthly (22-day) lagged RV averages. It is
estimated **one series at a time**, so the "universe" is a reporting choice, not a sample-size driver:
each ticker contributes its own time-series regression. GARCH-family models are likewise per-series.
For these models, the number of stocks affects only the breadth of the cross-sectional average you
report, not the estimation sample of any single fit. Corsi (2009), *A Simple Approximate Long-Memory
Model of Realized Volatility*, J. Financial Econometrics 7(2):174-196 — summary and specification:
<https://medium.com/@simomenaldo/realized-volatility-and-har-models-a-new-paradigm-for-volatility-forecasting-4a660f2530f3>
and the methodology review in Financial Innovation (2025):
<https://link.springer.com/article/10.1186/s40854-025-00809-5>.

### 1.2 Pooled / panel / graph models: more tickers = more samples AND cross-sectional structure

For deep and graph models the picture reverses — the cross-section is the source of both extra training
samples and the relational structure the model exploits:

- **GNNHAR (Zhang, Pu, Cucuringu et al., International Journal of Forecasting, 2025).** Primary sample:
  **DJIA components trading continuously over 2007-07-01..2021-06-30 → 27 stocks** (following the
  Bollerslev et al. continuous-trading filter). Robustness on a larger **S&P 100** panel. The graph is
  a GLASSO volatility-spillover graph with diameter 3, so a 3-layer GNN already reaches every node — the
  authors explicitly find **no gain from multi-hop neighbours**, while the **nonlinearity** of the GNN
  (trained with QLIKE loss) is what improves accuracy. On S&P 100, the graph-augmented linear GHAR
  consistently beats plain HAR. Paper: <https://web.media.mit.edu/~xdong/paper/ijf25.pdf>;
  ScienceDirect: <https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967>; code:
  <https://github.com/chaozhang-ox/GNNHAR>. Takeaway: a well-known graph-vol result runs on ~27-30
  liquid, continuously-traded names — a small, clean, homogeneous panel is a legitimate and published
  design point.
- **Panel-ML realized volatility (Zhang et al., Pacific-Basin Finance J., 2023).** Uses **all stocks
  ever in the S&P 500**, dropping any name with a no-trade day in the window (1998-2020, 1-min data).
  Rationale: "panel data ... accumulate more informative data over a short period, hence making the
  models more efficient." Note the explicit **no-trade / continuous-trading filter** — illiquid names
  are removed, not added. <https://www.sciencedirect.com/science/article/abs/pii/S0927539823000683>.
- **Full cross-section realized-variance benchmark (arXiv 2506.07928).** Runs the *entire* S&P 500
  cross-section at daily frequency (1993-2019), evaluating with MSE/MAE/QLIKE/Mincer-Zarnowitz — an
  example of the large end (~500 names) being used specifically to stress-test whether anything beats
  the HAR benchmark. <https://arxiv.org/pdf/2506.07928>.
- **GNN portfolio study (arXiv 2605.19278, code github.com/waderylan/sp500-gnn).** Weekly RV for
  **465 S&P 500 equities, 2015-2025**; GraphSAGE on correlation/sector/Granger graphs vs HAR & LSTM
  baselines. Explicit condition for when the cross-section pays off: "GNNs are worth the added
  complexity when the universe is **large**, market relationships are informative, macro regime features
  are available, and the portfolio rule can use cross-sectional structure." They pick S&P 500 because it
  is "large enough to make cross-sectional graph structure meaningful **while remaining liquid enough**."
  <https://arxiv.org/html/2605.19278> and <https://github.com/waderylan/sp500-gnn>.
- **Transformer pooling (Frank 2023, via the Financial Innovation review).** Weekly/monthly RV of S&P
  500 stocks; pooling across assets — and especially **sectoral pooling** — improved every ML model
  over per-series fitting. <https://link.springer.com/article/10.1186/s40854-025-00809-5>.

### 1.3 Concrete range and the diminishing-returns / heterogeneity caveat

- **Sufficiency range for a pooled/graph daily-vol model: ~25-30 liquid names is already enough to
  publish** (GNNHAR = 27; DJIA-30 is a recurring anchor). Larger, liquid panels (~100 = S&P 100;
  ~465-500 = S&P 500) are used **when the goal is to make the cross-sectional graph richer** or to
  stress-test the HAR benchmark on the full cross-section.
- **The added stocks in every large study are still liquid, continuously-traded names.** The recurring
  design pattern is a **continuous-trading / no-trade filter** (GNNHAR: DJIA names trading throughout;
  Panel-ML: drop any stock with a no-trade day). No mainstream daily-vol study adds illiquid micro-caps
  to inflate sample size — they are filtered out because no-trade days corrupt the realized-vol target.
- **Diminishing returns / heterogeneity:** GNNHAR finds multi-hop graph information adds no clear
  advantage and that most of the lift is nonlinearity, not breadth; the portfolio-GNN paper finds the
  best-MSE model, best-ranking model, and best-Sharpe model are three *different* models — i.e. adding
  scale/graph complexity does not monotonically help and interacts with the objective. The consistent
  message: **more liquid tickers help through cross-sectional structure and sample count, but with
  diminishing returns, and only if the added names do not degrade target quality.**

---

## 2. Vietnamese practice — which universe do VN papers pick?

Search was run in both Vietnamese and English. Findings:

- **VN30 is the de-facto benchmark universe** for constituent-level VN volatility/return studies:
  - GARCH+LSTM Value-at-Risk on **the full VN30 basket (30 stocks), 2018-2024** — Tạp chí Công Thương:
    <https://tapchicongthuong.vn/ung-dung-hoc-may-trong-uoc-tinh-value-at-risk--phuong-phap-ket-hop-garch-va-lstm-417219.htm>.
  - Bayesian GARCH(1,1) fitted to **each of the 30 VN30 constituents**, then mixture-combined to the
    index (referenced in the VN30 ARIMA study): <https://crimsonpublishers.com/siam/pdf/SIAM.000598.pdf>.
  - LSTM + Ichimoku directional forecasting over **VN30-Index constituents, 2012-2020**:
    <https://link.springer.com/chapter/10.1007/978-981-19-2130-8_19>.
  - ARCH/GARCH on the **VN30F1M** futures contract (derivative of VN30):
    <https://tapchinganhang.gov.vn/ung-dung-mo-hinh-arch-garch-phan-tich-do-bien-dong-cua-hop-dong-tuong-lai-vn30f1m-tren-thi-truong-chung-khoan-phai-sinh-viet-nam-10191.html>.
- **Index-level work uses the VN-Index / VN30 index series** (GARCH vs LSTM/TCN, LSTM-GRU hybrids):
  Kinh tế và Dự báo <https://kinhtevadubao.vn/du-bao-chi-so-chung-khoan-bang-hoc-may-bang-chung-thuc-nghiem-tu-thi-truong-chung-khoan-viet-nam-29030.html>;
  Can Tho Univ. LSTM-GRU on the HOSE price-trend index
  <https://ctujsvn.ctu.edu.vn/index.php/ctujsvn/article/view/5347>.
- **Broader-HOSE studies exist but narrow to a liquid/sector slice**, not the full exchange — e.g. ML +
  DEA return prediction on **26 real-estate firms on HOSE, 2019-2024**:
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12463195/>.
- **VN100 as a modelling universe: not found in the volatility literature.** The VN100 appears mainly as
  an *index/ETF* product (VinaCapital FUEVN100; VN100 = VN30 large-cap + VNMidcap 70 mid-cap), described
  as the top-100 largest and most-liquid HOSE names covering ~85% of the VN-Index:
  <https://vinacapital.com/investment-solutions/onshore-funds/vinacapital-vn100-etf/> and
  <https://www.tradingview.com/symbols/HOSE-VN100/components/>.
- **No public GitHub repo applying a GNN to VN100/HOSE was found.** The transferable graph-vol pipelines
  are foreign (e.g. S&P 100 GNN <https://github.com/timothewt/SP100AnalysisWithGNNs>).
- **Data-quality caveat VN papers/sources flag:** Vietnam lists 1,600+ names across HOSE/HNX/UPCoM,
  "most are illiquid small-caps with limited disclosure" (742 small-caps on HOSE alone); liquidity and a
  shortage of quality stocks are cited as barriers to MSCI EM upgrade. This is exactly why VN modelling
  studies default to VN30 (or a liquid sector slice): liquidity + data availability + benchmark
  standard. Sources: <https://thevietnamyield.com/vn-index-explained/> and
  <https://thevietnamyield.com/the-vietnam-10-blue-chips/>.

**Most common VN choice: VN30** (or the VN-Index at index level), chosen for liquidity, data
availability, and being the standard liquid benchmark. VN100/HOSE-wide constituent-level volatility
modelling is rare in the literature reviewed.

---

## 3. This project's specifics and measured data-quality facts

| Fact | Source in repo |
|---|---|
| Model is pooled cross-sectional: per-ticker LSTM + cross-stock GAT over a directed volume->volatility edge + gated news branch; target = single-day Parkinson **variance** (sigma^2) at h=1/5/10/22 | task spec + `MEMORY.md` (Track B pooled architecture; Parkinson target is variance) |
| VN30 processed universe = **33 tickers**; ~73k pooled ticker-days | `docs/reports/2026-08-16_processed_data_quality_audit_report.md` (33 per-ticker rows) |
| VN30 processed data is clean: no ticker with >10% zero-fraction; only SSI hits the clip ceiling once; no high<low rows | same audit (Flags section) |
| VN100 crawl = **104 tickers** (VN30 + ~70 VNMidcap; 1 fetch failure BCG), ~3x pooled samples | `docs/reports/2026-08-16_vn100_crawl_report.md` |
| VN100 adds short-history mid-caps (e.g. DSE 551 rows from 2024-07, GEE 523 rows from 2024-08) | same crawl report (short-history table) |
| Mid-caps carry more illiquid/no-trade + limit-locked + feed-glitch rows; project measured **~47 VN100 tickers with spurious zero-volume-with-price-move rows vs only 2 in VN30** | task-provided project measurement |
| News panel (PhoBERT features) covers **only 32 VN30-era tickers** | verified: `data/features/dual_group_news_panel.parquet`, `ticker` n=32 |
| Consequence: for VN100/HOSE, ~70%+ of tickers have **no news** -> gated news branch is zero-filled (effectively disabled) for them | derived: 32 of 104 VN100 tickers have news => 69% have none |
| Paper framing: VN30 positioned as a **case study of the VN market** | `MEMORY.md` (paper writing style / submission targets) |
| A quick VN100-vs-VN30 ablation is planned/running to give empirical evidence on whether more data helps | task-provided; not yet in `docs/reports/` at time of writing |

---

## 4. Comparison table

| Universe | # stocks | ~pooled samples | Liquidity / data quality | News coverage | History length | Heterogeneity |
|---|---|---|---|---|---|---|
| **VN30** | 33 | ~73k ticker-days | High; measured clean (only 2 zero-vol-glitch tickers; no >10% zero-frac; no high<low) | Full (32/33 tickers have PhoBERT news) | Long (many from 2006-2014) | Low — homogeneous large-caps |
| **VN100** | ~104 (crawled 104) | ~3x VN30 (~200k+) | Mixed; ~47 tickers with spurious zero-vol rows; some 2024 IPOs (<600 rows) | ~31% (only 32/104 have news; ~70% zero-filled) | Mixed; several short mid-cap histories | Higher — large + mid-cap mix |
| **HOSE (all)** | ~400+ (742 small-caps alone) | Much larger | Poor tail; many illiquid small-caps, limited disclosure, no-trade/limit-locked days corrupt Parkinson target | Very low (news only for ~32 large-caps) | Very mixed; many short/new listings | Very high |
| **HNX** | ~300+ | Large | Worst of the three exchanges; thin trading, short histories, weaker feed quality | ~none | Mostly short | Very high; different microstructure vs HOSE |

Notes: "samples" for a pooled model = sum over tickers of usable ticker-days (after dropping no-trade /
zero-variance rows and horizon lookahead). Illiquid names contribute fewer *usable* rows than their raw
length suggests, because no-trade / limit-locked days must be dropped or they inject a spurious zero into
the Parkinson variance target — the same reason international panel studies apply a continuous-trading
filter (Section 1.2-1.3).

---

## 5. Recommendation for THIS project

**Primary: VN30 as the news-inclusive benchmark. Secondary: VN100 price+graph-only as a scale-robustness
check. Do not use full HOSE or HNX.**

### Rationale (tied to Sections 1-3)

1. **The full model is only meaningful where news exists.** The gated news branch is a first-class
   component of this architecture, and PhoBERT news covers only 32 VN30-era tickers. On VN100/HOSE ~70%
   of tickers would run with a zero-filled news branch — that is not "more data for the same model," it
   is a *different, crippled model* evaluated on a different distribution. Any headline result about the
   news+gate+graph model must therefore be reported on VN30. (Project fact §3; verified 32-ticker panel.)
2. **Literature says ~27-33 liquid names is already a sufficient, publishable panel for a graph/HAR
   daily-vol model** — GNNHAR runs on 27 DJIA names; DJIA-30 is a standard anchor. VN30's 33 tickers sit
   squarely in that range. The scale benefit of going bigger comes with **diminishing returns** and only
   when the added names stay liquid (Sections 1.2-1.3). VN30 is also the standard liquid benchmark in the
   VN literature (Section 2), so it is the defensible, comparable choice and matches the paper's stated
   "VN30 as a case study" framing.
3. **Adding VN100's mid-caps trades quantity for target quality and homogeneity.** Every large
   international panel study filters *out* illiquid/no-trade names precisely because no-trade and
   limit-locked days corrupt the realized-vol target. This project has *measured* that degradation:
   ~47 VN100 tickers show spurious zero-volume-with-price-move rows vs 2 in VN30, plus short 2024-IPO
   histories. HOSE-wide (742+ small-caps) and HNX make this strictly worse and add cross-exchange
   microstructure heterogeneity. So the ~3x sample gain from VN100 is partly illusory (fewer *usable*
   rows per added ticker) and risks contaminating both the target and the learned graph.

### Why the hybrid rather than VN30-only

Keep VN100 in the study as a **price + directed-volume-graph-only robustness check** (news branch
disabled by construction, so the ~70% no-news problem is not a confound). This directly answers the
scientific question "does more pooled cross-sectional data improve the deep/graph model?" without
letting missing news distort the comparison — and the project already has a VN100-vs-VN30 ablation
in flight (§3) that will supply the empirical evidence. If that ablation shows a clear, honest lift
from VN100 on the price/graph-only model, report it as a scale finding; if not, it corroborates the
GNNHAR-style diminishing-returns result. Either outcome strengthens the paper.

**Explicitly not recommended:** full HOSE or HNX — illiquid tail corrupts the Parkinson target, adds
short/heterogeneous histories, near-zero news coverage, and no VN precedent for exchange-wide
constituent-level volatility modelling.

### Guardrails if VN100 is used for the robustness check
- Apply a continuous-trading / no-trade filter (drop zero-volume-with-price-move rows and limit-locked
  days) before computing Parkinson variance — mirrors Bollerslev/Panel-ML practice.
- Drop or flag <~750-row histories (e.g. DSE, GEE) so 2024 IPOs do not dominate the recent test window.
- Report VN100 results only for the news-free variant; never compare a zero-news VN100 full-model number
  against a VN30 full-model number.

---

## 6. Verification / limitations

- Verified in-repo: VN30 = 33 clean tickers; VN100 crawl = 104 tickers with the mid-cap
  short-history/quality caveats; news panel = exactly 32 VN30-era tickers (so ~70% of VN100 has no news).
- The "~47 VN100 zero-volume-glitch tickers vs 2 in VN30" figure and the "VN100-vs-VN30 ablation is
  running" status are taken as project-provided facts; the ablation result report was not present in
  `docs/reports/` at time of writing and should be cited once it lands.
- External sources are real and linked. Web access returned abstracts/summaries; exact per-study stock
  counts quoted (GNNHAR=27, Panel-ML=all-S&P500-ever, portfolio-GNN=465, full-cross-section benchmark)
  come from the fetched search summaries — the primary PDFs (linked) should be checked before quoting a
  number verbatim in the paper.

## Sources

- Corsi (2009) HAR-RV: <https://medium.com/@simomenaldo/realized-volatility-and-har-models-a-new-paradigm-for-volatility-forecasting-4a660f2530f3>
- RV methodology review (Financial Innovation 2025): <https://link.springer.com/article/10.1186/s40854-025-00809-5>
- GNNHAR (IJF 2025) PDF: <https://web.media.mit.edu/~xdong/paper/ijf25.pdf> · ScienceDirect: <https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967> · code: <https://github.com/chaozhang-ox/GNNHAR>
- Panel-ML realized volatility: <https://www.sciencedirect.com/science/article/abs/pii/S0927539823000683>
- Full cross-section RV benchmark (arXiv 2506.07928): <https://arxiv.org/pdf/2506.07928>
- GNN portfolio / S&P 500 (arXiv 2605.19278): <https://arxiv.org/html/2605.19278> · code: <https://github.com/waderylan/sp500-gnn>
- S&P 100 GNN pipeline (GitHub): <https://github.com/timothewt/SP100AnalysisWithGNNs>
- VN GARCH+LSTM VaR on VN30: <https://tapchicongthuong.vn/ung-dung-hoc-may-trong-uoc-tinh-value-at-risk--phuong-phap-ket-hop-garch-va-lstm-417219.htm>
- VN30 ARIMA / Bayesian-GARCH: <https://crimsonpublishers.com/siam/pdf/SIAM.000598.pdf>
- VN30 LSTM + Ichimoku: <https://link.springer.com/chapter/10.1007/978-981-19-2130-8_19>
- VN30F1M ARCH/GARCH: <https://tapchinganhang.gov.vn/ung-dung-mo-hinh-arch-garch-phan-tich-do-bien-dong-cua-hop-dong-tuong-lai-vn30f1m-tren-thi-truong-chung-khoan-phai-sinh-viet-nam-10191.html>
- VNIndex ML forecasting: <https://kinhtevadubao.vn/du-bao-chi-so-chung-khoan-bang-hoc-may-bang-chung-thuc-nghiem-tu-thi-truong-chung-khoan-viet-nam-29030.html>
- LSTM-GRU on HOSE index: <https://ctujsvn.ctu.edu.vn/index.php/ctujsvn/article/view/5347>
- ML+DEA returns on 26 HOSE real-estate firms: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12463195/>
- VN100 index/ETF definition: <https://vinacapital.com/investment-solutions/onshore-funds/vinacapital-vn100-etf/> · <https://www.tradingview.com/symbols/HOSE-VN100/components/>
- VN market liquidity / small-cap data quality: <https://thevietnamyield.com/vn-index-explained/> · <https://thevietnamyield.com/the-vietnam-10-blue-chips/>
