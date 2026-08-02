# Code Review — Sentiment ↔ Price EDA (2026-07-11)

**Tool:** Adversarial review (general-purpose agent, cynical mode), scoped to NEW baseline code only.
**Files reviewed:** `code/sentiment_price_eda.py`, `test/test_smoke.py`.
**Reason for scoped review:** Working tree contains many pre-existing unrelated modifications; `/code-review` over full tree would be polluted. Review focused on the 2 new files.

## Findings & actions

| # | Sev | Finding | Action |
|---|-----|---------|--------|
| F1 | HIGH | Pooled Mann-Whitney pseudoreplication + 23 neg events from only 14 tickers (6 with ≥2) → ticker-confounded | **Fixed.** Added per-ticker-demeaned MW (`p_demeaned`) alongside raw; added `neg_composition()` reported in summary. |
| F2 | HIGH | 5 horizons tested at α=0.05 with no multiple-comparison correction → ~23% false-GO under null | **Fixed.** Bonferroni (α/n=0.01); pre-registered primary horizon T+5; GO requires primary significant after correction AND spread≥30bp. |
| F3 | HIGH | Winsorize on full per-ticker series (incl. non-event days) before selecting events → bias toward null; 30bp threshold undocumented | **Fixed.** `compute_forward_returns` is now a pure shift (no winsorize); winsorize moved to event level (`winsorize_events`, per-horizon pooled). Threshold documented as heuristic ~0.3%/period. |
| F4 | MED | Sentiment asymmetric (min −0.5, max +0.75); neg is extreme-tail, not mirror of pos | **Documented.** `note_class_imbalance` + `sentiment_median` per group in summary. |
| F5 | MED | `volatility_comparison` ret vs vol means over different ticker sets (apples-to-oranges) | **Fixed.** Now computes per-ticker (ret_rho, vol_rho) PAIRS over intersection; `n_tickers_paired` reported. |
| F6 | MED | `per_ticker_corr` includes neutral days; interpretation misleading | **Documented + added.** `per_ticker_corr_note` in summary; added `corr_posneg_ret5d` (pos+neg only). |
| F7 | MED | Fig 3 comment says "spread" but plots corr; guard checked wrong DataFrame | **Fixed.** Guard now `if "corr_ret5d" in tc.columns`; comment corrected. |
| F8 | MED | bp unit undocumented; silent if data unit changes | **Fixed.** `units` field in summary; assertion `max abs return < 2.0` guards decimal assumption. |
| F9 | LOW | vol dict-key lookup fragile (`str(k) if ... else k`) | **Fixed.** Keys normalized to int; lookup simplified. |
| F10 | LOW | No guard against close≤0 → inf | **Fixed.** `close = np.where(close<=0, nan, close)` + errstate. Test added. |
| F11 | LOW | Property test circular (re-used winsorize) | **Fixed.** compute_forward_returns no longer winsorizes → test compares to raw formula directly. |
| F12 | LOW | No tests for statistical functions | **Fixed.** Added tests: MW detects/doesn't detect known effect, `_spearman_safe` degenerate→NaN, `neg_composition` counts. |

## Result
- All HIGH/MEDIUM findings fixed or honestly documented.
- Tests: **10/10 pass** (`pytest test/ -v`).
- Re-run verdict: **NO-GO** (robust under stricter Bonferroni α=0.01 and per-ticker demeaning; primary T+5 p_demeaned=0.7472).

## Honest caveat carried into report
With only 23 negative events across 14 tickers (6 with ≥2), **no pooled test can produce a trustworthy directional conclusion**. NO-GO here means "insufficient evidence of a usable price-direction signal" — not "sentiment definitely has no relationship."

---

## Market-level script review (sentiment_market_eda.py + test_market_smoke.py)

Second adversarial pass (same scope rules). 9 findings: **2 HIGH, 4 MEDIUM, 3 LOW**.

| # | Sev | Finding | Action |
|---|-----|---------|--------|
| F1 | HIGH | Event study used `mkt_realized_vol` (trailing 22d, backward) as a "next-day" target → mislabeled; claim "lower subsequent vol" was actually "lower contemporaneous trailing vol" | **Fixed.** Added forward vol `mkt_fwd_vol_22d` (window [T+1..T+22]); event study now uses forward targets; keys/titles corrected. Re-ran: result **robust under forward vol** (ratio 0.865, p=1e-11) — the directional finding survives correct methodology. |
| F2 | HIGH | `mkt_realized_vol` as corr target mixed backward window with forward lags | **Fixed.** Corr vol-targets now use `mkt_fwd_vol_22d` (predictive); trailing realized_vol kept only for the time-series visualization (plot 4). |
| F3 | MED | `mkt_news_count` corr on 4889 days (75% zeros) vs other measures on 1197 days — non-comparable samples | **Documented** in caveats (sample-mismatch row). |
| F4 | MED | 80 correlations, no multiple-testing correction | **Documented** in caveats (~4 false positives expected under null; treat as exploratory). |
| F5 | MED | Basket composition drifts (no min stock count) | **Documented** in caveats (composition varies; equal-weighted, not cap-weighted). |
| F6 | MED | Detrend checks only covered trailing vol, not forward-return headline targets | **Fixed.** Detrend now checks sent_mean.diff vs mkt_ret_1d/5d/fwd_vol (all ~0.06 or lower → no spurious trend signal). |
| F7 | LOW | `ratio_news_over_none` truthiness guard fragile | **Fixed.** `denom = mean(none); ratio if denom>0`. |
| F8 | LOW | `mkt_ret_1d` (fwd) vs `mkt_ret_1d_realized` (backward) naming collision | **Mitigated** — added fwd_vol with explicit name; direction documented in caveats. |
| F9 | LOW | Event-study test couldn't detect the backward-vol bug (pre-aligned input) | **Updated** test to feed `mkt_fwd_vol_22d`; remains a structural test. |

Result: all HIGH/MEDIUM fixed or documented; tests **14/14 pass**; market verdict unchanged (no return-direction signal; weak negative attention→forward-vol link).

---

## News-type script review (sentiment_newstype_eda.py + test_newstype_smoke.py)

Third adversarial pass. 10 findings: **3 HIGH, 4 MEDIUM, 3 LOW**.

| # | Sev | Finding | Action |
|---|-----|---------|--------|
| H1 | HIGH | NEG bucket ~62% false positives — bare `'bán'` matched *bán lẻ/bán hàng/bán nhà/doanh số bán*; `'tranh'` matched *cạnh tranh* | **Fixed.** Added `NONRATING_COMPOUNDS` strip before rating extraction; removed `'tranh'` (use accented `'tránh'`/`'né tránh'`). NEG 21→**7** genuine sells (matches manual audit). |
| H2 | HIGH | "Giữ KN MUA" (maintain BUY) mislabeled NEG via *bán hàng* + NEG-precedence | **Fixed** by H1 compound strip (verified by regression test). |
| H3 | HIGH | "news→lower vol" pooled result confounded by coverage trend (news sparse pre-2018, dense post-2018 = different vol regimes) | **Fixed.** Added **year-matched** ratio (news vs no-news WITHIN same year). Result: pooled 0.865 → **year-matched median 1.034** → effect **vanishes**. The earlier "news→13% lower vol, p=1e-11" was largely a coverage artifact. |
| M1 | MED | `' q1'` (leading space) missed `2Q20`/`1Q21` quarterly format | **Fixed.** `q1..q4` + `1q..4q` variants. earnings 171→182. |
| M2 | MED | No multiple-testing caveat | **Documented** in interpretation_notes. |
| M3 | MED | NEG→higher-return reported without mean-reversion caveat | **Documented** (sells issued post-drop; not predictive). |
| M4 | MED | `'kiện'` substring matches *điều kiện* | **Fixed.** → `'tố kiện'`/`'vụ kiện'`. |
| L1-3 | LOW | NaN→"nan" string; `'nắm'` redundant; diacritic normalization | Documented/minor; not blocking. |

**Decisive outcome:** the year-matched analysis (H3) **overturns** the market-level "news→lower vol" finding. After controlling for coverage period, news has essentially NO effect on subsequent market volatility (median ratio ≈ 1.0). Only earnings-updates show a mild residual (~6% lower within-year, consistent with scheduled-info resolution). Tests **24/24 pass**.


