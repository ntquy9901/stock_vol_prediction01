# SOTA backbones to replace the LSTM encoder — deep-research synthesis (2026-08-30)

Deep-research workflow (102 agents, 5 search angles → fetch 20 sources → 3-vote adversarial verification;
25/25 claims confirmed, 0 refuted). **Verification caveat:** the safety classifier was unavailable during
the verify pass — re-open every arXiv/DOI link before citing any of these in the paper.

## Headline answer

For generic multivariate time-series benchmarks, several 2023–2024 backbones verifiably beat older
Transformers/LSTM. **But for the CRITICAL question — does any deep backbone reliably beat HAR/GARCH on
DAILY volatility with daily inputs — the evidence is MIXED and conditional: deep wins hold mainly at
MONTHLY/longer horizons or with RICHER inputs (intraday RV, cross-stock commonality), not in like-for-like
daily-RV-only comparisons, where HAR remains very hard to beat.** This directly corroborates the project's
own finding (HAR-X beats the deep models on VN100; the GAT graph adds no robust OOS value).

## Verified backbone candidates (arXiv, 3-0 confirmed)

| Backbone | Mechanism | Evidence | Citation |
|---|---|---|---|
| **DLinear/NLinear** | 1-layer linear + trend/seasonal decomp | beats pre-2023 Transformers "often by a large margin" (9 datasets) — parsimony prior for noisy data | Zeng+ AAAI'23, arXiv:2205.13504 |
| **PatchTST** | subseries patch tokens + channel-independence | significantly improves long-term accuracy vs SOTA Transformers; independently reused as strong baseline | Nie+ ICLR'23, arXiv:2211.14730 |
| **iTransformer** | per-variate tokens, attention ACROSS variates | SOTA multivariate; independently corroborated by TFB benchmark | Liu+ ICLR'24, arXiv:2310.06625 |
| **S-Mamba** | bidirectional selective-SSM + FFN | leading perf on 13 datasets at low compute (self-reported); ⚠ bidirectional = look-ahead leakage risk | Wang+ 2024, arXiv:2403.11144 |
| **xLSTM** | sLSTM + mLSTM, exponential gating | favorable vs Transformers/Mamba — but on LANGUAGE, not TS/vol | Beck+ NeurIPS'24, arXiv:2405.04517 |
| **Chronos / TimesFM** | foundation models (tokenize+T5 / decoder-only) | strong in-corpus + zero-shot on 42+ datasets; no vol-vs-HAR head-to-head | 2403.07815 / 2310.10688 |
| **TFB** (independent benchmark) | 3rd-party fair benchmark, 25 multivariate datasets | the tool to compare backbones fairly | Qiu+ PVLDB'24, arXiv:2403.20150 |

## Critical: deep-vs-HAR on volatility (verified)

- **Positive but SCOPED:** Bucci 2020 (JFEC 18(3):502) RNNs>econometric **for MONTHLY RV**; Christensen/Siggaard/Veliyev 2023 (JFEC 21(5):1680) ML>HAR on DJIA RV but **"gains more pronounced at longer horizons"**; Zhang+ 2022 (arXiv:2202.08962) NNs>daily-RV baselines **only via intraday RV + cross-stock commonality** (input asymmetry); HARNet (arXiv:2205.07719) beats HAR but is **HAR-structured**.
- **Graph/spillover corroboration:** Zhang/Pu/Cucuringu/Dong 2023 (arXiv:2308.01419, = GNNHAR, already in our refs) — spillover helps **only short horizons (≤1 week)**; multi-hop alone gives no clear advantage; much gain is QLIKE-vs-MSE-loss driven. **Exactly our VN result** (graph helps h1 only; 1-hop ≥ 2-hop).
- **Do NOT use LLM backbones:** Tan+ NeurIPS'24 (arXiv:2406.16964) — removing the LLM from Time-LLM/GPT4TS does not degrade (usually improves) forecasting.
- **Mamba-in-graph precedents** (movement/price, not vol): FinMamba (arXiv:2502.06707), SAMBA (arXiv:2410.03707).

## Ranked recommendation (if pursuing a backbone swap)

1. **Causal Mamba / S-Mamba** — near-linear cost, financial graph-hybrid precedents; MUST use a unidirectional/causal variant (bidirectional S-Mamba/SAMBA leak future).
2. **iTransformer** — SOTA multivariate, independently corroborated, cross-variate design aligns with the cross-stock graph.
3. **PatchTST** — channel-independence = robustness prior for a noisy panel.
4. **xLSTM (mLSTM)** — minimal-change swap from the current LSTM, but TS/vol evidence thin.
5. **DLinear/NLinear** — mandatory simple control (may win on small noisy data).

**Overriding caveat:** on this small, noisy, ~104-stock/~4000-day emerging-market DAILY-Parkinson panel,
**no backbone is likely to reliably beat HAR with daily inputs alone.** The literature's deep wins are
horizon- or richer-input-conditional; the project's own GAT graph hurts. Highest-value next step is likely
NOT a fancier backbone but (a) the walk-forward retrain test, and (b) richer inputs (if intraday data were
available) — the one lever the literature shows actually flips deep-vs-HAR.

## Open question (the core risk)
No verified study shows a deep backbone with ONLY daily RV/Parkinson inputs beating HAR OOS on daily
volatility for a small emerging-market panel. That gap is exactly this project's contribution space.
