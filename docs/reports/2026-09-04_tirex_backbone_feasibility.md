# TiRex as a Temporal Backbone for the VolGA (LSTM+GAT) Architecture — Feasibility and Integration Design

Date: 2026-09-04
Status: Research / feasibility assessment only. No model was implemented, installed into a project
environment, or trained. GPU was not touched (a walk-forward chain is training on
`.venv_gpu_encode`).

Scope: assess whether the TiRex time-series foundation model (NX-AI / Sepp Hochreiter group, xLSTM-based)
could replace the per-node LSTM temporal encoder in the delivered VolGA model
(`baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py`, class `MaskedRichNet`).

Items that could not be verified from public docs during this pass are marked
**[VERIFY LIVE]** and must be confirmed against the installed package / source before any build.

---

## 1. What TiRex is

**Origin / lineage.** TiRex is a pretrained time-series foundation model from NX-AI (the company
commercialising xLSTM; scientific lead Sepp Hochreiter, co-inventor of the LSTM). It is described in
the NeurIPS-2025 paper "TiRex: Zero-Shot Forecasting across Long and Short Horizons with Enhanced
In-Context Learning" (Auer, Podest, Klotz, Böck, Klambauer, Hochreiter; arXiv:2505.23719).

**Architecture.** Decoder-only model built on **xLSTM** (a recurrent architecture that scales
linearly in sequence length, contrast with quadratic attention). It includes custom **sLSTM CUDA
kernels** that compile on first load. It forecasts by *masking* future steps (marking them missing)
rather than autoregressive feedback; the xLSTM hidden state carries forward both the point ("median")
and uncertainty. The paper highlights "state tracking" as a capability transformers lack. Training
technique is called CPM (a training-time masking strategy).

**Size.** ~35M parameters — small relative to peers (Chronos-Bolt-Base ~200M, Toto ~151M,
TimesFM-2.0 ~500M). Memory-efficient; NX-AI markets edge/PLC deployment.

**Pretraining corpus.** Generic time-series mixtures: the Chronos training corpus, GIFT-Eval data, and
synthetic series. There is **no indication the pretraining set is finance-specific**, and in
particular nothing Vietnamese-equity or realized-volatility specific. **[VERIFY LIVE]** exact corpus
composition against the paper's data section.

**Zero-shot vs fine-tune.** Primary mode is **zero-shot** (download and forecast, no training on your
data). **Public fine-tuning is NOT offered** — the model card / repo direct custom-tuning requests to
`contact@nx-ai.com`. So "fine-tune the backbone" is, in practice, not a supported first-class path for
the open weights; only frozen zero-shot use is documented.

**Input / output format.**
- Input: context tensor shaped `(batch_size, sequence_length)` — i.e. **univariate** series, batched.
  Maximum context length **2048**. The quick-start uses length-128 contexts.
- Output: `quantiles, mean = model.forecast(context=..., prediction_length=L)`. Returns **9 quantile
  predictions plus a point mean** over the requested horizon. Multivariate / covariate support does
  **not** exist in TiRex-1; it was added only in the successor **TiRex-2** (38.4M active +44.1M for
  multivariate mode), which is a separate model.

**License.** **NXAI Community License Agreement.** Research/academic use and result publication are
permitted; redistribution permitted **with attribution** ("prominently display 'Built with technology
from NXAI'" and retain notices). Commercial restriction applies only to organisations with
consolidated annual revenue > €100M (they must obtain a separate commercial license). Includes an
IP-litigation termination clause, indemnification, no trademark grant, Austrian jurisdiction (Linz).
For a public academic thesis repo this is **usable**, provided the attribution string is displayed and
the license text is retained. It is **not** OSI/MIT/Apache — reviewers who require a permissive
license should be told it is a source-available community license.

**Weights openly downloadable?** Yes — `NX-AI/TiRex` on Hugging Face; package `tirex-ts` on PyPI;
code at `github.com/NX-AI/tirex`; docs at `nx-ai.github.io/tirex`. No gated access observed
**[VERIFY LIVE]** (some HF models require accepting terms before download).

**Dependencies / install.**
- `pip install tirex-ts` (extras: `[cuda,gluonts,hfdataset]`). Depends on the **xlstm** library.
- A conda spec `requirements_cu124.yaml` targets **CUDA 12.4**. Exact torch/python pins not stated in
  the README **[VERIFY LIVE]** from the package metadata.

**Hardware / platform.**
- GPU with **CUDA compute capability ≥ 8.0** (Ampere+, e.g. RTX 30xx / 40xx). An RTX 4060 (Ada, cc 8.9)
  qualifies.
- CPU execution is possible but the custom kernels accelerate GPU inference.
- **Tested on Linux and macOS only.** Windows is not listed as supported — the custom sLSTM CUDA
  kernel compilation is the specific risk (see §4).

---

## 2. Integration design — TiRex as the per-node temporal encoder

### 2.1 The interface it must satisfy

`MaskedRichNet.forward(x, adj_b)` receives `x` of shape **[B, N, seq, 5]** (B snapshots, N nodes,
`lookback` days, 5 node features) and does:

```
out, _ = self.lstm(x.reshape(B*N, seq, 5))   # per-node sequence -> hidden
h = out[:, -1].reshape(B, N, hidden)          # last-step embedding, hidden=64
parts = [h] (+ GAT branch on x[:, :, -1, :])
return self.head(cat(parts))                  # per-node scalar volatility
```

The LSTM's only job is: **per-node sequence [seq,5] → a fixed embedding vector** (64-d) that the head
(and, concatenated, the GAT branch) consumes. Any replacement must produce a per-node embedding of
known width from the per-node window. Note two properties of the current design that constrain the
swap:
- `lookback` default is **10** (`run(..., lookback=10)`); sequences are short.
- The **GAT branch does not use the LSTM output** — it reads raw features at day t
  (`x[:, :, -1, :]`). So replacing the LSTM only changes the temporal branch `h`; the graph branch is
  untouched. This means a TiRex swap tests "better temporal embedding" in isolation, which is clean.

### 2.2 The mismatches that must be adapted

1. **Univariate vs 5-dim features.** TiRex ingests a univariate context `(B, L)`. The node has 5
   features `[parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore]`. Options:
   - (a) Feed only the **base series** (parkinson_variance, or its sqrt = the vol series) to TiRex and
     keep HAR/market/volume features on a side path concatenated after. This is the most faithful use
     of a univariate foundation model.
     - (b) Run TiRex **once per feature channel** (5 passes) and concatenate the resulting
       embeddings/forecasts. 5× inference cost; the 4 engineered features are not natural "series" for
       a forecaster (e.g. volume_zscore), so semantics are dubious.
   - (c) Use **TiRex-2** (native multivariate/covariate) instead — but that is a different, larger
     model and a bigger dependency; out of scope for a first probe.
   - Recommended for a first probe: **(a)**.

2. **No documented embedding/`encode()` API.** The public surface is `forecast(context, prediction_length)`
   returning `(quantiles, mean)` — **not** a hidden-state extractor. Two ways to still get a per-node
   vector:
   - **Forecast-as-embedding:** use the 9 quantiles + mean at the target horizon (a 10-d vector, or a
     [L×10] block for multi-step) as the per-node "embedding" feeding the GAT/head. This uses only the
     public API and is robust to version changes.
   - **Hidden-state hook:** register a forward hook on the last xLSTM block to pull the latent state.
     This is **undocumented / private** and will break across versions. NX-AI does ship downstream
     "TiRex classification" and "TiRex regression" models, and there is external work
     ("Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for Time Series
     Classification") confirming the internal representations are usable as features — but **no stable
     public extraction API is documented**. **[VERIFY LIVE]** whether `tirex-ts` exposes any
     `embed()` / `output_hidden_states` before relying on it.

3. **Context-length regime mismatch (the deepest concern).** TiRex is built for and evaluated on
   contexts up to 2048; its headline strength is *in-context learning over long histories*. The VolGA
   node window is **~10 days**. Feeding 10-step contexts uses the model far outside the regime where it
   is strong, and the pretraining prior over generic series may not transfer to a 10-step
   Parkinson-variance snippet. If TiRex is adopted, the natural change is to give it a **long per-node
   context** (e.g. 250–2048 days of the raw vol series) rather than the 10-day window — which is a
   deviation from the current design and must be kept leakage-safe (context must end at or before the
   anchor day t, never crossing into the horizon).

4. **Frozen vs fine-tune.** Because public fine-tuning is unsupported, the realistic design is a
   **frozen TiRex encoder**: run it in inference (no grad), cache per-node embeddings/forecasts once
   per fold, then train only the (small) GAT + head on top. This is also the cheapest and most
   leakage-controllable option.

### 2.3 Leakage-safety and the walk-forward loop

The existing `build_masked_rich` estimates every scaler and both edges on **train rows only**, per
fold. A frozen TiRex is inference-only and learns nothing from the data, so it introduces **no fold
leakage** by construction — provided each per-node context window is truncated to end at the anchor
day t (no peeking at t+1..t+horizon). Concretely:
- For each fold and each anchor date t, build the per-node context as the ticker's own realized-vol
  series up to and including t (length ≤ 2048), run `model.forecast(context, prediction_length=horizon)`,
  take the quantiles/mean (or hooked embedding) as the node vector.
- Cache these `[n_anchors, N, emb_dim]` arrays once (they do not depend on any trainable weights) and
  feed them where `h` currently is. The GAT + head then train exactly as now.
- The per-node target scaler, HAR/HAR-X baselines, edges, purge, and DM machinery stay unchanged, so
  the comparison to HAR remains apples-to-apples.

### 2.4 Engineering effort and risks

- **Effort:** New baseline folder `baselines/YYYY-MM-DD_tirex_backbone/` per §3.F, with: a thin TiRex
  wrapper (load once, batched forecast over N series, cache), a variant of `MaskedRichNet` where the
  LSTM branch is replaced by the cached embedding, and reuse of the existing trainer/DM/report path.
  Estimated **2–4 focused days** *if* installation succeeds on this machine; most of the risk is in
  install/platform, not modelling.
- **Risks (ranked):**
  1. Windows + custom sLSTM CUDA-kernel compilation may fail (unsupported platform) — see §4.
  2. Dependency conflict with the project's torch / CUDA stack — see §4.
  3. No stable embedding API → dependence on forecast-outputs (acceptable) or private hooks (fragile).
  4. Context-regime mismatch → design pressure to lengthen the node window, changing the experiment.
  5. Most likely outcome is **no lift over HAR** (see §3), so effort should be gated behind a cheap
     zero-shot pre-check.

---

## 3. Honest assessment — is a foundation-model backbone likely to help here?

**The established finding.** Across this project's experiments, a linear **HAR** model is very hard to
beat on the Vietnamese-market volatility target; per-node LSTM, +GAT graph (correlation and directed
vol→PK edges), news, gates, and edge-construction variants have repeatedly failed to beat HAR under
Diebold-Mariano on OOS QLIKE. The diagnosed bottleneck is **signal/data**, not backbone capacity:
the target is a derived Parkinson **variance** series with roughly **−0.30 anti-persistence
autocorrelation** and (on thin panels like HNX) large fractions of near-zero targets. Overnight range
noise is not persistent, so there is little for any temporal model to forecast beyond the HAR mean.

**What prior non-LSTM backbones showed.**
- **CryptoMamba (Mamba/SSM) enhanced** — result JSON at
  `results/cryptomamba_enhanced_2026-06-20_002016/cryptomamba_enhanced_results.json`: test
  **R² = −1.10**, test **QLIKE = 1.00**, directional accuracy ~2%. This is catastrophically worse than
  HAR — a Mamba SSM backbone did not help; it hurt. (Multiple later cryptomamba runs exist as `.pth`
  only, no better JSON located.)
- **TimesNet** — checkpoints exist (`results/timesnet_baseline_2026-06-20_*/best_timesnet_model.pth`)
  but **no result JSON demonstrating a beat-HAR outcome was located**. **[VERIFY LIVE]** if a metrics
  file exists elsewhere.
- **TimesFM (LoRA fine-tune)** — implemented (`src/timesfm_baseline/`, docs in `docs/timesfm_*`,
  transformers `timesfm`/`timesfm2_5` are installed in `.venv_gpu_encode`), but **no delivered result
  JSON showing it beat HAR on the walk-forward eval was located**. All of these predate the current
  HAR-anchored walk-forward protocol, so even where numbers exist they are not directly comparable.

**Where TiRex could plausibly differ.** Its one genuinely new ingredient is **external pretrained
knowledge** — every prior backbone (LSTM, Mamba, TimesNet) learned only from the scarce in-domain
data and hit the same data ceiling. A foundation model brings a prior learned from millions of
external series. That is the *only* mechanism by which it could help where in-domain deep models did
not. However, the counter-arguments are strong: (i) TiRex's prior is generic (Chronos/GIFT/synthetic),
not finance/vol-specific; (ii) the target's anti-persistence means the "true" best forecast is close
to a shrinkage-to-mean that HAR already captures — a richer prior cannot manufacture signal that is
not in the series; (iii) recent literature is openly skeptical that TS foundation models are truly
"foundational" for forecasting (arXiv:2510.00742, "How Foundational are Foundation Models for Time
Series Forecasting?"). The realistic expectation is **another null**, with a small chance of a modest
QLIKE improvement at short horizons where some vol persistence exists.

**Recommendation: run a scoped, cheap pre-check — do NOT build the full integration first.**
The integration (encoder swap + cached embeddings + GAT retrain) is only worth building if TiRex shows
any edge on the *raw univariate task* it is actually designed for. So:

**Minimal experiment (no GAT, no integration, frozen model, ~hours not days):**
1. Take each ticker's realized-vol series (sqrt of parkinson_variance) on the existing VN30/VN100
   walk-forward test folds.
2. For each test anchor, zero-shot `model.forecast(context = series up to t, prediction_length = horizon)`,
   take the mean (and/or an appropriate quantile) as the volatility point forecast.
3. Score QLIKE / RMSE / MSE / MAE / R² against the same targets, at horizons {1, 5, 10, 22}, using the
   **same positivity floor** as HAR, and run the same date-clustered DM test **TiRex-zero-shot vs HAR**.
4. Decision gate: if zero-shot TiRex cannot at least match HAR on QLIKE at any horizon, the
   encoder-swap-into-GAT will not help either (the graph branch has already been shown not to add OOS
   value), and the effort should stop. If it matches/beats HAR at some horizon, proceed to the frozen
   -embedding-into-GAT design in §2.

This pre-check needs only CPU/GPU inference of a 35M model over a few hundred short series — cheap,
leakage-safe, and it directly tests the single hypothesis (external prior beats the data ceiling)
before any architecture work.

---

## 4. Practical blockers

1. **Platform (highest risk).** TiRex is **tested on Linux/macOS only**; the project runs on
   **Windows 11**. The custom **sLSTM CUDA kernels compile on first load** — on Windows this commonly
   fails (toolchain/Triton/ninja/MSVC issues). Mitigations, in order: run under **WSL2**; use a CPU/
   pure-PyTorch fallback path if `tirex-ts` provides one **[VERIFY LIVE]**; or run the pre-check on a
   Linux box. This should be resolved *before* any modelling effort is scheduled.

2. **Environment / dependency conflicts.** Do **not** install into `.venv_gpu_encode` (the training
   venv) or any existing project venv. Note the GPU venv is **Python 3.10** (`cp310` artifacts observed)
   and already carries a specific torch + `transformers` (with `timesfm`, `timesfm2_5`) — CLAUDE.md
   states the project standard is Python 3.11, so there is already version heterogeneity. `tirex-ts`
   depends on the `xlstm` library and targets CUDA 12.4; its torch pin **[VERIFY LIVE]** may not match
   the training venv's torch. Correct approach: a **fresh throwaway venv** (or WSL2 env) dedicated to
   TiRex, embeddings cached to disk, so nothing touches the training stack. Exact torch/python pins
   must be read from the installed package before committing.

3. **VRAM (8GB RTX 4060; delivered model already uses ~5–7GB).** TiRex at 35M params is small; frozen
   inference over N univariate series (N up to ~30 VN30 / ~100 VN100 / ~500 SP500) is memory-cheap if
   batched and run **before/separately** from GAT training, with embeddings cached to disk. The 8GB
   budget is a problem only if one tries to hold TiRex resident **and** train the GAT simultaneously —
   which the cache-then-train design avoids. If fine-tuning were attempted (unsupported anyway),
   activation memory over length-2048 contexts could pressure 8GB; frozen inference will not.

4. **License compatibility with a public repo.** NXAI Community License permits research use,
   publication, and redistribution **with attribution**; the €100M-revenue commercial gate does not
   affect an academic thesis. Requirements to honor: retain the license text, display
   "Built with technology from NXAI", and note in the paper that TiRex is source-available under the
   NXAI Community License (not a permissive OSI license). No weights need be committed to the repo
   (load from Hugging Face at runtime), which keeps the public repo clean.

---

## Sources

- TiRex GitHub: https://github.com/NX-AI/tirex
- TiRex on Hugging Face: https://huggingface.co/NX-AI/TiRex
- NXAI Community License: https://raw.githubusercontent.com/NX-AI/tirex/main/LICENSE
- TiRex paper (arXiv:2505.23719): https://arxiv.org/abs/2505.23719
- TiRex product page: https://www.nx-ai.com/en/tirex
- tirex-ts on PyPI: https://pypi.org/project/tirex-ts/
- Analysis: https://aihorizonforecast.substack.com/p/tirex-lstms-take-the-lead-again-in
- Skepticism ref (arXiv:2510.00742): https://arxiv.org/pdf/2510.00742
- TiRex-2 (multivariate successor): https://www.nx-ai.com/en/news/tirex-2
- Local evidence: `results/cryptomamba_enhanced_2026-06-20_002016/cryptomamba_enhanced_results.json`;
  `baselines/2026-08-21_har_anchored_residual/code/{masked_rich.py,run_masked_rich.py}`

Verification still required (marked [VERIFY LIVE] above): exact torch/python pins of `tirex-ts`;
whether any public embedding/`encode` API exists; whether HF weights are gated; whether a CPU/no-kernel
fallback exists for Windows; TiRex pretraining-corpus composition; existence of any TimesNet/TimesFM
beat-HAR result JSON in the repo.
