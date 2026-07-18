# Summary of Update — 2026-07-11 Latent Noise Baseline (Tier A)

**Date:** 2026-07-11
**Trigger:** Teacher's hint #2 — "add randomly-generated vectors for sparse-news case". User asked to build + train 5 epoch, then continue 5 more (total 10).

## What changed
New isolated baseline `baselines/2026-07-11_latent_noise/` (rule §3.F: 5 sub-folders). Implements **Tier A Latent Noise Injection** = subclass `EmbeddingBaseline` + add `news_rep += σ·ε` (σ=0.1) on the news representation **train-only** (off at eval). No loss change.

## Files (path → purpose)
- `baselines/2026-07-11_latent_noise/requirements/requirements.md` — goal, success criteria, go/no-go
- `baselines/2026-07-11_latent_noise/design/design.md` — subclass-reuse decision, data flow, hyperparams
- `baselines/2026-07-11_latent_noise/code/model_latent_noise.py` — `LatentNoiseBaseline(EmbeddingBaseline)`, noise on news_rep
- `baselines/2026-07-11_latent_noise/code/train_latent_noise.py` — train loop (reuse sibling's train_epoch/validate/dataloaders), `--noise_std`, `--resume_from`
- `baselines/2026-07-11_latent_noise/test/test_latent_noise.py` — 7 tests (shape, eval determinism, train stochasticity, randn-call-count ×3, backward)
- `baselines/2026-07-11_latent_noise/code_review/code_review_2026-07-11.md` — adversarial self-review artifact
- `docs/report_2026-07-11/BAO_CAO_TUAN_CHO_THAY.md` — updated §0, §1.1, §1.5, §2.4.6, §5 with results

## Tests + coverage
- **pytest:** `python -m pytest baselines/2026-07-11_latent_noise/test/ -v` → **7/7 pass**.
- **diff-coverage:** Not measured (small subclass; tests cover the noise property directly via randn-call-count, which is the only new logic). Follow-up: add if diff-cover tooling wired.

## Code-review result + actions (REQUIRED)
- **Method:** adversarial self-review + pytest. No HIGH bug.
- **Fixed during build (1):** confounded `noise_std=0 → train==eval` test (Dropout in inherited news_pool/news_temporal also makes train≠eval) → replaced with precise `randn_like` call-counter (noise uses randn, dropout uses bernoulli).
- **Inherited/accepted (documented in design + review):** hardcoded dropout 0.2 in sibling's pooling/temporal (not noise-related, sibling already reviewed); subclass duplicates ~5-line forward (no inject hook in parent); σ=0.1 untuned.
- **Artifact:** `baselines/2026-07-11_latent_noise/code_review/code_review_2026-07-11.md`.

## Commands actually run
- `python -m pytest baselines/2026-07-11_latent_noise/test/ -v` → 7 pass ✓
- `python .../train_latent_noise.py --epochs 5` → results/latent_noise_2026-07-11_121518/ ✓
- `python .../train_latent_noise.py --epochs 5 --resume_from models/latent_noise_2026-07-11_121518/best.pt` → results/latent_noise_2026-07-11_124004/ ✓

## Results (val DirAcc curve, resume = epochs 6-10)
`68.48→68.12→69.43→70.40→69.58` | `68.25→69.14→69.26→69.43→71.28(best)`

| Model | Epochs | Test DirAcc | R² | QLIKE | vs HAR-only 69.98% |
|-------|:------:|:-----------:|:--:|:-----:|:------------------:|
| Latent-noise (5ep) | 5 | 69.09% | 0.712 | 0.560 | −0.89 |
| **Latent-noise (10ep)** | 10 | **69.33%** | 0.713 | 0.544 | −0.65 |
| Embedding (40ep, ref) | 40 | 68.76% | 0.717 | 0.553 | −1.22 |
| HAR-only (70ep, ref) | 70 | 69.98% | 0.714 | 0.529 | — (best) |

→ Latent-noise = **highest among 5 news variants** (+0.57% over embedding, better QLIKE), but still −0.65% vs HAR-only. **Marginal positive signal** (teacher's hint shows a small effect), not conclusive (epoch mismatch).

## DoD checklist
- [x] Code satisfies request (Tier A latent noise, 10 epoch, reported)
- [x] Tests run (7/7 pass)
- [x] Code review run + findings addressed (self-review, 1 fixed, rest inherited/documented)
- [x] Summary report generated (this file)
- [ ] diff-coverage ≥80% — Not measured (only new logic = noise line, covered by randn-counter test)
- [ ] Smoke gate (tag `smoke`) — no `@smoke` test written (follow-up); pytest covers happy path
- [x] Isolation verified — no edits to src/ or sibling baselines (subclass + read-only imports)

## Risks / follow-ups
1. **Not matched-epoch** (10 vs 40 vs 70) — ±0.5% could be noise. Need matched-epoch control to confirm lift is real.
2. **σ=0.1 untuned** — sweep 0.05/0.1/0.2.
3. **Train more** — curve still rising (best val 71.28% at ep10); 40-epoch run may clarify.
4. **Bug SSB** still open — affects all multi-stock numbers' reproducibility (§4.1).

## Honest note
Latent noise is the **first news variant to (marginally) beat the embedding baseline** — a small vindication of the teacher's hint. But it does NOT beat HAR-only, and the comparison isn't epoch-matched yet. Reported honestly with caveats, not claimed as a breakthrough.
