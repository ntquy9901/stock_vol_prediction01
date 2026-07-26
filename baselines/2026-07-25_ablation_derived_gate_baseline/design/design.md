# Design (Plan) — Ablation-Derived Gate Baseline

Identical structure to `2026-07-25_top3_news_gate_baseline` (subclass `SelectiveGateNewsBaseline`,
override the ticker allowlist only). File list:

| File | Trách nhiệm |
|---|---|
| `code/model_ablation_gate.py` | `AblationDerivedGateBaseline(SelectiveGateNewsBaseline)` — mask = 11-ticker allowlist from the ablation |
| `code/train_ablation_gate.py` | Train loop, copy of `train_top3_gate.py` with model swapped |
| `test/test_mask_correctness.py` | Same exact-zero-contribution test pattern as the two sibling gate baselines |

Isolation: read-only import from `2026-07-25_selective_news_gate_baseline` and
`2026-07-25_dual_group_news_embedding_baseline`. No edits to siblings or the ablation baseline's
result files.
