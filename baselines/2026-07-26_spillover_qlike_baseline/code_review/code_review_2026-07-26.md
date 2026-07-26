# Code Review (self-adversarial) — 2026-07-26_spillover_qlike_baseline

**Reviewer:** self-review (Claude), no `/code-review` interactive checkpoint — user explicitly
asked for this whole session to run unattended ("tự làm hết ... không cần tôi approve ... tôi đi
đây"), matching the precedent set by `2026-07-25_macro_news_baseline`'s overnight self-review.
3-layer adversarial pass (Blind Hunter / Edge Case Hunter / Acceptance Auditor) applied to the 4
new files.

## Findings

### 1. [MINOR, FIXED] Spurious `noqa: F401` on a fully-used import
`dataset_spillover_news.py` imported `HAR_COLS, load_news_panel, _norm_date` from the sibling's
`dataset_dual_news` with a `# noqa: F401 (re-exported)` comment, but all three names ARE used
directly inside this file (not just re-exported) — the noqa was misleading/unnecessary.
**Fix applied:** removed the noqa comment.

### 2. [MINOR, ACCEPTED — not a bug in this baseline's scope] `build_denorm_tensors` assumes fitted normalizers
`losses.build_denorm_tensors(dataset, device)` indexes `dataset.target_normalizers[sname]` for
every stock without a defensive check. If called on a dataset built with `normalize=False`, or
before `create_spillover_news_dataloaders`'s fit loop runs, this raises `KeyError`. In this
baseline's actual entrypoint (`train_spillover_qlike.py::main`), `create_spillover_news_dataloaders`
always fits normalizers before `build_denorm_tensors` is called (verified by reading the call
order), so this cannot happen in practice. Per CLAUDE.md §1 ("no error handling for impossible
scenarios"), not adding a guard here — noted for anyone reusing `losses.py` outside this
entrypoint.

### 3. [VERIFIED, not a finding] Directed-graph semantics match the GAT layer's masking convention
Traced `src/lstm_gat_hybrid/model.py`'s `GraphAttentionLayer.forward` (lines ~160-191, NOT
modified by this baseline): attention score `e[i,j] = LeakyReLU(a1·h_i + a2·h_j)`, softmax
normalized over dim=2 (`j`), masked by `adj[i,j]==0`. This means node `i` (row/query) aggregates
information FROM node `j` (column/key) wherever `adj[i,j] != 0`. `construct_directed_spillover_graph`
sets `adj[i,j]` = strength of `corr(vol_j[t], vol_i[t+1])` — i.e., receiver `i`'s row lists its
strongest incoming (top-k) transmitters `j`. This is the correct assignment for "i listens to j's
earlier shock"; a swapped convention (`adj[j,i]` instead) would have silently trained a model
listening to the WRONG causal direction with no shape/crash-level symptom — worth flagging
explicitly since this is exactly the kind of directionality bug that would NOT show up in a
shape-only test. Confirmed correct via the asymmetry+leader-follower test
(`test_asymmetric_on_lead_lag_data`), which asserts `adj[0,1] > adj[1,0]` when stock 1 provably
leads stock 0 by one day — this fails loudly if the convention is ever swapped.

### 4. [VERIFIED, not a finding] No new data leakage introduced
The directed spillover graph is built inside the SAME per-window loop
(`sequence_volatility = all_volatility[i:i+seq_length]`) the sibling's symmetric graph already
used — i.e., built independently for every train/val/test window from only that window's own
lookback slice. No global fit, no cross-split statistics. Confirmed by reading
`dataset_spillover_news.py::_create_sequences` side-by-side with the sibling's
`dataset_dual_news.py::_create_sequences` — identical windowing, only the graph-construction call
differs.

### 5. [VERIFIED, not a finding] QLIKE loss term cannot silently collapse predictions like the
documented 2026-06-21 LSTM-GNN Softplus incident (CLAUDE.md "Normalization Failure" lesson)
The model's output layer is unchanged (linear, on the normalized scale) — QLIKE is computed only
as a loss-side regularizer on the denormalized (inverse-transformed, affine) prediction, clamped
with `eps=1e-6`, weighted at `qlike_weight=0.1` (MSE still dominates). `test_combined_loss_qlike_weight_changes_value`
and `test_combined_loss_is_finite_and_differentiable` confirm gradients stay finite even with
`qlike_weight=1.0`. The real training smoke run (2 epochs, real HAR data) showed no NaN/Inf loss
and a healthy decreasing train/val loss (1.044→1.009 train, 1.210→1.182 val) — see
`test/` run log referenced in the summary report.

### 6. [MINOR, ACCEPTED] Lag-1 correlation has fewer effective points than the sibling's
contemporaneous correlation
`seq_length=22` gives 21 lag-1 pairs vs. 22 contemporaneous pairs for the existing graph — one
fewer data point per window. Documented as an accepted risk in `design.md` §5 (both methods are
already low-power on a 22-day window; this is a 4.5% reduction, not a qualitative change).

## Verdict

No HIGH/MEDIUM issues found requiring a fix beyond item #1 (already applied). Items #2 and #6 are
documented, accepted limitations, not defects. Items #3-#5 are verification notes recording what
was specifically checked to rule out the two failure modes most likely for this kind of change
(wrong graph direction; loss-side positivity collapse) given this project's own history.

## Tests run

`pytest baselines/2026-07-26_spillover_qlike_baseline/test/ -v` → **20/20 passed**
(`test_graph_spillover.py` ×9 [incl. 3 parametrized degenerate-window cases], `test_losses.py` ×6,
`test_train_smoke.py` ×5).

Real smoke run: `python train_spillover_qlike.py --epochs 2 --smoke` → completed, 2 epochs, no
NaN/crash, `results.json` + learning-curve PNG written to
`results/spillover_qlike_2026-07-26_191055/`.

diff-cover: **Not run** (tooling gap, already documented project-wide in CLAUDE.md — pytest-cov/
diff-cover not installed in this environment).
