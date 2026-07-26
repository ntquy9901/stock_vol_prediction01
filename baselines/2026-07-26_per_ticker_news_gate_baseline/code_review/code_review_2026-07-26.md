# Code Review (self-adversarial) — 2026-07-26_per_ticker_news_gate_baseline

**Reviewer:** self-review (interactive session, user present). 3-layer adversarial pass applied
to the 2 new files (`model_per_ticker_gate.py`, `train_per_ticker_gate.py`).

## Findings

### 1. [VERIFIED, not a finding] Gradient isolation claim — the entire point of this baseline —
holds under direct test, not just architectural reasoning
This is the one property that MUST be true for this baseline to mean what it claims. Verified via
2 independent property tests (`test_gate_gradient_isolated_per_ticker`,
`..._also_holds_for_feature_perturbation`): perturbing ticker j's target OR its news features
leaves `gate_logits[i]`'s gradient (i≠j) byte-for-byte unchanged (`pytest.approx(..., abs=1e-6)`).
A third test (`test_gate_gradient_does_change_for_the_affected_ticker`) confirms the perturbation
used is actually meaningful (rules out the isolation tests passing trivially because nothing
propagates gradient at all — an Edge Case Hunter concern: an all-zero-gradient bug would also
"pass" a naive isolation test).

### 2. [MINOR, ACCEPTED] `gate_lr=0.05` is a judgment call, not derived from data
Design.md documents WHY a higher LR for `gate_logits` is used (30 scalars need to move
meaningfully within the 10-epoch cap), but the specific value (0.05, 10× the base LR) wasn't
tuned — it's a reasonable default, not a swept-and-selected optimum. Acceptable for a first
experimental run (requirements.md §6 Out of scope explicitly excludes deep tuning); the debug
log (console + `gate_history.json`) lets the user SEE if 10 epochs was enough for gates to move
meaningfully, rather than silently assuming it was.

### 3. [VERIFIED, not a finding] `plot_gate_evolution`'s colormap indexing is safe for the real
30-ticker case
`matplotlib.colormaps.get_cmap("tab20").resampled(len(stock_names))` — `tab20` has 20 discrete
colors; `resampled(30)` interpolates a continuous version rather than raising or silently reusing
only 20 distinct colors for 30 lines. Confirmed via the smoke run (32 real tickers from
`data/processed`, not a synthetic small count) — plot saved without error, file size >100KB
(non-trivial content, not a blank/broken plot).

### 4. [MINOR, FIXED during initial implementation, not a review finding] matplotlib API
The installed matplotlib (3.11.0) removed `matplotlib.cm.get_cmap` (deprecated/removed upstream);
initial implementation used it and failed `test_plot_gate_evolution_writes_file`. Fixed by
switching to `matplotlib.colormaps.get_cmap(...).resampled(...)` before this review pass (caught
by the test itself, not by inspection — exactly the point of running tests before claiming done).

### 5. [MINOR, ACCEPTED] Best-checkpoint gate values in `results.json`/final console table reflect
the EARLY-STOPPING-SELECTED epoch, not the LAST epoch
`model.load_state_dict(torch.load(best_path))` reloads the epoch with the best val loss before
computing final gate values — this is intentional (consistent with how `best_vm`/test metrics are
also drawn from the best checkpoint, not the last epoch) but worth flagging: if the user expects
the FINAL-epoch gate trend (from `gate_history.json`, which DOES have every epoch), the printed
"Final per-ticker gate values" table is from a possibly-earlier epoch, not necessarily the last
one trained. `gate_history.json` is the source of truth for the full trajectory; the console
table after training is a checkpoint-specific snapshot.

### 6. [VERIFIED, not a finding] No leakage / no change to sibling files
`create_dual_news_dataloaders`, `NewsFeatureLSTM`, `evaluate_predictions`, `EarlyStopping`,
`plot_learning_curves_with_analysis` are all imported read-only from the sibling baseline / `src`.
`grep`-confirmed no edits to any file outside this baseline's own `code/` folder.

## Verdict

No HIGH/MEDIUM issues. Item #2 and #5 are documented, accepted design choices (not defects);
item #3 and #6 are verification notes; item #4 was fixed during implementation before this review
pass (test-driven catch).

## Tests run

`pytest baselines/2026-07-26_per_ticker_news_gate_baseline/test/ -v` → **12/12 passed**:
- `test_model_per_ticker_gate.py` ×7 (shape, neutral init, sigmoid range, **2 gradient-isolation
  property tests**, 1 isolation-is-non-trivial sanity check, all-zero-news forward).
- `test_train_smoke.py` ×5 (debug console table ×2, gate-evolution plot file write,
  gate_history.json roundtrip, end-to-end train_epoch smoke).

Real smoke run: `python train_per_ticker_gate.py --epochs 3 --smoke` → completed, gate values
visibly changed epoch-to-epoch (e.g. POW 0.5058→0.5437→0.4643 across 3 epochs), console table +
`gate_history.json` + `gate_evolution_final_*.png` + `learning_curves_epoch_3_*.png` all written
correctly to `results/per_ticker_gate_2026-07-26_221512/`.

diff-cover: **Not run** (tooling gap, documented project-wide in CLAUDE.md).
