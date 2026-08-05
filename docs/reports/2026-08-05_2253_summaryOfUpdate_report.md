# Summary of update — HAR baseline split fix + paper primary-baseline reframing

Date: 2026-08-05. Scope: two linked changes — (1) fix a temporal-split bug in the classical HAR
baseline and rerun it, (2) restructure the SOICT 2026 paper draft so classical HAR is the primary
baseline and the deep no-news backbone becomes an ablation.

## 1. HAR baseline temporal-split fix

**Bug.** `src/har_baseline/train.py::train_har_baseline()` concatenated every ticker's full series
end-to-end in arbitrary `os.listdir` order, then applied one global 80/20 cut. Because tickers were
concatenated whole, some tickers landed entirely in train and others entirely in test, which is not
a temporal split (violated CLAUDE.md §3.A). Any HAR number from the old script was not a valid
temporal-forecast evaluation.

**Fix.** Added `load_har_train_test_split()`, which builds HAR features + the 5-day target per
ticker (unchanged, causal), drops warm-up/trailing NaN rows, and cuts each ticker at 80% of its own
chronological rows. Pooled train holds the early portion of every ticker; pooled test holds the late
portion of every ticker. Pattern mirrors `HARVolatilityDataset._load_all_data`. Rows stay
ticker-blocked, so `evaluate_predictions()`'s `np.diff`-based directional accuracy is already correct
without an `n_stocks` argument (verified against `src/common/evaluation.py::directional_accuracy`).
No change to `src/common/` shared code.

**Files:** `src/har_baseline/train.py` (helper + rewired loader/split section);
`tests/har_baseline/test_train_split_fix.py`, `tests/har_baseline/__init__.py` (new).

## 2. Rerun numbers (classical HAR, primary baseline)

`results/har_baseline_2026-08-05_224208/` (33 tickers, 84,549 train / 21,154 test):

| Metric | Value |
|---|---|
| QLIKE | 0.5493 |
| RMSE | 0.002182 |
| MAE | 0.000575 |
| R² | 0.7419 |
| DirAcc | 48.65% |

Comparison to the full gated news-fusion model (3-seed mean from the readiness report): news lower
QLIKE (0.4430) and higher R² (0.8031); classical HAR lower RMSE (vs 0.002734) and MAE (vs
0.0007930); DirAcc near-random for both. Mixed result, reported honestly in the paper.

## 3. Tests

`python -m pytest tests/har_baseline/test_train_split_fix.py -v` — 4 passed:
- every ticker contributes to both splits (the direct regression for the bug; old code fails it),
- per-ticker chronological no-leakage (test dates after train dates per ticker),
- per-ticker 80/20 ratio,
- `smoke`: real-data slice runs end-to-end, metrics finite, RMSE in the ~1e-3 range.

Coverage gate (diff-cover `--fail-under=100`): Not run — pytest-cov/diff-cover not set up in this
repo (documented tooling gap in CLAUDE.md §Per-project setup). Changed lines are exercised by the
four tests above (helper via all four, `train_har_baseline` rewired section via the smoke test).

## 4. Paper restructure (v2)

Created `docs/paper/soict2026_draft_v2.tex` and `docs/paper/soict2026_draft_v2_summary.md`; v1 kept
as the frozen prior version. Changes: classical HAR is now the primary external baseline (new
Table 1, classical HAR vs full model); the price-only LSTM--GAT backbone (former v1 headline) moved
to an ablation (Table 2) isolating the news branch; v1's "HAR-only backbone" term renamed
"price-only backbone" throughout to avoid collision with "classical HAR". Abstract, Introduction
contributions/preview, Background §2, Method §3.1, Setup §4, Results §5, Discussion §6 (new §6.2),
Related Work, and Limitations updated. Honest positioning: the classical HAR keeping lower RMSE/MAE
is stated in the abstract, Table 1, §5.1 takeaway, §6.2, Related Work, and Limitations. A protocol
caveat (HAR point-wise 80/20 vs deep windowed 70/15/15) is stated in §4/§6.2/§8; the causal news
claim rests on the protocol-matched ablation. See `soict2026_draft_v2_summary.md` for the full diff.

## 5. Code review

Paper edits ran the paper-writing skill's mechanical gate (`gate_mechanical.md`, greps) and semantic
gate (`gate_semantic.md`): M1 em-dashes 0 in prose, M11 passive 0 in prose (3 introduced hits
fixed), M4/M5/M6/M12/M15/M18 clean; define-before-use and honest-positioning verified; all
refs/cites resolve; table numbers trace to the rerun and macros. Adversarial `/code-review` on the
Python change: not run as a separate pass in this session; the split logic is covered by the
regression + smoke tests, and the change is confined to the data-loading/splitting section with no
edits to shared `src/common/` code.

## 6. Commands run

- `python -m pytest tests/har_baseline/test_train_split_fix.py -v` → 4 passed.
- `python -m src.har_baseline.train` → wrote `results/har_baseline_2026-08-05_224208/`.
- Mechanical-gate greps on `soict2026_draft_v2.tex` (counts above).

## 7. Risks / follow-ups

- diff-cover coverage gate not runnable (repo tooling gap); relied on the four tests.
- Adversarial `/code-review` on the Python diff deferred; low blast radius (one file, tested).
- Optional: rerun classical HAR under the deep models' 70/15/15 windowed protocol to remove the
  protocol caveat and make Table 1 fully protocol-matched.
- Paper not compiled (no LaTeX toolchain); confirm page count ≤ 12 excl. refs on compile.
