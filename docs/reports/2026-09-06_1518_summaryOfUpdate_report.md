# Summary — root-cause of the --dump-cells gate miss + gate hardening

## Root cause (why the auto-gate passed a real bug)
The `--dump-cells` wiring bug (`getattr(D,'d_tr')` AttributeError; MaskedRichData has only d_va/d_te) shipped
green because the buggy block lived inside `run()`, which carries `# pragma: no cover`. Consequence per gate
layer:
- diff-cover C0/C1: `# pragma: no cover` excluded the new logic from coverage; it measured only the pure helper
  `_cell_rows` ("Total: 2 lines", 100%) and never executed the wiring.
- PostToolUse (ruff-F + config-hardcode): static — cannot see a runtime AttributeError.
- No smoke executes the driver `run()`, so the wiring was never run by any test.
Net: substantive logic hidden in a `# pragma: no cover` function is invisible to every gate layer. The CLAUDE.md
rule "test the I/O runner, not just pure helpers" existed only as prose, not mechanically enforced.

## Fixes
1. **Bug fix (commit aba6282):** extracted `_fold_cell_rows` + `_split_dates` (covered helpers; dates from
   `panel.target_dates[fold.<split>]`) out of `run()`; added a wiring integration test with a realistic mock D
   that LACKS d_tr (regression). Verified end-to-end via live `--smoke --dump-cells` (299k rows, 3 splits, 4
   models).
2. **Gate rule (this change):** new `scripts/quality_gate/pragma_logic_guard.py` — parses changed pipeline
   files, finds functions whose def-header carries `# pragma: no cover`, and counts NEW logic lines (from the
   push diff) added inside them. WARN at >=4, BLOCK at >=8. Wired into pre-push as step 6b (BLOCK). Scope:
   existing files only (new baselines are covered by §3.F review), `main` excluded (genuine argparse glue).
   Verified: it BLOCKS the original buggy commit (9 lines in run()) and PASSES the fixed commit (logic moved to
   a helper -> 1 glue line).

## Tests
- `pytest baselines/2026-09-05_edge_horizon_matched/test/ -q` -> 19 passed (incl. the no-d_tr wiring test).
- `pytest scripts/quality_gate/test_pragma_logic_guard.py -q` -> 5 passed.

## Effect
From now, adding non-trivial logic to a `# pragma: no cover` runner is BLOCKED at pre-push, forcing extraction
into a tested helper (or an executing smoke) — the class of bug that shipped --dump-cells cannot recur silently.

## DoD
- [x] Root cause identified + documented
- [x] Bug fixed + regression test + live smoke
- [x] Gate rule added + tests + wired into pre-push, verified against the real failure
- [ ] Push (next)
