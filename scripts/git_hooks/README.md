# Git hooks — enforced quality gate

`pre-push` runs the quality gate automatically before every `git push`, so it cannot be forgotten.

## Activate (once per clone / worktree share the repo config)
```bash
git config core.hooksPath scripts/git_hooks
chmod +x scripts/git_hooks/pre-push
```

## What it enforces
| Step | Behaviour |
|---|---|
| TDD gate | BLOCKS if implementation `.py` changed with NO accompanying test change (proxy for test-first; cannot prove ordering, but forbids untested code) |
| pytest (+coverage) | BLOCKS push on any test failure |
| diff-coverage on changed lines | BLOCKS if below `QG_MIN_COVER` (default 80; **target 100** per CLAUDE.md C0) |
| ruff on changed `.py` | Reports findings (warn) |
| Pandera schema + Evidently drift | Runs + BLOCKS on schema fail **when data/manifest/pipeline files changed** (per the DoD data-quality rule); `N/A` otherwise |

## What it does NOT enforce (not automatable in a git hook)
- **Code review** (3-layer adversarial `/code-review`) — remains a CLAUDE.md process rule the author must run; the hook cannot perform an LLM review.

## Knobs
- `QG_MIN_COVER=100` — raise the diff-coverage floor toward the CLAUDE.md target once coverage debt is cleared.
- `QG_SKIP=1 git push ...` — emergency bypass; must be justified in the summary report.

## Current known debt (2026-08-08)
- Track B pilot diff-coverage vs master = **87%** (176/1459 changed lines uncovered — mostly error/CUDA/plot branches). Floor is set to 80 so it does not hard-block while the debt is worked down to 100.
