"""Evidence-capture runner for the verify-audit-fixes skill.

Real, mechanical, non-fakeable evidence for repository identity (Gate 1),
static repository checks (Gate 2), test discovery (Gate 3), full tests
(Gate 4), smoke tests (Gate 5), and coverage (Gate 6).

Does not implement fixes, alter tests, or start training. See
``.claude/skills/verify-audit-fixes/SKILL.md`` for the orchestration
process that wraps this runner and handles Gates 7-11.
"""
