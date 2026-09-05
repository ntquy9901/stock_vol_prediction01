"""Post-generate quality gate for a single .py file, invoked by the PostToolUse hook after Write|Edit.

Runs the FAST, working-tree checks the moment source code is generated (not only at pre-push), so a
newly written file is gated immediately:
  * ruff pyflakes (F, excl. cosmetic F541) -> BLOCK  (real bugs: unused import / undefined name / redef)
  * config-hardcode whole-file scan         -> BLOCK  (tunable constant outside pipeline_config; skips tests)

Both checks read the file from disk, so they work on the just-written (uncommitted) file. The FULL suite
(pytest + diff-cover C0/C1 + data-quality Pandera/Evidently + config-hardcode diff + overfit-evidence)
stays at the pre-push gate; mypy/interrogate run at pre-commit. This hook is the fast first tier only.

Emits a Claude Code PostToolUse hook JSON on stdout: {"decision":"block","reason":...} when a BLOCK check
fails (fed back to the model to fix now), otherwise nothing.

Run (normally via the hook): python scripts/quality_gate/postgen_gate.py <file.py>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / ".venv_gpu_encode" / "Scripts" / "python.exe"
EXCLUDE_PREFIXES = ("archive/", ".claude/", ".agents/", "_bmad/", "data/", ".venv")
sys.path.insert(0, str(HERE))


def rel_in_scope(path: str) -> str | None:
    """Return the repo-relative POSIX path if this file is an in-scope .py source, else None.

    Out of scope: non-.py, files outside the repo, and the vendored/generated/data trees that the
    project excludes from all gates (archive/.claude/.agents/_bmad/data/.venv)."""
    try:
        rel = str(Path(path).resolve().relative_to(REPO)).replace("\\", "/")
    except (ValueError, OSError):
        return None
    if not rel.endswith(".py"):
        return None
    if any(rel.startswith(x) for x in EXCLUDE_PREFIXES):
        return None
    if not (REPO / rel).exists():
        return None
    return rel


def is_test_file(rel: str) -> bool:
    """Test files are exempt from the config-hardcode check (fixtures legitimately hold literals)."""
    return "/test" in rel or Path(rel).name.startswith("test_")


def build_decision(rel: str, ruff_out: str | None, hardcode_out: str | None) -> dict | None:
    """Assemble the PostToolUse hook payload from the two check results (pure; no I/O).

    Returns a {"decision":"block","reason":...} dict when either check reported a problem, else None."""
    blocks = []
    if ruff_out:
        blocks.append("ruff pyflakes (F) — real bugs:\n" + ruff_out)
    if hardcode_out:
        blocks.append("config-hardcode — tunable constant(s) outside pipeline_config:\n" + hardcode_out)
    if not blocks:
        return None
    reason = (f"[post-generate gate] {rel} failed fast quality checks — fix these now before continuing "
              f"(the full suite also runs at pre-push):\n\n" + "\n\n".join(blocks))
    return {"decision": "block", "reason": reason}


def _run_ruff_f(rel: str) -> str | None:  # pragma: no cover - subprocess glue
    r = subprocess.run([str(PY), "-m", "ruff", "check", "--select", "F", "--ignore", "F541", rel],
                       cwd=REPO, capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stdout or r.stderr).strip()


def _run_hardcode(rel: str) -> str | None:  # pragma: no cover - subprocess/import glue
    if is_test_file(rel):
        return None
    try:
        import check_config_hardcode as cc
    except ImportError:
        return None
    text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    added = [(i + 1, ln) for i, ln in enumerate(text.splitlines())]
    blocks = [f for f in cc.scan_added(rel, added) if f.severity == "BLOCK"]
    if not blocks:
        return None
    return "\n".join(f"    {rel}:{f.lineno}: {f.reason.strip()}" for f in blocks)


def main(argv=None) -> int:  # pragma: no cover - entry driver; pure logic tested via rel_in_scope/build_decision
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return 0
    rel = rel_in_scope(argv[0])
    if rel is None:
        return 0
    decision = build_decision(rel, _run_ruff_f(rel), _run_hardcode(rel))
    if decision is not None:
        print(json.dumps(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
