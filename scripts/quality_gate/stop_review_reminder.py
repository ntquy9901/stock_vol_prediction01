"""Stop-hook reminder: if uncommitted .py SOURCE changed, surface a non-blocking systemMessage reminding
to run the mandatory 3-layer /code-review (DoD) and commit before finishing.

Claude Code does not support ``type: agent`` hooks on the Stop event (agent/prompt hooks are tool-event
only), so an auto-agent review at turn end is not available; this lightweight command hook is the closest
enforcement — the /code-review itself is still run by the developer/agent per the Definition of Done.
"""
from __future__ import annotations

import json
import subprocess
import sys

EXCLUDE = ("archive/", "data/", ".claude/", "_research/", "temp/", "_master_ref/")


def changed_py_sources(porcelain: str) -> list[str]:
    """Repo-relative .py SOURCE paths from ``git status --porcelain`` output — skips test files and the
    vendored/generated/data trees the project excludes from gates."""
    out = []
    for line in porcelain.splitlines():
        path = line[3:].strip()
        if not path.endswith(".py"):
            continue
        if any(path.startswith(x) for x in EXCLUDE):
            continue
        base = path.rsplit("/", 1)[-1]
        if "/test" in path or base.startswith("test_"):
            continue
        out.append(path)
    return out


def reminder(paths: list[str]) -> dict | None:
    """Build the Stop-hook systemMessage payload, or None when there is nothing uncommitted to review."""
    if not paths:
        return None
    return {"systemMessage": "DoD reminder: uncommitted .py source changed — run /code-review (3-layer) and "
            "commit before finishing: " + ", ".join(paths)}


def main():  # pragma: no cover - entry driver (git subprocess glue)
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    msg = reminder(changed_py_sources(porcelain))
    if msg:
        print(json.dumps(msg))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
