"""Gate guard: flag substantive NEW logic added inside a `# pragma: no cover` function.

Root cause it closes (2026-09-06): the `--dump-cells` wiring bug (`getattr(D,'d_tr')` AttributeError) shipped
green because the buggy block lived inside `run()`, which carries `# pragma: no cover`. diff-cover therefore
measured only the pure helper and never executed the wiring, so no gate layer could see the bug. `# pragma:
no cover` is a legitimate escape for an untestable entry driver (needs GPU/IO), but real logic added there
escapes coverage. This guard makes that visible: extract new logic into a covered helper (or add an executing
smoke) instead of hiding it in the runner.

Heuristic: parse each changed pipeline file, find functions whose def-header carries `# pragma: no cover`, and
count NEW added lines (from the push diff) inside those functions' bodies that are executable logic (not blank,
comment, or docstring). WARN at >= WARN_T such lines; BLOCK at >= BLOCK_T.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

WARN_T = 4          # >= this many new logic lines under a pragma-no-cover function -> WARN
BLOCK_T = 8         # >= this many -> BLOCK (extract to a tested helper or add an executing smoke)
_EXCLUDE = ("archive/", ".agents/", ".claude/", "_bmad/", "data/")


def added_line_nums(diff_text: str) -> set:
    """New-file line numbers added in a `git diff --unified=0` (only '+' lines; '-' lines do not advance)."""
    out = set()
    cur = None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            plus = line.split("+", 1)[1].split(" ", 1)[0]        # c or c,d
            cur = int(plus.split(",")[0])
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            if cur is not None:
                out.add(cur); cur += 1
    return out


def pragma_functions(source: str):
    """List of (name, body_start_line, end_line, docstring_end_line) for functions whose def-header carries
    `# pragma: no cover`. The header spans the def line(s) up to (not including) the first body statement."""
    lines = source.splitlines()
    tree = ast.parse(source)
    funcs = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        body_start = node.body[0].lineno
        header = "\n".join(lines[node.lineno - 1:body_start - 1])
        if "# pragma: no cover" not in header:
            continue
        if node.name == "main":                   # argparse entry driver is genuine untestable glue -> allowed
            continue
        doc_end = 0
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                and isinstance(first.value.value, str):
            doc_end = first.end_lineno            # skip the docstring span when counting logic
        funcs.append((node.name, body_start, node.end_lineno, doc_end))
    return funcs


def _is_logic(text: str) -> bool:
    s = text.strip()
    return bool(s) and not s.startswith("#")


def analyze(source: str, added: set):
    """Per-pragma-function count of added logic lines. Returns [(name, count)] for functions with count>0."""
    findings = []
    for name, start, end, doc_end in pragma_functions(source):
        lines = source.splitlines()
        n = sum(1 for ln in added if start <= ln <= (end or start)
                and ln > doc_end and _is_logic(lines[ln - 1]))
        if n:
            findings.append((name, n))
    return findings


def _changed_pipeline_files(base: str):  # pragma: no cover - git I/O glue (analysis core is tested)
    out = subprocess.run(["git", "diff", "--name-only", f"{base}..HEAD", "--", "*.py"],
                         capture_output=True, text=True).stdout.splitlines()
    keep = []
    for f in out:
        if any(f.startswith(x) or f.startswith("./" + x) for x in _EXCLUDE):
            continue
        if "/test" in f or f.endswith("_test.py") or Path(f).name.startswith("test_"):
            continue                                             # tests are exempt (they are the coverage)
        keep.append(f)
    return keep


def run_guard(base: str):  # pragma: no cover - git I/O glue (per-file diff -> tested analyze())
    """Scan changed pipeline files; return (worst_count, report_lines)."""
    report = []
    worst = 0
    for f in _changed_pipeline_files(base):
        p = Path(f)
        if not p.exists():
            continue
        # target logic SLIPPED INTO an existing runner; a brand-new file is reviewed via the §3.F baseline
        # process, so skip files absent at base to avoid false-blocking new baselines.
        if subprocess.run(["git", "cat-file", "-e", f"{base}:{f}"], capture_output=True).returncode != 0:
            continue
        diff = subprocess.run(["git", "diff", "--unified=0", f"{base}..HEAD", "--", f],
                              capture_output=True, text=True).stdout
        added = added_line_nums(diff)
        try:
            findings = analyze(p.read_text(encoding="utf-8"), added)
        except SyntaxError:
            continue
        for name, n in findings:
            worst = max(worst, n)
            report.append(f"{f}: {n} new logic line(s) under `# pragma: no cover` in {name}()")
    return worst, report


def main(argv=None):  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="@{upstream}")
    a = ap.parse_args(argv)
    base = subprocess.run(["git", "rev-parse", a.base], capture_output=True, text=True).stdout.strip() \
        or subprocess.run(["git", "merge-base", "HEAD", "master"], capture_output=True, text=True).stdout.strip() \
        or "master"
    worst, report = run_guard(base)
    for line in report:
        print("[pragma-guard] " + line)
    if worst >= BLOCK_T:
        print(f"[pragma-guard] BLOCK: >= {BLOCK_T} new logic lines added inside a # pragma: no cover function. "
              "Extract them into a tested helper (or add an executing smoke) -- untested runner logic shipped "
              "the --dump-cells d_tr bug on 2026-09-06.")
        return 2
    if worst >= WARN_T:
        print(f"[pragma-guard] WARN: new logic added inside a # pragma: no cover function (< {BLOCK_T}); "
              "prefer extracting it into a covered helper.")
    else:
        print("[pragma-guard] no substantive logic hidden in pragma-no-cover functions.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
