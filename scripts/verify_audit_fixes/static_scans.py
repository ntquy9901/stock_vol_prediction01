"""Mechanical grep-based scans used as part of Gate 2 (static repository checks).

Each scan is a pure function over a list of files so it can be unit-tested
without touching the filesystem beyond the fixture the test provides. These
scans report evidence; they do not decide pass/fail policy for the caller.
"""
from __future__ import annotations

import re
from pathlib import Path

# Directories excluded from lint/scan scope per CLAUDE.md ("Lint excludes").
EXCLUDE_DIR_NAMES = {".agents", ".claude", "_bmad", "archive", "data"}
# Directories that are never source (tooling/vcs/venv noise), always excluded.
ALWAYS_EXCLUDE_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}

_HARDCODED_PATH_RE = re.compile(
    r"""["']((?:[A-Za-z]:[\\/])|(?:/(?:home|Users)/))[^"']*["']"""
)
_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*(#.*)?$")
_RANDOM_SPLIT_RE = re.compile(r"\b(random_split|train_test_split)\s*\(")


def iter_scan_files(repo_root: Path, extra_exclude_dirs: set[str] | None = None) -> list[Path]:
    """List ``*.py`` files under ``repo_root`` in scope for the mechanical scans."""
    exclude = EXCLUDE_DIR_NAMES | ALWAYS_EXCLUDE_DIR_NAMES | (extra_exclude_dirs or set())
    files = []
    for path in sorted(repo_root.rglob("*.py")):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in exclude for part in rel_parts):
            continue
        files.append(path)
    return files


def _scan_lines(files: list[Path], repo_root: Path, pattern: re.Pattern) -> list[dict]:
    matches = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(
                    {
                        "file": str(path.relative_to(repo_root)).replace("\\", "/"),
                        "line": lineno,
                        "text": line.strip(),
                    }
                )
    return matches


def scan_hardcoded_paths(files: list[Path], repo_root: Path) -> list[dict]:
    """Flag string literals containing an absolute local filesystem path."""
    return _scan_lines(files, repo_root, _HARDCODED_PATH_RE)


def scan_bare_except(files: list[Path], repo_root: Path) -> list[dict]:
    """Flag bare ``except:`` clauses (CLAUDE.md: avoid bare except)."""
    return _scan_lines(files, repo_root, _BARE_EXCEPT_RE)


def scan_random_split(files: list[Path], repo_root: Path) -> list[dict]:
    """Flag ``random_split``/``train_test_split`` calls (CLAUDE.md 3.A: temporal split mandatory)."""
    return _scan_lines(files, repo_root, _RANDOM_SPLIT_RE)


def scan_duplicate_module_names(files: list[Path], repo_root: Path) -> list[dict]:
    """Flag ``.py`` basenames (other than ``__init__.py``) that appear in more than one directory.

    Relevant to VER-004: cross-baseline module name collisions under
    ``--import-mode=importlib`` do not raise ImportError, but same-named
    modules across directories are still worth surfacing as evidence.
    """
    by_name: dict[str, list[str]] = {}
    for path in files:
        if path.name == "__init__.py":
            continue
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        by_name.setdefault(path.name, []).append(rel)

    duplicates = []
    for name, paths in sorted(by_name.items()):
        if len(paths) > 1:
            duplicates.append({"module_name": name, "count": len(paths), "files": paths})
    return duplicates


def run_all_scans(repo_root: Path) -> dict:
    """Run every mechanical scan and return a JSON-serializable summary."""
    files = iter_scan_files(repo_root)
    return {
        "files_scanned": len(files),
        "hardcoded_paths": scan_hardcoded_paths(files, repo_root),
        "bare_except": scan_bare_except(files, repo_root),
        "random_split": scan_random_split(files, repo_root),
        "duplicate_module_names": scan_duplicate_module_names(files, repo_root),
    }


def format_scan_report(scan_result: dict) -> str:
    """Render ``run_all_scans`` output as a human-readable text report."""
    lines = [f"files_scanned: {scan_result['files_scanned']}", ""]
    for key in ("hardcoded_paths", "bare_except", "random_split"):
        matches = scan_result[key]
        lines.append(f"=== {key} ({len(matches)} matches) ===")
        for m in matches:
            lines.append(f"{m['file']}:{m['line']}: {m['text']}")
        lines.append("")

    dups = scan_result["duplicate_module_names"]
    lines.append(f"=== duplicate_module_names ({len(dups)} names) ===")
    for d in dups:
        lines.append(f"{d['module_name']} ({d['count']}x): {', '.join(d['files'])}")
    lines.append("")
    return "\n".join(lines)
