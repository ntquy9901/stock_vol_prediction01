"""Pre-push heuristic: flag NEW scattered hardcoded tunable constants in changed pipeline files.

Enforces CLAUDE.md "Single-source-of-truth app config (ENFORCED)": tunable constants (windows,
thresholds/floors, edge params, hyperparameters, horizons) must live in the ONE canonical
``pipeline_config`` and be imported -- not re-hardcoded as scattered magic-numbers. This is a
LOW-FALSE-POSITIVE heuristic on the ADDED lines of the push diff only (pre-existing literals are
grandfathered; the refactor's own ``= pc.X`` lines are clean because they reference ``pc.``).

Decision (WARN vs BLOCK):
  * BLOCK -- a line that is CLEARLY a tunable pipeline constant:
      - ``.rolling(<int-literal>)`` (a rolling window),
      - ``top_k = <int-literal>`` / ``top_k=<int-literal>`` (edge Top-K),
      - ``NAME = <numeric-literal>`` where NAME is a window/threshold/floor/hyperparameter name
        (WIN/WINDOW/LOOKBACK/SEQ/HORIZON/TOP_K/PATIENCE/EPOCHS/DROPOUT/HIDDEN/HEADS/LR/FLOOR/THRESHOLD/
        BATCH/WEIGHT_DECAY/GRAD_CLIP/MIN_OVERLAP/MIN_PAIRS/MIN_TRAIN/MIN_VALID/MIN_COMMON/MIN_ROWS/...).
  * WARN -- a bare ``1e-N`` float literal on a changed pipeline line (could be a floor; could be a
    legitimate local numerical guard).

Exceptions (never flagged): literals 0 / 1, lines that reference ``pc.`` / ``config`` /
``pipeline_config`` (already centralized), lines carrying ``# noqa`` or ``# config-ok``, pure comments,
and excluded paths (the canonical config module itself, tests, ``archive/``, ``.agents``, ``.claude``,
``_bmad``, vendored, ``data/``).

CLI (used by scripts/git_hooks/pre-push):
    python check_config_hardcode.py --base <ref> <file.py> [<file.py> ...]
Exit 1 if any BLOCK finding (bypassable only via QG_SKIP in the hook); WARN never fails.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

# --- path exclusions (the config module itself, tests, vendored / retired trees) ---
_EXCLUDE_PATH = re.compile(
    r"(^|/)(archive|\.agents|\.claude|_bmad|data)/"
    r"|(^|/)pipeline_config\.py$"
    r"|(^|/)tests?/"
    r"|(^|/)test_[^/]*\.py$"
    r"|_test\.py$"
    r"|(^|/)conftest\.py$"
)

# --- tunable-name vocabulary (BLOCK when a numeric literal is assigned to one of these) ---
_KEYWORD_TOKENS = {
    "win", "window", "lookback", "seq", "horizon", "horizons", "topk", "patience", "epoch", "epochs",
    "dropout", "hidden", "heads", "lr", "floor", "threshold", "thresh", "batch", "momentum", "decay",
    "clip", "overlap", "pairs", "frac",
}
_KEYWORD_SUBSTR = (
    "top_k", "min_train", "min_valid", "min_overlap", "min_pairs", "min_common", "min_rows",
    "min_anchors", "min_valid_nodes", "weight_decay", "grad_clip",
)

_NUM = r"[0-9][0-9_]*\.?[0-9]*(?:[eE]-?[0-9]+)?"
_ASSIGN_RE = re.compile(rf"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=]+?)?=\s*({_NUM})\s*$")
_ROLLING_RE = re.compile(r"\.rolling\(\s*([0-9]+)")
_TOPK_RE = re.compile(r"\btop_?k\s*[=:]\s*([0-9]+)")
_SCI_RE = re.compile(r"\b[0-9]\.?[0-9]*[eE]-?[0-9]+\b")


@dataclass(frozen=True)
class Finding:
    path: str
    lineno: int
    severity: str        # "BLOCK" or "WARN"
    reason: str
    text: str


def is_excluded_path(path: str) -> bool:
    """True if the file is out of scope for the hardcode scan (config module, tests, vendored, ...)."""
    return bool(_EXCLUDE_PATH.search(path.replace("\\", "/")))


def _name_is_tunable(name: str) -> bool:
    lname = name.lower()
    if any(sub in lname for sub in _KEYWORD_SUBSTR):
        return True
    return any(tok in _KEYWORD_TOKENS for tok in lname.split("_"))


def _is_trivial_number(num: str) -> bool:
    """0 / 1 (and their float forms) are indices / identities, never flagged."""
    try:
        return float(num.replace("_", "")) in (0.0, 1.0)
    except ValueError:
        return False


def classify_line(text: str) -> tuple[str, str] | None:
    """Classify ONE added source line. Returns (severity, reason) or None (clean / excepted)."""
    stripped = text.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "# noqa" in text or "# config-ok" in text:
        return None
    if "pc." in text or "pipeline_config" in text or re.search(r"\bconfig\.", text) or "cfg." in text:
        return None                                            # already sourced from the canonical config

    m = _ROLLING_RE.search(text)
    if m and not _is_trivial_number(m.group(1)):
        return ("BLOCK", f".rolling({m.group(1)}) hardcoded window -> move to pipeline_config")

    m = _TOPK_RE.search(text)
    if m and not _is_trivial_number(m.group(1)):
        return ("BLOCK", f"top_k={m.group(1)} hardcoded edge Top-K -> move to pipeline_config")

    m = _ASSIGN_RE.match(text)
    if m and _name_is_tunable(m.group(1)) and not _is_trivial_number(m.group(2)):
        return ("BLOCK", f"tunable constant {m.group(1)}={m.group(2)} hardcoded -> move to pipeline_config")

    if _SCI_RE.search(text):
        return ("WARN", "bare 1e-N float literal on a pipeline line -> consider a named pipeline_config floor")
    return None


def scan_added(path: str, added: list[tuple[int, str]]) -> list[Finding]:
    """Scan the ADDED lines ``(lineno, text)`` of one file; excluded files yield no findings."""
    if is_excluded_path(path):
        return []
    out: list[Finding] = []
    for lineno, text in added:
        verdict = classify_line(text)
        if verdict is not None:
            sev, reason = verdict
            out.append(Finding(path, lineno, sev, reason, text.rstrip()))
    return out


def added_lines_from_diff(diff_text: str) -> list[tuple[int, str]]:
    """Parse a unified diff (``--unified=0``) into ``(new_lineno, added_text)`` for '+' lines."""
    out: list[tuple[int, str]] = []
    new_ln = 0
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            out.append((new_ln, line[1:]))
            new_ln += 1
        elif not line.startswith("-") and not line.startswith("\\"):
            new_ln += 1
    return out


def _git_diff(base: str, path: str) -> str:  # pragma: no cover - subprocess glue exercised in the hook
    try:
        return subprocess.check_output(
            ["git", "diff", "--unified=0", "--no-color", f"{base}..HEAD", "--", path],
            text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""


def main(argv=None) -> int:  # pragma: no cover - entry driver; the scan/parse logic is unit-tested
    ap = argparse.ArgumentParser(description="Flag new hardcoded tunable constants in changed pipeline files.")
    ap.add_argument("--base", default="HEAD~1", help="diff base ref (added lines = base..HEAD)")
    ap.add_argument("files", nargs="*", help="changed .py files to scan")
    a = ap.parse_args(argv)
    findings: list[Finding] = []
    for f in a.files:
        if is_excluded_path(f):
            continue
        findings.extend(scan_added(f, added_lines_from_diff(_git_diff(a.base, f))))
    blocks = [x for x in findings if x.severity == "BLOCK"]
    warns = [x for x in findings if x.severity == "WARN"]
    for x in findings:
        print(f"[config-hardcode] {x.severity} {x.path}:{x.lineno}: {x.reason}\n    {x.text.strip()}")
    if not findings:
        print("[config-hardcode] no new hardcoded tunable constants in changed pipeline files.")
    print(f"[config-hardcode] summary: {len(blocks)} BLOCK, {len(warns)} WARN")
    return 1 if blocks else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
