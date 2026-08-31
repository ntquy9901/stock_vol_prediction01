"""Regression guard for the hard-rename of the mislabeled column identifier.

The processed-data column holds a *variance* (sigma^2 = ln(H/L)^2 / (4 ln2)), not a
volatility, so it was renamed to ``parkinson_variance`` project-wide. This guard fails
if the old identifier reappears anywhere in the active code/data/schema scope.

The needle is assembled at runtime so this guard file itself does not contain the
literal old token (which would otherwise self-trip the grep).
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Assemble the retired identifier without embedding the contiguous literal here.
_OLD = "parkinson_" + "volatility"
# The one permitted residue: a *document filename* that references the old name and
# points at an actual .md experiment-plan file (not the column identifier).
_DOC_FILENAME = _OLD + "_gnn_eda_experiment_plan"

# Active scope that was hard-renamed. Historical/report artifacts (docs/reports *.md,
# *.html, results/*.json), retired code (archive/, src/archive/) and frozen delivery
# snapshots (deliverables*/, backlog/) are intentionally out of scope.
_SCOPE = [
    "src/",
    "baselines/",
    "submission/",
    "scripts/",
    "tests/",
    "graph_eda/",
    "data/processed/",
    "docs/eda/",
    ":(exclude)src/archive/",
    # Narrative docs (spec/design/README/report prose) keep the old name in prose by
    # design; the rename targets live code, tests, data headers and schema only.
    ":(exclude)*.md",
    ":(exclude)*.html",
    ":(exclude)*.txt",
]


def test_old_identifier_absent_from_active_scope():
    proc = subprocess.run(
        ["git", "grep", "-n", _OLD, "--", *_SCOPE],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    # git grep: rc==1 means "no match" (the desired state); rc==0 means matches found.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    offenders = [ln for ln in lines if _DOC_FILENAME not in ln]
    assert not offenders, (
        "Retired identifier still present in active scope:\n"
        + "\n".join(offenders[:40])
    )
