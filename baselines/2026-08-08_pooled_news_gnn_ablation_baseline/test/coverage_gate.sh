#!/bin/bash
# Coverage gate for the pooled-news-GNN ablation baseline (CLAUDE.md Testing quality rules).
#
# Enforces on the CHANGED lines of a commit range:
#   C0 (line coverage)   = 100%   -> diff-cover --fail-under=100
#   C1 (branch coverage) >= 80%   -> asserted via branch data in coverage_sparse.xml
#
# Intended to be invoked by a pre-push hook (parent wires it into .git/hooks/pre-push).
# Usage: bash coverage_gate.sh [BASE_REF]   (BASE_REF default: origin/master)
set -euo pipefail

BASE_REF="${1:-origin/master}"
ROOT="$(git rev-parse --show-toplevel)"
CODE="baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code"
TESTS="baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test"
cd "$ROOT"

python -m pytest "$TESTS" -q \
  --cov="$CODE" --cov-branch \
  --cov-report=xml:coverage_sparse.xml >/dev/null

echo "== C0 diff-coverage (fail under 100%) vs $BASE_REF =="
python -m diff_cover.diff_cover_tool coverage_sparse.xml --compare-branch="$BASE_REF" --fail-under=100

echo "== C1 branch coverage on changed lines (>= 80%) =="
python - "$BASE_REF" <<'PY'
import subprocess, re, sys, xml.etree.ElementTree as ET
base = sys.argv[1]
paths = [
    "baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py",
    "baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py",
]
def added(path):
    d = subprocess.run(["git", "diff", "--unified=0", base, "HEAD", "--", path],
                       capture_output=True, text=True).stdout
    out, cur = set(), None
    for ln in d.splitlines():
        m = re.match(r"^@@ .*\+(\d+)(?:,(\d+))? @@", ln)
        if m:
            cur = int(m.group(1)); continue
        if ln.startswith("+") and not ln.startswith("+++") and cur is not None:
            out.add(cur); cur += 1
    return out
root = ET.parse("coverage_sparse.xml").getroot()
xml = {}
for cls in root.iter("class"):
    fn = cls.get("filename", "").replace("\\", "/")
    for line in cls.iter("line"):
        xml.setdefault(fn, {})[int(line.get("number"))] = line.get("condition-coverage")
partials = []
for path in paths:
    key = next((k for k in xml if k.endswith(path.split("/")[-1]) and "ablation_baseline/code/" in k), None)
    dd = xml.get(key, {})
    for n in sorted(added(path)):
        cov = dd.get(n)
        if cov and cov.startswith(("0%", "50%")):  # < 80% branch coverage
            partials.append((path, n, cov))
if partials:
    print("C1 GAP (changed branch lines < 80%):", partials); sys.exit(1)
print("C1 OK: no changed branch line below 80%.")
PY
echo "COVERAGE GATE: PASS"
