"""Build a self-contained HTML dashboard from the task ledger.

Reads ``task_ledger.json`` (a JSON array of task entries) and writes a single
self-contained ``docs/reports/task_dashboard.html`` (inline CSS/JS, no external
deps, no server). Open the HTML by double-clicking.

Ledger entry schema (all keys optional at render time; missing keys degrade
gracefully): see ``README.md`` for the documented contract.

Run:
    python scripts/task_dashboard/build_dashboard.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LEDGER = HERE / "task_ledger.json"
DEFAULT_OUT = HERE.parent.parent / "docs" / "reports" / "task_dashboard.html"
DEFAULT_GATE_RESULTS = HERE / "gate_results"

STATUS_ORDER = ["done", "running", "blocked", "planned"]
STATUS_ICON = {"done": "✔", "running": "◐", "blocked": "✖", "planned": "○"}

# Gate status vocabulary. A gate value may be a canonical token string, a
# {status, detail} object, or legacy free-text prose; all resolve to one of
# these statuses via ``_resolve_gate``.
GATE_ICON = {"pass": "✔", "warn": "⚠", "fail": "✖", "miss": "✖", "na": "—"}
GATE_FIELDS = [
    ("tests", "Tests"),
    ("lint", "Lint"),
    ("code_review", "Review"),
    ("diff_cover", "Diff-cover"),
    ("data_quality", "Data-quality"),
]

# Which gates a task_type must have run (status ``pass``) to count as fully
# gated. ``data_quality`` is required only when it is applicable (its status is
# not ``na``) — enforced separately in ``validate_task``.
REQUIRED_GATES = {
    "code": ["tests", "lint", "code_review", "diff_cover"],
    "doc": [],
    "research": [],
    "git-op": [],
}
DATA_GATE = "data_quality"

# diff-coverage floor the pre-push hook enforces (QG_MIN_COVER default). A real
# run at/above this counts as gate-run; below it is a coverage failure.
_COVER_FLOOR = 80.0


def load_ledger(path: Path) -> list[dict]:
    """Read the JSON ledger array. Missing/empty file -> empty list."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Ledger must be a JSON array, got {type(data).__name__}")
    return data


def status_counts(tasks: list[dict]) -> dict[str, int]:
    """Count tasks per status, always covering the four canonical statuses."""
    counts = {s: 0 for s in STATUS_ORDER}
    for task in tasks:
        status = str(task.get("status", "planned")).lower()
        if status in counts:
            counts[status] += 1
        else:
            counts["planned"] += 1
    return counts


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _norm_status(status: object) -> str:
    s = str(status).lower()
    return s if s in STATUS_ORDER else "planned"


# --- Gate resolution -------------------------------------------------------
# Canonical string tokens that map directly onto a gate status.
_GATE_TOKENS = {
    "pass": "pass",
    "passed": "pass",
    "warn": "warn",
    "fail": "fail",
    "failed": "fail",
    "miss": "miss",
    "n/a": "na",
    "na": "na",
    "skip": "na",
    "skipped": "na",
}


def _derive_status(text: str) -> str:
    """Best-effort status for legacy free-text prose (never collapse to dash)."""
    low = text.strip().lower()
    if not low:
        return "na"
    if low.startswith("n/a") or low.startswith("na ") or "not applicable" in low:
        return "na"
    if "fail" in low:
        return "fail"
    if low.startswith("pass") or "passed" in low:
        return "pass"
    # Anything else is real content (a number, a note): show it with a warn icon
    # rather than silently downgrading it to the muted "—" dash.
    return "warn"


def _resolve_gate(value: object) -> tuple[str, str]:
    """Resolve a gate cell to ``(status, detail_text)``.

    Handles three shapes: a ``{status, detail}`` object, a canonical token
    string, and legacy free-text prose. A non-empty value never resolves to a
    bare dash with no text.
    """
    if isinstance(value, dict):
        raw_status = str(value.get("status", "")).strip().lower()
        detail = str(value.get("detail", "")).strip()
        status = raw_status if raw_status in GATE_ICON else _GATE_TOKENS.get(
            raw_status, _derive_status(detail or raw_status)
        )
        return status, detail
    s = str(value).strip()
    low = s.lower()
    if low in _GATE_TOKENS:
        return _GATE_TOKENS[low], s
    return _derive_status(s), s


def _task_type(task: dict) -> str:
    t = str(task.get("task_type", "")).strip().lower()
    return t if t in REQUIRED_GATES else ""


def validate_task(task: dict) -> tuple[bool, list[str]]:
    """Return ``(is_incomplete, missing_gates)`` for a task.

    A task_type with required gates (currently only ``code``) is incomplete when
    any required gate does not resolve to ``pass``, or when the data-quality gate
    is an explicit ``miss``/``fail``. Doc/research/git-op tasks (no required
    gates) are never flagged incomplete.
    """
    ttype = _task_type(task)
    required = REQUIRED_GATES.get(ttype, [])
    if not required:
        return False, []
    gate = task.get("quality_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    missing: list[str] = []
    for key in required:
        status, _ = _resolve_gate(gate.get(key, "na"))
        if status != "pass":
            missing.append(key)
    data_status, _ = _resolve_gate(gate.get(DATA_GATE, "na"))
    if data_status in ("miss", "fail"):
        missing.append(DATA_GATE)
    return (len(missing) > 0), missing


# --- Timestamp ordering ----------------------------------------------------
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def _parse_ts(value: object) -> datetime:
    s = str(value).strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.min


# --- Per-commit gate-result overlay ---------------------------------------
def load_gate_results(dir_path: Path) -> dict[str, dict]:
    """Load ``gate_results/<short-sha>.json`` files into a ``{sha: payload}`` map.

    Missing dir -> empty map. Unreadable/invalid files are skipped, not fatal.
    """
    dir_path = Path(dir_path)
    out: dict[str, dict] = {}
    if not dir_path.is_dir():
        return out
    for jf in sorted(dir_path.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            out[jf.stem] = payload
    return out


def _overlay_gate(task: dict, gate_results: dict[str, dict]) -> dict:
    """Return a task copy whose quality_gate is overlaid with real run values.

    For any of the task's commits that has a gate-result JSON, the machine
    statuses win over hand-entered values (hand-entry is fallback only).
    """
    commits = task.get("commits")
    if not gate_results or not isinstance(commits, list):
        return task
    matched = [gate_results[c] for c in commits if isinstance(c, str) and c in gate_results]
    if not matched:
        return task
    payload = max(matched, key=lambda p: str(p.get("timestamp", "")))
    gate = dict(task.get("quality_gate", {}) if isinstance(task.get("quality_gate"), dict) else {})

    if "tests_passed" in payload:
        ok = bool(payload["tests_passed"])
        gate["tests"] = {"status": "pass" if ok else "fail", "detail": f"hook: {'passed' if ok else 'failed'}"}
    if payload.get("diff_cover_pct") is not None:
        pct = float(payload["diff_cover_pct"])
        # A real diff-cover run counts as gate-run; below the enforced floor (80)
        # it is a genuine coverage failure. Target remains 100% per CLAUDE.md C0.
        gate["diff_cover"] = {"status": "pass" if pct >= _COVER_FLOOR else "fail",
                              "detail": f"hook: {pct:g}%"}
    if payload.get("ruff"):
        gate["lint"] = {"status": "pass" if str(payload["ruff"]).lower() == "pass" else "warn",
                        "detail": f"hook: ruff {payload['ruff']}"}
    if payload.get("pandera"):
        p = str(payload["pandera"]).lower()
        gate["data_quality"] = {"status": "pass" if p == "pass" else ("fail" if p == "fail" else "na"),
                                "detail": f"hook: pandera {payload['pandera']}"}

    new_task = dict(task)
    new_task["quality_gate"] = gate
    return new_task


def _render_badge(status: object) -> str:
    s = _norm_status(status)
    return (
        f'<span class="status-badge st-{s}">'
        f'<span class="ic" aria-hidden="true">{STATUS_ICON[s]}</span>{_esc(s)}</span>'
    )


def _render_overview(counts: dict[str, int], fully_gated: int = 0, total_code: int = 0) -> str:
    total = sum(counts.values())
    tiles = "".join(
        f'<div class="tile st-{s}-tile"><div class="tile-n">{counts[s]}</div>'
        f'<div class="tile-l">{STATUS_ICON[s]} {s}</div></div>'
        for s in STATUS_ORDER
    )
    if total == 0:
        bar = ""
    else:
        segs = []
        for s in STATUS_ORDER:
            n = counts[s]
            if n == 0:
                continue
            pct = 100.0 * n / total
            segs.append(
                f'<div class="seg seg-{s}" style="flex:{n} 1 0" '
                f'title="{s}: {n}"><span class="seg-n">{n}</span>'
                f'<span class="sr-only"> {s}, {pct:.0f} percent</span></div>'
            )
        bar = f'<div class="stackbar" role="img" aria-label="Task status distribution">{"".join(segs)}</div>'
    legend = "".join(
        f'<span class="lg"><span class="sw sw-{s}" aria-hidden="true"></span>'
        f'{STATUS_ICON[s]} {s} ({counts[s]})</span>'
        for s in STATUS_ORDER
    )
    incomplete_code = total_code - fully_gated
    kpi_cls = "kpi-ok" if incomplete_code == 0 else "kpi-warn"
    gate_kpi = (
        f'<div class="gate-kpi {kpi_cls}">Code tasks fully gated: '
        f'<strong>{fully_gated} / {total_code}</strong>'
        + (f' &middot; <span class="kpi-flag">{incomplete_code} INCOMPLETE</span>' if incomplete_code else "")
        + "</div>"
    )
    return (
        '<section class="overview">'
        f'<div class="tiles">{tiles}</div>'
        f'{bar}'
        f'<div class="legend">{legend}</div>'
        f'{gate_kpi}'
        "</section>"
    )


def _render_gate_row(gate: dict, missing: list[str] | None = None) -> str:
    gate = gate if isinstance(gate, dict) else {}
    missing = missing or []
    cells = []
    for key, label in GATE_FIELDS:
        status, detail = _resolve_gate(gate.get(key, "na"))
        # A required-but-not-run gate is flagged MISS even if its own value was a
        # bland "n/a" — the validator decides, the cell shows it red.
        if key in missing and status not in ("miss", "fail"):
            status = "miss"
            detail = detail or "required, not run"
        text = f"{label}: {_esc(detail)}" if detail else label
        cells.append(
            f'<span class="gate gate-{status}">'
            f'<span class="ic" aria-hidden="true">{GATE_ICON[status]}</span>'
            f'{text}</span>'
        )
    return f'<div class="gate-row">{"".join(cells)}</div>'


def _render_chips(skills: object) -> str:
    if not isinstance(skills, list) or not skills:
        return ""
    chips = "".join(f'<span class="chip">{_esc(s)}</span>' for s in skills)
    return f'<div class="chips">{chips}</div>'


def _render_evidence(evidence: object) -> str:
    if not isinstance(evidence, list) or not evidence:
        return ""
    blocks = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        cmd = _esc(ev.get("cmd", ""))
        result = _esc(ev.get("result", ""))
        blocks.append(
            f'<div class="ev"><div class="ev-cmd">$ {cmd}</div>'
            f'<pre class="ev-out">{result}</pre></div>'
        )
    if not blocks:
        return ""
    return f'<div class="evidence"><div class="sub">Evidence</div>{"".join(blocks)}</div>'


def _render_dod(dod: object) -> str:
    if not isinstance(dod, list) or not dod:
        return ""
    items = []
    for d in dod:
        if not isinstance(d, dict):
            continue
        ok = bool(d.get("ok"))
        mark = "✔" if ok else "☐"
        cls = "dod-ok" if ok else "dod-no"
        item_text = _esc(d.get("item", ""))
        items.append(f'<li class="{cls}"><span class="ic" aria-hidden="true">{mark}</span>{item_text}</li>')
    if not items:
        return ""
    return f'<div class="dod"><div class="sub">Definition of Done</div><ul>{"".join(items)}</ul></div>'


def _render_review(review: object) -> str:
    if not isinstance(review, dict) or not review:
        return ""
    layers = review.get("layers", "?")
    fixed = review.get("findings_fixed", 0)
    deferred = review.get("findings_deferred", 0)
    return (
        f'<div class="review">Code review: {_esc(layers)}-layer &middot; '
        f'{_esc(fixed)} fixed &middot; {_esc(deferred)} deferred</div>'
    )


def _render_meta(task: dict) -> str:
    parts = []
    branch = task.get("branch")
    if branch:
        parts.append(f'<span class="meta-item">branch <code>{_esc(branch)}</code></span>')
    commits = task.get("commits")
    if isinstance(commits, list) and commits:
        shas = " ".join(f'<code>{_esc(c)}</code>' for c in commits)
        parts.append(f'<span class="meta-item">commits {shas}</span>')
    ts = task.get("timestamp")
    if ts:
        parts.append(f'<span class="meta-item">{_esc(ts)}</span>')
    report = task.get("report_md")
    if report:
        parts.append(f'<span class="meta-item">report <code>{_esc(report)}</code></span>')
    if not parts:
        return ""
    return f'<div class="meta">{"".join(parts)}</div>'


def _render_task_card(task: dict) -> str:
    if not isinstance(task, dict):
        task = {}
    title = _esc(task.get("title") or task.get("id") or "(untitled task)")
    tid = _esc(task.get("id", ""))
    summary = _esc(task.get("result_summary", ""))
    status = _norm_status(task.get("status", "planned"))
    incomplete, missing = validate_task(task)
    phase = task.get("phase")
    phase_tag = f'<span class="phase-tag">{_esc(phase)}</span>' if phase else ""
    incomplete_badge = (
        f'<span class="incomplete-badge" title="missing required gate(s): '
        f'{_esc(", ".join(missing))}">INCOMPLETE</span>'
        if incomplete else ""
    )
    body = "".join(
        [
            f'<p class="summary">{summary}</p>' if summary else "",
            _render_chips(task.get("skills_applied")),
            _render_gate_row(task.get("quality_gate", {}), missing),
            _render_review(task.get("code_review")),
            _render_dod(task.get("dod")),
            _render_evidence(task.get("evidence")),
            _render_meta(task),
        ]
    )
    card_cls = f"task-card card-{status}" + (" card-incomplete" if incomplete else "")
    return (
        f'<article class="{card_cls}">'
        f'<header class="card-head">{_render_badge(task.get("status", "planned"))}'
        f'{incomplete_badge}'
        f'<h3 class="card-title">{title}</h3>'
        f'{phase_tag}'
        f'<span class="card-id">{tid}</span></header>'
        f"{body}</article>"
    )


def _order_newest_first(tasks: list[dict]) -> list[dict]:
    """Sort tasks newest-first by timestamp; ties fall back to reverse ledger order."""
    indexed = list(enumerate(tasks))
    indexed.sort(key=lambda pair: (_parse_ts(pair[1].get("timestamp", "")), pair[0]), reverse=True)
    return [t for _, t in indexed]


def _gated_kpi(tasks: list[dict]) -> tuple[int, int]:
    """Return ``(fully_gated_code, total_code)`` counts."""
    code = [t for t in tasks if _task_type(t) == "code"]
    fully = sum(1 for t in code if not validate_task(t)[0])
    return fully, len(code)


def build_html(tasks: list[dict], gate_results: dict[str, dict] | None = None) -> str:
    """Render the full self-contained dashboard HTML string.

    ``gate_results`` overlays real per-commit gate output onto matching tasks
    (real run wins over hand-entry).
    """
    tasks = [_overlay_gate(t, gate_results) for t in tasks if isinstance(t, dict)]
    counts = status_counts(tasks)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not tasks:
        content = (
            '<section class="empty-state"><h2>No tasks yet</h2>'
            "<p>The ledger is empty. Add entries to "
            "<code>scripts/task_dashboard/task_ledger.json</code> and re-run "
            "<code>python scripts/task_dashboard/build_dashboard.py</code>.</p></section>"
        )
    else:
        ordered = _order_newest_first(tasks)
        cards = "".join(_render_task_card(t) for t in ordered)
        fully, total_code = _gated_kpi(tasks)
        content = _render_overview(counts, fully, total_code) + f'<section class="tasks">{cards}</section>'

    return _PAGE.format(css=_CSS, js=_JS, content=content, generated=_esc(generated), total=sum(counts.values()))


_CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --st-done:#0ca30c; --st-running:#fab219; --st-blocked:#d03b3b; --st-planned:#898781;
  --chip-bg:#eef1f4; --chip-ink:#2a3138; --ev-bg:#f4f4f1; --incomplete-bg:#fdf2f2;
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
  --st-done:#0ca30c; --st-running:#fab219; --st-blocked:#d03b3b; --st-planned:#898781;
  --chip-bg:#26262a; --chip-ink:#d7d7cf; --ev-bg:#121211; --incomplete-bg:#2a1414;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
    --chip-bg:#26262a; --chip-ink:#d7d7cf; --ev-bg:#121211; --incomplete-bg:#2a1414;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;}
.wrap{max-width:980px;margin:0 auto;padding:24px 20px 64px;}
.top{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;
  border-bottom:1px solid var(--grid);padding-bottom:12px;margin-bottom:20px;}
h1{font-size:22px;margin:0;}
.subtitle{color:var(--text-secondary);font-size:13px;}
.theme-btn{border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);
  border-radius:8px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer;}
.theme-btn:hover{color:var(--text-primary);}
.overview{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
  padding:16px;margin-bottom:24px;}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;}
.tile{flex:1 1 120px;border:1px solid var(--border);border-radius:10px;padding:12px 14px;
  background:var(--page);}
.tile-n{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;}
.tile-l{font-size:12px;color:var(--text-secondary);text-transform:capitalize;}
.st-done-tile{border-left:3px solid var(--st-done);}
.st-running-tile{border-left:3px solid var(--st-running);}
.st-blocked-tile{border-left:3px solid var(--st-blocked);}
.st-planned-tile{border-left:3px solid var(--st-planned);}
.stackbar{display:flex;gap:2px;height:26px;border-radius:6px;overflow:hidden;
  background:var(--surface-1);margin-bottom:10px;}
.seg{display:flex;align-items:center;justify-content:center;min-width:26px;border-radius:4px;}
.seg-n{font-size:12px;font-weight:600;color:#0b0b0b;font-variant-numeric:tabular-nums;}
.seg-done{background:var(--st-done);} .seg-running{background:var(--st-running);}
.seg-blocked{background:var(--st-blocked);} .seg-planned{background:var(--st-planned);}
.seg-blocked .seg-n,.seg-done .seg-n{color:#fff;}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text-secondary);}
.lg{display:inline-flex;align-items:center;gap:6px;text-transform:capitalize;}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;}
.sw-done{background:var(--st-done);} .sw-running{background:var(--st-running);}
.sw-blocked{background:var(--st-blocked);} .sw-planned{background:var(--st-planned);}
.phase{margin-bottom:28px;}
.phase-title{font-size:15px;border-bottom:1px solid var(--grid);padding-bottom:6px;
  display:flex;align-items:center;gap:8px;}
.phase-count{font-size:12px;color:var(--muted);border:1px solid var(--border);
  border-radius:20px;padding:1px 8px;font-variant-numeric:tabular-nums;}
.task-card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
  padding:16px;margin:14px 0;border-left:4px solid var(--baseline);}
.card-done{border-left-color:var(--st-done);}
.card-running{border-left-color:var(--st-running);}
.card-blocked{border-left-color:var(--st-blocked);}
.card-planned{border-left-color:var(--st-planned);}
.card-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.card-title{font-size:16px;margin:0;flex:1 1 auto;}
.card-id{font-size:12px;color:var(--muted);font-family:ui-monospace,monospace;}
.status-badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;
  text-transform:capitalize;padding:3px 9px;border-radius:20px;color:#fff;}
.status-badge .ic{font-size:11px;}
.st-done{background:var(--st-done);} .st-running{background:var(--st-running);color:#0b0b0b;}
.st-blocked{background:var(--st-blocked);} .st-planned{background:var(--st-planned);}
.summary{color:var(--text-secondary);font-size:14px;margin:10px 0;}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0;}
.chip{background:var(--chip-bg);color:var(--chip-ink);font-size:11px;border-radius:20px;
  padding:3px 10px;}
.gate-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0;}
.gate{display:inline-flex;align-items:center;gap:5px;font-size:12px;border-radius:6px;
  padding:3px 9px;border:1px solid var(--border);}
.gate .ic{font-size:11px;}
.gate-pass{color:var(--st-done);}
.gate-fail,.gate-miss{color:var(--st-blocked);border-color:var(--st-blocked);}
.gate-warn{color:var(--st-running);}
.gate-na{color:var(--muted);}
.card-incomplete{border-color:var(--st-blocked);border-left-color:var(--st-blocked);
  background:var(--incomplete-bg);}
.incomplete-badge{display:inline-flex;align-items:center;font-size:11px;font-weight:700;
  letter-spacing:0.04em;padding:3px 9px;border-radius:20px;background:var(--st-blocked);color:#fff;}
.phase-tag{font-size:11px;color:var(--text-secondary);background:var(--chip-bg);
  border-radius:20px;padding:2px 9px;}
.gate-kpi{margin-top:12px;font-size:13px;color:var(--text-secondary);
  border-top:1px solid var(--grid);padding-top:10px;}
.gate-kpi strong{font-variant-numeric:tabular-nums;color:var(--text-primary);}
.kpi-flag{color:var(--st-blocked);font-weight:600;}
.review{font-size:12px;color:var(--text-secondary);margin:8px 0;}
.sub{font-size:12px;font-weight:600;color:var(--text-secondary);margin:12px 0 6px;
  text-transform:uppercase;letter-spacing:0.03em;}
.dod ul{list-style:none;padding:0;margin:0;}
.dod li{font-size:13px;display:flex;align-items:flex-start;gap:8px;padding:2px 0;}
.dod-ok .ic{color:var(--st-done);} .dod-no .ic{color:var(--muted);}
.evidence{margin-top:12px;}
.ev{margin:8px 0;border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.ev-cmd{background:var(--chip-bg);color:var(--chip-ink);font-family:ui-monospace,monospace;
  font-size:12px;padding:6px 10px;}
.ev-out{background:var(--ev-bg);margin:0;padding:10px;font-family:ui-monospace,monospace;
  font-size:12px;white-space:pre-wrap;word-break:break-word;color:var(--text-primary);}
.meta{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;padding-top:10px;
  border-top:1px solid var(--grid);font-size:12px;color:var(--text-secondary);}
.meta code{font-family:ui-monospace,monospace;background:var(--chip-bg);color:var(--chip-ink);
  border-radius:4px;padding:1px 5px;}
.empty-state{background:var(--surface-1);border:1px dashed var(--baseline);border-radius:12px;
  padding:40px;text-align:center;color:var(--text-secondary);}
.empty-state code{font-family:ui-monospace,monospace;}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);}
"""

_JS = """
(function(){
  var KEY='task-dashboard-theme';
  var root=document.documentElement;
  var saved=null; try{saved=localStorage.getItem(KEY);}catch(e){}
  if(saved){root.setAttribute('data-theme',saved);}
  var btn=document.getElementById('theme-toggle');
  function label(){
    var dark=root.getAttribute('data-theme')==='dark' ||
      (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme:dark)').matches);
    btn.textContent=dark?'\\u2600 Light':'\\u263D Dark';
  }
  btn.addEventListener('click',function(){
    var dark=root.getAttribute('data-theme')==='dark' ||
      (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme:dark)').matches);
    var next=dark?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem(KEY,next);}catch(e){}
    label();
  });
  label();
})();
"""

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Track A+B Task Dashboard</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<div class="top">
<div>
<h1>Track A+B Campaign - Task Dashboard</h1>
<div class="subtitle">{total} task(s) &middot; generated {generated}</div>
</div>
<button id="theme-toggle" class="theme-btn" type="button">Theme</button>
</div>
{content}
</div>
<script>{js}</script>
</body>
</html>
"""


def main(ledger_path: Path = DEFAULT_LEDGER, out_path: Path = DEFAULT_OUT) -> Path:
    """Read the ledger and write the dashboard HTML. Returns the output path."""
    tasks = load_ledger(ledger_path)
    gate_results = load_gate_results(DEFAULT_GATE_RESULTS)
    html_str = build_html(tasks, gate_results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    written = main()
    print(f"Wrote {written} ({written.stat().st_size} bytes)")
