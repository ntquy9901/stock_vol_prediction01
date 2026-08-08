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

STATUS_ORDER = ["done", "running", "blocked", "planned"]
STATUS_ICON = {"done": "✔", "running": "◐", "blocked": "✖", "planned": "○"}
GATE_ICON = {"pass": "✔", "skip": "—", "fail": "✖", "n/a": "—"}
GATE_FIELDS = [("tests", "Tests"), ("lint", "Lint"), ("code_review", "Review"), ("diff_cover", "Diff-cover")]


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


def _render_badge(status: object) -> str:
    s = _norm_status(status)
    return (
        f'<span class="status-badge st-{s}">'
        f'<span class="ic" aria-hidden="true">{STATUS_ICON[s]}</span>{_esc(s)}</span>'
    )


def _render_overview(counts: dict[str, int]) -> str:
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
    return (
        '<section class="overview">'
        f'<div class="tiles">{tiles}</div>'
        f'{bar}'
        f'<div class="legend">{legend}</div>'
        "</section>"
    )


def _render_gate_row(gate: dict) -> str:
    cells = []
    for key, label in GATE_FIELDS:
        raw = str(gate.get(key, "n/a")).lower() if isinstance(gate, dict) else "n/a"
        state = raw if raw in GATE_ICON else "n/a"
        cells.append(
            f'<span class="gate gate-{state}">'
            f'<span class="ic" aria-hidden="true">{GATE_ICON[state]}</span>'
            f'{label}: {_esc(raw)}</span>'
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
    body = "".join(
        [
            f'<p class="summary">{summary}</p>' if summary else "",
            _render_chips(task.get("skills_applied")),
            _render_gate_row(task.get("quality_gate", {})),
            _render_review(task.get("code_review")),
            _render_dod(task.get("dod")),
            _render_evidence(task.get("evidence")),
            _render_meta(task),
        ]
    )
    return (
        f'<article class="task-card card-{status}">'
        f'<header class="card-head">{_render_badge(task.get("status", "planned"))}'
        f'<h3 class="card-title">{title}</h3>'
        f'<span class="card-id">{tid}</span></header>'
        f"{body}</article>"
    )


def _group_by_phase(tasks: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for task in tasks:
        phase = str(task.get("phase", "Unassigned")) if isinstance(task, dict) else "Unassigned"
        if phase not in groups:
            groups[phase] = []
            order.append(phase)
        groups[phase].append(task)
    return [(phase, groups[phase]) for phase in order]


def build_html(tasks: list[dict]) -> str:
    """Render the full self-contained dashboard HTML string."""
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
        phase_blocks = []
        for phase, phase_tasks in _group_by_phase(tasks):
            cards = "".join(_render_task_card(t) for t in phase_tasks)
            phase_blocks.append(
                f'<section class="phase"><h2 class="phase-title">{_esc(phase)}'
                f'<span class="phase-count">{len(phase_tasks)}</span></h2>{cards}</section>'
            )
        content = _render_overview(counts) + "".join(phase_blocks)

    return _PAGE.format(css=_CSS, js=_JS, content=content, generated=_esc(generated), total=sum(counts.values()))


_CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --st-done:#0ca30c; --st-running:#fab219; --st-blocked:#d03b3b; --st-planned:#898781;
  --chip-bg:#eef1f4; --chip-ink:#2a3138; --ev-bg:#f4f4f1;
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
  --st-done:#0ca30c; --st-running:#fab219; --st-blocked:#d03b3b; --st-planned:#898781;
  --chip-bg:#26262a; --chip-ink:#d7d7cf; --ev-bg:#121211;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
    --chip-bg:#26262a; --chip-ink:#d7d7cf; --ev-bg:#121211;
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
.gate-pass{color:var(--st-done);} .gate-fail{color:var(--st-blocked);}
.gate-skip,.gate-na{color:var(--muted);}
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
    html_str = build_html(tasks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    written = main()
    print(f"Wrote {written} ({written.stat().st_size} bytes)")
