"""Assemble the self-contained architecture review page.

Inlines three matplotlib-generated SVGs (current combined architecture, real G0/G1
graph snapshots, proposed-updates diagram) directly into a single HTML file so it opens
with no external files, and renders the proposed-updates table. Objective tone.

Run: python docs/paper/diagrams/build_review_html.py
Regenerate the SVGs first if they changed:
    python docs/paper/diagrams/generate_final_combined.py
    python docs/paper/diagrams/generate_graph_snapshots.py   # needs baseline code + data
    python docs/paper/diagrams/generate_proposed_updates.py
"""
import re
from pathlib import Path

DIAGRAMS = Path(__file__).resolve().parent
OUT = DIAGRAMS.parent / "architecture_review.html"


def inline_svg(name: str, prefix: str) -> str:
    """Read an SVG, drop the XML/DOCTYPE prolog, make it responsive, and namespace
    every id/reference with `prefix` so three SVGs can coexist in one document."""
    raw = (DIAGRAMS / name).read_text(encoding="utf-8")
    svg = raw[raw.index("<svg"):]  # strip <?xml?> + DOCTYPE prolog
    # Namespace ids and their references (matplotlib clip-paths + glyph defs).
    svg = re.sub(r'id="([^"]+)"', rf'id="{prefix}\1"', svg)
    svg = re.sub(r'xlink:href="#([^"]+)"', rf'xlink:href="#{prefix}\1"', svg)
    svg = re.sub(r'(?<!xlink:)href="#([^"]+)"', rf'href="#{prefix}\1"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', rf'url(#{prefix}\1)', svg)
    # Drop the fixed pt width/height on the root so the viewBox scales to the card.
    svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*pt"\s+height="[^"]*pt"',
                 r'\1', svg, count=1)
    svg = svg.replace("<svg", '<svg class="diagram"', 1)
    return svg


# Proposed-updates table: (Update, Stage, Rationale + report, Expected impact, Status).
STATUS_CLASS = {"DONE": "done", "PENDING-DM": "pending", "PROPOSED": "proposed"}
ROWS = [
    ("Masked availability-aware message passing + presence mask",
     "Graph manifest + message passing (G0/G1)",
     "Per-day present node set over 4,941 dates vs 1,296 intersection; absent tickers "
     "masked, never imputed. The news-on-graph novelty. "
     "(2026-08-08_gnn_sparse_data_research.md, Rank 1)",
     "Removes the data-scarcity confound behind the earlier graph-null.",
     "DONE"),
    ("k-NN top-8 sparse adjacency",
     "Graph adjacency (masked GNN)",
     "Mutual top-8 correlation sparsification of the masked graph. Single-seed hint: "
     "&Delta; val-loss &minus;0.00253, best on 5/6 metrics (seed42). "
     "(2026-08-09_0747_all_metrics_consolidated.md, cluster 5; "
     "2026-08-08_gnn_sparse_data_research.md, sparse adjacency)",
     "Possible first non-null graph result; separates a starved graph from an "
     "uninformative one. Claim awaits multi-seed + DM.",
     "PENDING-DM"),
    ("Cache frozen-encoder embeddings",
     "Graph training loop",
     "96.7% of the dominant stage's encoder forward work is redundant recompute of "
     "frozen, unchanging outputs across epochs and across G0/G1. "
     "(2026-08-09_g0g1_training_slowness.md, rec #1)",
     "~15&times; faster graph training (de-duplicate epochs + share cache G0/G1).",
     "PROPOSED"),
    ("Encode present nodes only",
     "Graph training loop",
     "~52% of padded-node encoding is spent on zero-padded absent tickers whose outputs "
     "are then zeroed. (2026-08-09_g0g1_training_slowness.md, rec #2)",
     "~2.1&times; on encoder cost (present &asymp; 48% of padded nodes).",
     "PROPOSED"),
    ("Cut epochs 15 &rarr; ~7",
     "Graph training schedule",
     "Masked knn-8 val loss essentially flat after ~epoch 5 (0.8399&rarr;0.8392 over 10 "
     "extra epochs). (2026-08-09_g0g1_training_slowness.md, rec #7)",
     "~2&times; on the dominant training stage, no accuracy loss.",
     "PROPOSED"),
    ("Add ReduceLROnPlateau LR scheduler",
     "Optimizer (pooled P1/P2/P3 + graph training)",
     "The one missing CLAUDE.md &sect;E technique; dropout 0.2 / weight-decay 1e-5 / "
     "grad-clip 1.0 are already present. (2026-08-09_pooled_convergence_rootcause.md, rec 1)",
     "Paper-grade defensibility; most likely deepens the shared ~0.846 basin equally "
     "for P1/P2/P3 (relative ordering unlikely to flip).",
     "PROPOSED"),
    ("Expanded baseline suite: HARQ + log-HAR + GARCH-family + EWMA/persistence",
     "Baseline comparison (beside P0 HAR classical)",
     "Plain HAR alone invites the reviewer objection 'you only beat a weak HAR'; HARQ + "
     "log-HAR are computable from the same inputs, no new data. "
     "(2026-08-08_har_benchmark_literature.md, secs 3-5)",
     "Main reviewer-risk fix; hardens the baseline that the deep/news model is judged "
     "against.",
     "PROPOSED"),
    ("Diebold-Mariano + Model Confidence Set",
     "Evaluation (beside the 6 metrics)",
     "Standard equal-predictive-accuracy testing under QLIKE + MSE (Patton 2011 robust "
     "losses). (2026-08-08_har_benchmark_literature.md, sec 4)",
     "Statistical significance for the model comparison; the test that confirms or "
     "rejects the knn-8 single-seed hint.",
     "PROPOSED"),
]


def table_html() -> str:
    head = (
        "<tr><th>Update</th><th>Stage</th><th>Rationale (source report)</th>"
        "<th>Expected impact</th><th>Status</th></tr>"
    )
    body = []
    for update, stage, rationale, impact, status in ROWS:
        cls = STATUS_CLASS[status]
        body.append(
            f"<tr><td><strong>{update}</strong></td><td>{stage}</td>"
            f"<td>{rationale}</td><td>{impact}</td>"
            f'<td><span class="badge {cls}">{status}</span></td></tr>'
        )
    return f"<table>{head}{''.join(body)}</table>"


def build_html() -> str:
    fig_a = inline_svg("final_combined_architecture.svg", "a_")
    fig_b = inline_svg("graph_snapshots_g0_g1.svg", "b_")
    fig_c = inline_svg("proposed_updates_architecture.svg", "c_")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Architecture review: current model + proposed updates</title>
<style>
  :root {{
    --bg: #f6f7f9; --fg: #1c1f24; --muted: #5b626c; --card: #ffffff;
    --border: #d7dbe0; --head: #eef1f4;
    --done: #cfe8cf; --done-fg: #1d5a1d;
    --pending: #fbe2b0; --pending-fg: #7a4e00;
    --proposed: #d9e8fb; --proposed-fg: #1a4d80;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14171b; --fg: #e6e8eb; --muted: #9aa2ad; --card: #1e2228;
      --border: #333a42; --head: #262b31;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 0 4rem; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  header {{
    padding: 1.6rem 1.5rem 1.2rem; border-bottom: 1px solid var(--border);
    background: var(--card);
  }}
  header h1 {{ margin: 0 0 .3rem; font-size: 1.45rem; }}
  header p {{ margin: 0; color: var(--muted); font-size: .95rem; }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 0 1.2rem; }}
  section {{ margin-top: 2.2rem; }}
  h2 {{
    font-size: 1.18rem; margin: 0 0 .2rem; padding-bottom: .35rem;
    border-bottom: 2px solid var(--border);
  }}
  section > p.note {{ color: var(--muted); font-size: .9rem; margin: .4rem 0 1rem; }}
  .card {{
    background: #ffffff; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; overflow-x: auto;
  }}
  svg.diagram {{ display: block; width: 100%; height: auto; }}
  table {{
    width: 100%; border-collapse: collapse; margin-top: .6rem;
    background: var(--card); font-size: .88rem;
  }}
  th, td {{
    border: 1px solid var(--border); padding: 8px 10px; vertical-align: top;
    text-align: left;
  }}
  th {{ background: var(--head); font-weight: 600; }}
  td:nth-child(1) {{ width: 15%; }}
  td:nth-child(2) {{ width: 15%; }}
  td:nth-child(5) {{ width: 9%; white-space: nowrap; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: .78rem; font-weight: 600; white-space: nowrap;
  }}
  .badge.done {{ background: var(--done); color: var(--done-fg); }}
  .badge.pending {{ background: var(--pending); color: var(--pending-fg); }}
  .badge.proposed {{ background: var(--proposed); color: var(--proposed-fg); }}
</style>
</head>
<body>
<header>
  <h1>Model architecture review: current design + proposed updates</h1>
  <p>Stock volatility prediction (VN30, Track B pooled news-GNN). Three inlined diagrams
     plus a proposed-updates table. Self-contained &mdash; no external files.</p>
</header>
<main>
  <section>
    <h2>(a) Current combined architecture</h2>
    <p class="note">Pooled news-GNN as built: external HAR baseline (P0), pooled
      shared-weight encoders (P1/P2/P3), and the masked availability-aware graph (G0/G1).
      Source: <code>diagrams/final_combined_architecture.svg</code>.</p>
    <div class="card">{fig_a}</div>
  </section>

  <section>
    <h2>(b) Real G0/G1 graph snapshots</h2>
    <p class="note">Actual masked cross-stock graphs the model operates on (train split):
      present tickers on each target date, signed-correlation edges, dense vs k-NN top-8
      adjacency. The same structure feeds G0 (message passing off) and G1 (on).
      Source: <code>diagrams/graph_snapshots_g0_g1.svg</code>.</p>
    <div class="card">{fig_b}</div>
  </section>

  <section>
    <h2>(c) Proposed updates diagram</h2>
    <p class="note">Current pipeline (neutral boxes) with each change overlaid on the stage
      it touches, tagged <span class="badge done">DONE</span>
      <span class="badge pending">PENDING-DM</span>
      <span class="badge proposed">PROPOSED</span>.
      Source: <code>diagrams/proposed_updates_architecture.svg</code>.</p>
    <div class="card">{fig_c}</div>
  </section>

  <section>
    <h2>(d) Proposed-updates table</h2>
    <p class="note">Each row cites the report it came from. Status: DONE (implemented),
      PENDING-DM (implemented, result claim awaits multi-seed + Diebold-Mariano),
      PROPOSED (not yet built).</p>
    {table_html()}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(build_html(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
