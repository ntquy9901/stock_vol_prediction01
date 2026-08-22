"""Generate a teaching diagram for the masked union-panel concept.

Concrete toy example (6 tickers A..F x 12 trading dates t1..t12) that contrasts the
common-date intersection design (drops early history for everyone) with the masked
union panel used in ``masked_snapshots.py`` (keeps every date; per-date node_mask /
target_mask decide which tickers are valid graph nodes / have a label).

Faithful to ``masked_snapshots.build_masked``:
  - a "panel" is rows = trading dates, cols = tickers; NaN where a ticker is missing;
  - a node is valid on a date only if its lookback window is complete (node_mask) AND
    its target is observed (target_mask); a listed-but-warmup day is masked;
  - loss + graph attention run only on valid nodes; missing nodes are skipped per date,
    not dropped for every ticker;
  - the graph snapshot's node set therefore adapts per date via the mask.

Pure-Python SVG emitter (no matplotlib): keeps text as real <text> elements so it
stays crisp, and avoids GPU / heavy deps. Run:
    .venv_gpu_encode/Scripts/python.exe draw_masked_panel.py
"""
from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------- toy example
TICKERS = ["A", "B", "C", "D", "E", "F"]
DATES = [f"t{i}" for i in range(1, 13)]           # t1 .. t12
LISTED_FROM = {"A": 1, "B": 1, "C": 1, "D": 4, "E": 6, "F": 8}   # 1-based date index
WARMUP = 1                                          # 1 day lookback warmup after listing

# common-date intersection keeps dates where EVERY ticker is listed -> t8..t12
INTERSECTION_FROM = max(LISTED_FROM.values())      # = 8

# graph snapshots to illustrate the adapting node set
SNAP_EARLY = 5    # t5 -> A,B,C,D valid
SNAP_LATE = 9     # t9 -> all six valid

# ---------------------------------------------------------------- palette
GREEN_FILL, GREEN_STROKE = "#c8e6c9", "#2e7d32"
MISS_FILL, MISS_STROKE = "#eceff1", "#b0bec5"
WARM_STROKE = "#ef6c00"
DISCARD_STROKE = "#78909c"
TEXT = "#263238"
MUTED = "#546e7a"

CELL_W, CELL_H = 56, 34
N_COL, N_ROW = len(TICKERS), len(DATES)


def cell_state(ticker: str, t: int) -> str:
    """'valid' | 'warmup' | 'missing' for the masked union panel."""
    lf = LISTED_FROM[ticker]
    if t < lf:
        return "missing"
    if t < lf + WARMUP:
        return "warmup"
    return "valid"


def valid_tickers(t: int) -> list[str]:
    return [tk for tk in TICKERS if cell_state(tk, t) == "valid"]


# ---------------------------------------------------------------- svg helpers
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, color=TEXT, anchor="start", weight="normal", style="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="{style}">{esc(s)}</text>'
    )


def rect(x, y, w, h, fill, stroke, sw=1.4, rx=4, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>'
    )


def line(x1, y1, x2, y2, stroke, sw=2, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>'
    )


def circle(cx, cy, r, fill, stroke, sw=2, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>'
    )


# ---------------------------------------------------------------- panel drawing
def draw_grid(gx: float, gy: float, mode: str) -> list[str]:
    """mode = 'union' or 'intersection'."""
    out: list[str] = []
    # column (ticker) headers
    for c, tk in enumerate(TICKERS):
        cx = gx + c * CELL_W + CELL_W / 2
        out.append(text(cx, gy - 10, tk, size=15, weight="bold", anchor="middle"))
    # row (date) labels + cells
    for r, dt in enumerate(DATES):
        t = r + 1
        ry = gy + r * CELL_H
        out.append(text(gx - 12, ry + CELL_H / 2 + 5, dt, size=13, color=MUTED, anchor="end"))
        for c, tk in enumerate(TICKERS):
            cx = gx + c * CELL_W
            if mode == "union":
                st = cell_state(tk, t)
                if st == "valid":
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, GREEN_FILL, GREEN_STROKE))
                    out.append(text(cx + CELL_W / 2, ry + CELL_H / 2 + 4, "✓",
                                    size=13, color=GREEN_STROKE, anchor="middle", weight="bold"))
                elif st == "warmup":
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, MISS_FILL,
                                    WARM_STROKE, sw=1.6, dash="4 2"))
                    out.append(text(cx + CELL_W / 2, ry + CELL_H / 2 + 4, "warm",
                                    size=10, color=WARM_STROKE, anchor="middle"))
                else:
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, MISS_FILL, MISS_STROKE))
            else:  # intersection
                listed = t >= LISTED_FROM[tk]
                kept = t >= INTERSECTION_FROM
                if kept:
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, GREEN_FILL, GREEN_STROKE))
                    out.append(text(cx + CELL_W / 2, ry + CELL_H / 2 + 4, "✓",
                                    size=13, color=GREEN_STROKE, anchor="middle", weight="bold"))
                elif listed:  # data existed on this date but the whole date is discarded
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, GREEN_FILL,
                                    GREEN_STROKE, opacity=0.35))
                    out.append(text(cx + CELL_W / 2, ry + CELL_H / 2 + 4, "✓",
                                    size=13, color=GREEN_STROKE, anchor="middle",
                                    weight="bold"))
                else:
                    out.append(rect(cx + 2, ry + 2, CELL_W - 4, CELL_H - 4, MISS_FILL, MISS_STROKE))
                    out.append(text(cx + CELL_W / 2, ry + CELL_H / 2 + 4, "–",
                                    size=13, color=MISS_STROKE, anchor="middle"))
        # discard overlay for intersection design
        if mode == "intersection" and (r + 1) < INTERSECTION_FROM:
            gw = N_COL * CELL_W
            out.append(line(gx + 2, ry + CELL_H / 2, gx + gw - 2, ry + CELL_H / 2,
                            DISCARD_STROKE, sw=2, opacity=0.65))
    return out


def draw_snapshot_graph(cx: float, cy: float, r: float, t: int, caption: str) -> list[str]:
    """Hexagon layout for A..F; active nodes solid + edges, masked nodes ghosted."""
    out: list[str] = []
    pos = {}
    for i, tk in enumerate(TICKERS):
        ang = math.radians(-90 + i * 60)
        pos[tk] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    active = valid_tickers(t)
    # edges only among active nodes (ring + one chord)
    ring = [(active[i], active[(i + 1) % len(active)]) for i in range(len(active))]
    chord = [(active[0], active[2])] if len(active) >= 3 else []
    for a, b in ring + chord:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        out.append(line(x1, y1, x2, y2, GREEN_STROKE, sw=2, opacity=0.55))
    # nodes
    nr = 22
    for tk in TICKERS:
        x, y = pos[tk]
        if tk in active:
            out.append(circle(x, y, nr, GREEN_FILL, GREEN_STROKE, sw=2.4))
            out.append(text(x, y + 5, tk, size=15, weight="bold", anchor="middle", color=GREEN_STROKE))
        else:
            out.append(circle(x, y, nr, MISS_FILL, MISS_STROKE, sw=1.8, dash="4 3", opacity=0.9))
            out.append(text(x, y + 5, tk, size=14, anchor="middle", color=MISS_STROKE))
    out.append(text(cx, cy - r - 40, f"Snapshot {DATES[t - 1]}", size=16, weight="bold", anchor="middle"))
    out.append(text(cx, cy - r - 20, f"{len(active)} valid nodes", size=13, color=MUTED, anchor="middle"))
    out.append(text(cx, cy + r + 46, caption, size=12.5, color=MUTED, anchor="middle"))
    return out


def legend_swatch(x, y, kind) -> list[str]:
    out = []
    if kind == "valid":
        out.append(rect(x, y, 26, 18, GREEN_FILL, GREEN_STROKE))
        out.append(text(x + 13, y + 13, "✓", size=11, color=GREEN_STROKE, anchor="middle", weight="bold"))
    elif kind == "warm":
        out.append(rect(x, y, 26, 18, MISS_FILL, WARM_STROKE, sw=1.6, dash="4 2"))
    else:
        out.append(rect(x, y, 26, 18, MISS_FILL, MISS_STROKE))
    return out


# ---------------------------------------------------------------- assemble
def build_svg() -> str:
    W, H = 1560, 1140
    LGX, RGX = 150, 900          # grid origins x
    GY = 190                     # grid origin y
    grid_w = N_COL * CELL_W
    grid_bottom = GY + N_ROW * CELL_H

    s: list[str] = []
    s.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">')
    s.append(rect(0, 0, W, H, "#ffffff", "#ffffff", sw=0, rx=0))

    # ---- title / subtitle
    s.append(text(W / 2, 40, "Common-date intersection vs. masked union panel "
                  "(concrete example: 6 tickers x 12 dates)",
                  size=23, weight="bold", anchor="middle"))
    s.append(text(W / 2, 68, "A panel = rows (trading dates) x columns (tickers). "
                  "Each date is a cross-sectional graph snapshot for the GAT; "
                  "tickers have different listing dates, so early dates have missing cells.",
                  size=13.5, color=MUTED, anchor="middle"))
    s.append(text(W / 2, 90, "Listing dates: A,B,C from t1  ·  D from t4  ·  "
                  "E from t6  ·  F from t8   (first listed day = lookback warmup, "
                  "not yet a valid node).",
                  size=13, color=MUTED, anchor="middle"))

    # ---- panel headers
    s.append(text(LGX + grid_w / 2, 130, "1) Common-date intersection  (old / biased)",
                  size=17, weight="bold", anchor="middle", color="#c62828"))
    s.append(text(LGX + grid_w / 2, 150, "keep only dates where EVERY ticker has data",
                  size=12.5, color=MUTED, anchor="middle"))
    s.append(text(LGX + grid_w / 2, 168, "(faded ✓ = data existed but the whole date is dropped)",
                  size=11, color=MUTED, anchor="middle", style="italic"))
    s.append(text(RGX + grid_w / 2, 130, "2) Masked union panel  (the fix)",
                  size=17, weight="bold", anchor="middle", color="#2e7d32"))
    s.append(text(RGX + grid_w / 2, 150, "keep ALL dates; per-date masks pick valid nodes",
                  size=12.5, color=MUTED, anchor="middle"))

    # ---- grids
    s += draw_grid(LGX, GY, "intersection")
    s += draw_grid(RGX, GY, "union")

    # bracket marking kept vs discarded on the intersection panel
    kept_top = GY + (INTERSECTION_FROM - 1) * CELL_H
    s.append(line(LGX - 34, kept_top, LGX - 34, grid_bottom, "#2e7d32", sw=3))
    s.append(text(LGX - 40, (kept_top + grid_bottom) / 2, "kept",
                  size=12, color="#2e7d32", anchor="end", weight="bold"))
    s.append(line(LGX - 34, GY, LGX - 34, kept_top, DISCARD_STROKE, sw=3))
    s.append(text(LGX - 40, (GY + kept_top) / 2, "discarded",
                  size=12, color=DISCARD_STROKE, anchor="end", weight="bold"))

    # ---- legend (right of union grid)
    lx = RGX + grid_w + 28
    ly = GY + 6
    s.append(text(lx, ly - 12, "Legend", size=14, weight="bold"))
    s += legend_swatch(lx, ly, "valid")
    s.append(text(lx + 34, ly + 13, "valid: full lookback window", size=12))
    s.append(text(lx + 34, ly + 28, "+ target here (green)", size=12, color=MUTED))
    s += legend_swatch(lx, ly + 44, "warm")
    s.append(text(lx + 34, ly + 57, "warmup: listed but window", size=12))
    s.append(text(lx + 34, ly + 72, "not yet full (masked)", size=12, color=MUTED))
    s += legend_swatch(lx, ly + 88, "miss")
    s.append(text(lx + 34, ly + 101, "not yet listed (masked)", size=12))
    s.append(text(lx, ly + 132, "node_mask: valid graph node", size=12.5, weight="bold", color=GREEN_STROKE))
    s.append(text(lx, ly + 148, "on this date?", size=12, color=MUTED))
    s.append(text(lx, ly + 170, "target_mask: has a label to", size=12.5, weight="bold", color=GREEN_STROKE))
    s.append(text(lx, ly + 186, "train / score here?", size=12, color=MUTED))
    s.append(text(lx, ly + 214, "Loss + graph attention run", size=12))
    s.append(text(lx, ly + 230, "on GREEN cells only. Missing", size=12))
    s.append(text(lx, ly + 246, "nodes are skipped per date,", size=12))
    s.append(text(lx, ly + 262, "not dropped for everyone.", size=12, weight="bold"))

    # ---- annotations under each grid
    ay = grid_bottom + 34
    s.append(text(LGX, ay, "Result:", size=14, weight="bold", color="#c62828"))
    for i, ln in enumerate([
        "Only t8-t12 survive = 5 usable dates; t1-t7 discarded for ALL tickers.",
        "F's late listing (t8) alone removes 7 dates from every ticker.",
        "-> low statistical power, early history lost,",
        "   biased toward long-history survivors.",
    ]):
        s.append(text(LGX, ay + 22 + i * 19, ln, size=13, color=TEXT))

    s.append(text(RGX, ay, "Result:", size=14, weight="bold", color="#2e7d32"))
    for i, ln in enumerate([
        "All 12 dates usable = more train/test dates -> higher power.",
        "Late-listed tickers included on the dates they exist",
        "-> no survivorship bias.",
        "Each date keeps only its valid (green) nodes for loss + attention.",
    ]):
        s.append(text(RGX, ay + 22 + i * 19, ln, size=13, color=TEXT))

    # ---- snapshot graphs
    sec_y = ay + 128
    s.append(line(60, sec_y - 20, W - 60, sec_y - 20, "#cfd8dc", sw=1))
    s.append(text(W / 2, sec_y + 4, "The graph snapshot adapts its node set per date via node_mask",
                  size=17, weight="bold", anchor="middle"))
    gcx1, gcx2 = 430, 1090
    gcy = sec_y + 150
    rad = 96
    s += draw_snapshot_graph(gcx1, gcy, rad, SNAP_EARLY,
                             "E, F not yet listed -> masked out (still present for A-D)")
    s += draw_snapshot_graph(gcx2, gcy, rad, SNAP_LATE,
                             "F listed at t8 -> now a valid node; graph grows to 6")
    # arrow between the two graphs
    s.append(line(gcx1 + rad + 60, gcy, gcx2 - rad - 60, gcy, MUTED, sw=2))
    s.append(f'<polygon points="{gcx2 - rad - 60},{gcy - 6} {gcx2 - rad - 48},{gcy} '
             f'{gcx2 - rad - 60},{gcy + 6}" fill="{MUTED}"/>')
    s.append(text((gcx1 + gcx2) / 2, gcy - 14, "later dates", size=12, color=MUTED, anchor="middle"))
    s.append(text((gcx1 + gcx2) / 2, gcy + 20, "more nodes", size=12, color=MUTED, anchor="middle"))

    s.append("</svg>")
    return "\n".join(s)


def main() -> Path:
    out = (Path(__file__).resolve().parents[3]
           / "docs" / "paper" / "diagrams" / "masked_panel_explained.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    main()
