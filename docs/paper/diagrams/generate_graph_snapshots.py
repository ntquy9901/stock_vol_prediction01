"""Render real masked cross-stock graph snapshots (the per-day graphs G0/G1 operate on).

The masked graph manifest places each pooled per-ticker window as one node of the
snapshot for its own target date; a ticker with no window that day is masked
(``presence_mask[id] == 0``) and never imputed. Edges are the signed correlation of
the present tickers' scaled Parkinson-volatility windows. G0 and G1 receive the SAME
snapshot structure: G0 ignores the edges (each node independent), G1 mixes cross-stock
information along them.

This module draws a few representative TRAIN snapshots, each in two adjacency modes:
  - dense: full present-node correlation graph
  - k-NN top-8: GAT-hybrid-style mutual top-8 sparsification of the same correlation

Rendering (``render_graph_figure``) is separated from the heavy real-data manifest
build (``build_panels``) so the smoke test can drive the drawing path on a tiny
synthetic panel. Text is kept as SVG ``<text>`` (svg.fonttype='none') so ticker and
mode labels stay searchable in the output.

Run (inside a tree that has the masked-graph baseline code):
    python docs/paper/diagrams/generate_graph_snapshots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Representative TRAIN target dates (present-node counts confirmed from the real
# masked manifest, horizon=5): few / mid / near-full.
_TARGET_DATES: tuple[str, ...] = ("2010-01-12", "2017-03-13", "2018-08-08")

_MODE_DENSE = "dense"
_MODE_KNN = "k-NN top-8"

_EDGE_POS = "#c0392b"
_EDGE_NEG = "#2c6fbb"
_NODE_PRESENT = "#1f4e79"
_NODE_ABSENT = "#cccccc"


def _node_positions(count: int) -> np.ndarray:
    """Circular layout: node 0 at the top, going clockwise."""

    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(count) / count
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _present_edges(adjacency: np.ndarray, present_ids: list[int]) -> list[tuple[int, int, float]]:
    """Off-diagonal nonzero present-present edges as (i, j, signed_corr) with i < j."""

    edges: list[tuple[int, int, float]] = []
    for a_index, i in enumerate(present_ids):
        for j in present_ids[a_index + 1:]:
            value = float(adjacency[i, j])
            if value != 0.0:
                edges.append((i, j, value))
    return edges


def _draw_graph(
    ax: plt.Axes, tickers_all: list[str], present_ids: list[int], adjacency: np.ndarray,
    mode_label: str, date: str,
) -> None:
    """Draw one snapshot: fixed 33-position ring, present nodes solid, absent nodes greyed."""

    count = len(tickers_all)
    positions = _node_positions(count)
    present = set(present_ids)

    edges = _present_edges(adjacency, present_ids)
    for i, j, value in edges:
        magnitude = min(abs(value), 1.0)
        ax.plot(
            [positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]],
            color=_EDGE_POS if value > 0 else _EDGE_NEG,
            linewidth=0.3 + 2.2 * magnitude, alpha=min(0.12 + 0.6 * magnitude, 0.85),
            zorder=1, solid_capstyle="round",
        )

    for index in range(count):
        x, y = positions[index]
        is_present = index in present
        ax.scatter(
            [x], [y], s=90 if is_present else 18,
            color=_NODE_PRESENT if is_present else _NODE_ABSENT,
            edgecolors="white" if is_present else "none", linewidths=0.6, zorder=2,
        )
        label_r = 1.13
        ax.text(
            x * label_r, y * label_r, tickers_all[index],
            fontsize=7 if is_present else 5,
            color="black" if is_present else "#b0b0b0",
            fontweight="bold" if is_present else "normal",
            ha="center", va="center", zorder=3,
        )

    avg_degree = 2.0 * len(edges) / len(present_ids)
    ax.set_title(
        f"{date}  |  {mode_label}\n{len(present_ids)} nodes, {len(edges)} edges, "
        f"avg deg {avg_degree:.1f}",
        fontsize=9,
    )
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_heatmap(
    ax: plt.Axes, tickers_all: list[str], present_ids: list[int], adjacency: np.ndarray,
    mode_label: str,
) -> None:
    """Draw the present-node adjacency submatrix (signed correlation, diverging colormap)."""

    labels = [tickers_all[i] for i in present_ids]
    sub = adjacency[np.ix_(present_ids, present_ids)]
    ax.imshow(sub, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=4, rotation=90)
    ax.set_yticklabels(labels, fontsize=4)
    ax.set_title(f"{mode_label} adjacency", fontsize=9)
    ax.tick_params(length=0)


def render_graph_figure(panels: list[dict], output_path: Path | str) -> None:
    """Render snapshot panels (rows) x {dense graph, dense heatmap, knn graph, knn heatmap}.

    Each panel dict carries ``date``, ``split``, ``tickers_all`` (full vocabulary order),
    ``present_ids``, and dense / knn adjacency matrices over the full vocabulary.
    """

    if not panels:
        raise ValueError("render_graph_figure requires at least one panel")

    # Keep text as <text> so ticker/mode labels are searchable in the SVG output.
    plt.rcParams["svg.fonttype"] = "none"

    rows = len(panels)
    figure, axes = plt.subplots(rows, 4, figsize=(19, 5.0 * rows), squeeze=False)
    for row, panel in enumerate(panels):
        tickers_all = panel["tickers_all"]
        present_ids = list(panel["present_ids"])
        _draw_graph(axes[row, 0], tickers_all, present_ids, panel["adjacency_dense"],
                    _MODE_DENSE, panel["date"])
        _draw_heatmap(axes[row, 1], tickers_all, present_ids, panel["adjacency_dense"], _MODE_DENSE)
        _draw_graph(axes[row, 2], tickers_all, present_ids, panel["adjacency_knn"],
                    _MODE_KNN, panel["date"])
        _draw_heatmap(axes[row, 3], tickers_all, present_ids, panel["adjacency_knn"], _MODE_KNN)

    figure.suptitle(
        "Masked cross-stock graph snapshots (train split): same structure feeds G0 and G1",
        fontsize=13, y=0.995,
    )
    figure.text(
        0.5, 0.008,
        "Nodes = present tickers on that target date (absent tickers greyed on the ring, never "
        "imputed). Edge width/opacity is proportional to |corr| of scaled Parkinson-volatility "
        "windows; red = positive, blue = negative correlation. Heatmap = signed correlation "
        "(red +, blue -, white 0/absent). G0 = message-passing OFF (edges drawn but ignored, each "
        "node independent); G1 = message-passing ON (the SAME edges mix cross-stock information).",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.98))
    figure.savefig(output_path)  # format inferred from the path suffix (.svg deliverable)
    plt.close(figure)


def build_panels(target_dates: tuple[str, ...] = _TARGET_DATES) -> list[dict]:
    """Build the real masked graph manifest (dense + knn-8) and extract the chosen TRAIN snapshots.

    Reproduces the ``run_pilot.py --phase graph --graph masked`` builder inputs: processed price
    data + the dual-group news panel, graph-store scalers fitted on the common-date 70% boundary,
    pooled manifest with news attached, then ``build_masked_graph_manifest`` over the full union.
    """

    import sys

    root = _repo_root()
    code_dir = root / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
    for path in (str(root), str(code_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import run_pilot as rp
    from data import (
        PooledManifest, attach_news, build_masked_graph_manifest, build_pooled_manifest,
        load_and_split_price_data,
    )

    horizon = 5
    inputs = rp.build_screening_inputs(False, None, "P3", horizon=horizon)
    selected = tuple(inputs.smoke_filter["selected_tickers"])
    raw = rp._select_tickers(load_and_split_price_data(root / "data" / "processed"), selected)
    full_frames = {
        ticker: rp.np_concat_frames(raw.frames[ticker][name] for name in ("train", "val", "test"))
        for ticker in selected
    }
    panel = rp.load_runner_news_panel(
        root / "data" / "features" / "dual_group_news_panel.parquet",
        selected, rp._train_news_cutoffs(inputs.manifest),
    )
    graph_store = rp._fit_graph_preprocessors(full_frames)
    pooled = build_pooled_manifest(raw, graph_store, horizon=horizon)
    pooled = PooledManifest(
        {split: tuple(attach_news(pooled.samples[split], panel, panel.feature_cols))
         for split in ("train", "val", "test")},
        pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash,
    )
    dense = build_masked_graph_manifest(pooled, graph_store, adjacency="dense")
    knn = build_masked_graph_manifest(pooled, graph_store, adjacency="knn", top_k=8)

    tickers_all = [ticker for ticker, _ in sorted(dense.ticker_to_id.items(), key=lambda kv: kv[1])]
    dense_train = {s.target_date: s for s in dense.snapshots if s.split == "train"}
    knn_train = {s.target_date: s for s in knn.snapshots if s.split == "train"}

    panels: list[dict] = []
    for date in target_dates:
        if date not in dense_train:
            raise ValueError(f"target date {date} is not a train snapshot")
        snapshot = dense_train[date]
        present_ids = [int(i) for i in np.flatnonzero(snapshot.presence_mask)]
        panels.append({
            "date": date, "split": "train", "tickers_all": tickers_all,
            "present_ids": present_ids,
            "adjacency_dense": np.asarray(snapshot.adjacency, dtype=float),
            "adjacency_knn": np.asarray(knn_train[date].adjacency, dtype=float),
        })
    return panels


def _repo_root() -> Path:
    """Locate the repo root that holds the baseline code, from this file's location."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code").is_dir():
            return parent
    raise FileNotFoundError("could not locate the pooled_news_gnn_ablation baseline code directory")


def main() -> None:
    panels = build_panels()
    for panel in panels:
        dense = panel["adjacency_dense"]
        knn = panel["adjacency_knn"]
        present = panel["present_ids"]
        d_edges = len(_present_edges(dense, present))
        k_edges = len(_present_edges(knn, present))
        print(f"{panel['date']}: present={len(present)} dense_edges={d_edges} knn8_edges={k_edges}")
    output = Path(__file__).resolve().parent / "graph_snapshots_g0_g1.svg"
    render_graph_figure(panels, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
