#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Plots VectorStoreBenchmark results: two charts per results directory, as SVG and PNG.

  uv run benchmark-results/plot.py benchmark-results/*/

- latency.svg/.png: median top-10 search latency by corpus size, one line per store (log scales).
- recall.svg/.png: recall@10 of each search type at the largest corpus size, one dot per store.

Reads the raw CSV files through summarize.py, so the numbers match summary.csv.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Stable SVG output: the same data gives the same file, so committed charts only change with it.
matplotlib.rcParams["svg.hashsalt"] = "vector-store-benchmark"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import summarize  # noqa: E402

# Each store keeps its color and marker in every chart (categorical slots in fixed order).
STORES = {
    "SqliteVectorStore": ("SQLite (MediaPipe)", "#2a78d6", "o"),
    "ObjectBoxVectorStore": ("ObjectBox HNSW", "#eb6834", "s"),
    "ObjectBoxVectorStoreWideSearch": ("ObjectBox HNSW, ≥100 candidates", "#1baf7a", "D"),
    "InMemoryVectorStore": ("In-memory (exact)", "#eda100", "^"),
}
SEARCHES = [
    ("k10", "top-10"),
    ("k100", "top-100"),
    ("k1000", "top-1000"),
    ("k10_filtered", "top-10, filtered (10% match)"),
]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# About 3,800 px wide: sharp even full-width on a 4K screen.
DPI = 400


def style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)


def save(fig, out: Path, **kwargs) -> None:
    """Saves [fig] as out.svg and out.png."""
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE, metadata={"Date": None}, **kwargs)
    fig.savefig(out.with_suffix(".png"), dpi=DPI, facecolor=SURFACE, **kwargs)
    plt.close(fig)


def ms_label(value: float, _=None) -> str:
    return f"{value:.3g} ms"


def plot_latency(rows: list[dict], device: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5))
    style(ax, "Search latency, top-10", f"Median of 100 queries · {device}")
    stores = [s for s in STORES if any(r["store"] == s for r in rows)]
    corpora = sorted({r["corpus"] for r in rows})
    for store in stores:
        name, color, marker = STORES[store]
        points = sorted((r["corpus"], r["k10_p50_ms"]) for r in rows if r["store"] == store)
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        # Direct label at the line's end, in text ink; the marker beside it carries the color.
        ax.annotate(f"{name}  {ms_label(ys[-1])}", (xs[-1], ys[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(corpora)
    ax.set_xticklabels([f"{c // 1000}k vectors" for c in corpora])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(ms_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(corpora[0] / 1.5, corpora[-1] * 1.5)
    lows = [r["k10_p50_ms"] for r in rows]
    highs = [r["k10_p50_ms"] for r in rows]
    ax.set_ylim(10 ** (int(f"{min(lows):e}".split("e")[1])), max(highs) * 2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)  # room for the direct labels
    save(fig, out)


def plot_recall(rows: list[dict], device: str, out: Path) -> None:
    corpus = max(r["corpus"] for r in rows)
    rows = [r for r in rows if r["corpus"] == corpus]
    stores = [s for s in STORES if any(r["store"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    style(ax, f"Recall@10 at {corpus:,} vectors",
          f"Share of the exact top 10 found, 100 queries · {device}")
    # Within each search type, stores run top to bottom in legend order.
    offsets = [((len(stores) - 1) / 2 - i) * 0.16 for i in range(len(stores))]
    for offset, store in zip(offsets, stores):
        name, color, marker = STORES[store]
        (row,) = [r for r in rows if r["store"] == store]
        ys = [len(SEARCHES) - 1 - i + offset for i in range(len(SEARCHES))]
        xs = [row[f"{key}_recall_at_10"] for key, _ in SEARCHES]
        ax.scatter(xs, ys, s=70, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5,
                   label=name, zorder=3)
    ax.set_yticks(range(len(SEARCHES)))
    ax.set_yticklabels([label for _, label in reversed(SEARCHES)], color=TEXT)
    lowest = min(r[f"{key}_recall_at_10"] for r in rows for key, _ in SEARCHES)
    ax.set_xlim(min(0.85, lowest - 0.01), 1.005)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=len(stores), frameon=False,
              fontsize=9, labelcolor=TEXT, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out, bbox_inches="tight")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        directory = Path(arg)
        rows = summarize(directory)
        if not rows:
            continue
        device = rows[0]["device"].split(",")[0]
        plot_latency(rows, device, directory / "latency")
        plot_recall(rows, device, directory / "recall")
        print(f"Wrote latency and recall charts (SVG, PNG) to {directory}")


if __name__ == "__main__":
    main()
