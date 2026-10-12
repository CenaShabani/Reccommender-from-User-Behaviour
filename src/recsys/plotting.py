"""A small, consistent plotting style for the notebooks.

The categorical colours are a fixed, colour-blind-checked order. Slot 1 always
goes to the first series, slot 2 to the second, and so on. A model keeps its
colour across every chart (see :data:`MODEL_COLORS`).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = "Blues"
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dd"

MODEL_COLORS = {
    "random": "#9a998f",  # a neutral grey: the floor, not a contender
    "popularity": PALETTE[3],
    "trending": PALETTE[4],
    "item_knn": PALETTE[0],
    "als": PALETTE[1],
    "bpr": PALETTE[2],
    "two_stage": PALETTE[6],
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 4),
            "figure.dpi": 110,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, PALETTE[len(MODEL_COLORS) % len(PALETTE)])


def bar_leaderboard(board, metric: str, ax=None, title: str | None = None):
    """Horizontal bars, best at the top, value labels at the bar ends."""
    ax = ax or plt.gca()
    s = board[metric].sort_values()
    ax.barh(s.index, s.values, color=[model_color(n) for n in s.index], height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel(metric)
    ax.grid(axis="y", visible=False)
    ax.set_title(title or f"{metric} by model")
    ax.set_xlim(0, s.max() * 1.18 if s.max() > 0 else 1)
    return ax
