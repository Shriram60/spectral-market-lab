"""Shared figure style: light surface, a CVD-validated categorical order, hairline chrome."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#cde2fb"  # lightest step of the blue ramp, for interval bands
# Fixed categorical order; assign slots in order, never cycle past eight.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def figures_dir() -> Path:
    return Path(os.environ.get("SPECTRAL_MARKET_FIGURES", "figures"))


def results_dir() -> Path:
    return Path(os.environ.get("SPECTRAL_MARKET_RESULTS", "results"))


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "text.color": INK,
        }
    )


def save(fig, name: str) -> Path:
    path = figures_dir() / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_results(name: str, payload: dict) -> Path:
    path = results_dir() / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def end_label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    """Direct label just right of a line's last point, in text ink (the line carries hue)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK_2,
        annotation_clip=False,
    )


def note(ax, text: str, loc: str = "upper left") -> None:
    """Small secondary-ink annotation box inside an axes."""
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        color=INK_2,
        bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": GRID, "lw": 0.6},
    )
