from collections.abc import Callable, Sequence
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-colorblind")

BACKGROUND_COLOR = "#ffffff"
PRIMARY_COLOR = "#365f91"
GRID_COLOR = "#e1e4ea"
TEXT_COLOR = "#1f2933"
SMALL_MULTIPLE_COLS = 5
SMALL_MULTIPLE_REPO_PER_ROW_HEIGHT = 2.4
SMALL_MULTIPLE_TITLE_PAD = 0.9
SMALL_MULTIPLE_BAR_ALPHA = 0.9
SMALL_MULTIPLE_TITLE_BG = "#f5f7fb"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_axes(ax: plt.Axes, *, xlabel: str | None) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel("")
    ax.set_axisbelow(True)
    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=0.6,
        color=GRID_COLOR,
        alpha=0.8,
        zorder=1,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="upper"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR, pad=4)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT_COLOR)


def horizontal_bar(
    data: pd.DataFrame,
    label_col: str,
    value_col: str,
    *,
    title: str,
    destination: Path,
    top: int | None = None,
    xlabel: str | None = None,
    annotations: Sequence[str] | None = None,
    value_formatter: str | Callable[[float], str] = "{:.0f}",
) -> bool:
    subset = data.head(top) if top is not None else data
    if subset.empty:
        return False
    subset = subset.copy()
    subset[label_col] = subset[label_col].astype(str)
    values = subset[value_col].astype(float)

    fig_height = max(2.0, 0.4 * len(subset) + 1.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    max_value = float(values.max()) if not values.empty else 0.0
    x_upper = max_value * 1.05 if max_value > 0 else 1.0
    ax.set_xlim(0, x_upper)
    bars = ax.barh(
        subset[label_col],
        values,
        color=PRIMARY_COLOR,
        edgecolor="none",
        zorder=3,
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, pad=14, color=TEXT_COLOR, weight="bold")

    _configure_axes(ax, xlabel=xlabel or value_col.replace("_", " ").title())

    if annotations is not None:
        labels = list(annotations)
    else:
        if isinstance(value_formatter, str):
            labels = [value_formatter.format(float(val)) for val in values]
        else:
            labels = [value_formatter(float(val)) for val in values]
    ax.bar_label(
        bars,
        labels=labels,
        padding=6,
        fontsize=9,
        color=TEXT_COLOR,
        label_type="edge",
    )

    fig.tight_layout(pad=1.0)
    ensure_directory(destination.parent)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def small_multiple_bars(
    pivot: pd.DataFrame,
    *,
    title: str,
    destination: Path,
    cols: int = SMALL_MULTIPLE_COLS,
    legend: bool = True,
) -> bool:
    if pivot.empty:
        return False

    methods = list(pivot.columns)
    sorted_pivot = pivot.assign(__avg=pivot.mean(axis=1)).sort_values("__avg", ascending=False).drop(columns="__avg")
    repos = list(sorted_pivot.index)
    if not methods or not repos:
        return False

    rows = math.ceil(len(repos) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.4, rows * SMALL_MULTIPLE_REPO_PER_ROW_HEIGHT),
        sharey=False,
    )
    axes_iter = np.atleast_1d(np.array(axes, dtype=object).ravel())
    color_map = plt.get_cmap("tab10")
    bar_width = 0.8 / len(methods)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_width
    x_positions = np.array([0.0], dtype=float)
    global_max = float(sorted_pivot.to_numpy(dtype=float).max(initial=0.0))
    min_height = max(global_max * 0.15, 0.02)

    for idx, repo in enumerate(repos):
        ax = axes_iter[idx]
        values = sorted_pivot.loc[repo].to_numpy(dtype=float)
        ax.set_facecolor(BACKGROUND_COLOR)
        local_max = float(values.max(initial=0.0))
        y_upper = max(local_max * 1.2 if local_max > 0 else 0.0, min_height)
        for method_idx, (value, method) in enumerate(zip(values, methods, strict=False)):
            color = color_map(method_idx % color_map.N)
            ax.bar(
                x_positions + offsets[method_idx],
                value,
                width=bar_width,
                color=color,
                edgecolor="none",
                alpha=SMALL_MULTIPLE_BAR_ALPHA,
                label=method if idx == 0 else None,
            )
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, y_upper)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="-", linewidth=0.4, color=GRID_COLOR, alpha=0.5)
        ax.set_title("")
        bbox = dict(boxstyle="round,pad=0.2", fc=SMALL_MULTIPLE_TITLE_BG, ec="none", alpha=0.85)
        ax.text(
            0.5,
            0.96,
            repo,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=TEXT_COLOR,
            bbox=bbox,
        )

    for idx in range(len(repos), len(axes_iter)):
        axes_iter[idx].axis("off")

    for ax in axes_iter[::cols]:
        ax.set_ylabel("Weight", fontsize=8, color=TEXT_COLOR)
    for ax in axes_iter:
        ax.tick_params(axis="y", labelsize=7, colors=TEXT_COLOR)

    fig.suptitle(title, fontsize=14, color=TEXT_COLOR)
    if legend:
        handles, labels = axes_iter[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(methods), 4),
                frameon=False,
                fontsize=9,
                bbox_to_anchor=(0.5, 0.92),
            )

    top = SMALL_MULTIPLE_TITLE_PAD if legend else 1.0
    fig.tight_layout(pad=1.0, rect=(0, 0, 1, top))
    if legend:
        fig.subplots_adjust(top=0.85)
    ensure_directory(destination.parent)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def histogram(
    values: pd.Series,
    *,
    title: str,
    destination: Path,
    xlabel: str,
    ylabel: str = "Entities",
    annotate: bool = False,
) -> bool:
    clean = values.dropna()
    if clean.empty:
        return False
    clean = clean.astype(float)
    min_val = int(clean.min())
    max_val = int(clean.max())
    if min_val == max_val:
        min_edge = min_val - 0.5
        max_edge = max_val + 0.5
        bins: Sequence[float] = [min_edge, max_edge]
    else:
        bins = np.arange(int(min_val), int(max_val) + 2)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    counts, bin_edges, patches = ax.hist(
        clean,
        bins=bins,
        color=PRIMARY_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    counts_array = np.asarray(counts, dtype=float)

    ax.set_title(title, fontsize=13, pad=12, color=TEXT_COLOR, weight="bold")
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    max_count = float(np.max(counts_array)) if counts_array.size else 0.0
    ax.set_ylim(bottom=0, top=max(max_count * 1.1, 1))
    ax.tick_params(axis="x", labelsize=9, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR)

    if annotate:
        patch_sources: list[Any]
        if hasattr(patches, "patches"):
            patch_sources = list(patches.patches)  # type: ignore[attr-defined]
        else:
            try:
                patch_sources = list(patches)  # type: ignore[arg-type]
            except TypeError:
                patch_sources = [patches]
        for count, patch in zip(counts_array, patch_sources, strict=False):
            if count <= 0:
                continue
            x_center = patch.get_x() + patch.get_width() / 2
            ax.annotate(
                f"{int(count)}",
                (x_center, count),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=1.0)
    ensure_directory(destination.parent)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
