#!/usr/bin/env python3
"""Create an RQ1 bubble matrix for eleven models and four outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


MODEL_ORDER = [
    "Llama-3-8B",
    "Llama-3-70B",
    "Qwen2.5-7B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct",
    "GLM-5.1",
    "GPT-5.4",
    "Gemini-3.1-Pro-Preview",
    "Claude-Opus-4-6",
    "Kimi-K2.6",
    "Grok-4.3",
    "DeepSeek-V4-Pro",
]

MODEL_LABELS = {
    "Llama-3-8B": "Llama 3 8B",
    "Llama-3-70B": "Llama 3 70B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next 80B-A3B",
    "GLM-5.1": "GLM-5.1",
    "GPT-5.4": "GPT-5.4",
    "Gemini-3.1-Pro-Preview": "Gemini 3.1 Pro",
    "Claude-Opus-4-6": "Claude Opus 4.6",
    "Kimi-K2.6": "Kimi K2.6",
    "Grok-4.3": "Grok 4.3",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
}

COLUMNS = [
    ("PD", "cooperation", "Prisoner's\ndilemma"),
    ("PGG", "contribution", "Public-goods\ngame"),
    ("TG", "sender", "Trust game\nSender"),
    ("TG", "receiver", "Trust game\nReceiver"),
]

TEXT = "#262626"
MUTED = "#777777"
ZERO_EDGE = "#BFC5CC"
SEPARATOR = "#D8DDE2"

# Soft cyan-to-indigo palette with enough contrast for values near zero while
# remaining readable in print and under common colour-vision deficiencies.
COOP_CMAP = LinearSegmentedColormap.from_list(
    "cooperation",
    ["#EAF5F2", "#8FD3C7", "#42A7B3", "#5470B3", "#59459B"],
)
NORM = Normalize(vmin=0, vmax=100)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "RQ1_figure_output" / "rq1_11model_mean_sd.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "RQ1_figure_output",
    )
    return parser.parse_args()


def set_style() -> None:
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_and_validate(path: Path) -> pd.DataFrame:
    required = {"game", "outcome", "model", "mean_pct", "sd_pct", "n_runs"}
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.copy()
    data["mean_pct"] = pd.to_numeric(data["mean_pct"], errors="raise")
    data["sd_pct"] = pd.to_numeric(data["sd_pct"], errors="coerce")
    if not data["mean_pct"].between(0, 100).all():
        raise ValueError("Means must lie between 0 and 100 percent.")
    if (data["sd_pct"].dropna() < 0).any():
        raise ValueError("SD values cannot be negative.")

    expected = {
        (game, outcome, model)
        for game, outcome, _ in COLUMNS
        for model in MODEL_ORDER
    }
    observed = set(zip(data["game"], data["outcome"], data["model"]))
    if expected != observed:
        raise ValueError(
            "Input cells do not match the 11-model by 4-outcome matrix: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return data.set_index(["game", "outcome", "model"])


def bubble_area(mean: float) -> float:
    # Matplotlib interprets s as area in points squared. A sub-linear exponent
    # prevents medium values from becoming visually negligible next to 100%.
    return 20.0 + 300.0 * (mean / 100.0) ** 0.72


def value_text(mean: float, sd: float) -> str:
    mean_text = "0" if mean == 0 else f"{mean:.1f}"
    if np.isfinite(sd):
        sd_text = "0" if sd == 0 else f"{sd:.1f}"
    else:
        sd_text = "n/a"
    return f"{mean_text} ± {sd_text}"


def draw_matrix(ax: plt.Axes, data: pd.DataFrame) -> None:
    for row, model in enumerate(MODEL_ORDER):
        for column, (game, outcome, _) in enumerate(COLUMNS):
            record = data.loc[(game, outcome, model)]
            mean = float(record["mean_pct"])
            sd = float(record["sd_pct"]) if pd.notna(record["sd_pct"]) else np.nan

            bubble_y = row - 0.105
            if mean == 0:
                ax.scatter(
                    column,
                    bubble_y,
                    s=bubble_area(mean),
                    facecolor="white",
                    edgecolor=ZERO_EDGE,
                    linewidth=0.8,
                    zorder=3,
                )
            else:
                ax.scatter(
                    column,
                    bubble_y,
                    s=bubble_area(mean),
                    facecolor=COOP_CMAP(NORM(mean)),
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

            # Exact mean +/- SD is printed under every bubble. This is more
            # faithful than inventing a second geometric scale for uncertainty.
            ax.text(
                column,
                row + 0.255,
                value_text(mean, sd),
                ha="center",
                va="center",
                fontsize=5.0,
                color=MUTED if np.isfinite(sd) else "#9A5F25",
                zorder=4,
            )

    # Separate the two one-outcome games from the two trust-game roles.
    ax.axvline(1.5, color=SEPARATOR, linewidth=0.7, zorder=0)
    ax.set_xlim(-0.62, len(COLUMNS) - 0.38)
    ax.set_ylim(-0.72, len(MODEL_ORDER) - 0.25)
    ax.invert_yaxis()
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([label for _, _, label in COLUMNS], fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=7, colors=TEXT)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.tick_params(axis="y", length=0, pad=7, colors=TEXT)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.text(
        -0.10,
        1.045,
        "a",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def write_caption(output: Path) -> None:
    caption = """**Figure 2 | Bubble-matrix view of no-mechanism cooperation in RQ1.**
Rows denote the 11 language models and columns denote the four game-specific
cooperation outcomes. Bubble area and colour both encode the reported mean;
larger and darker bubbles indicate higher cooperation. Numbers below bubbles
report mean ± SD across three independent experimental runs. An open bubble
denotes a mean of zero. `n/a` denotes the two Gemini 3.1 Pro trust-game cells
for which the supplied summary table reports a mean but no SD. Because each
column uses a game-specific behavioural denominator, values should not be
interpreted as a pooled psychological scale.
"""
    (output / "Figure2_RQ1_11model_bubble_matrix_caption.md").write_text(
        caption,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data = read_and_validate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()

    fig, ax = plt.subplots(figsize=(7.20, 4.85), layout="constrained")
    draw_matrix(ax, data)

    colorbar = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=COOP_CMAP),
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
        aspect=38,
    )
    colorbar.set_ticks([0, 25, 50, 75, 100])
    colorbar.set_label("Mean cooperation (%)", labelpad=3)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(labelsize=6, width=0.5, length=2.2)

    stem = "Figure2_RQ1_11model_bubble_matrix"
    fig.savefig(args.output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.svg", bbox_inches="tight")
    write_caption(args.output)
    plt.close(fig)
    print(f"Saved RQ1 bubble-matrix files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
