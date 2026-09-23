#!/usr/bin/env python3
"""Create the RQ1 no-mechanism baseline cooperation figure.

The figure uses three aligned subplots: one for each game family.  The trust
game retains a paired-dot comparison because its Sender and Receiver outcomes
are role specific.  The input contains retained RQ1 point estimates only, so
error bars are intentionally not drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


MODEL_ORDER = [
    "GLM-5.1",
    "GPT-5.4",
    "Gemini-3.1-Pro-Preview",
    "Claude-Opus-4-6",
    "Kimi-K2.6",
    "Grok-4.3",
]

MODEL_LABELS = {
    "GLM-5.1": "GLM-5.1",
    "GPT-5.4": "GPT-5.4",
    "Gemini-3.1-Pro-Preview": "Gemini 3.1 Pro",
    "Claude-Opus-4-6": "Claude Opus 4.6",
    "Kimi-K2.6": "Kimi K2.6",
    "Grok-4.3": "Grok 4.3",
}

# A clear but restrained scientific palette. Marker shape is redundant with
# colour in the only multi-series panel, so interpretation is colour-blind safe.
BLUE = "#4C78A8"
GREEN = "#59A14F"
PURPLE = "#8F6BB3"
CORAL = "#E97855"
CONNECTOR = "#C8CDD3"
TEXT = "#262626"
AXIS = "#4D4D4D"

PANELS = [
    {
        "letter": "a",
        "title": "Prisoner's dilemma",
        "xlabel": "Cooperative actions (%)",
        "series": [
            {
                "game": "PD",
                "outcome": "cooperation",
                "label": "Cooperation",
                "color": BLUE,
                "marker": "o",
            },
        ],
    },
    {
        "letter": "b",
        "title": "Public-goods game",
        "xlabel": "Contribution / endowment (%)",
        "series": [
            {
                "game": "PGG",
                "outcome": "contribution",
                "label": "Contribution",
                "color": GREEN,
                "marker": "o",
            },
        ],
    },
    {
        "letter": "c",
        "title": "Trust game",
        "xlabel": "Normalized role behaviour (%)",
        "series": [
            {
                "game": "TG",
                "outcome": "sender",
                "label": "Sender transfer",
                "color": PURPLE,
                "marker": "o",
            },
            {
                "game": "TG",
                "outcome": "receiver",
                "label": "Receiver return",
                "color": CORAL,
                "marker": "s",
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "RQ2_figure_output" / "rq1_fixed_cooperation_baselines.csv",
        help="Retained RQ1 baseline CSV used by the manuscript analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "RQ1_figure_output",
        help="Directory for publication figure files.",
    )
    return parser.parse_args()


def set_publication_style() -> None:
    # Fail loudly instead of silently substituting a different typeface.
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 0,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_and_validate(path: Path) -> pd.DataFrame:
    required = {"game", "outcome", "model", "baseline_pp"}
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["baseline_pp"] = pd.to_numeric(data["baseline_pp"], errors="raise")
    if not data["baseline_pp"].between(0, 100).all():
        raise ValueError("All baseline_pp values must lie between 0 and 100.")

    expected = {
        (series["game"], series["outcome"], model)
        for panel in PANELS
        for series in panel["series"]
        for model in MODEL_ORDER
    }
    observed = set(zip(data["game"], data["outcome"], data["model"]))
    missing_cells = sorted(expected.difference(observed))
    duplicate_cells = data.duplicated(["game", "outcome", "model"], keep=False)
    if missing_cells:
        raise ValueError(f"Missing RQ1 baseline cells: {missing_cells}")
    if duplicate_cells.any():
        duplicates = data.loc[duplicate_cells, ["game", "outcome", "model"]]
        raise ValueError(f"Duplicate RQ1 baseline cells:\n{duplicates.to_string(index=False)}")
    return data


def values_for(data: pd.DataFrame, series: dict[str, str]) -> np.ndarray:
    subset = data[
        (data["game"] == series["game"])
        & (data["outcome"] == series["outcome"])
    ].set_index("model")
    return np.asarray(
        [float(subset.loc[model, "baseline_pp"]) for model in MODEL_ORDER],
        dtype=float,
    )


def value_label(value: float) -> str:
    if value in {0.0, 100.0}:
        return f"{value:.0f}"
    return f"{value:.1f}"


def label_point(ax: plt.Axes, value: float, y: float, color: str) -> None:
    # Zeroes are already unambiguous at the origin; suppressing their labels
    # keeps the many tied baseline points visually quiet.
    if value == 0:
        return
    if value >= 94:
        x, ha = value - 2.2, "right"
    else:
        x, ha = value + 2.0, "left"
    ax.text(
        x,
        y,
        value_label(value),
        ha=ha,
        va="center",
        fontsize=5.8,
        color=color,
    )


def plot_panel(ax: plt.Axes, data: pd.DataFrame, panel: dict) -> None:
    series_list = panel["series"]
    arrays = [values_for(data, series) for series in series_list]
    y = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = [0.0] if len(series_list) == 1 else [-0.09, 0.09]

    if len(series_list) == 2:
        for row, x1, x2 in zip(y, arrays[0], arrays[1]):
            ax.plot(
                [x1, x2],
                [row + offsets[0], row + offsets[1]],
                color=CONNECTOR,
                linewidth=1.0,
                solid_capstyle="round",
                zorder=1,
            )

    for series, values, offset in zip(series_list, arrays, offsets):
        ax.scatter(
            values,
            y + offset,
            s=33 if series["marker"] == "o" else 31,
            marker=series["marker"],
            facecolor=series["color"],
            edgecolor="white",
            linewidth=0.55,
            label=series["label"],
            zorder=3,
        )
        for row, value in zip(y + offset, values):
            label_point(ax, float(value), float(row), series["color"])

    ax.set_xlim(-3.2, 103.2)
    ax.set_ylim(-0.62, len(MODEL_ORDER) - 0.38)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_xlabel(panel["xlabel"], labelpad=3)
    title_pad = 15 if len(series_list) == 2 else 8
    ax.set_title(panel["title"], loc="left", fontweight="bold", pad=title_pad)
    ax.text(
        -0.12,
        1.075,
        panel["letter"],
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )
    if len(series_list) == 2:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.035),
            ncol=2,
            frameon=False,
            handletextpad=0.3,
            columnspacing=0.7,
            borderaxespad=0,
        )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=TEXT, direction="out")
    ax.tick_params(axis="y", colors=TEXT, pad=3)


def write_caption(output: Path) -> None:
    caption = """**Figure 2 | No-mechanism cooperation baselines in RQ1.**
**a**, Proportion of cooperative actions in the prisoner's dilemma. **b**,
Public-goods contributions relative to the per-round endowment. **c**,
Within-model comparison of trust-game transfers relative to the Sender's
endowment (purple circles) and returns relative to the amount received by the
Receiver (coral squares). Thin grey lines link the two role-specific estimates
for the same model. Values are the retained RQ1 baseline point estimates. Error
bars are not shown because the retained summary table does not provide run-level
baseline observations or uncertainty estimates. Percentages use game-specific
denominators and should not be interpreted as a pooled psychological scale.
"""
    (output / "Figure2_RQ1_no_mechanism_baseline_caption.md").write_text(
        caption,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data = read_and_validate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    # Approximately 183 mm wide: the standard double-column target width.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 3.25),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for index, (ax, panel) in enumerate(zip(axes, PANELS)):
        plot_panel(ax, data, panel)
        ax.tick_params(labelbottom=True, labelleft=(index == 0))

    stem = "Figure2_RQ1_no_mechanism_baseline"
    fig.savefig(args.output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.svg", bbox_inches="tight")
    write_caption(args.output)
    plt.close(fig)

    print(f"Saved RQ1 figure files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
