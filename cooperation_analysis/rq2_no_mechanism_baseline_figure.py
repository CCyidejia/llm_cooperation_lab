#!/usr/bin/env python3
"""Plot the no-mechanism cooperation baselines used by the RQ2 analysis.

The retained RQ2 baseline table contains one point estimate per
model/game/outcome cell.  It does not contain run-level observations, so this
figure deliberately shows no error bars.  Supplying uncertainty that is not
present in the source data would make the figure misleading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

# Okabe-Ito-inspired, colour-vision-deficiency-friendly colours.  Colour is
# redundant with panel position and therefore never carries meaning alone.
PANELS = [
    {
        "game": "PD",
        "outcome": "cooperation",
        "letter": "a",
        "title": "Prisoner's dilemma",
        "xlabel": "Cooperative actions (%)",
        "color": "#0072B2",
    },
    {
        "game": "PGG",
        "outcome": "contribution",
        "letter": "b",
        "title": "Public-goods game",
        "xlabel": "Contribution / endowment (%)",
        "color": "#009E73",
    },
    {
        "game": "TG",
        "outcome": "sender",
        "letter": "c",
        "title": "Trust game: Sender",
        "xlabel": "Transfer / endowment (%)",
        "color": "#E69F00",
    },
    {
        "game": "TG",
        "outcome": "receiver",
        "letter": "d",
        "title": "Trust game: Receiver",
        "xlabel": "Return / amount received (%)",
        "color": "#CC79A7",
    },
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "RQ2_figure_output" / "rq1_fixed_cooperation_baselines.csv",
        help="CSV containing game, outcome, model, and baseline_pp columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "RQ2_figure_output",
        help="Directory for PNG, PDF, SVG, and caption outputs.",
    )
    return parser.parse_args()


def set_nature_style() -> None:
    """Use a compact, accessible journal-style Matplotlib theme."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
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
        (panel["game"], panel["outcome"], model)
        for panel in PANELS
        for model in MODEL_ORDER
    }
    observed = set(zip(data["game"], data["outcome"], data["model"]))
    missing_cells = sorted(expected.difference(observed))
    duplicate_cells = data.duplicated(["game", "outcome", "model"], keep=False)
    if missing_cells:
        raise ValueError(f"Missing baseline cells: {missing_cells}")
    if duplicate_cells.any():
        duplicates = data.loc[duplicate_cells, ["game", "outcome", "model"]]
        raise ValueError(f"Duplicate baseline cells:\n{duplicates.to_string(index=False)}")
    return data


def format_value(value: float) -> str:
    if value in {0.0, 100.0}:
        return f"{value:.0f}"
    return f"{value:.1f}"


def plot_panel(ax: plt.Axes, data: pd.DataFrame, panel: dict[str, str]) -> None:
    subset = data[
        (data["game"] == panel["game"])
        & (data["outcome"] == panel["outcome"])
    ].set_index("model")
    values = [float(subset.loc[model, "baseline_pp"]) for model in MODEL_ORDER]
    y_positions = list(range(len(MODEL_ORDER)))
    color = panel["color"]

    # Stems encode distance from zero; they are data marks, not grid lines.
    for y, value in zip(y_positions, values):
        ax.hlines(y, 0, value, color=color, alpha=0.30, linewidth=1.2, zorder=1)
    ax.scatter(
        values,
        y_positions,
        s=25,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )

    for y, value in zip(y_positions, values):
        if value >= 94:
            x, alignment = value - 2.2, "right"
        else:
            x, alignment = value + 2.2, "left"
        ax.text(x, y, format_value(value), ha=alignment, va="center", fontsize=6, color="#333333")

    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.65, len(MODEL_ORDER) - 0.35)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_xlabel(panel["xlabel"], labelpad=3)
    ax.set_title(panel["title"], loc="left", fontweight="bold", pad=7)
    ax.text(
        -0.12,
        1.06,
        panel["letter"],
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4D4D4D")
    ax.spines["bottom"].set_color("#4D4D4D")
    ax.tick_params(axis="x", colors="#333333", direction="out")
    ax.tick_params(axis="y", colors="#333333", pad=3)


def write_caption(output: Path) -> None:
    caption = """**Figure 2 | No-mechanism cooperation baselines used in RQ2.**
**a**, Proportion of cooperative actions in the prisoner's dilemma.
**b**, Public-goods contributions as a percentage of the per-round endowment.
**c**, Trust-game transfers as a percentage of the sender's endowment.
**d**, Trust-game returns as a percentage of the amount received by the receiver.
Points show the retained baseline point estimate for each model. Error bars are
not shown because the retained RQ2 baseline table contains summary means rather
than run-level baseline observations or uncertainty estimates. Measures are
normalized to percentages for display but represent game-specific behaviours
and are not pooled across games.
"""
    (output / "Figure2_RQ2_no_mechanism_baseline_caption.md").write_text(
        caption,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data = read_and_validate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    set_nature_style()

    # 183 mm double-column width, with a compact height suitable for a
    # four-panel Nature-style figure.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.20, 5.05),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, panel in zip(axes.flat, PANELS):
        plot_panel(ax, data, panel)
        ax.tick_params(labelbottom=True, labelleft=True)

    stem = "Figure2_RQ2_no_mechanism_baseline"
    fig.savefig(args.output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.svg", bbox_inches="tight")
    write_caption(args.output)
    plt.close(fig)

    print(f"Saved figure files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
