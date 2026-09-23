#!/usr/bin/env python3
"""Plot RQ1 cooperation means and between-run SDs for eleven models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


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

BLUE = "#4C78A8"
GREEN = "#59A14F"
PURPLE = "#8F6BB3"
CORAL = "#E97855"
AXIS = "#4D4D4D"
TEXT = "#262626"

PANELS = [
    {
        "letter": "a",
        "title": "Prisoner's dilemma",
        "xlabel": "Cooperative actions (%)",
        "series": [
            {"game": "PD", "outcome": "cooperation", "label": "Cooperation", "color": BLUE, "marker": "o"},
        ],
    },
    {
        "letter": "b",
        "title": "Public-goods game",
        "xlabel": "Contribution / endowment (%)",
        "series": [
            {"game": "PGG", "outcome": "contribution", "label": "Contribution", "color": GREEN, "marker": "o"},
        ],
    },
    {
        "letter": "c",
        "title": "Trust game",
        "xlabel": "Normalized role behaviour (%)",
        "series": [
            {"game": "TG", "outcome": "sender", "label": "Sender transfer", "color": PURPLE, "marker": "o"},
            {"game": "TG", "outcome": "receiver", "label": "Receiver return", "color": CORAL, "marker": "s"},
        ],
    },
]


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
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 0,
            "legend.fontsize": 6.2,
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
    data["n_runs"] = pd.to_numeric(data["n_runs"], errors="raise").astype(int)
    if not data["mean_pct"].between(0, 100).all():
        raise ValueError("All means must lie between 0 and 100 percent.")
    if (data["sd_pct"].dropna() < 0).any():
        raise ValueError("SD values cannot be negative.")

    expected = {
        (series["game"], series["outcome"], model)
        for panel in PANELS
        for series in panel["series"]
        for model in MODEL_ORDER
    }
    observed = set(zip(data["game"], data["outcome"], data["model"]))
    if expected != observed:
        missing_cells = sorted(expected.difference(observed))
        extra_cells = sorted(observed.difference(expected))
        raise ValueError(f"Cell mismatch; missing={missing_cells}, extra={extra_cells}")
    if data.duplicated(["game", "outcome", "model"]).any():
        raise ValueError("Duplicate model/game/outcome rows found.")
    return data


def select_series(data: pd.DataFrame, series: dict[str, str]) -> pd.DataFrame:
    selected = data[
        (data["game"] == series["game"])
        & (data["outcome"] == series["outcome"])
    ].set_index("model")
    return selected.loc[MODEL_ORDER]


def draw_series(
    ax: plt.Axes,
    selected: pd.DataFrame,
    series: dict[str, str],
    y: np.ndarray,
    offset: float,
) -> None:
    means = selected["mean_pct"].to_numpy(dtype=float)
    sds = selected["sd_pct"].to_numpy(dtype=float)
    available = np.isfinite(sds)

    # Draw the reported mean +/- between-run SD. Error bars are clipped only by
    # the displayed axis, not numerically altered.
    if available.any():
        ax.errorbar(
            means[available],
            y[available] + offset,
            xerr=sds[available],
            fmt=series["marker"],
            markersize=4.8,
            markerfacecolor=series["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            ecolor=series["color"],
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            linestyle="none",
            label=series["label"],
            zorder=3,
        )

    # A hollow marker explicitly denotes a reported mean whose SD is absent.
    if (~available).any():
        ax.scatter(
            means[~available],
            y[~available] + offset,
            s=27,
            marker=series["marker"],
            facecolor="white",
            edgecolor=series["color"],
            linewidth=1.0,
            zorder=4,
        )


def plot_panel(ax: plt.Axes, data: pd.DataFrame, panel: dict) -> None:
    y = np.arange(len(MODEL_ORDER), dtype=float)
    series_list = panel["series"]
    offsets = [0.0] if len(series_list) == 1 else [-0.11, 0.11]

    for series, offset in zip(series_list, offsets):
        draw_series(ax, select_series(data, series), series, y, offset)

    ax.set_xlim(-4, 104)
    ax.set_ylim(-0.65, len(MODEL_ORDER) - 0.35)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_xlabel(panel["xlabel"], labelpad=3)
    ax.set_title(
        panel["title"],
        loc="left",
        fontweight="bold",
        pad=15 if len(series_list) == 2 else 8,
    )
    ax.text(
        -0.13,
        1.065,
        panel["letter"],
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    if len(series_list) == 2:
        handles = [
            Line2D(
                [0],
                [0],
                color=series["color"],
                marker=series["marker"],
                linestyle="none",
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=series["label"],
            )
            for series in series_list
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(-0.025, 1.035),
            ncol=2,
            frameon=False,
            handletextpad=0.25,
            columnspacing=0.65,
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
    caption = """**Figure 2 | No-mechanism cooperation baselines across 11 models in RQ1.**
**a**, Cooperative-action rates in the prisoner's dilemma. **b**, Public-goods
contributions relative to the per-round endowment. **c**, Trust-game transfers
relative to the Sender's endowment (purple circles) and returns relative to the
amount received by the Receiver (coral squares). Points denote means and
horizontal whiskers denote the reported standard deviation across three
independent experimental runs. Hollow markers denote the Gemini 3.1 Pro
trust-game means for which the supplied summary table does not report a
standard deviation. Percentages use game-specific denominators and should not
be interpreted as a pooled psychological scale.
"""
    (output / "Figure2_RQ1_11model_mean_sd_caption.md").write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = read_and_validate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.20, 4.55),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for index, (ax, panel) in enumerate(zip(axes, PANELS)):
        plot_panel(ax, data, panel)
        ax.tick_params(labelbottom=True, labelleft=(index == 0))

    stem = "Figure2_RQ1_11model_mean_sd"
    fig.savefig(args.output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output / f"{stem}.svg", bbox_inches="tight")
    write_caption(args.output)
    plt.close(fig)
    print(f"Saved 11-model RQ1 figure files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
