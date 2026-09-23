#!/usr/bin/env python3
"""Create a labelled scenario with Llama 3 70B PGG round 30 estimated from 25-29.

Original experiment logs and confirmed-result figures are preserved. This is
a sensitivity illustration, not a correction established by observed data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_rq1_cooperation_trends as base


MODEL = "Llama 3 70B"
NOTE = (
    "ESTIMATED SCENARIO: Llama 3 70B, public-goods round 30 = each run's mean of rounds 25-29.\n"
    "Dashed segment / diamond: estimate, not observed data; no empirical CI at the estimated point."
)


def estimate_round30(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = runs.copy()
    result["observed_value_pct"] = result["value_pct"]
    result["value_origin"] = "observed"
    result["estimate_source_rounds"] = ""
    result["observed_n_observations"] = result["n_observations"]
    changes = []
    selected = result[(result["model"] == MODEL) & (result["game"] == "PGG")]
    for run_name, run in selected.groupby("run", sort=False):
        previous = run[run["round"].between(25, 29)]
        target = run[run["round"] == 30]
        if len(previous) != 5 or len(target) != 1 or previous["value_pct"].isna().any():
            raise ValueError(f"Expected five source rounds and one final round: {run_name}")
        index = target.index[0]
        estimate = float(previous["value_pct"].mean())
        changes.append({
            "model": MODEL, "game": "PGG", "run": run_name, "round": 30,
            "observed_value_pct": float(target.iloc[0]["value_pct"]),
            "estimated_value_pct": estimate, "source_rounds": "25-29",
            "method": "within-run arithmetic mean of rounds 25-29",
            "source_file": target.iloc[0]["source_file"],
        })
        result.loc[index, "value_pct"] = estimate
        result.loc[index, "value_origin"] = "estimated"
        result.loc[index, "estimate_source_rounds"] = "25-29"
        # This plotted point has no observed round-30 decisions behind it.
        result.loc[index, "n_observations"] = 0
    return result, pd.DataFrame(changes)


def draw(ax, data: pd.DataFrame, model: str, color, marker: str, offset: int) -> None:
    part = data[data["model"] == model].sort_values("round")
    estimated = part["value_origin"] == "estimated"
    base.draw_curve(ax, part.loc[~estimated], model, color, marker, marker_offset=offset)
    if estimated.any():
        end = part.loc[estimated].iloc[0]
        start = part[part["round"] == end["round"] - 1].iloc[0]
        ax.plot([start["round"], end["round"]], [start["mean_pct"], end["mean_pct"]],
                color=color, linestyle="--", linewidth=1.8)
        ax.scatter([end["round"]], [end["mean_pct"]], marker="D", s=35,
                   facecolor="white", edgecolor=color, linewidth=1.5, zorder=5)


def make_figures(summary: pd.DataFrame, models: list[str], output: Path) -> None:
    base.set_style()
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    styles = {model: (colors[i], markers[i]) for i, model in enumerate(models)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.15,
                        wspace=0.22, hspace=0.40)
    fig.suptitle("RQ1 | Cooperation over rounds — estimated round-30 scenario",
                 fontsize=14, fontweight="bold", y=0.985)
    handles = [Line2D([], [], color=styles[model][0], marker=styles[model][1],
                      markerfacecolor="white", linewidth=1.5, markersize=5, label=model)
               for model in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.955),
               ncol=5, frameon=False, fontsize=9)
    for index, (ax, panel) in enumerate(zip(axes.flat, base.PANELS)):
        game, outcome, title, ylabel, n_rounds = panel
        data = summary[(summary["game"] == game) & (summary["outcome"] == outcome)]
        for model_index, model in enumerate(models):
            draw(ax, data, model, *styles[model], offset=model_index % 5)
        base.style_axis(ax, n_rounds)
        ax.set_title(f"{'abcd'[index]}  {title}", loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
    fig.text(0.075, 0.035, NOTE + "\nOther points: observed run means and pointwise 95% t CIs.",
             fontsize=9, color="#555555", linespacing=1.6)
    base.save_figure(fig, output, "RQ1_10model_cooperation_trends_round30_estimated")

    data = summary[(summary["game"] == "PGG") & (summary["outcome"] == "contribution")]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.19,
                        wspace=0.18, hspace=0.45)
    fig.suptitle("RQ1 | Public-goods game — estimated round-30 scenario",
                 fontsize=15, fontweight="bold", y=0.975)
    for index, (ax, model) in enumerate(zip(axes.flat, models)):
        draw(ax, data, model, *styles[model], offset=0)
        base.style_axis(ax, 30)
        ax.set_title(model, color=styles[model][0], fontweight="bold", fontsize=10)
        if index % 5 == 0:
            ax.set_ylabel("Contribution / endowment (%)")
    fig.text(0.06, 0.045, NOTE, fontsize=9, color="#555555", linespacing=1.6)
    base.save_figure(fig, output, "RQ1_10model_PGG_contribution_by_model_round30_estimated")


def main() -> None:
    models = list(base.SOURCES)
    original, _ = base.load_runs(base.ROOT, models)
    base.validate_confirmed_results(original)
    estimated, changes = estimate_round30(original)
    summary = base.summarize(estimated)
    summary["value_origin"] = "observed"
    selected = ((summary["model"] == MODEL) & (summary["game"] == "PGG")
                & (summary["round"] == 30))
    summary.loc[selected, "value_origin"] = "estimated"
    # Across-run variability of preceding-round averages is not an empirical
    # confidence interval for the unobserved alternative round-30 behavior.
    summary.loc[selected, ["sd_pct", "ci95_lower_pct", "ci95_upper_pct"]] = np.nan
    summary["n_estimated_runs"] = 0
    summary.loc[selected, "n_estimated_runs"] = summary.loc[selected, "n_runs"]
    summary.loc[selected, "n_runs"] = 0
    output = base.ROOT / "RQ1_figure_output" / "time_trends_10models" / "round30_estimated"
    output.mkdir(parents=True, exist_ok=True)
    estimated.to_csv(output / "rq1_run_round_values_estimated.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "rq1_round_summary_estimated.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(output / "llama70b_pgg_round30_changes.csv", index=False, encoding="utf-8-sig")
    make_figures(summary, models, output)

    observed_mean = float(changes["observed_value_pct"].mean())
    estimated_mean = float(changes["estimated_value_pct"].mean())
    overall = estimated[(estimated["model"] == MODEL) & (estimated["game"] == "PGG")]
    overall_runs = overall.groupby("run")["value_pct"].mean()
    report = f"""# Llama 3 70B 公共物品博弈：末轮平稳情景（估算展示）

截图中出现末轮骤降的是 Llama 3 70B（橙色），不是 Llama 3 8B。
原始日志中的第 30 轮均值为 {observed_mean:.2f}%，三次实验分别为
{', '.join(f'{value:.2f}%' for value in changes['observed_value_pct'])}。
各日志的贡献总额与逐个玩家贡献求和相符；日志解释中有玩家说明因最后一轮而停止贡献。
目前未发现支持将这些记录认定为读取或计算错误的证据。

## 展示情景

仅将 Llama 3 70B 的 PGG 第 30 轮绘图值，替换为同一次实验第 25–29 轮合作率的平均值。
估算末轮均值为 {estimated_mean:.2f}%，三次实验的估算值分别为
{', '.join(f'{value:.2f}%' for value in changes['estimated_value_pct'])}。
这是人为设定的平稳情景，不是实测数据的已验证更正。其他模型、博弈和轮次均不改变。

图中虚线和空心菱形表示估算点。该点不显示实测置信区间；此方法没有估计替代情景的不确定性。
其他点保留原始逐轮均值及 95% t 置信区间。
`value_origin` 区分 observed 与 estimated，原始值保留在 `observed_value_pct` 列。
估算行的 `n_observations` 为 0；汇总中 `n_runs` 为 0、`n_estimated_runs` 为 3。

## 与已确认表格的关系

原始总体合作率为 47.39% ± 0.54%（三次实验均值 ± 样本标准差）。
替换末轮后的情景总体值为 {overall_runs.mean():.2f}% ± {overall_runs.std(ddof=1):.2f}%，
因此该情景不再复现附件中的原始总体值。此处的总体标准差仅描述三条调整后的曲线。
已确认表格、原始日志和上一层目录中的原始结果图均保留。

## 文件

- 两组带 `_round30_estimated` 后缀的图：总览图与 PGG 分模型图，均提供 PNG、PDF、SVG。
- `rq1_run_round_values_estimated.csv`：绘图数据及原始值、估算标记。
- `rq1_round_summary_estimated.csv`：逐轮汇总；估算点不报告实测 CI。
- `llama70b_pgg_round30_changes.csv`：三条替换记录和方法。

重新生成：`python plot_rq1_llama70b_round30_estimated.py`。
如需正式更正实验结果，应使用经核实的第 30 轮正确记录重新计算，而不是将本情景当作实测值。
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(f"Observed round 30: {observed_mean:.6f}% -> scenario estimate: {estimated_mean:.6f}%")
    print(f"Scenario overall mean +/- SD: {overall_runs.mean():.6f} +/- {overall_runs.std(ddof=1):.6f}%")
    print(f"Saved two estimated-scenario figures (PNG/PDF/SVG) and labelled data to: {output}")


if __name__ == "__main__":
    main()
