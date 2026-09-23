#!/usr/bin/env python3
"""Plot RQ1 round-level cooperation and pointwise 95% CIs across independent runs.

Run ``python plot_rq1_cooperation_trends.py`` for the ten confirmed models,
including DeepSeek V4 Pro and excluding Qwen3-Next 80B-A3B. Whole-run means
and sample SDs must match the user's confirmed table before figures are saved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
OLD = "RQ1_old_result"
NEW = "RQ1_result/result"

# Explicit selections avoid mixing other treatments, reruns, or backup logs.
# The three paths per model are PD, PGG, and TG, respectively.
SOURCES = {
    "Llama 3 8B": (
        f"{OLD}/result_prisoners_dilemma_group/*_012404_llama3-8b",
        f"{OLD}/result_public_goods_group/llama3-8b-three",
        f"{OLD}/result_trust_game_population/30lun_3ci_012401_llama3-8b",
    ),
    "Llama 3 70B": (
        f"{OLD}/result_prisoners_dilemma_group/*_012403_llama3-70b",
        f"{OLD}/result_public_goods_group/llama3-70b-three",
        f"{OLD}/result_trust_game_population/30lun_3ci_012402_llama3-70b",
    ),
    "Qwen 2.5 7B": (
        f"{OLD}/result_prisoners_dilemma_group/10lun_3ci_020501_Qwen2.5-7B-Instruct",
        f"{OLD}/result_public_goods_group/qwen2.5-7b-instruct-three",
        f"{OLD}/result_trust_game_population/020601_30lun_3ci_Qwen2.5-7B-Instruct",
    ),
    "GLM-5.1": (
        f"{NEW}/prisoners_dilemma/090201_glm-5.1",
        f"{NEW}/public_goods/result_090801_glm-5.1",
        f"{NEW}/trust_game/082904_glm-5.1_final",
    ),
    "GPT-5.4": (
        f"{NEW}/prisoners_dilemma/082004_gpt-5.4",
        f"{NEW}/public_goods/result_070108_gpt-5.4",
        f"{NEW}/trust_game/083001_gpt-5.4",
    ),
    "Gemini 3.1 Pro": (
        f"{NEW}/prisoners_dilemma/082003_gemini-3.1-pro-preview",
        f"{NEW}/public_goods/result_082001_gemini-3.1-pro-preview",
        f"{NEW}/trust_game/091601_gemini-3.1-pro-preview",
    ),
    "Claude Opus 4.6": (
        f"{NEW}/prisoners_dilemma/070101_claude-opus-4-6",
        f"{NEW}/public_goods/result_082001_claude-opus-4-6",
        f"{NEW}/trust_game/083102_claude-opus-4-6",
    ),
    "Kimi K2.6": (
        f"{NEW}/prisoners_dilemma/072201_kimi-k2.6",
        f"{NEW}/public_goods/result_072301_kimi-k2.6",
        f"{NEW}/trust_game/072201_kimi-k2.6",
    ),
    "Grok 4.3": (
        f"{NEW}/prisoners_dilemma/072701_grok-4.3",
        f"{NEW}/public_goods/result_072202_grok-4.3",
        f"{NEW}/trust_game/072601_grok-4.3",
    ),
    "DeepSeek V4 Pro": (
        f"{NEW}/prisoners_dilemma/091501_deepseek-v4-pro",
        f"{NEW}/public_goods/result_063004_deepseek-v4-pro",
        f"{NEW}/trust_game/091601_deepseek-v4-pro",
    ),
}

PANELS = (
    ("PD", "cooperation", "Prisoner's dilemma", "Cooperative actions (%)", 10),
    ("PGG", "contribution", "Public-goods game", "Contribution / endowment (%)", 30),
    ("TG", "sender", "Trust game: Sender", "Transfer / endowment (%)", 30),
    ("TG", "receiver", "Trust game: Receiver", "Return / amount received (%)", 30),
)

# Transcribed from the user's confirmed PPT table, in PANELS order:
# PD, PGG, TG Sender, TG Receiver. Each pair is (mean %, sample SD %).
CONFIRMED_RESULTS = {
    "Llama 3 8B": ((2.22, 1.20), (60.18, 3.66), (55.65, 0.34), (57.99, 2.32)),
    "Llama 3 70B": ((0.14, 0.24), (47.39, 0.54), (54.32, 0.62), (36.51, 0.47)),
    "Qwen 2.5 7B": ((0.00, 0.00), (3.75, 2.97), (46.19, 1.69), (37.50, 2.97)),
    "GLM-5.1": ((0.00, 0.00), (0.00, 0.00), (30.21, 0.97), (9.85, 0.57)),
    "GPT-5.4": ((0.00, 0.00), (0.00, 0.00), (10.60, 0.25), (0.00, 0.00)),
    "Gemini 3.1 Pro": ((3.19, 0.87), (0.00, 0.00), (61.06, 2.09), (40.82, 1.78)),
    "Claude Opus 4.6": ((100.00, 0.00), (0.00, 0.00), (32.29, 2.29), (31.92, 0.49)),
    "Kimi K2.6": ((0.00, 0.00), (0.00, 0.00), (23.45, 0.91), (20.69, 1.47)),
    "Grok 4.3": ((0.00, 0.00), (0.00, 0.00), (7.41, 0.88), (0.74, 0.83)),
    "DeepSeek V4 Pro": ((0.14, 0.24), (0.00, 0.00), (40.52, 2.05), (1.74, 1.19)),
}


def round_scores(logs: list[dict], game: str) -> list[dict]:
    """Return one estimate per run/round/outcome; never pool runs' decisions."""
    n_rounds = 10 if game == "PD" else 30
    per_round = 1 if game == "PGG" else 12
    if len(logs) != n_rounds * per_round:
        raise ValueError(f"{game}: expected {n_rounds * per_round} records, got {len(logs)}")
    # Use the experiment interaction sequence, not concurrent completion times.
    logs = sorted(logs, key=lambda record: record["interaction"])
    if [record["interaction"] for record in logs] != list(range(1, len(logs) + 1)):
        raise ValueError(f"{game}: duplicate, missing, or invalid interaction IDs")

    rows = []
    for round_num in range(1, n_rounds + 1):
        current = logs[(round_num - 1) * per_round : round_num * per_round]
        if game == "PD":
            choices = [str(row["detail"][key]).strip().lower()
                       for row in current for key in ("choice1", "choice2")]
            if any(choice not in {"yes", "no"} for choice in choices):
                raise ValueError("PD: an action is neither Yes nor No")
            outcomes = {"cooperation": [float(choice == "yes") for choice in choices]}
        elif game == "PGG":
            record = current[0]
            contributions = list(record["agent_contributions"].values())
            if len(contributions) != record["num_agents"] or len(contributions) != 24:
                raise ValueError("PGG: expected contributions from 24 agents")
            if not np.isclose(sum(contributions), record["total_contribution"]):
                raise ValueError("PGG: individual contributions do not match the total")
            outcomes = {"contribution": [amount / 20 for amount in contributions]}
        else:
            outcomes = {"sender": [], "receiver": []}
            for record in current:
                detail = record["detail"]
                sent = float(detail["sent_amount"])
                returned = float(detail["returned_amount"])
                received = float(detail.get("trustee_received", 3 * sent))
                outcomes["sender"].append(sent / 10)
                if received > 0:
                    outcomes["receiver"].append(returned / received)
                elif received < 0 or returned != 0:
                    raise ValueError("TG: invalid received/returned amounts")

        for outcome, scores in outcomes.items():
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
                raise ValueError(f"{game}/{outcome}: scores must be finite and within [0, 1]")
            rows.append({
                "round": round_num,
                "outcome": outcome,
                "value_pct": float(np.mean(scores) * 100) if scores else math.nan,
                "n_observations": len(scores),
            })
    return rows


def load_runs(root: Path, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, manifest = [], []
    for model in models:
        for game, pattern in zip(("PD", "PGG", "TG"), SOURCES[model]):
            matches = [path for path in root.glob(pattern) if path.is_dir()]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one directory for {pattern}: {matches}")
            directory = matches[0]
            data_dir = directory / "data" if (directory / "data").is_dir() else directory
            files = sorted(data_dir.glob("game_log*.json"))  # No recursive backup search.
            if len(files) != 3:
                raise ValueError(f"Expected 3 independent runs in {data_dir}, found {len(files)}")
            for path in files:
                logs = json.loads(path.read_text(encoding="utf-8"))
                try:
                    scores = round_scores(logs, game)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid log {path}: {exc}") from exc
                source = path.relative_to(root).as_posix()
                rows.extend({"model": model, "game": game, "run": path.stem,
                             "source_file": source, **score} for score in scores)
                manifest.append({"model": model, "game": game, "run": path.stem,
                                 "records": len(logs), "source_file": source})
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in runs.groupby(["model", "game", "outcome", "round"], sort=False):
        values = group["value_pct"].dropna().to_numpy()
        n = len(values)
        mean = float(np.mean(values)) if n else math.nan
        sd = float(np.std(values, ddof=1)) if n >= 2 else math.nan
        # Multiplication handles zero variance without scipy's zero-scale NaNs.
        margin = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n >= 2 else math.nan
        rows.append(dict(zip(("model", "game", "outcome", "round"), keys)) | {
            "mean_pct": mean, "sd_pct": sd,
            "ci95_lower_pct": mean - margin, "ci95_upper_pct": mean + margin,
            "n_runs": n, "n_runs_total": len(group),
            "n_observations": int(group["n_observations"].sum()),
        })
    return pd.DataFrame(rows)


def validate_confirmed_results(runs: pd.DataFrame) -> pd.DataFrame:
    """Match whole-experiment means and sample SDs to the confirmed PPT table."""
    if set(runs["model"]) != set(CONFIRMED_RESULTS):
        raise ValueError("The loaded models do not match the ten confirmed models")
    rows = []
    for model, expected in CONFIRMED_RESULTS.items():
        for panel, (expected_mean, expected_sd) in zip(PANELS, expected):
            game, outcome = panel[:2]
            selected = runs[(runs["model"] == model) & (runs["game"] == game)
                            & (runs["outcome"] == outcome)]
            estimates = []
            for _, run in selected.groupby("run", sort=False):
                valid = run["n_observations"] > 0
                if not valid.any():
                    raise ValueError(f"No valid observations for {model}/{game}/{outcome}")
                # Recover each run's overall mean from all its valid decisions.
                # Receiver rounds can contain different numbers of valid pairs.
                estimates.append(float(np.average(
                    run.loc[valid, "value_pct"], weights=run.loc[valid, "n_observations"]
                )))
            if len(estimates) != 3:
                raise ValueError(f"Expected three runs for {model}/{game}/{outcome}")
            mean, sd = float(np.mean(estimates)), float(np.std(estimates, ddof=1))
            rows.append({
                "model": model, "game": game, "outcome": outcome,
                "n_runs": len(estimates),
                "confirmed_mean_pct": expected_mean, "confirmed_sd_pct": expected_sd,
                "calculated_mean_pct": mean, "calculated_sd_pct": sd,
                "mean_difference_pp": mean - expected_mean,
                "sd_difference_pp": sd - expected_sd,
                # The attachment reports two decimal places (half-unit tolerance).
                "matches_confirmed_table": bool(np.allclose(
                    [mean, sd], [expected_mean, expected_sd], rtol=0, atol=0.005000001
                )),
            })
    audit = pd.DataFrame(rows)
    failures = audit[~audit["matches_confirmed_table"]]
    if not failures.empty:
        raise ValueError("Raw logs do not reproduce the confirmed table:\n"
                         + failures.to_string(index=False))
    return audit


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#777777", "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def style_axis(ax: plt.Axes, n_rounds: int) -> None:
    ax.set_xlim(0.7, n_rounds + 0.3)
    ax.set_ylim(-2, 102)
    ax.set_xticks([1, 5, 10] if n_rounds == 10 else [1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("Round")


def draw_curve(ax: plt.Axes, data: pd.DataFrame, model: str, color,
               marker: str, marker_offset: int = 0, alpha: float = 0.14) -> None:
    part = data.loc[data["model"] == model].sort_values("round")
    x = part["round"].to_numpy()
    # Raw t bounds remain in the CSV; display their intersection with [0,100].
    ax.fill_between(x, np.clip(part["ci95_lower_pct"].to_numpy(), 0, 100),
                    np.clip(part["ci95_upper_pct"].to_numpy(), 0, 100),
                    color=color, alpha=alpha, linewidth=0)
    ax.plot(x, part["mean_pct"].to_numpy(), color=color, linewidth=1.5,
            marker=marker, markersize=3.6, markevery=(marker_offset, 5),
            markerfacecolor="white", markeredgewidth=0.9, label=model)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(summary: pd.DataFrame, models: list[str], output: Path) -> None:
    set_style()
    colors = list(plt.get_cmap("tab10").colors) + [(0.12, 0.16, 0.20)]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
    styles = {model: (colors[i], markers[i]) for i, model in enumerate(models)}
    stem = f"RQ1_{len(models)}model"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.12,
                        wspace=0.22, hspace=0.40)
    fig.suptitle(f"RQ1 | Cooperation over rounds across {len(models)} models",
                 fontsize=15, fontweight="bold", y=0.985)
    handles = [Line2D([], [], color=styles[model][0], marker=styles[model][1],
                      markerfacecolor="white", linewidth=1.5, markersize=5, label=model)
               for model in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.955),
               ncol=5 if len(models) == 10 else 4, frameon=False, fontsize=9)

    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        game, outcome, title, ylabel, n_rounds = panel
        data = summary[(summary["game"] == game) & (summary["outcome"] == outcome)]
        for model_index, model in enumerate(models):
            draw_curve(ax, data, model, *styles[model], marker_offset=model_index % 5)
        style_axis(ax, n_rounds)
        ax.set_title(f"{'abcd'[index]}  {title}", loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
    fig.text(0.075, 0.035,
             "Lines: mean across runs; shading: pointwise 95% t CI (up to 3 runs). "
             "Coincident lines overlap; see model panels.\n"
             "Receiver: conditional on a positive amount received. "
             "No valid observation: gap; fewer than 2 valid runs: no CI.",
             fontsize=9, color="#555555", linespacing=1.6)
    save_figure(fig, output, f"{stem}_cooperation_trends")

    for game, outcome, title, ylabel, n_rounds in PANELS:
        data = summary[(summary["game"] == game) & (summary["outcome"] == outcome)]
        nrows = math.ceil(len(models) / 5)
        fig, axes = plt.subplots(nrows, 5, figsize=(15, 2.8 * nrows + 1),
                                 sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.17,
                            wspace=0.18, hspace=0.45)
        fig.suptitle(f"RQ1 | {title}", fontsize=15, fontweight="bold", y=0.975)
        fig.text(0.5, 0.91, "Mean and pointwise 95% confidence interval across independent runs",
                 ha="center", color="#555555", fontsize=10)
        for index, (ax, model) in enumerate(zip(axes.flat, models)):
            draw_curve(ax, data, model, *styles[model], alpha=0.24)
            style_axis(ax, n_rounds)
            ax.set_title(model, color=styles[model][0], fontweight="bold", fontsize=10)
            if index % 5 == 0:
                ax.set_ylabel(ylabel)
        for ax in list(axes.flat)[len(models):]:
            ax.set_visible(False)
        fig.text(0.06, 0.05,
                 "Three independent runs per model. Zero between-run variance gives a zero-width CI. "
                 "Bands displayed within 0-100%.\n"
                 "Receiver ratios exclude zero receipts; gaps indicate undefined means; "
                 "fewer than two valid runs gives no CI.",
                 color="#555555", fontsize=9, linespacing=1.6)
        save_figure(fig, output, f"{stem}_{game}_{outcome}_by_model")


def write_readme(output: Path, models: list[str], summary: pd.DataFrame) -> None:
    counts = summary["n_runs"].value_counts().sort_index().to_dict()
    text = f"""# RQ1 逐轮合作率及 95% 置信区间

模型（{len(models)} 个）：{', '.join(models)}。

## 图与数据

- `RQ1_{len(models)}model_cooperation_trends.*`：四个指标的总览图。
- `RQ1_{len(models)}model_*_by_model.*`：每个指标按模型分面，便于查看重合的零值曲线。
- 每张图提供 PNG（300 dpi）、PDF 和 SVG。
- `rq1_round_summary.csv`：逐轮均值、样本标准差、未经截断的 CI 上下界及有效实验数。
- `rq1_run_round_values.csv`：每次实验、每轮的比例与有效行为数；缺失值为空。
- `rq1_source_manifest.csv`：本次绘图使用的全部原始日志路径。
- `rq1_confirmed_table_check.csv`：附件中 40 项均值及样本标准差与原始日志的核对结果。

## 计算口径

横轴是实验轮次。PD 为 10 轮，每轮 12 对；PGG 和 TG 为 30 轮。
每个模型、每种博弈读取 3 份独立实验日志，仅读取指定目录或其 `data` 子目录，
不读取备份目录。使用日志的 `interaction` 编号排序并校验完整性。

- PD：一轮中两名玩家的所有 `Yes` 行为数 / 24。
- PGG：一轮中 24 名玩家贡献额的均值 / 20。
- TG Sender：一轮中 12 次发送金额的均值 / 10。
- TG Receiver：先计算各次 `returned_amount / trustee_received`，
  再在每轮、每次实验内取均值；收到金额为 0 的记录不参与计算。
  旧日志缺少 `trustee_received` 时使用 `3 * sent_amount`。

折线为同一模型同一轮的有效实验均值，阴影为点态 95% Student t 均值置信区间：
`mean ± t(0.975, n-1) * SD(ddof=1) / sqrt(n)`。
样本单位为独立实验；同一实验中的玩家/配对不作为独立重复。
这些是逐轮区间，不是覆盖整条曲线的同时置信带。
全部实验相同时置信区间宽度为 0；有效实验少于 2 次时不画区间；
没有有效实验时不画均值，不填零、不插值。各模型/指标/轮次的有效实验数分布：{counts}。
图中 CI 与比例取值范围 [0,100] 相交，CSV 保留未截断的上下界。

模型数据来自 `RQ1_result`、`RQ1_old_result`。
精确来源见 manifest。图中数值均由当前原始日志计算，不根据汇总表反推轨迹。
Receiver 每轮只对有效实验等权平均，因此对这条曲线再作简单时间平均，
不一定等于将每次实验全部有效配对汇总后得到的总体均值。

## 与用户确认附件的核对

本次采用附件中的 10 个模型：包含 DeepSeek V4 Pro，不包含 Qwen3-Next 80B-A3B。
附件中的 40 项结果（10 个模型，每个模型 4 个指标）均由对应的原始日志复算核对，
所有均值和样本标准差均与附件保留两位小数的精度一致。
绘图前脚本会自动执行该检查；若日志与确认值不符，将报错并停止生成图片。

附件中的“均值 ± 标准差”先计算每次实验全部有效行为的总体合作率，
再对 3 次实验计算均值和样本标准差（ddof=1）。
对照值保存在脚本 `CONFIRMED_RESULTS` 中；核对结果见 `rq1_confirmed_table_check.csv`。
逐轮图的阴影仍为 95% 置信区间，与附件报告的总体标准差含义不同。
例如，Gemini 的 TG 发送/返还总体均值为 61.06% / 40.82%，
DeepSeek 的 PD 总体均值为 0.14%，TG 发送/返还总体均值为 40.52% / 1.74%。
本次使用附件确认的数值匹配日志，不使用旧的 11 模型汇总表作为绘图依据。

## 重新生成

在项目根目录执行：

```powershell
python plot_rq1_cooperation_trends.py
```

默认绘制附件确认的 10 个模型，并覆盖 `time_trends_10models` 中的同名输出。
`--output` 可指定输出目录，`--root` 可指定原始项目目录。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    models = list(SOURCES)
    output = args.output or root / "RQ1_figure_output" / f"time_trends_{len(models)}models"
    runs, manifest = load_runs(root, models)
    audit = validate_confirmed_results(runs)
    summary = summarize(runs)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("rq1_run_round_values", runs), ("rq1_round_summary", summary),
                        ("rq1_source_manifest", manifest),
                        ("rq1_confirmed_table_check", audit)):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    make_figures(summary, models, output)
    write_readme(output, models, summary)
    print(f"Models: {len(models)}; log files: {len(manifest)}; round estimates: {len(summary)}")
    print(f"Confirmed table: {len(audit)}/{len(audit)} mean/SD pairs match at reported precision")
    print(f"Valid run counts per estimate: {summary['n_runs'].value_counts().sort_index().to_dict()}")
    print(f"Saved five figures (PNG/PDF/SVG), four CSVs and README to: {output.resolve()}")


if __name__ == "__main__":
    main()
