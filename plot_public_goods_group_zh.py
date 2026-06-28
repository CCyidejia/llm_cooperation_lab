#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from plot_zh_style import (
    FIG_SIZE,
    FONT_SIZE_TITLE,
    FONT_SIZE_TICK,
    configure_matplotlib_fonts,
    chinese_label_font,
    chinese_legend_font,
    savefig_kw,
)

import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re
from scipy import stats
# ------- 全局配置 -------
MAX_CONTRIBUTION = 20
INITIAL_ENDOWMENT = 20
NUM_AGENTS = 24

result_dir = r"d:\常用\agentsociety2\result_public_goods_group_punishment\result_032801_deepseek-r1"
data_dir = result_dir


def extract_llm_name(directory_path):
    last_dir = os.path.basename(directory_path)
    match = re.search(r"\d+_(.+)", last_dir)
    if match:
        return match.group(1)
    return last_dir


llm_name = extract_llm_name(result_dir)
experiment_id = llm_name


def get_max_rounds_to_plot():
    """
    按结果目录名返回需要截断的最大轮次。
    仅对指定实验目录生效，其他情况返回 None（不截断）。
    """
    target_dir_name = "result_50轮_3次_012701_llama3-70b"
    current_dir_name = os.path.basename(os.path.normpath(result_dir))
    if current_dir_name == target_dir_name:
        return 30
    return None


def resolve_plot_options(num_log_files):
    """
    当仅有一个 game_log*.json 时：只绘制前 10 轮，且不绘制/不显示 95% 置信区间。
    否则：按目录规则截断轮次（若有），并显示置信区间。
    """
    if num_log_files == 1:
        return 10, False
    return get_max_rounds_to_plot(), True


def odd_round_xticks(rounds):
    """
    横轴刻度仅显示奇数轮：1, 3, 5, …, 29（仅包含数据中实际存在的轮次）。
    若无奇数轮则退回全部 rounds。
    """
    odd = [r for r in rounds if r % 2 == 1]
    return odd if odd else rounds


def load_game_logs(data_directory):
    try:
        game_logs_dict = {}
        log_files = [f for f in os.listdir(data_directory) if f.endswith(".json") and f.startswith("game_log")]

        if not log_files:
            print(f"在 {data_directory} 中未找到游戏日志文件")
            return game_logs_dict

        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, "r", encoding="utf-8") as f:
                game_logs = json.load(f)
            game_logs_dict[log_file] = game_logs
            print(f"成功加载游戏日志: {log_file}")
            print(f"总交互次数: {len(game_logs)}")

        return game_logs_dict
    except Exception as e:
        print(f"加载日志文件时出错: {e}")
        return {}


def process_game_data(game_logs_dict):
    if not game_logs_dict:
        print("没有可用的游戏日志数据进行处理。")
        return {}, {}, {}

    total_contributions_by_round = defaultdict(dict)
    avg_contributions_by_round = defaultdict(dict)
    non_zero_contributors_by_round = defaultdict(dict)

    for exp_name, game_logs in game_logs_dict.items():
        for log in game_logs:
            round_num = log["interaction"]
            total_contrib = log["total_contribution"]
            avg_contrib = total_contrib / log["num_agents"]

            non_zero_count = 0
            if "agent_contributions" in log:
                for contrib in log["agent_contributions"].values():
                    if contrib > 0:
                        non_zero_count += 1

            if exp_name not in total_contributions_by_round[round_num]:
                total_contributions_by_round[round_num][exp_name] = []
            total_contributions_by_round[round_num][exp_name].append(total_contrib)

            if exp_name not in avg_contributions_by_round[round_num]:
                avg_contributions_by_round[round_num][exp_name] = []
            avg_contributions_by_round[round_num][exp_name].append(avg_contrib)

            if exp_name not in non_zero_contributors_by_round[round_num]:
                non_zero_contributors_by_round[round_num][exp_name] = []
            non_zero_contributors_by_round[round_num][exp_name].append(non_zero_count)

    return total_contributions_by_round, avg_contributions_by_round, non_zero_contributors_by_round


def calculate_confidence_interval(data, confidence=0.95):
    if not data:
        return 0, 0, 0

    n = len(data)
    mean_val = np.mean(data)
    if n == 1:
        return mean_val, mean_val, mean_val

    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean_val, scale=std_err)
    return mean_val, ci[0], ci[1]


def plot_total_contribution_trend(total_contributions_data, save_dir, max_rounds=None, show_ci=True):
    if not total_contributions_data:
        print("没有总贡献量数据可用于绘制趋势图。")
        return

    rounds = sorted(total_contributions_data.keys())
    if max_rounds is not None:
        rounds = [r for r in rounds if r <= max_rounds]
        if not rounds:
            print(f"目录 {os.path.basename(os.path.normpath(result_dir))} 在前 {max_rounds} 轮无数据，跳过绘图。")
            return
    avg_total_contributions, ci_lower, ci_upper = [], [], []

    for r in rounds:
        all_amounts = []
        for exp_data in total_contributions_data[r].values():
            all_amounts.extend(exp_data)
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        avg_total_contributions.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(rounds, avg_total_contributions, marker="o", color="#FFB482", linestyle="-", linewidth=2, label="平均总贡献金额")
    if show_ci:
        plt.fill_between(rounds, ci_lower, ci_upper, color="#FFB482", alpha=0.3, label="95%置信区间")

    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("总贡献金额（金币）", fontproperties=zh_label)
    plt.title(f"总贡献金额随轮次变化\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(0, NUM_AGENTS * MAX_CONTRIBUTION)
    plt.xticks(odd_round_xticks(rounds), fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"total_contribution_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"总贡献量随轮次变化图表已保存至: {save_path}")


def plot_average_contribution_trend(avg_contributions_data, save_dir, max_rounds=None, show_ci=True):
    if not avg_contributions_data:
        print("没有平均贡献量数据可用于绘制趋势图。")
        return

    rounds = sorted(avg_contributions_data.keys())
    if max_rounds is not None:
        rounds = [r for r in rounds if r <= max_rounds]
        if not rounds:
            print(f"目录 {os.path.basename(os.path.normpath(result_dir))} 在前 {max_rounds} 轮无数据，跳过绘图。")
            return
    avg_avg_contributions, ci_lower, ci_upper = [], [], []

    for r in rounds:
        all_amounts = []
        for exp_data in avg_contributions_data[r].values():
            all_amounts.extend(exp_data)
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        avg_avg_contributions.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(rounds, avg_avg_contributions, marker="o", color="#A1C9F4", linestyle="-", linewidth=2, label="每个智能体平均贡献金额")
    if show_ci:
        plt.fill_between(rounds, ci_lower, ci_upper, color="#A1C9F4", alpha=0.3, label="95%置信区间")

    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("每个智能体平均贡献金额（金币）", fontproperties=zh_label)
    # 第一行按要求固定，第二行模型名保持 Times New Roman
    plt.title(f"每个智能体平均贡献金额\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(0, MAX_CONTRIBUTION)
    plt.xticks(odd_round_xticks(rounds), fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"average_contribution_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"每轮平均贡献量图表已保存至: {save_path}")


def plot_non_zero_contributors(non_zero_contributors_data, save_dir, max_rounds=None, show_ci=True):
    if not non_zero_contributors_data:
        print("没有贡献金额大于0的人数数据可用于绘制图表。")
        return

    rounds = sorted(non_zero_contributors_data.keys())
    if max_rounds is not None:
        rounds = [r for r in rounds if r <= max_rounds]
        if not rounds:
            print(f"目录 {os.path.basename(os.path.normpath(result_dir))} 在前 {max_rounds} 轮无数据，跳过绘图。")
            return
    avg_non_zero_contributors, ci_lower, ci_upper = [], [], []

    for r in rounds:
        all_counts = []
        for exp_data in non_zero_contributors_data[r].values():
            all_counts.extend(exp_data)
        mean_val, lower, upper = calculate_confidence_interval(all_counts)
        avg_non_zero_contributors.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(rounds, avg_non_zero_contributors, marker="o", color="#95E1D3", linestyle="-", linewidth=2, label="平均有贡献智能体人数")
    if show_ci:
        plt.fill_between(rounds, ci_lower, ci_upper, color="#95E1D3", alpha=0.3, label="95%置信区间")

    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("贡献金额大于0的智能体人数", fontproperties=zh_label)
    plt.title(f"贡献金额大于0的智能体人数随轮次变化\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(0, NUM_AGENTS + 1)
    plt.xticks(odd_round_xticks(rounds), fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"non_zero_contributors_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"每轮贡献金额大于0的人数图表已保存至: {save_path}")


def main():
    configure_matplotlib_fonts()
    print("\n=== 公共物品博弈群体实验可视化工具（中文） ===")
    print(f"结果目录: {result_dir}")
    print(f"数据目录: {data_dir}")
    print(f"LLM名称: {llm_name}")

    print("\n正在加载游戏日志...")
    game_logs_dict = load_game_logs(data_dir)
    if not game_logs_dict:
        print("没有找到游戏日志文件，程序退出。")
        return

    print("\n正在处理游戏数据...")
    total_contributions_data, avg_contributions_data, non_zero_contributors_data = process_game_data(game_logs_dict)

    num_log_files = len(game_logs_dict)
    max_rounds, show_ci = resolve_plot_options(num_log_files)
    if num_log_files == 1:
        print(f"检测到仅 1 个 game_log JSON：仅绘制前 {max_rounds} 轮，图例不显示 95% 置信区间。")

    os.makedirs(result_dir, exist_ok=True)

    print("\n正在生成可视化图表...")
    plot_total_contribution_trend(total_contributions_data, result_dir, max_rounds=max_rounds, show_ci=show_ci)
    plot_average_contribution_trend(avg_contributions_data, result_dir, max_rounds=max_rounds, show_ci=show_ci)
    plot_non_zero_contributors(non_zero_contributors_data, result_dir, max_rounds=max_rounds, show_ci=show_ci)

    print("\n所有可视化图表生成完成！")
    print(f"生成图表保存在: {result_dir}")


if __name__ == "__main__":
    main()
