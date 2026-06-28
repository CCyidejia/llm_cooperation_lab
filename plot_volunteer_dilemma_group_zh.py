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
NUM_AGENTS = 24
NUM_ROUNDS = 10

result_dir = r"d:\常用\agentsociety\agentsociety\packages\agentsociety2\result_volunteer_dilemma_group_punishment\040501_deepseek-r1"
data_file = os.path.join(result_dir, "data", "game_logs.json")


def extract_llm_name(directory_path):
    last_dir = os.path.basename(directory_path)
    match = re.search(r"\d+_(.+)", last_dir)
    if match:
        return match.group(1)
    return last_dir


llm_name = extract_llm_name(result_dir)
experiment_id = llm_name


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


def odd_round_xticks(rounds):
    """
    横轴刻度仅显示奇数轮：1, 3, 5, …, 29（仅包含数据中实际存在的轮次）。
    若无奇数轮则退回全部 rounds。
    """
    odd = [r for r in rounds if r % 2 == 1]
    return odd if odd else rounds


def load_game_data_from_directory(data_directory):
    try:
        log_files = [f for f in os.listdir(data_directory) if f.endswith(".json") and f.startswith("game_log")]

        if not log_files:
            print(f"在 {data_directory} 中未找到游戏日志文件")
            return defaultdict(dict), defaultdict(dict)

        num_volunteers_by_round = defaultdict(dict)
        at_least_one_volunteer_by_round = defaultdict(dict)

        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, "r", encoding="utf-8") as f:
                game_logs = json.load(f)

            print(f"成功加载游戏日志: {log_file}")
            print(f"总交互次数: {len(game_logs)}")

            for log in game_logs:
                round_num = log["interaction"]
                num_volunteers = log["num_volunteers"]
                is_someone_volunteering = log["is_someone_volunteering"]

                if log_file not in num_volunteers_by_round[round_num]:
                    num_volunteers_by_round[round_num][log_file] = []
                num_volunteers_by_round[round_num][log_file].append(num_volunteers)

                if log_file not in at_least_one_volunteer_by_round[round_num]:
                    at_least_one_volunteer_by_round[round_num][log_file] = []
                at_least_one_volunteer_by_round[round_num][log_file].append(1 if is_someone_volunteering else 0)

        return num_volunteers_by_round, at_least_one_volunteer_by_round
    except Exception as e:
        print(f"读取游戏日志文件时出错: {str(e)}")
        return defaultdict(dict), defaultdict(dict)


def plot_volunteer_count_trend_with_ci(num_volunteers_data, save_dir):
    if not num_volunteers_data:
        print("没有志愿者数量数据可用于绘制趋势图。")
        return

    rounds = sorted(num_volunteers_data.keys())
    avg_volunteers, ci_lower, ci_upper = [], [], []

    for r in rounds:
        all_counts = []
        for exp_data in num_volunteers_data[r].values():
            all_counts.extend(exp_data)
        mean_val, lower, upper = calculate_confidence_interval(all_counts)
        avg_volunteers.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(rounds, avg_volunteers, marker="o", color="#66CCFF", linestyle="-", linewidth=2, label="平均志愿者人数")
    plt.fill_between(rounds, ci_lower, ci_upper, color="#66CCFF", alpha=0.3, label="95%置信区间")

    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("志愿者人数", fontproperties=zh_label)
    plt.title(f"每轮平均志愿者数量\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(-1, NUM_AGENTS + 1)

    plt.xticks(odd_round_xticks(rounds), fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"volunteer_count_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"每轮志愿者人数趋势图表已保存至: {save_path}")


def plot_at_least_one_volunteer_frequency_with_ci(at_least_one_data, save_dir):
    if not at_least_one_data:
        print("没有至少有一个志愿者的数据可用于绘制图表。")
        return

    rounds = sorted(at_least_one_data.keys())
    avg_frequency, ci_lower, ci_upper = [], [], []

    for r in rounds:
        all_statuses = []
        for exp_data in at_least_one_data[r].values():
            all_statuses.extend(exp_data)
        mean_val, lower, upper = calculate_confidence_interval(all_statuses)
        avg_frequency.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(rounds, avg_frequency, marker="o", color="#FFCC99", linestyle="-", linewidth=2, label="至少有一个志愿者的平均频率")
    plt.fill_between(rounds, ci_lower, ci_upper, color="#FFCC99", alpha=0.3, label="95%置信区间")

    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("频率（0 到 1）", fontproperties=zh_label)
    plt.title(f"每轮至少有一个志愿者的频率\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(-0.1, 1.1)

    plt.xticks(odd_round_xticks(rounds), fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"at_least_one_volunteer_frequency_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"每轮至少有一个志愿者频率图表已保存至: {save_path}")


def main():
    data_dir = os.path.join(result_dir, "data")
    configure_matplotlib_fonts()

    print("\n=== 志愿者博弈群体实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"数据目录: {data_dir}")
    print(f"LLM名称: {llm_name}")

    print("\n正在加载游戏数据...")
    num_volunteers_by_round, at_least_one_volunteer_by_round = load_game_data_from_directory(data_dir)

    if not num_volunteers_by_round:
        print("没有找到游戏日志数据，程序退出。")
        return

    print(f"成功加载数据: {len(num_volunteers_by_round)} 轮")

    print("\n正在生成可视化图表...")
    plot_volunteer_count_trend_with_ci(num_volunteers_by_round, result_dir)
    plot_at_least_one_volunteer_frequency_with_ci(at_least_one_volunteer_by_round, result_dir)

    print("\n所有可视化图表生成完成！")
    print(f"生成图表保存在: {result_dir}")


if __name__ == "__main__":
    main()
