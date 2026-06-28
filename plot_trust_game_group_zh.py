#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信任博弈群组实验可视化工具
基于群组实验日志格式（120 次交互，24 个智能体 × 10 轮）
支持多实验日志并绘制 95% 置信区间
"""
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
import numpy as np
import re
from collections import defaultdict
from scipy import stats

# ------- 全局配置 -------
# 设置结果目录路径
result_dir = r"d:/常用/agentsociety2/result_trust_game_population_reward/040503_deepseek-v3"
data_dir = os.path.join(result_dir, "data")


# ------- 辅助函数 -------
def extract_llm_name(directory_path):
    """
    从结果目录路径中提取 LLM 名称
    """
    last_dir = os.path.basename(directory_path)
    match = re.search(r"[^_]+$", last_dir)
    if match:
        return match.group(0)
    return last_dir


llm_name = extract_llm_name(result_dir)


# ------- 数据加载函数 -------
def load_game_logs(data_directory):
    """
    从目录中加载所有实验日志文件
    """
    try:
        game_logs_dict = {}
        log_files = [f for f in os.listdir(data_directory) if f.endswith(".json") and f.startswith("game_log")]

        if not log_files:
            print(f"在目录中未找到博弈日志文件: {data_directory}")
            return game_logs_dict

        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, "r", encoding="utf-8") as f:
                game_logs = json.load(f)
            game_logs_dict[log_file] = game_logs
            print(f"成功加载博弈日志: {log_file}")
            print(f"交互总数: {len(game_logs)}")

        return game_logs_dict
    except Exception as e:
        print(f"加载日志文件失败: {e}")
        return {}


# ------- 轮次分配函数 -------
def assign_rounds(game_logs, agents_per_round=24):
    """
    按时间戳分配轮次
    """
    sorted_logs = sorted(game_logs, key=lambda x: x["timestamp"])
    interactions_per_round = agents_per_round // 2

    round_data = defaultdict(list)
    for i, log in enumerate(sorted_logs):
        round_num = i // interactions_per_round + 1
        round_data[round_num].append(log)

    print(f"轮次分配完成: {len(round_data)} 轮")
    for rnd, logs in round_data.items():
        print(f"第 {rnd} 轮: {len(logs)} 条交互")

    return round_data


# ------- 数据处理函数 -------
def process_game_data(game_logs_dict):
    """
    处理多实验日志，提取发送金额与返还金额
    """
    if not game_logs_dict:
        print("没有可处理的博弈日志数据。")
        return {}, {}

    all_sent_amounts_by_experiment = defaultdict(dict)
    all_returned_amounts_by_experiment = defaultdict(dict)

    for exp_name, game_logs in game_logs_dict.items():
        round_data = assign_rounds(game_logs)
        for round_num, logs in round_data.items():
            sent_amounts = []
            returned_amounts = []

            for log in logs:
                sent_amount = log["detail"]["sent_amount"]
                sent_amounts.append(sent_amount)

                returned_amount = log["detail"]["returned_amount"]
                returned_amounts.append(returned_amount)

            all_sent_amounts_by_experiment[round_num][exp_name] = sent_amounts
            all_returned_amounts_by_experiment[round_num][exp_name] = returned_amounts

    return all_sent_amounts_by_experiment, all_returned_amounts_by_experiment


# ------- 置信区间计算 -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    计算 95% 置信区间
    """
    if not data:
        return 0, 0, 0

    n = len(data)
    mean_val = np.mean(data)

    if n == 1:
        return mean_val, mean_val, mean_val

    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean_val, scale=std_err)

    return mean_val, ci[0], ci[1]


# ------- 可视化函数 -------
def plot_average_investment_trend(sent_amounts_data, save_dir):
    """
    绘制发送方每轮平均发送金额趋势（含95%置信区间）
    """
    if not sent_amounts_data:
        print("没有可用于绘制趋势图的发送金额数据。")
        return

    rounds = sorted(sent_amounts_data.keys())
    average_sent_amounts_per_round = []
    ci_lower_per_round = []
    ci_upper_per_round = []

    for r in rounds:
        all_amounts = []
        for exp_data in sent_amounts_data[r].values():
            all_amounts.extend(exp_data)

        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        average_sent_amounts_per_round.append(mean_val)
        ci_lower_per_round.append(lower)
        ci_upper_per_round.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(
        rounds,
        average_sent_amounts_per_round,
        marker="o",
        linestyle="-",
        color="#FFCC99",
        label="发送方发送金额均值",
    )
    plt.fill_between(
        rounds,
        ci_lower_per_round,
        ci_upper_per_round,
        color="#FFCC99",
        alpha=0.3,
        label="95%置信区间",
    )

    plt.title(f"发送方发送的平均金额\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("平均发送金额（金币）", fontproperties=zh_label)

    max_ticks = 20
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step], fontsize=FONT_SIZE_TICK)
    else:
        plt.xticks(rounds, fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)

    plt.ylim(-0.5, 10.5)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"average_investment_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"发送方平均发送金额趋势图（带95%置信区间）已保存至: {save_path}")


def plot_average_returned_amount_trend(returned_amounts_data, save_dir):
    """
    绘制返还方每轮平均返还金额趋势（含95%置信区间）
    """
    if not returned_amounts_data:
        print("没有可用于绘制趋势图的返还金额数据。")
        return

    rounds = sorted(returned_amounts_data.keys())
    average_returned_amounts_per_round = []
    ci_lower_per_round = []
    ci_upper_per_round = []

    for r in rounds:
        all_amounts = []
        for exp_data in returned_amounts_data[r].values():
            filtered_amounts = [amount for amount in exp_data if not np.isnan(amount) and not np.isinf(amount)]
            all_amounts.extend(filtered_amounts)

        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        average_returned_amounts_per_round.append(mean_val)
        ci_lower_per_round.append(lower)
        ci_upper_per_round.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    plt.plot(
        rounds,
        average_returned_amounts_per_round,
        marker="o",
        linestyle="-",
        color="#A1C9F4",
        label="返还方返还金额均值",
    )
    plt.fill_between(
        rounds,
        ci_lower_per_round,
        ci_upper_per_round,
        color="#A1C9F4",
        alpha=0.3,
        label="95%置信区间",
    )

    plt.title(f"返还方返还的平均金额\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("平均返还金额（金币）", fontproperties=zh_label)

    max_ticks = 20
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step], fontsize=FONT_SIZE_TICK)
    else:
        plt.xticks(rounds, fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)

    plt.ylim(-1, 30)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"average_returned_amount_trend_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"返还方平均返还金额趋势图（带95%置信区间）已保存至: {save_path}")


# ------- 主函数 -------
def main():
    """
    主函数，执行可视化流程
    """
    configure_matplotlib_fonts()
    print("\n=== 信任博弈群组实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"数据目录: {data_dir}")
    print(f"模型名称: {llm_name}")

    print("\n正在加载博弈日志...")
    game_logs_dict = load_game_logs(data_dir)

    if not game_logs_dict:
        print("未找到博弈日志文件，程序退出。")
        return

    print("\n正在处理博弈数据...")
    all_sent_amounts, all_returned_amounts = process_game_data(game_logs_dict)

    os.makedirs(result_dir, exist_ok=True)

    print("\n正在生成可视化图表（95%置信区间）...")
    plot_average_investment_trend(all_sent_amounts, result_dir)
    plot_average_returned_amount_trend(all_returned_amounts, result_dir)

    print("\n所有图表已成功生成并保存！")


if __name__ == "__main__":
    main()
