#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
囚徒困境群组实验可视化工具
基于群组实验日志格式（多个智能体 × 多轮）
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
result_dir = r"d:/常用/agentsociety/agentsociety/packages/agentsociety2/result_prisoners_dilemma_group/35轮_3次_012701_deepseek-r1"
data_dir = os.path.join(result_dir, "data")
# 仅当结果目录名为下列之一时，绘图只使用前 N 轮（路径最后一级目录名需完全一致）
RESULT_DIR_MAX_ROUNDS = {
    "35轮_3次_012701_deepseek-r1": 10,
}


def max_rounds_for_current_result_dir():
    """若当前 result_dir 在配置表中，返回最大轮次（含）；否则不截断。"""
    base = os.path.basename(os.path.normpath(result_dir))
    return RESULT_DIR_MAX_ROUNDS.get(base)

# ------- 辅助函数 -------
def extract_llm_name(directory_path):
    """
    从结果目录路径中提取 LLM 名称

    参数:
        directory_path (str): 结果目录路径

    返回:
        str: 提取出的 LLM 名称
    """
    # 获取路径最后一级目录名
    last_dir = os.path.basename(directory_path)

    # 用正则匹配并提取模型名
    match = re.search(r"[^_]+$", last_dir)
    if match:
        return match.group(0)

    # 若未匹配，则返回完整目录名
    return last_dir


# LLM 名称（从结果目录动态提取）
llm_name = extract_llm_name(result_dir)


# ------- 数据加载函数 -------
def load_game_logs(data_dir):
    """
    从多个 JSON 文件加载群组实验博弈日志

    参数:
        data_dir (str): 日志文件所在目录

    返回:
        list: 所有文件的博弈日志列表，每个元素为 (file_name, logs)
    """
    experiments = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json") and f.startswith("game_log")]

    if not json_files:
        print(f"在目录中未找到博弈日志文件: {data_dir}")
        return []

    print(f"找到 {len(json_files)} 个博弈日志文件:")
    for file_name in sorted(json_files):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            experiments.append((file_name, logs))
            print(f"成功加载: {file_name}（{len(logs)} 条交互）")
        except Exception as e:
            print(f"加载失败 {file_name}: {e}")

    print(f"总共加载实验数: {len(experiments)}")
    return experiments


# ------- 轮次分配函数 -------
def assign_rounds(game_logs, agents_per_round=24):
    """
    基于时间戳将交互分配到各轮次

    参数:
        game_logs (list): 博弈日志数据
        agents_per_round (int): 每轮智能体数量

    返回:
        dict: 键为轮次，值为该轮交互列表
    """
    # 按时间戳排序
    sorted_logs = sorted(game_logs, key=lambda x: x["timestamp"])

    # 每轮交互数 = 智能体数的一半
    interactions_per_round = agents_per_round // 2

    # 分配轮次
    round_data = defaultdict(list)
    for i, log in enumerate(sorted_logs):
        round_num = i // interactions_per_round + 1  # 轮次从1开始
        round_data[round_num].append(log)

    print(f"轮次分配完成: {len(round_data)} 轮")
    for rnd, logs in round_data.items():
        print(f"第 {rnd} 轮: {len(logs)} 条交互")

    return round_data


# ------- 数据处理函数 -------
def process_game_data(round_data):
    """
    处理博弈日志，提取所需信息

    参数:
        round_data (dict): 键为轮次，值为该轮交互列表

    返回:
        tuple: (total_actions, round_action_counts, all_payoffs, agent_actions, agent_payoffs)
    """
    if not round_data:
        print("没有可处理的博弈日志数据。")
        return {}, {}, [], {}, {}

    # 统计所有游戏中所有智能体的动作（总计）
    total_actions = []

    # 统计每轮动作，用于绘制每轮趋势图
    round_action_counts = defaultdict(list)  # key: round_num, value: list of actions

    # 统计收益信息
    all_payoffs = []  # 所有轮次收益

    # 统计每个智能体的动作与收益
    agent_actions = defaultdict(list)  # key: agent_name, value: list of actions
    agent_payoffs = defaultdict(list)  # key: agent_name, value: list of payoffs

    for round_num, logs in round_data.items():
        round_actions = []

        for log in logs:
            detail = log["detail"]

            # 提取双方数据
            agent1_name = detail["agent1_name"]
            action1 = detail["choice1"]
            payoff1 = detail["payoff1"]

            agent2_name = detail["agent2_name"]
            action2 = detail["choice2"]
            payoff2 = detail["payoff2"]

            # 统一动作格式
            normalized_action1 = "Yes" if action1.lower() == "yes" else "No"
            normalized_action2 = "Yes" if action2.lower() == "yes" else "No"

            total_actions.extend([normalized_action1, normalized_action2])
            round_actions.extend([normalized_action1, normalized_action2])

            agent_actions[agent1_name].append(normalized_action1)
            agent_actions[agent2_name].append(normalized_action2)

            agent_payoffs[agent1_name].append(payoff1)
            agent_payoffs[agent2_name].append(payoff2)

            all_payoffs.extend([payoff1, payoff2])

        round_action_counts[round_num] = round_actions

    return total_actions, round_action_counts, all_payoffs, agent_actions, agent_payoffs


# ------- 置信区间计算 -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    计算一组数据的 95% 置信区间

    参数:
        data: 数值列表
        confidence: 置信水平（默认 0.95）

    返回:
        tuple: (mean, lower_bound, upper_bound)
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
# 绘制每轮“合作”智能体数量折线图（带95%置信区间）
def plot_yes_count_by_round(round_yes_counts, save_dir):
    """
    绘制每轮选择合作（Yes）的智能体数量，并显示 95% 置信区间

    参数:
        round_yes_counts: dict，键为轮次，值为该轮在各实验中的 Yes 数量列表
        save_dir: 图片保存目录
    """
    if not round_yes_counts:
        print("没有可用于绘图的轮次动作数据。")
        return

    rounds = sorted(round_yes_counts.keys())
    max_r = max_rounds_for_current_result_dir()
    if max_r is not None:
        rounds = [r for r in rounds if r <= max_r]
        if not rounds:
            print(f"当前结果目录仅绘制前 {max_r} 轮：无可用数据，跳过绘图。")
            return

    # 计算每轮均值与95%置信区间
    mean_yes_counts = []
    ci_lower = []
    ci_upper = []

    for r in rounds:
        yes_counts = round_yes_counts[r]
        mean, lower, upper = calculate_confidence_interval(yes_counts)
        mean_yes_counts.append(mean)
        ci_lower.append(lower)
        ci_upper.append(upper)

    plt.figure(figsize=FIG_SIZE)
    zh_label = chinese_label_font()

    # 绘制均值折线图
    plt.plot(rounds, mean_yes_counts, marker="o", label="合作智能体平均数量", color="#FFB482")

    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#FFB482", alpha=0.3, label="95%置信区间")

    # 统一放大图中所有文字
    plt.xlabel("轮次", fontproperties=zh_label)
    plt.ylabel("合作智能体数量", fontproperties=zh_label)
    plt.title(f"每轮选择合作的智能体数量\n{llm_name}", fontsize=FONT_SIZE_TITLE)
    plt.ylim(0, 24)

    # 自适应横坐标刻度，避免数字重叠
    max_ticks = 20  # 最大显示刻度数
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step], fontsize=FONT_SIZE_TICK)
    else:
        plt.xticks(rounds, fontsize=FONT_SIZE_TICK)
    plt.yticks(range(0, 25, 2), fontsize=FONT_SIZE_TICK)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(prop=chinese_legend_font())
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"yes_count_by_round_with_ci_{llm_name}_zh.png")
    plt.savefig(save_path, **savefig_kw())
    plt.close()
    print(f"每轮选择合作的智能体数量折线图（带95%置信区间）已保存: {save_path}")


# ------- 主函数 -------
def main():
    """
    执行可视化流程的主函数
    """
    print("\n=== 囚徒困境群组实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"模型名称: {llm_name}")
    configure_matplotlib_fonts()

    # 加载实验数据
    print("\n正在加载实验数据...")
    experiments = load_game_logs(data_dir)

    if not experiments:
        print("未找到博弈日志文件，程序退出。")
        return

    # 处理每个实验并收集轮次数据
    print("\n正在处理实验数据...")
    experiment_round_data = []

    for file_name, logs in experiments:
        print(f"\n处理实验文件: {file_name}")
        round_data = assign_rounds(logs)
        _, round_action_counts, _, _, _ = process_game_data(round_data)
        experiment_round_data.append((file_name, round_action_counts))

    # 汇总所有实验中每轮的 Yes 数量
    round_yes_counts = defaultdict(list)
    for _, round_action_counts in experiment_round_data:
        for round_num, actions in round_action_counts.items():
            yes_count = actions.count("Yes") if actions else 0
            round_yes_counts[round_num].append(yes_count)

    # 确保保存目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 生成图表
    print("\n正在生成可视化图表...")
    plot_yes_count_by_round(round_yes_counts, result_dir)
    print("\n图表已全部生成并保存完成！")


if __name__ == "__main__":
    main()
