#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re
from scipy import stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ------- 全局配置 -------
# 设置必要的常量
MAX_CONTRIBUTION = 20           # 每个代理的最大贡献量
INITIAL_ENDOWMENT = 20          # 初始禀赋
NUM_AGENTS = 24                 # 代理数量

# 设置结果目录路径
result_dir = r"d:\常用\agentsociety2\result_public_goods_group_reputation\result_032103_deepseek-v3"

# 数据目录路径
data_dir = result_dir

def extract_llm_name(directory_path):
    """
    从结果目录路径中提取LLM名称
    
    参数:
        directory_path (str): 结果目录路径
    
    返回:
        str: 提取的LLM名称，如"llama3-8b"
    """
    # 从路径中提取最后一个目录名
    last_dir = os.path.basename(directory_path)
    
    # 使用正则表达式匹配"数字_"后面的部分作为LLM名称
    match = re.search(r'\d+_(.+)', last_dir)
    if match:
        return match.group(1)
    
    # 如果没有匹配到，使用整个目录名
    return last_dir


# 从结果目录中提取LLM名称
llm_name = extract_llm_name(result_dir)
# 兼容原代码的experiment_id变量
experiment_id = llm_name

# ------- 数据加载函数 -------
def load_game_logs(data_directory):
    """
    从指定目录加载所有游戏日志文件
    
    参数:
        data_directory (str): 包含游戏日志文件的目录路径
    
    返回:
        dict: 键为日志文件名，值为日志数据列表
    """
    try:
        game_logs_dict = {}
        # 查找所有游戏日志文件
        log_files = [f for f in os.listdir(data_directory) if f.endswith('.json') and f.startswith('game_log')]
        
        if not log_files:
            print(f"在 {data_directory} 中未找到游戏日志文件")
            return game_logs_dict
        
        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                game_logs = json.load(f)
            game_logs_dict[log_file] = game_logs
            print(f"成功加载游戏日志: {log_file}")
            print(f"总交互次数: {len(game_logs)}")
        
        return game_logs_dict
    except Exception as e:
        print(f"加载日志文件时出错: {e}")
        return {}

# ------- 数据处理函数 -------
def process_game_data(game_logs_dict):
    """
    处理多个实验的游戏日志数据
    
    参数:
        game_logs_dict (dict): 键为日志文件名，值为日志数据列表
    
    返回:
        tuple: (total_contributions_by_round, avg_contributions_by_round, non_zero_contributors_by_round)
    """
    if not game_logs_dict:
        print("没有可用的游戏日志数据进行处理。")
        return {}, {}, {}
    
    total_contributions_by_round = defaultdict(dict)
    avg_contributions_by_round = defaultdict(dict)
    non_zero_contributors_by_round = defaultdict(dict)
    
    # 处理每个实验
    for exp_name, game_logs in game_logs_dict.items():
        # 处理每轮数据
        for log in game_logs:
            round_num = log["interaction"]
            total_contrib = log["total_contribution"]
            avg_contrib = total_contrib / log["num_agents"]
            
            # 计算贡献金额大于0的人数
            non_zero_count = 0
            if "agent_contributions" in log:
                for contrib in log["agent_contributions"].values():
                    if contrib > 0:
                        non_zero_count += 1
            
            # 存储总贡献量
            if exp_name not in total_contributions_by_round[round_num]:
                total_contributions_by_round[round_num][exp_name] = []
            total_contributions_by_round[round_num][exp_name].append(total_contrib)
            
            # 存储平均贡献量
            if exp_name not in avg_contributions_by_round[round_num]:
                avg_contributions_by_round[round_num][exp_name] = []
            avg_contributions_by_round[round_num][exp_name].append(avg_contrib)
            
            # 存储贡献金额大于0的人数
            if exp_name not in non_zero_contributors_by_round[round_num]:
                non_zero_contributors_by_round[round_num][exp_name] = []
            non_zero_contributors_by_round[round_num][exp_name].append(non_zero_count)
    
    return total_contributions_by_round, avg_contributions_by_round, non_zero_contributors_by_round

# ------- 置信区间计算函数 -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    计算数据的95%置信区间
    
    参数:
        data (list): 数值数据列表
        confidence (float): 置信水平，默认为0.95
    
    返回:
        tuple: (均值, 下限, 上限)
    """
    if not data:
        return 0, 0, 0
    
    n = len(data)
    mean_val = np.mean(data)
    
    # 计算置信区间
    if n == 1:
        # 只有一个数据点，无置信区间
        return mean_val, mean_val, mean_val
    
    std_err = stats.sem(data)  # 标准误
    ci = stats.t.interval(confidence, df=n-1, loc=mean_val, scale=std_err)
    
    return mean_val, ci[0], ci[1]

# ------- 可视化函数 -------
def plot_total_contribution_trend(total_contributions_data, save_dir):
    """
    绘制总贡献量随轮次变化的图表，包含95%置信区间
    
    参数:
        total_contributions_data: 字典，键为轮次，值为{实验名称: 贡献量列表}
        save_dir: 图表保存目录
    """
    if not total_contributions_data:
        print("没有总贡献量数据可用于绘制趋势图。")
        return
    
    rounds = sorted(total_contributions_data.keys())
    avg_total_contributions = []
    ci_lower = []
    ci_upper = []
    
    # 计算每轮的均值和置信区间
    for r in rounds:
        # 收集所有实验的总贡献量
        all_amounts = []
        for exp_data in total_contributions_data[r].values():
            all_amounts.extend(exp_data)
        
        # 计算均值和置信区间
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        avg_total_contributions.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制总贡献量折线
    plt.plot(rounds, avg_total_contributions, marker="o", color="#FFB482", linestyle='-', linewidth=2,
             label="Average Total Contribution")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#FFB482", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.xlabel("Round Number")
    plt.ylabel("Total Contribution (Coins)")
    plt.title(f"Total Contribution to Public Fund Over Rounds\nLLM: {llm_name}")
    plt.ylim(0, NUM_AGENTS * MAX_CONTRIBUTION)
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"total_contribution_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"总贡献量随轮次变化图表已保存至: {save_path}")


def plot_average_contribution_trend(avg_contributions_data, save_dir):
    """
    绘制每轮平均贡献量图表，包含95%置信区间
    
    参数:
        avg_contributions_data: 字典，键为轮次，值为{实验名称: 平均贡献量列表}
        save_dir: 图表保存目录
    """
    if not avg_contributions_data:
        print("没有平均贡献量数据可用于绘制趋势图。")
        return
    
    rounds = sorted(avg_contributions_data.keys())
    avg_avg_contributions = []
    ci_lower = []
    ci_upper = []
    
    # 计算每轮的均值和置信区间
    for r in rounds:
        # 收集所有实验的平均贡献量
        all_amounts = []
        for exp_data in avg_contributions_data[r].values():
            all_amounts.extend(exp_data)
        
        # 计算均值和置信区间
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        avg_avg_contributions.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制平均贡献量折线
    plt.plot(rounds, avg_avg_contributions, marker="o", color="#A1C9F4", linestyle='-', linewidth=2,
             label="Average Contribution per Agent")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#A1C9F4", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.xlabel("Round Number")
    plt.ylabel("Average Contribution per Agent (Coins)")
    plt.title(f"Average Contribution per Agent Over Rounds\nLLM: {llm_name}")
    plt.ylim(0, MAX_CONTRIBUTION)
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"average_contribution_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每轮平均贡献量图表已保存至: {save_path}")


def plot_individual_agent_contributions(agent_contributions, save_dir):
    """
    绘制每个Agent的贡献量随轮次变化的图表
    
    参数:
        agent_contributions: 字典，键为Agent ID，值为该Agent各轮次的贡献量列表
        save_dir: 图表保存目录
    """
    if not agent_contributions:
        print("没有Agent贡献数据可用于绘制图表。")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 获取轮次范围
    rounds_range = list(range(1, len(list(agent_contributions.values())[0]) + 1))
    
    # 为每个Agent绘制贡献量曲线
    colors = plt.cm.rainbow(np.linspace(0, 1, len(agent_contributions)))
    for (agent_id, contribs), color in zip(agent_contributions.items(), colors):
        plt.plot(rounds_range, contribs, marker="o", alpha=0.7, label=f"{agent_id}", color=color)
    
    plt.xlabel("Round Number")
    plt.ylabel("Contribution Amount (Coins)")
    plt.title(f"Individual Agent Contributions Over Rounds\nLLM: {llm_name} (24 Agents)")
    plt.ylim(0, MAX_CONTRIBUTION + 1)
    plt.xticks(rounds_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 优化图例显示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"individual_agent_contributions_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每个Agent的贡献量随轮次变化图表已保存至: {save_path}")


def plot_individual_agent_payoffs(agent_payoffs, save_dir):
    """
    绘制每个Agent的收益随轮次变化的图表
    
    参数:
        agent_payoffs: 字典，键为Agent ID，值为该Agent各轮次的收益列表
        save_dir: 图表保存目录
    """
    if not agent_payoffs:
        print("没有Agent收益数据可用于绘制图表。")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 获取轮次范围
    rounds_range = list(range(1, len(list(agent_payoffs.values())[0]) + 1))
    
    # 为每个Agent绘制收益曲线
    colors = plt.cm.rainbow(np.linspace(0, 1, len(agent_payoffs)))
    for (agent_id, payoffs), color in zip(agent_payoffs.items(), colors):
        plt.plot(rounds_range, payoffs, marker="o", alpha=0.7, label=f"{agent_id}", color=color)
    
    plt.xlabel("Round Number")
    plt.ylabel("Payoff (Coins)")
    plt.title(f"Individual Agent Payoffs Over Rounds\nLLM: {llm_name} (24 Agents)")
    plt.ylim(0, INITIAL_ENDOWMENT * 2)  # 设置y轴范围
    plt.xticks(rounds_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 优化图例显示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"individual_agent_payoffs_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每个Agent的收益随轮次变化图表已保存至: {save_path}")


def plot_non_zero_contributors(non_zero_contributors_data, save_dir):
    """
    绘制每轮贡献金额大于0的智能体人数，包含95%置信区间
    
    参数:
        non_zero_contributors_data: 字典，键为轮次，值为{实验名称: 贡献金额大于0的人数列表}
        save_dir: 图表保存目录
    """
    if not non_zero_contributors_data:
        print("没有贡献金额大于0的人数数据可用于绘制图表。")
        return
    
    rounds = sorted(non_zero_contributors_data.keys())
    avg_non_zero_contributors = []
    ci_lower = []
    ci_upper = []
    
    # 计算每轮的均值和置信区间
    for r in rounds:
        # 收集所有实验的贡献金额大于0的人数
        all_counts = []
        for exp_data in non_zero_contributors_data[r].values():
            all_counts.extend(exp_data)
        
        # 计算均值和置信区间
        mean_val, lower, upper = calculate_confidence_interval(all_counts)
        avg_non_zero_contributors.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制贡献金额大于0的人数折线
    plt.plot(rounds, avg_non_zero_contributors, marker="o", color="#95E1D3", linestyle='-', linewidth=2,
             label="Average Number of Contributors")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#95E1D3", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.xlabel("Round Number")
    plt.ylabel("Number of Agents with Non-Zero Contributions")
    plt.title(f"Number of Agents with Non-Zero Contributions Over Rounds\nLLM: {llm_name}")
    plt.ylim(0, NUM_AGENTS + 1)
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"non_zero_contributors_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每轮贡献金额大于0的人数图表已保存至: {save_path}")

# ------- 主函数 -------
def main():
    """
    主函数，执行可视化流程
    """
    print(f"\n=== 公共物品博弈群体实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"数据目录: {data_dir}")
    print(f"LLM名称: {llm_name}")
    
    # 加载游戏日志
    print("\n正在加载游戏日志...")
    game_logs_dict = load_game_logs(data_dir)
    
    if not game_logs_dict:
        print("没有找到游戏日志文件，程序退出。")
        return
    
    # 处理游戏数据
    print("\n正在处理游戏数据...")
    total_contributions_data, avg_contributions_data, non_zero_contributors_data = process_game_data(game_logs_dict)
    
    # 确保保存目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成带有置信区间的可视化图表
    print("\n正在生成带有95%置信区间的可视化图表...")
    plot_total_contribution_trend(total_contributions_data, result_dir)
    plot_average_contribution_trend(avg_contributions_data, result_dir)
    plot_non_zero_contributors(non_zero_contributors_data, result_dir)
    
    print(f"\n所有可视化图表生成完成！")
    print(f"生成图表保存在: {result_dir}")


if __name__ == "__main__":
    # 调用主函数
    main()