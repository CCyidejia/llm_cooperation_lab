#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import scipy.stats
import re

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ------- 全局配置 -------
# 设置必要的常量
MAX_EXTRACTION_PER_AGENT = 10  # 每个代理的最大提取量
INITIAL_POOL_RESOURCES = 600    # 初始资源池数量（群体实验中是600）
CONFIDENCE_LEVEL = 0.95         # 置信水平

# 设置结果目录路径（可配置）
# 使用原始字符串避免反斜杠转义问题
result_dir = r"d:\常用\agentsociety2\result_commons_tragedy_group\result_011403_qwen3-8b"

# 数据文件路径
data_file = os.path.join(result_dir, "data/game_logs.json")

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
    
    # 使用正则表达式匹配"result_"后面的部分作为LLM名称
    match = re.search(r'result_(.+)', last_dir)
    if match:
        return match.group(1)
    
    # 如果没有匹配到，使用整个目录名
    return last_dir


# 从结果目录中提取LLM名称
llm_name = extract_llm_name(result_dir)
# 兼容原代码的experiment_id变量
experiment_id = llm_name

# ------- 置信区间计算函数 -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    计算数据的95%置信区间
    
    参数:
        data: 数值列表
        confidence: 置信水平，默认为0.95
    
    返回:
        tuple: (均值, 下界, 上界)
    """
    if not data or len(data) == 0:
        return 0, 0, 0
    
    n = len(data)
    mean_val = np.mean(data)
    
    # 特殊情况处理
    if n < 2:
        # 样本量不足，返回均值和零误差
        return mean_val, mean_val, mean_val
    
    # 使用t检验计算置信区间
    std_err = scipy.stats.sem(data)  # 标准误差
    confidence_interval = scipy.stats.t.interval(
        confidence,
        df=n-1,  # 自由度
        loc=mean_val,
        scale=std_err
    )
    
    return mean_val, confidence_interval[0], confidence_interval[1]

# ------- 可视化函数 -------
def plot_overall_extraction_distribution(extractions, save_dir):
    """
    绘制总体提取量分布直方图
    
    参数:
        extractions: 所有代理的提取量列表
        save_dir: 图表保存目录
    """
    if not extractions:
        print("没有提取数据可用于绘制分布图。")
        return
    
    # 计算每个提取量的频率和置信区间
    plt.figure(figsize=(8, 5))
    bins = [i + 0.5 for i in range(MAX_EXTRACTION_PER_AGENT + 1)]
    n, bins, patches = plt.hist(extractions, bins=bins, rwidth=0.8, color="#A1C9F4", edgecolor='black')
    
    # 添加标题，包含LLM名称
    plt.title(f"Overall Extraction Amount Distribution\nLLM: {llm_name}")
    plt.xlabel("Extraction Amount (Units)")
    plt.ylabel("Frequency")
    plt.xticks(range(1, MAX_EXTRACTION_PER_AGENT + 1))
    plt.grid(axis='y', alpha=0.75)
    
    # 在每个柱子上方添加数值标签
    for i in range(len(n)):
        if n[i] > 0:
            plt.text(bins[i] + 0.5, n[i], str(int(n[i])), ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"overall_extraction_distribution_{llm_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"总体提取分布图表已保存至: {plot_path}")


def plot_average_extraction_by_round(round_extractions_data, save_dir):
    """
    按轮次绘制平均提取量图表
    
    参数:
        round_extractions_data: 字典，键为轮次，值为该轮次所有代理的提取量列表
        save_dir: 图表保存目录
    """
    if not round_extractions_data:
        print("没有轮次提取数据可用于绘制平均提取量图表。")
        return
    
    rounds = sorted(round_extractions_data.keys())
    average_extractions_per_round = []
    
    # 计算每轮平均提取量
    for r in rounds:
        extractions_in_round = round_extractions_data[r]
        mean_val = np.mean(extractions_in_round)
        average_extractions_per_round.append(mean_val)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制平均提取量折线
    plt.plot(rounds, average_extractions_per_round, marker="o", color="#FFB482", linestyle='-',
             label="Average Extraction")
    
    plt.xlabel("Round Number")
    plt.ylabel("Average Extraction Amount (Units)")
    plt.title(f"Average Extraction Amount per Round\nLLM: {llm_name}")
    plt.ylim(0, MAX_EXTRACTION_PER_AGENT + 1)
    plt.xticks(rounds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"average_extraction_by_round_{llm_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"每轮平均提取量图表已保存至: {save_path}")


def plot_pool_resources_over_time(pool_history, save_dir):
    """
    绘制公共资源池随轮次变化的图表
    
    参数:
        pool_history: 每轮资源池剩余量列表
        save_dir: 图表保存目录
    """
    if not pool_history:
        print("没有资源池历史数据可用于绘制图表。")
        return
    
    plt.figure(figsize=(10, 6))
    rounds_range = list(range(1, len(pool_history) + 1))
    
    # 绘制资源池变化曲线
    plt.plot(rounds_range, pool_history, marker="o", color="blue", linewidth=3, label="Pool Resources")
    
    plt.xlabel("Round Number")
    plt.ylabel("Remaining Pool Resources (Units)")
    plt.title(f"Public Resource Pool Remaining Over Rounds\nLLM: {llm_name}")
    plt.ylim(0, INITIAL_POOL_RESOURCES * 1.1)
    plt.xticks(rounds_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"pool_resources_over_rounds_{llm_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"资源池随轮次变化图表已保存至: {save_path}")


def plot_individual_agent_extraction(agent_extractions, save_dir):
    """
    绘制每个Agent的提取量随轮次变化的图表
    
    参数:
        agent_extractions: 字典，键为Agent ID，值为该Agent各轮次的提取量列表
        save_dir: 图表保存目录
    """
    if not agent_extractions:
        print("没有Agent提取数据可用于绘制图表。")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 获取轮次范围
    num_rounds = max(len(exts) for exts in agent_extractions.values())
    rounds_range = list(range(1, num_rounds + 1))
    
    # 为每个Agent绘制提取量曲线
    colors = plt.cm.rainbow(np.linspace(0, 1, len(agent_extractions)))
    for (agent_id, exts), color in zip(agent_extractions.items(), colors):
        plt.plot(rounds_range, exts, marker="o", alpha=0.7, label=f"{agent_id}", color=color)
    
    plt.xlabel("Round Number")
    plt.ylabel("Extraction Amount (Units)")
    plt.title(f"Individual Agent Extraction Over Rounds\nLLM: {llm_name} (24 Agents)")
    plt.ylim(0, MAX_EXTRACTION_PER_AGENT + 1)
    plt.xticks(rounds_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 优化图例显示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"individual_agent_extraction_{llm_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Agent个体提取量变化图表已保存至: {save_path}")

# ------- 数据加载函数 -------
def load_game_data(file_path):
    """
    从游戏日志文件加载数据
    
    参数:
        file_path: 游戏日志文件路径
    
    返回:
        tuple: (提取量列表, 按轮次组织的提取量字典, 资源池历史记录列表, 按Agent组织的提取量字典)
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            game_logs = json.load(f)
        
        extractions = []           # 所有提取量数据
        round_extractions = defaultdict(list)  # 按轮次组织的提取量
        pool_history = []          # 资源池历史记录
        agent_extractions = defaultdict(list)  # 按Agent组织的提取量
        
        # 处理轮次数据
        for round_data in game_logs:
            round_num = round_data["round"]
            pool_after = round_data["pool_after_round"]
            pool_history.append(pool_after)
            
            # 处理每个Agent的提取量
            for agent_name, extraction in round_data["extractions"].items():
                extractions.append(extraction)
                round_extractions[round_num].append(extraction)
                agent_extractions[agent_name].append(extraction)
        
        return extractions, round_extractions, pool_history, agent_extractions
    except Exception as e:
        print(f"读取游戏日志文件时出错: {file_path}，错误: {str(e)}")
        return [], defaultdict(list), [], defaultdict(list)

# ------- 主函数 -------
def main():
    """
    主函数，执行可视化流程
    """
    print(f"\n=== 公地悲剧群体实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"数据文件: {data_file}")
    print(f"LLM名称: {llm_name}")
    
    # 加载游戏数据
    print("\n正在加载游戏数据...")
    extractions, round_extractions, pool_history, agent_extractions = load_game_data(data_file)
    
    if not extractions:
        print("没有找到游戏日志数据，程序退出。")
        return
    
    print(f"成功加载数据: {len(extractions)} 个提取记录")
    print(f"总轮数: {len(pool_history)}")
    print(f"Agent数量: {len(agent_extractions)}")
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    plot_overall_extraction_distribution(extractions, result_dir)
    plot_average_extraction_by_round(round_extractions, result_dir)
    plot_pool_resources_over_time(pool_history, result_dir)
    plot_individual_agent_extraction(agent_extractions, result_dir)
    
    print("\n所有可视化图表生成完成！")
    print(f"LLM名称: {llm_name}")
    print(f"生成图表保存在: {result_dir}")


if __name__ == "__main__":
    # 调用主函数
    main()