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
NUM_AGENTS = 24                 # 代理数量
NUM_ROUNDS = 10                 # 轮次数量

# 设置结果目录路径
result_dir = r"d:\常用\agentsociety2\result_volunteer_dilemma_group_reputation\032202_qwen3-next-80b-a3b-instruct"

# 数据文件路径
data_file = os.path.join(result_dir, "data", "game_logs.json")

def extract_llm_name(directory_path):
    """
    从结果目录路径中提取LLM名称
    
    参数:
        directory_path (str): 结果目录路径
    
    返回:
        str: 提取的LLM名称，如"llama3-70b"
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

# ------- 数据加载函数 -------
def load_game_data_from_directory(data_directory):
    """
    从数据目录加载多个游戏日志文件
    
    参数:
        data_directory: 包含游戏日志文件的目录路径
    
    返回:
        tuple: (志愿者选择数据字典, 至少有一个志愿者数据字典)
    """
    try:
        # 查找所有游戏日志文件
        log_files = [f for f in os.listdir(data_directory) if f.endswith('.json') and f.startswith('game_log')]
        
        if not log_files:
            print(f"在 {data_directory} 中未找到游戏日志文件")
            return defaultdict(dict), defaultdict(dict)
        
        # 按轮次组织的志愿者数量字典 {轮次: {文件名: [志愿者数量]}}
        num_volunteers_by_round = defaultdict(dict)
        # 按轮次组织的是否有志愿者字典 {轮次: {文件名: [0或1]}}
        at_least_one_volunteer_by_round = defaultdict(dict)
        
        # 处理每个日志文件
        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                game_logs = json.load(f)
            
            print(f"成功加载游戏日志: {log_file}")
            print(f"总交互次数: {len(game_logs)}")
            
            # 处理每轮数据
            for log in game_logs:
                round_num = log["interaction"]
                num_volunteers = log["num_volunteers"]
                is_someone_volunteering = log["is_someone_volunteering"]
                
                # 存储志愿者数量
                if log_file not in num_volunteers_by_round[round_num]:
                    num_volunteers_by_round[round_num][log_file] = []
                num_volunteers_by_round[round_num][log_file].append(num_volunteers)
                
                # 存储是否有志愿者
                if log_file not in at_least_one_volunteer_by_round[round_num]:
                    at_least_one_volunteer_by_round[round_num][log_file] = []
                at_least_one_volunteer_by_round[round_num][log_file].append(1 if is_someone_volunteering else 0)
        
        return num_volunteers_by_round, at_least_one_volunteer_by_round
    except Exception as e:
        print(f"读取游戏日志文件时出错: {str(e)}")
        return defaultdict(dict), defaultdict(dict)

# ------- 可视化函数 -------
def plot_volunteer_count_trend_with_ci(num_volunteers_data, save_dir):
    """
    按轮次绘制志愿者人数趋势图，包含95%置信区间
    
    参数:
        num_volunteers_data: 字典，键为轮次，值为{文件名: [志愿者数量]}
        save_dir: 图表保存目录
    """
    if not num_volunteers_data:
        print("没有志愿者数量数据可用于绘制趋势图。")
        return
    
    rounds = sorted(num_volunteers_data.keys())
    avg_volunteers = []
    ci_lower = []
    ci_upper = []
    
    # 计算每轮的均值和置信区间
    for r in rounds:
        # 收集所有实验的志愿者数量
        all_counts = []
        for exp_data in num_volunteers_data[r].values():
            all_counts.extend(exp_data)
        
        # 计算均值和置信区间
        mean_val, lower, upper = calculate_confidence_interval(all_counts)
        avg_volunteers.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制志愿者人数折线
    plt.plot(rounds, avg_volunteers, marker="o", color="#66CCFF", linestyle='-', linewidth=2,
             label="Average Number of Volunteers")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#66CCFF", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.xlabel("Round Number")
    plt.ylabel("Number of Volunteers")
    plt.title(f"Number of Volunteers per Round\nLLM: {llm_name} (24 Agents)")
    plt.ylim(-1, NUM_AGENTS + 1)
    # 当轮数大于50时，只显示偶数轮次的标签
    if len(rounds) > 50:
        xticks = [r for r in rounds if r % 2 == 0]
    else:
        xticks = rounds
    plt.xticks(xticks, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"volunteer_count_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每轮志愿者人数趋势图表已保存至: {save_path}")


def plot_at_least_one_volunteer_frequency_with_ci(at_least_one_data, save_dir):
    """
    按轮次绘制至少有一个志愿者的频率图，包含95%置信区间
    
    参数:
        at_least_one_data: 字典，键为轮次，值为{文件名: 是否有志愿者的列表（0或1）}
        save_dir: 图表保存目录
    """
    if not at_least_one_data:
        print("没有至少有一个志愿者的数据可用于绘制图表。")
        return
    
    rounds = sorted(at_least_one_data.keys())
    avg_frequency = []
    ci_lower = []
    ci_upper = []
    
    # 计算每轮的均值和置信区间
    for r in rounds:
        # 收集所有实验的是否有志愿者数据
        all_statuses = []
        for exp_data in at_least_one_data[r].values():
            all_statuses.extend(exp_data)
        
        # 计算均值和置信区间
        mean_val, lower, upper = calculate_confidence_interval(all_statuses)
        avg_frequency.append(mean_val)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制至少有一个志愿者的频率折线
    plt.plot(rounds, avg_frequency, marker="o", color="#FFCC99", linestyle='-', linewidth=2,
             label="Average Frequency of At Least One Volunteer")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#FFCC99", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.xlabel("Round Number")
    plt.ylabel("Frequency (0 to 1)")
    plt.title(f"At Least One Volunteer per Round\nLLM: {llm_name} (24 Agents)")
    plt.ylim(-0.1, 1.1)
    # 当轮数大于50时，只显示偶数轮次的标签
    if len(rounds) > 50:
        xticks = [r for r in rounds if r % 2 == 0]
    else:
        xticks = rounds
    plt.xticks(xticks, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"at_least_one_volunteer_frequency_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每轮至少有一个志愿者频率图表已保存至: {save_path}")




# ------- 主函数 -------
def main():
    """
    主函数，执行可视化流程
    """
    # 数据目录路径
    data_dir = os.path.join(result_dir, "data")
    
    print(f"\n=== 志愿者博弈群体实验可视化工具 ===")
    print(f"结果目录: {result_dir}")
    print(f"数据目录: {data_dir}")
    print(f"LLM名称: {llm_name}")
    
    # 加载游戏数据
    print("\n正在加载游戏数据...")
    num_volunteers_by_round, at_least_one_volunteer_by_round = load_game_data_from_directory(data_dir)
    
    if not num_volunteers_by_round:
        print("没有找到游戏日志数据，程序退出。")
        return
    
    print(f"成功加载数据: {len(num_volunteers_by_round)} 轮")
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    plot_volunteer_count_trend_with_ci(num_volunteers_by_round, result_dir)
    plot_at_least_one_volunteer_frequency_with_ci(at_least_one_volunteer_by_round, result_dir)
    
    print(f"\n所有可视化图表生成完成！")
    print(f"生成图表保存在: {result_dir}")


if __name__ == "__main__":
    # 调用主函数
    main()