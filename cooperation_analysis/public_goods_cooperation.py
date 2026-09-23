#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共物品博弈群体实验合作率计算工具

该脚本用于计算公共物品博弈群体实验的合作率，
支持分析多个实验文件，计算每轮的合作率以及特定轮次的平均合作率。

与非群体实验计算规则一致：
- 合作率 = 贡献金额 / 最大贡献金额（20）
- 每轮合作率 = 该轮所有玩家合作分数的平均值
- 总体合作率 = (实验1合作率 + 实验2合作率 + 实验3合作率) / 3
- 标准差 = 三次实验总体合作率的样本标准差（ddof=1）
"""

import json
import numpy as np
import os

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据目录路径
# 这里使用指定的llama3-8b模型结果目录
data_dir = r"D:\常用\agentsociety2\result_public_goods_group\result_090801_glm-5.1"

# 定义实验配置
NUM_ROUNDS_PER_GAME = 30  # 每个实验的轮数
NUM_AGENTS = 24  # 每轮的agent数量
MAX_CONTRIBUTION = 20  # 每个agent的最大贡献金额

# 尝试获取数据目录中的所有game_logs文件
try:
    file_names = [f for f in os.listdir(data_dir) if f.startswith('game_logs') and f.endswith('.json')]
    file_names = sorted(file_names)  # 排序以确保顺序一致
    if not file_names:
        print(f"警告: 在目录 '{data_dir}' 中未找到任何game_logs文件。")
except FileNotFoundError:
    print(f"错误: 目录 '{data_dir}' 未找到。请检查路径是否正确。")
    file_names = []

# 用于存储每轮合作率的字典
# 键为轮次(1-30)，值为该轮次在所有实验中的合作率列表
round_cooperation_rates = {i: [] for i in range(1, NUM_ROUNDS_PER_GAME + 1)}

# 存储每次实验的总体合作率
experiment_cooperation_rates = []

# 遍历所有文件
for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            game_logs = json.load(f)

        print(f"处理文件: {file_name}")
        print(f"总轮数: {len(game_logs)}")
        
        # 验证数据完整性
        if len(game_logs) != NUM_ROUNDS_PER_GAME:
            print(f"警告: 文件 {file_name} 的轮数({len(game_logs)})与预期({NUM_ROUNDS_PER_GAME})不符。")

        # 统计该实验的总合作分数
        experiment_cooperation_scores = []
        
        # 遍历每轮数据
        for round_idx, round_data in enumerate(game_logs):
            # 轮次从1开始计数
            round_num = round_idx + 1
            
            # 只处理前NUM_ROUNDS_PER_GAME轮
            if round_num > NUM_ROUNDS_PER_GAME:
                continue
            
            # 获取该轮所有agent的贡献
            if 'agent_contributions' in round_data:
                contributions = round_data['agent_contributions']
                
                # 计算该轮每个agent的合作分数
                round_cooperation_scores = []
                for agent_id, contribution in contributions.items():
                    # 单人单轮合作分数 = 贡献量 / 最大贡献金额
                    cooperation_score = contribution / MAX_CONTRIBUTION
                    round_cooperation_scores.append(cooperation_score)
                    experiment_cooperation_scores.append(cooperation_score)
                
                # 计算该轮的合作率（该轮所有玩家合作分数的平均值）
                if round_cooperation_scores:
                    round_cooperation_rate = np.mean(round_cooperation_scores)
                    round_cooperation_rates[round_num].append(round_cooperation_rate)
        
        # 计算该实验的总体合作率
        if experiment_cooperation_scores:
            experiment_rate = np.mean(experiment_cooperation_scores)
            experiment_cooperation_rates.append(experiment_rate)
            print(f"  该实验合作率: {experiment_rate:.2%}")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是一个有效的JSON文件。")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

# 计算平均值并输出结果
print("-" * 70)
print("公共物品博弈群体实验合作率分析结果")
print("-" * 70)

num_games = len(file_names)
print(f"分析文件数量: {num_games}")
print(f"每实验轮数: {NUM_ROUNDS_PER_GAME}")
print(f"每轮Agent数量: {NUM_AGENTS}")
print(f"最大贡献金额: {MAX_CONTRIBUTION}")

# 计算每轮的平均合作率
if round_cooperation_rates:
    print("\n每轮平均合作率:")
    print("轮次 | 平均合作率 | 样本数")
    print("-" * 35)
    
    for round_num in range(1, NUM_ROUNDS_PER_GAME + 1):
        rates = round_cooperation_rates[round_num]
        avg_rate = np.mean(rates) if rates else 0
        sample_count = len(rates)
        print(f"{round_num:4d} | {avg_rate:10.2%} | {sample_count:6d}")

    # 计算特定轮次的平均合作率
    print("\n特定轮次平均合作率:")
    print("-" * 50)
    
    # 第1轮
    if round_cooperation_rates[1]:
        avg_round_1 = np.mean(round_cooperation_rates[1])
        print(f"第1轮的平均合作率（{num_games}次实验平均）: {avg_round_1:.2%}")
    
    # 第15轮
    if round_cooperation_rates[15]:
        avg_round_15 = np.mean(round_cooperation_rates[15])
        print(f"第15轮的平均合作率（{num_games}次实验平均）: {avg_round_15:.2%}")
    
    # 最后一轮（第30轮）
    if round_cooperation_rates[NUM_ROUNDS_PER_GAME]:
        avg_last_round = np.mean(round_cooperation_rates[NUM_ROUNDS_PER_GAME])
        print(f"最后一轮（第{NUM_ROUNDS_PER_GAME}轮）的平均合作率（{num_games}次实验平均）: {avg_last_round:.2%}")

# 计算总体平均合作率（简单平均：各实验合作率相加再除以实验次数）
if experiment_cooperation_rates:
    overall_cooperation_rate = np.mean(experiment_cooperation_rates)
    experiment_count = len(experiment_cooperation_rates)
    print("\n各实验合作率:")
    for i, rate in enumerate(experiment_cooperation_rates, 1):
        print(f"  实验{i}: {rate:.2%}")
    print(
        f"{NUM_ROUNDS_PER_GAME}轮的平均合作率"
        f"（{experiment_count}次实验平均）: "
        f"{overall_cooperation_rate:.2%}"
    )
    if experiment_count >= 2:
        cooperation_std = np.std(
            experiment_cooperation_rates,
            ddof=1,
        )
        print(
            "三次实验合作率的样本标准差（ddof=1）: "
            f"{cooperation_std:.2%}"
        )
        print(
            "合作率（Mean ± SD）: "
            f"{overall_cooperation_rate:.2%} ± {cooperation_std:.2%}"
        )
    else:
        print("警告: 至少需要2次有效实验才能计算合作率的样本标准差。")
    if experiment_count != 3:
        print(
            f"警告: 预期3次实验，实际得到{experiment_count}次；"
            "当前均值和标准差基于实际有效实验计算。"
        )
else:
    print("\n警告: 无法计算总体平均合作率，因为没有数据。")

print("-" * 70)
print("分析完成！")
print("-" * 70)
