#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信任博弈群体实验合作率计算工具

该脚本用于计算信任博弈群体实验的合作率，
支持分析多个实验文件，计算每轮的合作率以及特定轮次的平均合作率。

修改后的计算规则：
- 发送方合作率 = 发送金额 / 最大发送金额（20）
- 接收方合作率 = 返还金额 / 收到金额（3 * 发送金额）
- 每个game_log文件代表一次完整的实验
- 实验包含多轮，每轮包含多个配对交互
- 总体合作率 = (实验1合作率 + 实验2合作率 + 实验3合作率) / 3
"""

import json
import numpy as np
import os

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据目录路径
# 这里使用指定的llama3-8b模型结果目录
data_dir = r"d:\常用\agentsociety\agentsociety\packages\agentsociety2\result_trust_game_population\053001_claude-haiku-4-5-20251001-thinking\data"

# 定义实验配置
NUM_ROUNDS_PER_GAME = 30  # 每个实验的轮数
NUM_PAIRS_PER_ROUND = 12  # 每轮的配对数（24个agent，每对2人）
TRUSTEE_MULTIPLIER = 3  # 受托者收到金额的乘数
MAX_SEND_AMOUNT = 20  # 最大发送金额

# 尝试获取数据目录中的所有game_log文件
try:
    file_names = [f for f in os.listdir(data_dir) if f.startswith('game_log') and f.endswith('.json')]
    file_names = sorted(file_names)  # 排序以确保顺序一致
    if not file_names:
        print(f"警告: 在目录 '{data_dir}' 中未找到任何game_log文件。")
except FileNotFoundError:
    print(f"错误: 目录 '{data_dir}' 未找到。请检查路径是否正确。")
    file_names = []

# 用于存储每轮合作率的字典
# 键为轮次(1-30)，值为该轮次在所有实验中的合作率列表
round_sender_cooperation_rates = {i: [] for i in range(1, NUM_ROUNDS_PER_GAME + 1)}
round_receiver_cooperation_rates = {i: [] for i in range(1, NUM_ROUNDS_PER_GAME + 1)}

# 存储每次实验的总体合作率
experiment_sender_cooperation_rates = []
experiment_receiver_cooperation_rates = []

# 遍历所有文件
for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            game_logs = json.load(f)

        print(f"处理文件: {file_name}")
        print(f"总交互次数: {len(game_logs)}")
        
        # 验证数据完整性
        expected_interactions = NUM_ROUNDS_PER_GAME * NUM_PAIRS_PER_ROUND
        if len(game_logs) != expected_interactions:
            print(f"警告: 文件 {file_name} 的交互次数({len(game_logs)})与预期({expected_interactions})不符。")

        # 统计该实验的总发送方合作分数和总接收方合作分数
        experiment_sender_scores = []
        experiment_receiver_scores = []
        
        # 按轮次分组处理交互数据
        # 每轮包含NUM_PAIRS_PER_ROUND个配对交互
        for interaction_idx, interaction_data in enumerate(game_logs):
            # 计算该交互属于第几轮（从1开始）
            round_num = (interaction_idx // NUM_PAIRS_PER_ROUND) + 1
            
            # 只处理前NUM_ROUNDS_PER_GAME轮
            if round_num > NUM_ROUNDS_PER_GAME:
                continue
            
            # 检查是否有detail字段
            if 'detail' in interaction_data:
                detail = interaction_data['detail']
                
                # 获取发送金额和返还金额
                sent_amount = detail.get('sent_amount', 0)
                returned_amount = detail.get('returned_amount', 0)
                
                # 计算发送方合作率：发送金额 / 最大发送金额
                if MAX_SEND_AMOUNT > 0:
                    sender_cooperation_rate = sent_amount / MAX_SEND_AMOUNT
                    round_sender_cooperation_rates[round_num].append(sender_cooperation_rate)
                    experiment_sender_scores.append(sender_cooperation_rate)
                
                # 计算接收方合作率：返还金额 / 收到金额（3 * 发送金额）
                received_amount = sent_amount * TRUSTEE_MULTIPLIER
                if received_amount > 0:
                    receiver_cooperation_rate = returned_amount / received_amount
                    round_receiver_cooperation_rates[round_num].append(receiver_cooperation_rate)
                    experiment_receiver_scores.append(receiver_cooperation_rate)
        
        # 计算该实验的总体合作率
        if experiment_sender_scores:
            experiment_sender_rate = np.mean(experiment_sender_scores)
            experiment_sender_cooperation_rates.append(experiment_sender_rate)
            print(f"  该实验发送方合作率: {experiment_sender_rate:.2%}")
        
        if experiment_receiver_scores:
            experiment_receiver_rate = np.mean(experiment_receiver_scores)
            experiment_receiver_cooperation_rates.append(experiment_receiver_rate)
            print(f"  该实验接收方合作率: {experiment_receiver_rate:.2%}")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是一个有效的JSON文件。")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

# 计算平均值并输出结果
print("-" * 70)
print("信任博弈群体实验合作率分析结果")
print("-" * 70)

num_games = len(file_names)
print(f"分析文件数量: {num_games}")
print(f"每实验轮数: {NUM_ROUNDS_PER_GAME}")
print(f"每轮配对数: {NUM_PAIRS_PER_ROUND}")
print(f"受托者乘数: {TRUSTEE_MULTIPLIER}x")
print(f"最大发送金额: {MAX_SEND_AMOUNT}")

# 计算发送方每轮的平均合作率
if round_sender_cooperation_rates:
    print("\n发送方每轮平均合作率:")
    print("轮次 | 平均合作率 | 样本数")
    print("-" * 35)
    
    for round_num in range(1, NUM_ROUNDS_PER_GAME + 1):
        rates = round_sender_cooperation_rates[round_num]
        avg_rate = np.mean(rates) if rates else 0
        sample_count = len(rates)
        print(f"{round_num:4d} | {avg_rate:10.2%} | {sample_count:6d}")

# 计算接收方每轮的平均合作率
if round_receiver_cooperation_rates:
    print("\n接收方每轮平均合作率:")
    print("轮次 | 平均合作率 | 样本数")
    print("-" * 35)
    
    for round_num in range(1, NUM_ROUNDS_PER_GAME + 1):
        rates = round_receiver_cooperation_rates[round_num]
        avg_rate = np.mean(rates) if rates else 0
        sample_count = len(rates)
        print(f"{round_num:4d} | {avg_rate:10.2%} | {sample_count:6d}")

# 计算特定轮次的平均合作率
print("\n特定轮次平均合作率:")
print("-" * 50)

# 第1轮
if round_sender_cooperation_rates[1]:
    avg_round_1_sender = np.mean(round_sender_cooperation_rates[1])
    print(f"第1轮的平均发送方合作率（{num_games}次实验平均）: {avg_round_1_sender:.2%}")

if round_receiver_cooperation_rates[1]:
    avg_round_1_receiver = np.mean(round_receiver_cooperation_rates[1])
    print(f"第1轮的平均接收方合作率（{num_games}次实验平均）: {avg_round_1_receiver:.2%}")

# 第15轮
if round_sender_cooperation_rates[15]:
    avg_round_15_sender = np.mean(round_sender_cooperation_rates[15])
    print(f"第15轮的平均发送方合作率（{num_games}次实验平均）: {avg_round_15_sender:.2%}")

if round_receiver_cooperation_rates[15]:
    avg_round_15_receiver = np.mean(round_receiver_cooperation_rates[15])
    print(f"第15轮的平均接收方合作率（{num_games}次实验平均）: {avg_round_15_receiver:.2%}")

# 最后一轮（第30轮）
if round_sender_cooperation_rates[NUM_ROUNDS_PER_GAME]:
    avg_last_round_sender = np.mean(round_sender_cooperation_rates[NUM_ROUNDS_PER_GAME])
    print(f"最后一轮（第{NUM_ROUNDS_PER_GAME}轮）的平均发送方合作率（{num_games}次实验平均）: {avg_last_round_sender:.2%}")

if round_receiver_cooperation_rates[NUM_ROUNDS_PER_GAME]:
    avg_last_round_receiver = np.mean(round_receiver_cooperation_rates[NUM_ROUNDS_PER_GAME])
    print(f"最后一轮（第{NUM_ROUNDS_PER_GAME}轮）的平均接收方合作率（{num_games}次实验平均）: {avg_last_round_receiver:.2%}")

# 计算总体平均合作率（简单平均：各实验合作率相加再除以实验次数）
print("\n总体平均合作率:")
print("-" * 50)

if experiment_sender_cooperation_rates:
    overall_sender_cooperation_rate = np.mean(experiment_sender_cooperation_rates)
    print(f"{NUM_ROUNDS_PER_GAME}轮的平均发送方合作率（{num_games}次实验平均）: {overall_sender_cooperation_rate:.2%}")
    print(f"各实验发送方合作率:")
    for i, rate in enumerate(experiment_sender_cooperation_rates, 1):
        print(f"  实验{i}: {rate:.2%}")
else:
    print("警告: 无法计算发送方总体平均合作率，因为没有数据。")

print()

if experiment_receiver_cooperation_rates:
    overall_receiver_cooperation_rate = np.mean(experiment_receiver_cooperation_rates)
    print(f"{NUM_ROUNDS_PER_GAME}轮的平均接收方合作率（{num_games}次实验平均）: {overall_receiver_cooperation_rate:.2%}")
    print(f"各实验接收方合作率:")
    for i, rate in enumerate(experiment_receiver_cooperation_rates, 1):
        print(f"  实验{i}: {rate:.2%}")
else:
    print("警告: 无法计算接收方总体平均合作率，因为没有数据。")

print("-" * 70)
print("分析完成！")
print("-" * 70)