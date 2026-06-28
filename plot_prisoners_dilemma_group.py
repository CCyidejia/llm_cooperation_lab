#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma Group Experiment Visualization Tool
Based on group experiment log format (multiple agents × multiple rounds)
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict, Counter
from scipy import stats

# ------- Global Configuration -------
# Set result directory path
result_dir = r"d:/常用/agentsociety2/result_prisoners_dilemma_group/10轮_3次_012404_llama3-8b"
data_dir = os.path.join(result_dir, "data")

# ------- Helper Functions -------
def extract_llm_name(directory_path):
    """
    Extract LLM name from the result directory path
    
    Parameters:
        directory_path (str): Result directory path
    
    Returns:
        str: Extracted LLM name
    """
    # Get the last directory name from the path
    last_dir = os.path.basename(directory_path)
    
    # Use regex to match the pattern and extract model name
    match = re.search(r'[^_]+$', last_dir)
    if match:
        return match.group(0)
    
    # If no match, use the entire directory name
    return last_dir

# LLM name (dynamically extracted from result directory)
llm_name = extract_llm_name(result_dir)

# ------- Data Loading Function -------
def load_game_logs(data_dir):
    """
    Load group experiment game logs from multiple JSON files
    
    Parameters:
        data_dir (str): Directory containing log files
    
    Returns:
        list: List of game log data from all files, each element is a tuple (file_name, logs)
    """
    experiments = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f.startswith('game_log')]
    
    if not json_files:
        print(f"No game log files found in: {data_dir}")
        return []
    
    print(f"Found {len(json_files)} game log files:")
    for file_name in sorted(json_files):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            experiments.append((file_name, logs))
            print(f"Successfully loaded: {file_name} ({len(logs)} interactions)")
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
    
    print(f"Total experiments loaded: {len(experiments)}")
    return experiments

# ------- Round Assignment Function -------
def assign_rounds(game_logs, agents_per_round=24):
    """
    Assign interactions to rounds based on timestamp
    
    Parameters:
        game_logs (list): Game log data
        agents_per_round (int): Number of agents per round
    
    Returns:
        dict: Key is round number, value is list of interactions in that round
    """
    # Sort all interactions by timestamp
    sorted_logs = sorted(game_logs, key=lambda x: x['timestamp'])
    
    # Calculate number of interactions per round (half the number of agents)
    interactions_per_round = agents_per_round // 2
    
    # Assign rounds
    round_data = defaultdict(list)
    for i, log in enumerate(sorted_logs):
        round_num = i // interactions_per_round + 1  # Rounds start from 1
        round_data[round_num].append(log)
    
    print(f"Round assignment completed: {len(round_data)} rounds")
    for rnd, logs in round_data.items():
        print(f"Round {rnd}: {len(logs)} interactions")
    
    return round_data

# ------- Data Processing Function -------
def process_game_data(round_data):
    """
    Process game log data to extract required information
    
    Parameters:
        round_data (dict): Key is round number, value is list of interactions in that round
    
    Returns:
        tuple: (total_actions, round_action_counts, all_payoffs)
    """
    if not round_data:
        print("No game log data available for processing.")
        return {}, {}, []
    
    # 统计所有游戏中所有智能体的动作（总计）
    total_actions = []
    
    # 统计每轮的动作，用于绘制每轮概率趋势图
    round_action_counts = defaultdict(list)  # key: round_num, value: list of actions
    
    # 统计收益信息
    all_payoffs = []  # 所有轮次的收益
    
    # 统计每个智能体的动作
    agent_actions = defaultdict(list)  # key: agent_name, value: list of actions
    agent_payoffs = defaultdict(list)  # key: agent_name, value: list of payoffs
    
    # Iterate through all rounds
    for round_num, logs in round_data.items():
        round_actions = []
        
        # Iterate through all interactions in the round
        for log in logs:
            detail = log['detail']
            
            # Extract agent 1 data
            agent1_name = detail['agent1_name']
            action1 = detail['choice1']
            payoff1 = detail['payoff1']
            
            # Extract agent 2 data
            agent2_name = detail['agent2_name']
            action2 = detail['choice2']
            payoff2 = detail['payoff2']
            
            # Normalize actions
            normalized_action1 = "Yes" if action1.lower() == "yes" else "No"
            normalized_action2 = "Yes" if action2.lower() == "yes" else "No"
            
            # Update total actions
            total_actions.extend([normalized_action1, normalized_action2])
            
            # Update round actions
            round_actions.extend([normalized_action1, normalized_action2])
            
            # Update agent-specific data
            agent_actions[agent1_name].append(normalized_action1)
            agent_actions[agent2_name].append(normalized_action2)
            
            agent_payoffs[agent1_name].append(payoff1)
            agent_payoffs[agent2_name].append(payoff2)
            
            # Update payoffs
            all_payoffs.extend([payoff1, payoff2])
        
        # Update round action counts
        round_action_counts[round_num] = round_actions
    
    return total_actions, round_action_counts, all_payoffs, agent_actions, agent_payoffs

# ------- Helper Functions for Confidence Intervals -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate 95% confidence interval for a list of data
    
    Parameters:
        data: list of numerical values
        confidence: confidence level (default 0.95)
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0, 0, 0
    
    n = len(data)
    mean_val = np.mean(data)
    
    if n == 1:
        return mean_val, mean_val, mean_val
    
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, df=n-1, loc=mean_val, scale=std_err)
    return mean_val, ci[0], ci[1]

# ------- Visualization Functions -------
# 绘制每轮"Yes"动作数量折线图（带95%置信区间）
def plot_yes_count_by_round(round_yes_counts, save_dir):
    """
    Plot the number of agents choosing "Yes" in each round with 95% confidence interval
    
    Parameters:
        round_yes_counts: dict, key is round number, value is list of "Yes" counts in that round across experiments
        save_dir: Directory to save the chart
    """
    if not round_yes_counts:
        print("No round action data available for plotting trend.")
        return
    
    rounds = sorted(round_yes_counts.keys())
    
    # 计算每轮的均值和95%置信区间
    mean_yes_counts = []
    ci_lower = []
    ci_upper = []
    
    for r in rounds:
        yes_counts = round_yes_counts[r]
        mean, lower, upper = calculate_confidence_interval(yes_counts)
        mean_yes_counts.append(mean)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # 绘制均值折线图
    plt.plot(rounds, mean_yes_counts, marker="o", label="Mean Number of Agents Choosing 'Yes'", color="#FFB482")
    
    # 添加95%置信区间
    plt.fill_between(rounds, ci_lower, ci_upper, color="#FFB482", alpha=0.3, label="95% Confidence Interval")
    
    plt.xlabel("Round Number", fontsize=12)
    plt.ylabel("Number of Agents Choosing 'Yes'", fontsize=12)
    plt.title(f"Number of Agents Choosing 'Yes' per Round with 95% CI\nLLM: {llm_name}", fontsize=16)
    plt.ylim(0, 24)
    plt.yticks(range(0, 25, 2))  # 设置纵坐标刻度为0到24，间隔2
    
    # 自适应横坐标刻度，避免数字重叠
    max_ticks = 20  # 最大显示的刻度数量
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step])
    else:
        plt.xticks(rounds)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"yes_count_by_round_with_ci_{llm_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"每轮选择'Yes'的智能体数量折线图（带95%置信区间）已保存: {save_path}")

# ------- Main Function -------
def main():
    """
    Main function to execute the visualization process
    """
    print("\n=== Prisoner's Dilemma Group Experiment Visualization Tool ===")
    print(f"Result directory: {result_dir}")
    print(f"LLM name: {llm_name}")
    
    # Load experiments
    print("\nLoading experiments...")
    experiments = load_game_logs(data_dir)
    
    if not experiments:
        print("No game log files found, program exiting.")
        return
    
    # Process each experiment and collect round data
    print("\nProcessing experiments...")
    experiment_round_data = []
    
    for file_name, logs in experiments:
        print(f"\nProcessing experiment: {file_name}")
        # Assign rounds for this experiment
        round_data = assign_rounds(logs)
        # Process game data for this experiment
        total_actions, round_action_counts, all_payoffs, agent_actions, agent_payoffs = process_game_data(round_data)
        experiment_round_data.append((file_name, round_action_counts))
    
    # Collect "Yes" counts per round across all experiments
    round_yes_counts = defaultdict(list)
    for file_name, round_action_counts in experiment_round_data:
        for round_num, actions in round_action_counts.items():
            yes_count = actions.count("Yes") if actions else 0
            round_yes_counts[round_num].append(yes_count)
    
    # Ensure save directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate visualization charts
    print("\nGenerating visualization charts...")
    
    # Plot number of agents choosing "Yes" per round with confidence intervals
    plot_yes_count_by_round(round_yes_counts, result_dir)
    
    print("\nAll charts have been successfully generated and saved!")

if __name__ == "__main__":
    main()
