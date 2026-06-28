#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trust Game Group Experiment Visualization Tool
Based on group experiment log format (120 interactions, 24 agents × 10 rounds)
Enhanced to support multiple experiment logs with 95% confidence intervals
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from scipy import stats

# ------- Global Configuration -------
# Set result directory path
result_dir = r"d:/常用/agentsociety2/result_trust_game_group_reputation/032103_qwen3-next-80b-a3b-instruct"
data_dir = os.path.join(result_dir, "data")  # Use data subfolder since data is stored there

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
    
    # Use regex to match the pattern like "20260110_121146_deepseek-v3"
    # and extract the part after the last underscore as LLM name
    match = re.search(r'[^_]+$', last_dir)
    if match:
        return match.group(0)
    
    # If no match, use the entire directory name
    return last_dir

# LLM name (dynamically extracted from result directory)
llm_name = extract_llm_name(result_dir)

# ------- Data Loading Function -------
def load_game_logs(data_directory):
    """
    Load group experiment game logs from all log files in the specified directory
    
    Parameters:
        data_directory (str): Path to the directory containing log files
    
    Returns:
        dict: Key is log file name, value is list of game log data
    """
    try:
        game_logs_dict = {}
        # Find all game log files
        log_files = [f for f in os.listdir(data_directory) if f.endswith('.json') and f.startswith('game_log')]
        
        if not log_files:
            print(f"No game log files found in {data_directory}")
            return game_logs_dict
        
        for log_file in sorted(log_files):
            file_path = os.path.join(data_directory, log_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                game_logs = json.load(f)
            game_logs_dict[log_file] = game_logs
            print(f"Successfully loaded game logs: {log_file}")
            print(f"Total interactions: {len(game_logs)}")
        
        return game_logs_dict
    except Exception as e:
        print(f"Failed to load log files: {e}")
        return {}

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
def process_game_data(game_logs_dict):
    """
    Process game log data from multiple experiments to extract required information
    
    Parameters:
        game_logs_dict (dict): Key is log file name, value is list of game log data
    
    Returns:
        tuple: (all_sent_amounts_by_experiment, all_returned_amounts_by_experiment)
    """
    if not game_logs_dict:
        print("No game log data available for processing.")
        return {}, {}
    
    all_sent_amounts_by_experiment = defaultdict(dict)
    all_returned_amounts_by_experiment = defaultdict(dict)
    
    # Process each experiment
    for exp_name, game_logs in game_logs_dict.items():
        # Assign rounds for this experiment
        round_data = assign_rounds(game_logs)
        
        # Process each round
        for round_num, logs in round_data.items():
            sent_amounts = []
            returned_amounts = []
            
            # Iterate through all interactions in the round
            for log in logs:
                # Extract trustor's sent amount
                sent_amount = log['detail']['sent_amount']
                sent_amounts.append(sent_amount)
                
                # Extract trustee's returned amount directly
                returned_amount = log['detail']['returned_amount']
                returned_amounts.append(returned_amount)
            
            # Add to the total data collection
            all_sent_amounts_by_experiment[round_num][exp_name] = sent_amounts
            all_returned_amounts_by_experiment[round_num][exp_name] = returned_amounts
    
    return all_sent_amounts_by_experiment, all_returned_amounts_by_experiment

# ------- Confidence Interval Calculation -------
def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate 95% confidence interval for a list of data
    
    Parameters:
        data (list): List of numerical data
        confidence (float): Confidence level (default 0.95)
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0, 0, 0
    
    n = len(data)
    mean_val = np.mean(data)
    
    # Calculate confidence interval
    if n == 1:
        # Only one data point, no confidence interval
        return mean_val, mean_val, mean_val
    
    std_err = stats.sem(data)  # Standard error
    ci = stats.t.interval(confidence, df=n-1, loc=mean_val, scale=std_err)
    
    return mean_val, ci[0], ci[1]

# ------- Visualization Functions -------
def plot_average_investment_trend(sent_amounts_data, save_dir):
    """
    Plot the average investment trend of trustors across rounds with 95% confidence intervals
    
    Parameters:
        sent_amounts_data: Dictionary where key is round number, value is dict of {experiment_name: list of amounts}
        save_dir: Directory to save the chart
    """
    if not sent_amounts_data:
        print("No investment amount data available for plotting trend.")
        return
    
    rounds = sorted(sent_amounts_data.keys())
    average_sent_amounts_per_round = []
    ci_lower_per_round = []
    ci_upper_per_round = []
    
    # Calculate average investment and confidence intervals per round
    for r in rounds:
        # Collect all amounts from all experiments for this round
        all_amounts = []
        for exp_data in sent_amounts_data[r].values():
            all_amounts.extend(exp_data)
        
        # Calculate mean and confidence interval
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        average_sent_amounts_per_round.append(mean_val)
        ci_lower_per_round.append(lower)
        ci_upper_per_round.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # Plot average investment amount line
    plt.plot(rounds, average_sent_amounts_per_round, marker="o", linestyle='-', color="#FFCC99",
             label="Average Trustor Investment")
    
    # Add 95% confidence interval
    plt.fill_between(rounds, ci_lower_per_round, ci_upper_per_round, color="#FFCC99", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.title(f"Trust Game Group Experiment - Trustor Average Investment Trend\nLLM: {llm_name}", fontsize=16)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Investment (Coins)", fontsize=12)
    # 自适应横坐标刻度，避免数字重叠
    max_ticks = 20  # 最大显示的刻度数量
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step])
    else:
        plt.xticks(rounds)
    
    # Set y-axis range (investment amount range 0-10)
    plt.ylim(-0.5, 10.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"average_investment_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Average investment trend chart with CI saved to: {save_path}")


def plot_average_returned_amount_trend(returned_amounts_data, save_dir):
    """
    Plot the average returned amount trend of trustees across rounds with 95% confidence intervals
    
    Parameters:
        returned_amounts_data: Dictionary where key is round number, value is dict of {experiment_name: list of amounts}
        save_dir: Directory to save the chart
    """
    if not returned_amounts_data:
        print("No returned amount data available for plotting trend.")
        return
    
    rounds = sorted(returned_amounts_data.keys())
    average_returned_amounts_per_round = []
    ci_lower_per_round = []
    ci_upper_per_round = []
    
    # Calculate average returned amount and confidence intervals per round
    for r in rounds:
        # Collect all amounts from all experiments for this round
        all_amounts = []
        for exp_data in returned_amounts_data[r].values():
            # Filter out NaN and infinity values (if any)
            filtered_amounts = [amount for amount in exp_data if not np.isnan(amount) and not np.isinf(amount)]
            all_amounts.extend(filtered_amounts)
        
        # Calculate mean and confidence interval
        mean_val, lower, upper = calculate_confidence_interval(all_amounts)
        average_returned_amounts_per_round.append(mean_val)
        ci_lower_per_round.append(lower)
        ci_upper_per_round.append(upper)
    
    plt.figure(figsize=(12, 7))
    
    # Plot average returned amount line
    plt.plot(rounds, average_returned_amounts_per_round, marker="o", linestyle='-', color="#A1C9F4",
             label="Average Trustee Returned Amount")
    
    # Add 95% confidence interval
    plt.fill_between(rounds, ci_lower_per_round, ci_upper_per_round, color="#A1C9F4", alpha=0.3,
                     label="95% Confidence Interval")
    
    plt.title(f"Trust Game Group Experiment - Trustee Average Returned Amount Trend\nLLM: {llm_name}", fontsize=16)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average Returned Amount (Coins)", fontsize=12)
    # 自适应横坐标刻度，避免数字重叠
    max_ticks = 20  # 最大显示的刻度数量
    if len(rounds) > max_ticks:
        step = len(rounds) // max_ticks + 1
        plt.xticks(rounds[::step])
    else:
        plt.xticks(rounds)
    
    # Set y-axis range based on data
    plt.ylim(-1, 30)  # Returned amount range
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"average_returned_amount_trend_with_ci_{llm_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Average returned amount trend chart with CI saved to: {save_path}")

# ------- Main Function -------
def main():
    """
    Main function to execute the visualization process
    """
    print("\n=== Trust Game Group Experiment Visualization Tool ===")
    print(f"Result directory: {result_dir}")
    print(f"Data directory: {data_dir}")
    print(f"LLM name: {llm_name}")
    
    # Load game logs from all files
    print("\nLoading game logs...")
    game_logs_dict = load_game_logs(data_dir)
    
    if not game_logs_dict:
        print("No game log files found, program exiting.")
        return
    
    # Process game data from all experiments
    print("\nProcessing game data...")
    all_sent_amounts, all_returned_amounts = process_game_data(game_logs_dict)
    
    # Ensure save directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate visualization charts with confidence intervals
    print("\nGenerating visualization charts with 95% confidence intervals...")
    plot_average_investment_trend(all_sent_amounts, result_dir)
    plot_average_returned_amount_trend(all_returned_amounts, result_dir)
    
    print("\nAll charts have been successfully generated and saved!")

if __name__ == "__main__":
    main()
