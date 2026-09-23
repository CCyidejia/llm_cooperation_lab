import json
import numpy as np
import os

# 数据文件所在目录
data_dir = r"D:\常用\agentsociety2\result_prisoners_dilemma_group\092001_qwen3-next-80b-a3b-instruct\data"

# 合作行为的标记
COOPERATE_ACTION = "Yes"

# 实验参数
NUM_ROUNDS_PER_GAME = 10
NUM_PAIRS_PER_ROUND = 12


def calculate_cooperation_rate_for_file(file_path):
    """计算单个日志文件的每轮合作率。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            game_logs = json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return [0.0] * NUM_ROUNDS_PER_GAME

    cooperation_rates = []

    for round_num in range(NUM_ROUNDS_PER_GAME):
        start_idx = round_num * NUM_PAIRS_PER_ROUND
        end_idx = start_idx + NUM_PAIRS_PER_ROUND
        round_logs = game_logs[start_idx:end_idx]

        total_decisions = 0
        cooperate_count = 0

        for interaction_data in round_logs:
            detail = interaction_data.get("detail", {})
            choice1 = detail.get("choice1")
            choice2 = detail.get("choice2")

            if choice1 is not None:
                total_decisions += 1
                if choice1 == COOPERATE_ACTION:
                    cooperate_count += 1

            if choice2 is not None:
                total_decisions += 1
                if choice2 == COOPERATE_ACTION:
                    cooperate_count += 1

        cooperation_rate = cooperate_count / total_decisions if total_decisions > 0 else 0.0
        cooperation_rates.append(cooperation_rate)

    return cooperation_rates


def main():
    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 未找到。请检查路径是否正确。")
        file_names = []
    else:
        file_names = sorted(
            [
                file_name
                for file_name in os.listdir(data_dir)
                if file_name.startswith("game_log") and file_name.endswith(".json")
            ]
        )

    all_cooperation_rates = []
    # 每个元素是一份 game_log（一次独立实验）的总体合作率。
    experiment_cooperation_rates = []

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        cooperation_rates = calculate_cooperation_rate_for_file(file_path)
        all_cooperation_rates.append(cooperation_rates)
        experiment_rate = np.mean(cooperation_rates)
        experiment_cooperation_rates.append(experiment_rate)
        print(f"处理文件: {file_name}")
        print(f"  该实验合作率: {experiment_rate:.2%}")

    if all_cooperation_rates:
        average_cooperation_rates = np.mean(all_cooperation_rates, axis=0)
    else:
        average_cooperation_rates = [0.0] * NUM_ROUNDS_PER_GAME

    print("-" * 70)
    print("囚徒困境群体实验合作率分析结果")
    print("-" * 70)
    print(f"分析文件数量: {len(file_names)}")
    print(f"每实验轮数: {NUM_ROUNDS_PER_GAME}")
    print(f"每轮配对数: {NUM_PAIRS_PER_ROUND}")

    print("\n每轮平均合作率:")
    print("轮次 | 平均合作率 | 样本数")
    print("-" * 35)
    for round_num, rate in enumerate(average_cooperation_rates, start=1):
        print(f"{round_num:>4} | {rate:>10.2%} | {len(file_names):>6}")

    print("\n特定轮次平均合作率:")
    print("-" * 50)
    for round_num in [1, 5, 10]:
        if round_num <= len(average_cooperation_rates):
            print(f"第 {round_num} 轮: {average_cooperation_rates[round_num - 1]:.2%}")

    if experiment_cooperation_rates:
        experiment_count = len(experiment_cooperation_rates)
        overall_average = np.mean(experiment_cooperation_rates)
        print("\n各实验合作率:")
        for index, rate in enumerate(experiment_cooperation_rates, start=1):
            print(f"  实验{index}: {rate:.2%}")
        print(
            f"三次实验的平均合作率（实际有效实验数: {experiment_count}）: "
            f"{overall_average:.2%}"
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
                f"{overall_average:.2%} ± {cooperation_std:.2%}"
            )
        else:
            print("警告: 至少需要2次有效实验才能计算合作率的样本标准差。")
        if experiment_count != 3:
            print(
                f"警告: 预期3次实验，实际得到{experiment_count}次；"
                "当前均值和标准差基于实际实验计算。"
            )
    else:
        print("\n警告: 无法计算总体平均合作率，因为没有数据。")


if __name__ == "__main__":
    main()
