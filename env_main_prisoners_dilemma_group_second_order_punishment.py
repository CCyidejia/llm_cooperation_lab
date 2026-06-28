#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma Game - Group Experiment with First- and Second-Order Punishment
基于 env_main_prisoners_dilemma_group_punishment.py

一阶惩罚（与原版相同）：
- 暴露概率 0.5：单方背叛时受害者必见；其余 22 名旁观者各自以 0.5 目击。
- LLM 决定是否惩罚背叛者：背叛者每次被惩罚 -9，惩罚者每次 -2。

二阶惩罚（元规范）：
- 一阶中“目击背叛但未惩罚背叛者”的智能体（受害者选不惩罚，或目击旁观者选不惩罚）记为「未执行惩罚者」。
- 其余每名智能体对每名「未执行惩罚者」独立以暴露概率 0.5 目击其袖手行为，目击后由 LLM 决定是否元惩罚：
  元惩罚对象 -9，元执行者 -2（与一阶数值一致，见附件）。
- 双方同时合作或同时背叛时，不触发一阶/二阶惩罚阶段。

与一阶相同：未实现附件表中的「胆量」「报复心」等 [0,1] 数值参数；是否惩罚 / 是否元惩罚
**完全由 LLM 在 `decide_punishment` / `decide_meta_punishment` 中的自主决策**决定，仅暴露概率为随机机制。
"""
import os
import json
import sys
import asyncio
import time
from datetime import datetime
import logging
import re
import random
from collections import defaultdict, Counter

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# V2 framework core components import
from agentsociety2.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv

# 从游戏规则模块导入收益计算函数
PAYOFF_MATRIX = {
    ("Yes", "Yes"): (3, 3),
    ("Yes", "No"): (0, 5),
    ("No", "Yes"): (5, 0),
    ("No", "No"): (1, 1),
}


def get_payoff(action1: str, action2: str):
    return PAYOFF_MATRIX.get((action1.capitalize(), action2.capitalize()), (0, 0))

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_prisoners_dilemma_group_second_order_punishment", exist_ok=True)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_api_log.txt"),
        logging.StreamHandler()
    ]
)

# --- Global Configuration and Seed ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --- 惩罚机制参数（与需求一致，其余实验参数与 enhanced 版保持相同）---
EXPOSURE_PROBABILITY = 0.5
PUNISHMENT_COST_TO_DEFECTOR = -9
EXECUTION_COST_TO_PUNISHER = -2
# 二阶惩罚（元惩罚）：与附件一致，与一阶同数值
META_PUNISHMENT_TO_NON_PUNISHER = -9
META_EXECUTION_COST_TO_META_PUNISHER = -2


class PrisonersDilemmaAgent(AgentBase):
    """
    Agent for Prisoner's Dilemma Game that manages its own state.
    """

    def __init__(self, id: int, name: str, profile: str = ""):
        """
        Initialize PrisonersDilemmaAgent
        """
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []  # 跟踪之前的选择，增强选择稳定性

        # --- 自主状态管理：代理独立维护交互历史和状态 ---
        self._my_history = []        # 自己的选择历史
        self._partner_history = []   # 伙伴的选择历史
        self._outcome_history = []   # 结果历史（收益）
        # -------------------------------------------------

        # --- 增强记忆机制：按智能体ID记录互动历史 ---
        from collections import defaultdict
        self._agent_memory = defaultdict(lambda: {
            "interaction_count": 0,
            "my_choices": [],        # 自己对该智能体的选择历史
            "their_choices": [],     # 该智能体的选择历史
            "my_payoffs": [],        # 自己的收益历史
            "their_payoffs": []      # 该智能体的收益历史
        })
        # -------------------------------------------------

    def set_environment(self, env):
        """
        设置智能体所在的环境
        """
        self._env = env

    # --- 自主状态管理：添加更新内部状态的方法 ---
    def update_state(self, partner_id: int, my_choice: str, partner_choice: str, my_payoff: int, their_payoff: int):
        """
        更新代理的内部状态
        """
        # 添加选择历史和结果
        self._my_history.append(my_choice)
        self._partner_history.append(partner_choice)
        self._outcome_history.append(my_payoff)

        # 更新增强记忆：按智能体ID记录互动历史
        agent_memory = self._agent_memory[partner_id]
        agent_memory["interaction_count"] += 1
        agent_memory["my_choices"].append(my_choice)
        agent_memory["their_choices"].append(partner_choice)
        agent_memory["my_payoffs"].append(my_payoff)
        agent_memory["their_payoffs"].append(their_payoff)

        logging.debug(f"[{self.name}] Updated state: partner_id={partner_id}, my_choice={my_choice}, partner_choice={partner_choice}, my_payoff={my_payoff}, their_payoff={their_payoff}")

    def get_state_summary(self) -> dict:
        """
        获取代理当前状态摘要
        """
        return {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'partner_history': self._partner_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history),
            'agent_memory': dict(self._agent_memory)  # 包含增强记忆数据
        }
    # -----------------------------------------------

    async def init(self, llm: AgentLLM):
        self._llm = llm

    @property
    def name(self):
        return self._name

    async def dump(self) -> dict:
        """Export agent state"""
        return {
            "profile": self._profile,
            "my_history": self._my_history,
            "partner_history": self._partner_history,
            "outcome_history": self._outcome_history,
            "previous_choices": self._previous_choices,
            "agent_memory": dict(self._agent_memory)  # 包含增强记忆数据
        }

    async def load(self, dump_data: dict):
        """Load agent state"""
        self._profile = dump_data.get("profile", "")
        self._my_history = dump_data.get("my_history", [])
        self._partner_history = dump_data.get("partner_history", [])
        self._outcome_history = dump_data.get("outcome_history", [])
        self._previous_choices = dump_data.get("previous_choices", [])

        # 加载增强记忆数据
        from collections import defaultdict
        agent_memory_data = dump_data.get("agent_memory", {})
        self._agent_memory = defaultdict(lambda: {
            "interaction_count": 0,
            "my_choices": [],
            "their_choices": [],
            "my_payoffs": [],
            "their_payoffs": []
        })
        for agent_id, memory_data in agent_memory_data.items():
            # 确保agent_id是整数
            try:
                agent_id_int = int(agent_id)
                self._agent_memory[agent_id_int] = memory_data
            except (ValueError, TypeError):
                continue

    async def ask(self, message: str, readonly: bool = True) -> str:
        """Answer questions"""
        prompt = f"{self._profile}\n\n{message}"
        # 使用LLMAPI目录下的实现
        response = await asyncio.to_thread(
            self._llm.get_llm_response,
            system_message="",
            user_prompt=prompt
        )
        return response

    async def step(self, tick: int, t: datetime) -> str:
        """Execute one step"""
        return f"{self.name} step executed at tick {tick}"

    async def make_decision(self, partner_id: int, partner_name: str, memory_size: int) -> tuple:
        """
        Make decision using LLM based on history and game rules
        With Retry Mechanism and Error Handling
        """
        # 构建历史记录字符串
        history_str = "History:\n"
        for i, (my_action, opponent_action) in enumerate(zip(self._my_history, self._partner_history)):
            formatted_my_action = my_action.capitalize()
            formatted_opponent_action = opponent_action.capitalize()
            history_str += f"Round {i+1}: Your choice={formatted_my_action}, {partner_name}'s choice={formatted_opponent_action}\n"

        if partner_id in self._agent_memory:
            agent_memory = self._agent_memory[partner_id]
            history_str += f"\nEnhanced Memory with {partner_name}:\n"
            history_str += f"Interaction count: {agent_memory['interaction_count']}\n"
            if agent_memory['interaction_count'] > 0:
                history_str += "All interactions:\n"
                for i in range(agent_memory['interaction_count']):
                    round_num = i + 1
                    my_choice = agent_memory['my_choices'][i].capitalize()
                    their_choice = agent_memory['their_choices'][i].capitalize()
                    my_payoff = agent_memory['my_payoffs'][i]
                    their_payoff = agent_memory['their_payoffs'][i]
                    history_str += f"  Memory Round {round_num}: Your choice={my_choice}, {partner_name}'s choice={their_choice}, Your payoff={my_payoff}, {partner_name}'s payoff={their_payoff}\n"

        prompt = (
            f"{self._profile}\n"
            f"{history_str}\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your action.\n"
            "Your decision must be either **Yes (Cooperate)** or **No (Defect/Betray)**.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n"
            "\n"
            "<output>\n"
            "  <action>Yes or No</action>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>"
            "\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure. Your entire output must consist of exactly the XML format shown above."
        )

        # --- 新增：重试参数配置 ---
        max_retries = 5           # 最大重试次数
        base_delay = 2            # 基础等待时间(秒)

        for attempt in range(max_retries):
            try:
                # 调用LLM
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    system_message="",
                    user_prompt=prompt
                )

                # --- 关键修改：检查响应是否为 API 错误信息 ---
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] 收到API错误 (尝试 {attempt+1}/{max_retries})，准备重试...")
                    raise ConnectionError(f"API returned error: {content[:100]}...")

                # --- 解析逻辑 ---
                action_match = re.search(r'<action>\s*(Yes|No)\s*</action>', content, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).capitalize()
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', content, re.DOTALL | re.IGNORECASE)
                    if explanation_match:
                        explanation = explanation_match.group(1).strip()
                    else:
                        explanation = f"选择了 '{action}'，但未在 XML 中提供解释"

                    return action, explanation

                else:
                    lines = content.strip().split('\n')
                    first_line = lines[0].strip()
                    match = re.search(r'^\s*(yes|no)\s*[.!]?$', first_line, re.IGNORECASE)

                    if match:
                        action = match.group(1).capitalize()
                        explanation = ' '.join(lines[1:]).strip() or "单行匹配成功"
                        return action, explanation

                    raise ValueError(f"无法解析有效格式: {content[:50]}...")

            except Exception as e:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)

                if attempt < max_retries - 1:
                    print(f"[{self.name}] 请求或解析出错: {type(e).__name__}。将在 {sleep_time:.2f}秒后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    error_message = f"重试耗尽。错误: {str(e)}"
                    print(f"[{self.name}] [ERROR] {error_message}")
                    logging.error(f"[{self.name}] {error_message}")
                    raise

    def _first_order_victim_situation_text(self, betrayer_name: str) -> str:
        """与一阶 decide_punishment（受害者）相同的情境段。"""
        return (
            f"You are **{self.name}** (the victim). In the match just played, you chose Yes (Cooperate) and "
            f"**{betrayer_name}** chose No (Defect): **{betrayer_name} betrayed you**."
        )

    def _first_order_bystander_situation_text(self, betrayer_name: str, victim_name: str) -> str:
        """与一阶 decide_punishment（旁观者）相同的情境段；二阶 prompt 在其上追加。"""
        return (
            f"You are **{self.name}**, a bystander. You **observed** that **{betrayer_name}** betrayed "
            f"**{victim_name}** (one cooperated, one defected in their favor). "
            "Other agents may or may not have observed this."
        )

    def _first_order_punishment_option_text(self, betrayer_name: str) -> str:
        """一阶惩罚选项说明（与 env_main_prisoners_dilemma_group_punishment 版一致，供一阶与二阶 prompt 共用）。"""
        return (
            "**Punishment option**: If you choose to **punish** (retaliate against the betrayer), "
            f"you pay **2 points** yourself (execution cost), and **{betrayer_name}** loses **9 points**. "
            "If you **do not punish**, your score is unchanged by this punishment decision."
        )

    async def decide_punishment(
        self,
        observer_role: str,
        betrayer_name: str,
        victim_name: str,
    ) -> tuple[str, str]:
        """
        第三者/受害者惩罚决策：由 LLM 决定是否付出 -2 执行成本去惩罚背叛者（对方 -9）。
        observer_role: 'victim' 或 'bystander'
        返回 (Yes/No, explanation)，Yes 表示选择惩罚。
        """
        if observer_role == "victim":
            situation = self._first_order_victim_situation_text(betrayer_name)
        else:
            situation = self._first_order_bystander_situation_text(betrayer_name, victim_name)

        prompt = (
            f"{self._profile}\n\n"
            f"{situation}\n\n"
            f"{self._first_order_punishment_option_text(betrayer_name)}\n\n"
            "Your goal is to maximize your own total points over the long run.\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules above, decide whether to punish.\n"
            "Your decision must be either **Yes (punish)** or **No (do not punish)**.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n\n"
            "<output>\n"
            "  <action>Yes or No</action>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>\n\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure."
        )

        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    system_message="",
                    user_prompt=prompt
                )
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] 惩罚决策API错误 (尝试 {attempt+1}/{max_retries})，重试...")
                    raise ConnectionError(f"API returned error: {content[:100]}...")

                action_match = re.search(r'<action>\s*(Yes|No)\s*</action>', content, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).capitalize()
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', content, re.DOTALL | re.IGNORECASE)
                    if explanation_match:
                        explanation = explanation_match.group(1).strip()
                    else:
                        explanation = f"选择了 '{action}'，但未在 XML 中提供解释"
                    return action, explanation

                lines = content.strip().split('\n')
                first_line = lines[0].strip()
                match = re.search(r'^\s*(yes|no)\s*[.!]?$', first_line, re.IGNORECASE)
                if match:
                    action = match.group(1).capitalize()
                    explanation = ' '.join(lines[1:]).strip() or "单行匹配成功"
                    return action, explanation

                raise ValueError(f"无法解析惩罚决策格式: {content[:50]}...")

            except Exception as e:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                if attempt < max_retries - 1:
                    print(f"[{self.name}] 惩罚决策出错: {type(e).__name__}。{sleep_time:.2f}秒后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    logging.error(f"[{self.name}] 惩罚决策重试耗尽: {e}")
                    raise

    async def decide_meta_punishment(
        self,
        target_name: str,
        betrayer_name: str,
        victim_name: str,
    ) -> tuple[str, str]:
        """
        二阶（元）惩罚：prompt 结构为「与一阶相同的旁观者情境 + 一阶惩罚选项」再追加二阶段落；
        本阶段仅决策是否对「未惩罚背叛者的目击者」实施元惩罚。
        """
        first_situation = self._first_order_bystander_situation_text(betrayer_name, victim_name)
        first_option = self._first_order_punishment_option_text(betrayer_name)
        second_order_addon = (
            "---\n\n"
            "**Additional (second-order — meta-punishment)**:\n"
            f"You also observed that **{target_name}** witnessed this defection but chose **NOT** to punish "
            f"**{betrayer_name}** (stood by).\n\n"
            "**Second-order punishment option** (directed at the **non-punisher**): If you choose to "
            "**meta-punish** this player for failing to enforce the norm, you pay **2 points** yourself "
            f"(meta-execution cost), and **{target_name}** loses **9 points** (meta-punishment penalty). "
            "If you **do not** meta-punish, your score is unchanged by this second-order decision.\n\n"
            "**Note**: In this **decision step** you are **not** choosing again about first-order punishment "
            "of the betrayer — that stage has already been completed. "
            "Your **only** decision here is whether to **second-order** meta-punish **"
            f"{target_name}**.\n\n"
        )
        prompt = (
            f"{self._profile}\n\n"
            f"{first_situation}\n\n"
            f"{first_option}\n\n"
            f"{second_order_addon}"
            "Your goal is to maximize your own total points over the long run.\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules above, decide whether to **second-order "
            "meta-punish** "
            f"(punish **{target_name}** for standing by). "
            "Your decision must be either **Yes (meta-punish)** or **No (do not meta-punish)**.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n\n"
            "<output>\n"
            "  <action>Yes or No</action>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>\n\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure."
        )
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    system_message="",
                    user_prompt=prompt
                )
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] 元惩罚决策API错误 (尝试 {attempt+1}/{max_retries})，重试...")
                    raise ConnectionError(f"API returned error: {content[:100]}...")

                action_match = re.search(r'<action>\s*(Yes|No)\s*</action>', content, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).capitalize()
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', content, re.DOTALL | re.IGNORECASE)
                    if explanation_match:
                        explanation = explanation_match.group(1).strip()
                    else:
                        explanation = f"选择了 '{action}'，但未在 XML 中提供解释"
                    return action, explanation

                lines = content.strip().split('\n')
                first_line = lines[0].strip()
                match = re.search(r'^\s*(yes|no)\s*[.!]?$', first_line, re.IGNORECASE)
                if match:
                    action = match.group(1).capitalize()
                    explanation = ' '.join(lines[1:]).strip() or "单行匹配成功"
                    return action, explanation

                raise ValueError(f"无法解析元惩罚决策格式: {content[:50]}...")

            except Exception as e:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                if attempt < max_retries - 1:
                    print(f"[{self.name}] 元惩罚决策出错: {type(e).__name__}。{sleep_time:.2f}秒后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    logging.error(f"[{self.name}] 元惩罚决策重试耗尽: {e}")
                    raise

    def _get_stable_choice(self, options: list) -> str:
        """
        增强选择稳定性的逻辑：
        1. 优先选择最频繁的之前选择
        2. 如果没有一致的之前选择，则选择环境中最流行的选项
        3. 最后才使用随机选择作为最后的备选
        """
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            if count > len(self._previous_choices) / 2 and most_common_choice in options:
                logging.info(f"[{self.name}] 使用最频繁的之前选择: {most_common_choice}")
                return most_common_choice

        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, count = self._env.choice_frequency.most_common(1)[0]
                if most_popular_choice in options:
                    logging.info(f"[{self.name}] 使用环境中最流行的选项: {most_popular_choice}")
                    return most_popular_choice

        logging.info(f"[{self.name}] 使用随机选择作为备选")
        return random.choice(options)

    async def report_token_usage(self):
        """Report token usage (if supported)"""
        pass

    def get_agent_memory(self, agent_id: int) -> dict:
        """
        获取与特定智能体的互动记忆
        """
        return self._agent_memory.get(agent_id, {
            "interaction_count": 0,
            "my_choices": [],
            "their_choices": [],
            "my_payoffs": [],
            "their_payoffs": []
        })

    def get_all_agents_memory(self) -> dict:
        """
        获取所有智能体的互动记忆
        """
        return dict(self._agent_memory)

    def get_memory_summary(self) -> dict:
        """
        获取记忆摘要统计
        """
        summary = {
            "total_agents_interacted": len(self._agent_memory),
            "agent_interaction_counts": {}
        }
        for agent_id, memory in self._agent_memory.items():
            summary["agent_interaction_counts"][agent_id] = memory["interaction_count"]
        return summary


class PrisonersDilemmaSecondOrderPunishmentEnvironment:
    """
    囚徒困境群体实验：一阶（第三方惩罚背叛者）+ 二阶（惩罚未惩罚背叛者的目击者）。
    """

    def __init__(self, num_agents: int, memory_size: int, total_interactions: int,
                 interaction_schedule: list,
                 exposure_probability: float = EXPOSURE_PROBABILITY,
                 punishment_to_defector: int = PUNISHMENT_COST_TO_DEFECTOR,
                 execution_cost_punisher: int = EXECUTION_COST_TO_PUNISHER,
                 meta_punishment_to_non_punisher: int = META_PUNISHMENT_TO_NON_PUNISHER,
                 meta_execution_cost: int = META_EXECUTION_COST_TO_META_PUNISHER):

        self.num_agents = num_agents
        self.memory_size = memory_size
        self.total_interactions = total_interactions
        self.agents = []  # List of Agent objects
        self.interaction_schedule = interaction_schedule

        self.exposure_probability = exposure_probability
        self.punishment_to_defector = punishment_to_defector
        self.execution_cost_punisher = execution_cost_punisher
        self.meta_punishment_to_non_punisher = meta_punishment_to_non_punisher
        self.meta_execution_cost = meta_execution_cost

        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'memory_size': memory_size,
            'interactions_per_run': 1,
            'exposure_probability': exposure_probability,
            'punishment_to_defector': punishment_to_defector,
            'execution_cost_punisher': execution_cost_punisher,
            'meta_punishment_to_non_punisher': meta_punishment_to_non_punisher,
            'meta_execution_cost': meta_execution_cost,
            'punishment_decision_by_llm': True,
            'second_order_punishment': True,
            'courage_vengefulness_params': False,
            'punishment_decisions_by_llm_only': True,
        }

        self.game_stats = {'total_interactions': 0, 'cooperation_count': 0}
        self.game_logs = []
        self.choice_frequency = Counter()

    def set_agents(self, agents: list):
        self.agents = agents

    def _get_agent_by_id(self, agent_id):
        """Helper to get Agent object from ID."""
        return next((a for a in self.agents if a._id == agent_id), None)

    async def _resolve_punishment_with_llm(
        self,
        agent1_id: int,
        agent2_id: int,
        action1: str,
        action2: str,
        payoff1: int,
        payoff2: int,
    ):
        """
        单方背叛时：受害者一定看到并调用 LLM 决定是否惩罚；
        22 名旁观者各自以 exposure_probability 目击，目击者各自调用 LLM 决定是否惩罚。
        返回 (payoff1, payoff2, punishment_detail)
        """
        a1, a2 = action1.capitalize(), action2.capitalize()
        punishment_detail = {
            "applies": False,
            "betrayer_id": None,
            "victim_id": None,
            "victim_always_observes": True,
            "bystander_ids": [],
            "victim_punish": None,
            "bystander_exposure": {},
            "bystander_punish": {},
            "punisher_ids": [],
            "num_punishers": 0,
            "defector_punishment_total": 0,
            "punisher_execution_costs": {},
        }

        if a1 == "Yes" and a2 == "No":
            betrayer_id, victim_id = agent2_id, agent1_id
        elif a1 == "No" and a2 == "Yes":
            betrayer_id, victim_id = agent1_id, agent2_id
        else:
            return payoff1, payoff2, punishment_detail

        betrayer = self._get_agent_by_id(betrayer_id)
        victim = self._get_agent_by_id(victim_id)
        if not betrayer or not victim:
            return payoff1, payoff2, punishment_detail

        punishment_detail["applies"] = True
        punishment_detail["betrayer_id"] = betrayer_id
        punishment_detail["victim_id"] = victim_id

        bystander_ids = [i for i in range(self.num_agents) if i not in (agent1_id, agent2_id)]
        punishment_detail["bystander_ids"] = list(bystander_ids)

        for bid in bystander_ids:
            punishment_detail["bystander_exposure"][bid] = random.random() < self.exposure_probability

        async def ask_victim():
            dec, expl = await victim.decide_punishment(
                "victim", betrayer.name, victim.name
            )
            return ("victim", victim_id, dec, expl)

        async def ask_bystander(bid: int):
            agent = self._get_agent_by_id(bid)
            dec, expl = await agent.decide_punishment(
                "bystander", betrayer.name, victim.name
            )
            return ("bystander", bid, dec, expl)

        tasks = [ask_victim()]
        for bid in bystander_ids:
            if punishment_detail["bystander_exposure"][bid]:
                tasks.append(ask_bystander(bid))

        results = await asyncio.gather(*tasks)

        punishers = []
        for role, aid, dec, expl in results:
            if role == "victim":
                punishment_detail["victim_punish"] = {"decision": dec, "explanation": expl}
                if dec == "Yes":
                    punishers.append(victim_id)
            else:
                punishment_detail["bystander_punish"][aid] = {"decision": dec, "explanation": expl}
                if dec == "Yes":
                    punishers.append(aid)

        k = len(punishers)
        punishment_detail["punisher_ids"] = list(punishers)
        punishment_detail["num_punishers"] = k
        total_punishment = self.punishment_to_defector * k
        punishment_detail["defector_punishment_total"] = total_punishment

        if betrayer_id == agent1_id:
            payoff1 += total_punishment
        else:
            payoff2 += total_punishment

        for pid in punishers:
            punishment_detail["punisher_execution_costs"][pid] = self.execution_cost_punisher

        return payoff1, payoff2, punishment_detail

    async def _resolve_second_order_punishment_with_llm(
        self,
        punishment_detail: dict,
        payoff1: int,
        payoff2: int,
        agent1_id: int,
        agent2_id: int,
    ) -> tuple:
        """
        二阶惩罚：对「目击背叛却未惩罚背叛者」的智能体，其他智能体以暴露概率目击其袖手行为，
        由 LLM 决定是否元惩罚（对象 -9，执行者 -2）。
        """
        second_order = {
            "applies": False,
            "non_punisher_ids": [],
            "meta_exposure": {},
            "meta_decisions": {},
            "meta_punisher_ids": [],
            "meta_punisher_execution_costs": {},
            "meta_target_penalties": {},
        }

        if not punishment_detail.get("applies"):
            return payoff1, payoff2, second_order

        betrayer_id = punishment_detail["betrayer_id"]
        victim_id = punishment_detail["victim_id"]
        betrayer = self._get_agent_by_id(betrayer_id)
        victim = self._get_agent_by_id(victim_id)
        if not betrayer or not victim:
            return payoff1, payoff2, second_order

        betrayer_name = betrayer.name
        victim_name = victim.name

        non_punisher_ids = []
        vp = punishment_detail.get("victim_punish") or {}
        if vp.get("decision") == "No":
            non_punisher_ids.append(victim_id)

        for bid in punishment_detail.get("bystander_ids", []):
            if not punishment_detail.get("bystander_exposure", {}).get(bid):
                continue
            bp = punishment_detail.get("bystander_punish", {}).get(bid) or {}
            if bp.get("decision") == "No":
                non_punisher_ids.append(bid)

        if not non_punisher_ids:
            return payoff1, payoff2, second_order

        second_order["applies"] = True
        second_order["non_punisher_ids"] = list(non_punisher_ids)

        coros = []

        async def meta_task(meta_id: int, target_id: int):
            meta_agent = self._get_agent_by_id(meta_id)
            target_agent = self._get_agent_by_id(target_id)
            dec, expl = await meta_agent.decide_meta_punishment(
                target_agent.name, betrayer_name, victim_name
            )
            return meta_id, target_id, dec, expl

        for target_id in non_punisher_ids:
            for meta_id in range(self.num_agents):
                if meta_id == target_id:
                    continue
                obs = random.random() < self.exposure_probability
                second_order["meta_exposure"][f"{meta_id}_{target_id}"] = obs
                if obs:
                    coros.append(meta_task(meta_id, target_id))

        if not coros:
            return payoff1, payoff2, second_order

        results = await asyncio.gather(*coros)

        meta_pairs = []
        for meta_id, target_id, dec, expl in results:
            key = f"{meta_id}_{target_id}"
            second_order["meta_decisions"][key] = {"decision": dec, "explanation": expl}
            if dec == "Yes":
                meta_pairs.append((meta_id, target_id))

        second_order["meta_punisher_ids"] = [m for m, _ in meta_pairs]

        meta_target_penalties = defaultdict(int)
        meta_punisher_costs = defaultdict(int)
        for meta_id, target_id in meta_pairs:
            meta_target_penalties[target_id] += self.meta_punishment_to_non_punisher
            meta_punisher_costs[meta_id] += self.meta_execution_cost

        second_order["meta_target_penalties"] = {k: v for k, v in meta_target_penalties.items()}
        second_order["meta_punisher_execution_costs"] = {k: v for k, v in meta_punisher_costs.items()}

        for tid, pen in meta_target_penalties.items():
            if tid == agent1_id:
                payoff1 += pen
            elif tid == agent2_id:
                payoff2 += pen

        for mid, cost in meta_punisher_costs.items():
            if mid == agent1_id:
                payoff1 += cost
            elif mid == agent2_id:
                payoff2 += cost

        return payoff1, payoff2, second_order

    async def run_interaction(self, interaction_num: int) -> dict:
        """
        执行单次交互（基础矩阵收益 + LLM 惩罚决策）
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()

        self.interaction_number = interaction_num

        if interaction_num > len(self.interaction_schedule):
            logging.error("Interaction number exceeds schedule length.")
            return {}

        agent1_id, agent2_id = self.interaction_schedule[interaction_num - 1]

        agent1 = self._get_agent_by_id(agent1_id)
        agent2 = self._get_agent_by_id(agent2_id)

        if not agent1 or not agent2:
            logging.error(f"Agent pair {agent1_id}, {agent2_id} not found.")
            return {}

        interaction_summary = {
            "interaction": interaction_num,
            "pair_ids": (agent1_id, agent2_id),
            "pair_names": (agent1.name, agent2.name),
            "timestamp": datetime.now().isoformat()
        }

        try:
            action1, explanation1 = await agent1.make_decision(agent2._id, agent2.name, self.memory_size)
            action2, explanation2 = await agent2.make_decision(agent1._id, agent1.name, self.memory_size)

            payoff1, payoff2 = get_payoff(action1, action2)

            payoff1, payoff2, punishment_detail = await self._resolve_punishment_with_llm(
                agent1_id, agent2_id, action1, action2, payoff1, payoff2
            )

            payoff1, payoff2, second_order_detail = await self._resolve_second_order_punishment_with_llm(
                punishment_detail, payoff1, payoff2, agent1_id, agent2_id
            )
            punishment_detail["second_order"] = second_order_detail

            agent1.update_state(agent2._id, action1, action2, payoff1, payoff2)
            agent2.update_state(agent1._id, action2, action1, payoff2, payoff1)

            interaction_detail = {
                "agent1_id": agent1_id,
                "agent1_name": agent1.name,
                "agent2_id": agent2_id,
                "agent2_name": agent2.name,
                "choice1": action1,
                "choice2": action2,
                "explanation1": explanation1,
                "explanation2": explanation2,
                "payoff1": payoff1,
                "payoff2": payoff2,
                "cooperation": (action1 == "Yes" and action2 == "Yes"),
                "punishment": punishment_detail,
            }

        except Exception as e:
            logging.error(f"Error in interaction {interaction_num}: {e}")
            return {}

        self.choice_frequency[action1] += 1
        self.choice_frequency[action2] += 1

        if interaction_detail["cooperation"]:
            self.game_stats["cooperation_count"] += 1

        interaction_summary['detail'] = interaction_detail
        self.game_logs.append(interaction_summary)
        self.game_stats['total_interactions'] += 1

        return interaction_summary

    async def save_game_results(self, experiment_name: str = None, experiment_num: int = 1, main_result_dir: str = None):
        """
        保存游戏结果
        """
        from datetime import datetime
        if experiment_name is None:
            experiment_name = f"PrisonersDilemmaPunishment_{datetime.now().strftime('%m%d%H%M')}"

        if main_result_dir:
            result_dir = main_result_dir
        else:
            result_dir = os.path.join("result_prisoners_dilemma_group_punishment", experiment_name)

        data_dir = os.path.join(result_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self._config['max_tick'] = self.total_interactions
        self._config['random_seed'] = RANDOM_SEED

        if experiment_num == 1:
            config_path = os.path.join(data_dir, "experiment_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)

        stats_path = os.path.join(data_dir, f"game_stats{experiment_num}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            self.game_stats['total_interactions'] = len(self.game_logs)
            json.dump(self.game_stats, f, indent=4, ensure_ascii=False)

        states_path = os.path.join(data_dir, f"agent_states{experiment_num}.json")
        with open(states_path, 'w', encoding='utf-8') as f:
            agent_states = {}
            for agent in self.agents:
                agent_states[agent._id] = agent.get_state_summary()

            json.dump(agent_states, f, indent=4, ensure_ascii=False)

        game_logs_path = os.path.join(data_dir, f"game_log{experiment_num}.json")
        with open(game_logs_path, "w", encoding="utf-8") as f:
            json.dump(self.game_logs, f, ensure_ascii=False, indent=2)

        print(f"Game {experiment_num} results saved to: {result_dir}")
        return result_dir


def print_payoff_statistics(env: PrisonersDilemmaSecondOrderPunishmentEnvironment):
    """计算并打印收益（矩阵 + 一阶惩罚 + 二阶元惩罚；非对局双方的侧支付在下方单独累加）"""
    if not env.game_logs:
        print("Warning: No valid game data")
        return

    total_payoffs = {agent.name: 0 for agent in env.agents}
    cooperation_count = 0

    for interaction in env.game_logs:
        detail = interaction['detail']
        total_payoffs[detail['agent1_name']] += detail['payoff1']
        total_payoffs[detail['agent2_name']] += detail['payoff2']
        if detail['cooperation']:
            cooperation_count += 1

        pun = detail.get('punishment') or {}
        exec_costs = pun.get('punisher_execution_costs') or pun.get('bystander_punishment_costs') or {}
        for pid, cost in exec_costs.items():
            agent = env._get_agent_by_id(int(pid))
            if agent:
                total_payoffs[agent.name] += cost

        so = pun.get('second_order') or {}
        if so.get('applies'):
            a1, a2 = detail['agent1_id'], detail['agent2_id']
            for pid, cost in (so.get('meta_punisher_execution_costs') or {}).items():
                pid = int(pid)
                if pid not in (a1, a2):
                    agent = env._get_agent_by_id(pid)
                    if agent:
                        total_payoffs[agent.name] += cost
            for tid, pen in (so.get('meta_target_penalties') or {}).items():
                tid = int(tid)
                if tid not in (a1, a2):
                    agent = env._get_agent_by_id(tid)
                    if agent:
                        total_payoffs[agent.name] += pen

    print("\n===== Basic Payoff Information (含一阶与二阶惩罚) =====")
    print(f"Total interactions: {len(env.game_logs)}")
    print(f"Cooperation rate: {cooperation_count / len(env.game_logs) * 100:.1f}%")
    print("\nTotal payoffs per agent:")
    for agent_name, total in total_payoffs.items():
        print(f"  {agent_name}: {total} pts")
    print(f"\nAverage agent payoff: {sum(total_payoffs.values()) / len(total_payoffs):.1f} pts")
    print(f"Total group payoff: {sum(total_payoffs.values())} pts")


def print_memory_statistics(env):
    """Calculate and print memory statistics"""
    print("\n===== Memory Statistics =====")
    print(f"Total agents: {len(env.agents)}")

    for agent in env.agents:
        memory_summary = agent.get_memory_summary()
        print(f"\nAgent {agent.name}:")
        print(f"  Total agents interacted: {memory_summary['total_agents_interacted']}")
        print(f"  Interaction counts per agent: {memory_summary['agent_interaction_counts']}")


async def main():
    print(f"Random seed set to: {RANDOM_SEED}")

    NUM_AGENTS = 24
    MEMORY_SIZE = 0
    TOTAL_POPULATION_ROUNDS = 10
    TOTAL_INTERACTIONS = TOTAL_POPULATION_ROUNDS * (NUM_AGENTS // 2)

    base_result_dir = "result_prisoners_dilemma_group_second_order_punishment"
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PD_punishment_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = f"PrisonersDilemmaPunishment_{datetime.now().strftime('%m%d%H%M')}"

    main_result_dir = os.path.join(base_result_dir, base_experiment_name)
    os.makedirs(main_result_dir, exist_ok=True)

    try:
        from LLMAPI.wuwen import LLMAgent
        llm_agent = LLMAgent(name="PrisonerDilemmaAgent")
    except ImportError:
        class MockLLMAgent:
            def __init__(self, name): self.name = name
            def __getattr__(self, name): return lambda *args, **kwargs: None
        llm_agent = MockLLMAgent(name="PrisonerDilemmaAgent")

    llm_agent.router = llm_agent
    llm_agent.model_name = getattr(llm_agent, 'model', 'mock_model')

    agent_llm = llm_agent

    NUM_RUNS = 1  # 独立重复实验次数（每次单独随机配对表与完整交互）
    for experiment_index in range(NUM_RUNS):
        experiment_num = experiment_index + 1
        experiment_suffix = f"第{experiment_num}次"
        experiment_name = f"{base_experiment_name}_{experiment_suffix}"

        print("\n" + "=" * 70)
        print(f"Prisoner's Dilemma Group + 1st/2nd-Order Punishment - {experiment_suffix}")
        print("=" * 70)
        print(f"Experiment number: {experiment_num}/{NUM_RUNS}")
        print(f"Number of agents (N): {NUM_AGENTS}")
        print(f"Memory size (H): {MEMORY_SIZE}")
        print(f"Total population rounds: {TOTAL_POPULATION_ROUNDS}")
        print(f"Total interactions: {TOTAL_INTERACTIONS}")
        print(f"Exposure probability (bystanders observe event): {EXPOSURE_PROBABILITY}")
        print(f"Punishment decision: LLM (victim always prompted; exposed bystanders prompted)")
        print(f"Punishment to defector (per punisher): {PUNISHMENT_COST_TO_DEFECTOR}")
        print(f"Execution cost to punisher: {EXECUTION_COST_TO_PUNISHER}")
        print(f"Second-order meta punishment to non-punisher: {META_PUNISHMENT_TO_NON_PUNISHER}")
        print(f"Meta-execution cost: {META_EXECUTION_COST_TO_META_PUNISHER}")
        print(f"Interaction Schedule: Pre-generated using seed {RANDOM_SEED}.")
        print(f"Experiment folder: {experiment_name}")
        print("=" * 70)

        agent_ids = list(range(NUM_AGENTS))
        interaction_schedule = []

        for _ in range(TOTAL_INTERACTIONS):
            p1_id, p2_id = random.sample(agent_ids, 2)
            interaction_schedule.append((p1_id, p2_id))

        env = PrisonersDilemmaSecondOrderPunishmentEnvironment(
            num_agents=NUM_AGENTS,
            memory_size=MEMORY_SIZE,
            total_interactions=TOTAL_INTERACTIONS,
            interaction_schedule=interaction_schedule
        )

        rational_profile = (
            "You are a rational decision-maker. You are playing a game that is played in multiple rounds with different players. "
            "You know your opponent's choices, and your opponent also knows your choices. "
            "In each round, both of you will simultaneously choose one of two actions: Yes (Cooperate) or No (Defect/Betray)."
            "Your goal is to maximize your benefits. "
            "The following is the payoff matrix that determines your benefits: "
            "- If both choose Yes: You get 3 points, [OPPONENT_NAME] gets 3 points "
            "- If you choose Yes and [OPPONENT_NAME] chooses No: You get 0 points, [OPPONENT_NAME] gets 5 points "
            "- If you choose No and [OPPONENT_NAME] chooses Yes: You get 5 points, [OPPONENT_NAME] gets 0 points "
            "- If both choose No: You get 1 point, [OPPONENT_NAME] gets 1 point "
            "Remember this matrix and make decisions to maximize your total points. You must output 1 sentences to explain your decision. "
        )

        agents = []
        for i in range(NUM_AGENTS):
            agent = PrisonersDilemmaAgent(
                id=i,
                name=f"Agent_{i}",
                profile=rational_profile.replace("[OPPONENT_NAME]", "Partner")
            )
            await agent.init(agent_llm)
            agents.append(agent)

        env.set_agents(agents)
        for agent in agents:
            agent.set_environment(env)

        start_time = time.time()

        for interaction_num in range(1, TOTAL_INTERACTIONS + 1):

            if interaction_num % (NUM_AGENTS // 2) == 1:
                current_round = (interaction_num - 1) // (NUM_AGENTS // 2) + 1
                print(f"\n--- Population Round {current_round} | Interaction {interaction_num} ---")

            await env.run_interaction(interaction_num)

        end_time = time.time()

        print(f"\nSimulation {experiment_num} finished.")
        print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Total interactions: {len(env.game_logs)}")

        print_payoff_statistics(env)

        print_memory_statistics(env)

        await env.save_game_results(experiment_name, experiment_num, main_result_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program error occurred: {e}")
        import traceback
        traceback.print_exc()
