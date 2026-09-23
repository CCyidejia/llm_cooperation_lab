#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma Game - Group Experiment Implementation
Based on env_main_囚徒困境baseline.py and 自发涌现.py
Supporting multi-agent interactions with decentralized state management
Enhanced with agent-specific memory mechanism
R-inspired condition: global visibility of each agent's recent public actions
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
from llm_cooperation_lab.agent.base import AgentBase, AgentLLM
from llm_cooperation_lab.config.llm import LLMConfig
from dotenv import load_dotenv

PAYOFF_MATRIX = {
    ("Yes", "Yes"): (3, 3),
    ("Yes", "No"): (0, 5),
    ("No", "Yes"): (5, 0),
    ("No", "No"): (1, 1),
}


def get_payoff(action1: str, action2: str):
    return PAYOFF_MATRIX.get((action1.capitalize(), action2.capitalize()), (0, 0))

# 从游戏规则模块导入收益计算函数

# 加载环境变量
load_dotenv()

# Ensure the reputation-mechanism results directory exists
os.makedirs("result_prisoners_dilemma_group_reputation", exist_ok=True)

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
        # 使用LLMAPI目录下的实现 - profile作为system_message
        response = await asyncio.to_thread(
            self._llm.get_llm_response,
            system_message=self._profile,
            user_prompt=message
        )
        return response
    
    async def step(self, tick: int, t: datetime) -> str:
        """Execute one step"""
        return f"{self.name} step executed at tick {tick}"
    
    def format_public_action_records(
        self,
        partner_id: int,
        reputation_snapshot: dict,
        reputation_history_size: int
    ) -> str:
        """
        Format the round-start public-action snapshot for the decision prompt.

        The focal agent's own record is omitted. Every other agent is shown in
        stable numeric-ID order, and the current partner is explicitly marked.
        """
        if reputation_snapshot is None:
            reputation_snapshot = {}

        displayed_agent_ids = [
            agent_id
            for agent_id in sorted(reputation_snapshot.keys(), key=int)
            if agent_id != self._id
        ]

        if self._env is not None:
            assert len(displayed_agent_ids) == self._env.num_agents - 1
        assert partner_id in displayed_agent_ids

        lines = [
            "Public action records:",
            (
                "For every other agent, you can see their last up to "
                f"{reputation_history_size} completed actions from previous population rounds."
            ),
            '"Yes" means the agent chose to cooperate. "No" means the agent chose not to cooperate.',
            "Records for each agent are ordered from earlier to later.",
            (
                "The agent marked [Your current partner] is the agent you are "
                "interacting with now."
            ),
            (
                "You are not shown other agents' pairings, their payoffs, or "
                "any future pairings."
            ),
            ""
        ]

        for agent_id in displayed_agent_ids:
            agent = self._env._get_agent_by_id(agent_id) if self._env is not None else None
            agent_name = agent.name if agent is not None else f"Agent_{agent_id}"
            if agent_id == partner_id:
                agent_name += " [Your current partner]"

            records = reputation_snapshot.get(agent_id, [])[-reputation_history_size:]
            if records:
                record_text = ", ".join(
                    f"Round {record['round']}: {str(record['action']).capitalize()}"
                    for record in records
                )
            else:
                record_text = "No previous actions available"
            lines.append(f"- {agent_name}: {record_text}")

        return "\n".join(lines)

    async def make_decision(
        self,
        partner_id: int,
        partner_name: str,
        memory_size: int,
        reputation_snapshot: dict = None,
        reputation_history_size: int = 5,
        enable_global_reputation: bool = False
    ) -> tuple:
        """
        Make a decision using the game rules and the information visible at the
        start of the current population round.

        Private partner-specific memory and global public-action records remain
        separate information channels.
        """
        # Private history remains partner-specific. MEMORY_SIZE=0 disables only
        # this private channel; it does not disable the R-group public records.
        history_str = "Private interaction history: Not provided.\n"
        if memory_size > 0 and partner_id in self._agent_memory:
            agent_memory = self._agent_memory[partner_id]
            visible_count = min(memory_size, agent_memory["interaction_count"])
            if visible_count > 0:
                history_str = (
                    f"Private interaction history with {partner_name} "
                    f"(last {visible_count} interactions):\n"
                )
                start = agent_memory["interaction_count"] - visible_count
                for i in range(start, agent_memory["interaction_count"]):
                    my_choice = agent_memory["my_choices"][i].capitalize()
                    their_choice = agent_memory["their_choices"][i].capitalize()
                    my_payoff = agent_memory["my_payoffs"][i]
                    their_payoff = agent_memory["their_payoffs"][i]
                    history_str += (
                        f"  Interaction {i + 1}: Your choice={my_choice}, "
                        f"{partner_name}'s choice={their_choice}, Your payoff={my_payoff}, "
                        f"{partner_name}'s payoff={their_payoff}\n"
                    )

        public_records_str = ""
        if enable_global_reputation:
            public_records_str = self.format_public_action_records(
                partner_id=partner_id,
                reputation_snapshot=reputation_snapshot,
                reputation_history_size=reputation_history_size
            )
        
        # --- 重试参数配置 ---
        max_retries = 5           # 最大重试次数
        base_delay = 2            # 基础等待时间(秒)
        
        action = "No"
        explanation = "LLM调用失败，重试耗尽"

        # 将profile作为system_message，决策任务作为user_prompt
        system_msg = self._profile
        user_msg = (
            f"{history_str}\n"
            f"{public_records_str}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the game rules and "
            "the information shown above, determine your action in the current interaction.\n"
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

        for attempt in range(max_retries):
            try:
                # 调用LLM - profile作为system_message，决策任务作为user_prompt
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    system_message=system_msg,
                    user_prompt=user_msg
                )
                
                # --- 关键修改：检查响应是否为 API 错误信息 ---
                # 你的日志显示错误信息包含 "API Call Failed" 或 "HTTP Error"
                # 如果发现这些关键词，手动抛出异常以触发重试，而不是去解析它
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    # 只有在非最后一次尝试时才打印简单的警告，避免刷屏
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
                    
                    # 解析成功，直接返回
                    return action, explanation
                
                else:
                    # XML 解析失败，尝试单行匹配
                    lines = content.strip().split('\n')
                    first_line = lines[0].strip()
                    match = re.search(r'^\s*(yes|no)\s*[.!]?$', first_line, re.IGNORECASE)
                    
                    if match:
                        action = match.group(1).capitalize()
                        explanation = ' '.join(lines[1:]).strip() or "单行匹配成功"
                        return action, explanation
                    
                    # 如果内容不是错误信息，但无法解析，也视为一次失败尝试，抛出异常以重试
                    # 有时候LLM只是发疯没按格式写，重试一次可能就好了
                    raise ValueError(f"无法解析有效格式: {content[:50]}...")

            except Exception as e:
                # 计算指数退避时间 (2s, 4s, 8s, ...) 加上一点随机抖动防止并发冲突
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                
                if attempt < max_retries - 1:
                    print(f"[{self.name}] 请求或解析出错: {type(e).__name__}。将在 {sleep_time:.2f}秒后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    # 重试耗尽，记录最终错误
                    error_message = f"重试耗尽。错误: {str(e)}"
                    print(f"[{self.name}] [ERROR] {error_message}")
                    logging.error(f"[{self.name}] {error_message}")
        
        # 保存选择
        self._previous_choices.append(action)
        return action, explanation
    
    def _get_stable_choice(self, options: list) -> str:
        """
        增强选择稳定性的逻辑：
        1. 优先选择最频繁的之前选择
        2. 如果没有一致的之前选择，则选择环境中最流行的选项
        3. 最后才使用随机选择作为最后的备选
        """
        # 1. 优先选择最频繁的之前选择
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            # 获取最频繁的选择
            most_common_choice, count = choice_counter.most_common(1)[0]
            # 如果最频繁的选择出现次数超过总次数的一半，则使用它
            if count > len(self._previous_choices) / 2 and most_common_choice in options:
                logging.info(f"[{self.name}] 使用最频繁的之前选择: {most_common_choice}")
                return most_common_choice
        
        # 2. 如果没有一致的之前选择，则选择环境中最流行的选项
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            # 检查环境中哪个选项最流行
            if self._env.choice_frequency:
                # 获取最流行的选项
                most_popular_choice, count = self._env.choice_frequency.most_common(1)[0]
                if most_popular_choice in options:
                    logging.info(f"[{self.name}] 使用环境中最流行的选项: {most_popular_choice}")
                    return most_popular_choice
        
        # 3. 最后才使用随机选择作为最后的备选
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

class PrisonersDilemmaEnvironment:
    """
    管理囚徒困境群体实验的环境
    """
    def __init__(
        self,
        num_agents: int,
        memory_size: int,
        total_interactions: int,
        interaction_schedule: list,
        reputation_history_size: int = 5,
        enable_global_reputation: bool = True
    ):
        
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.total_interactions = total_interactions
        self.agents = [] # List of Agent objects
        self.interaction_schedule = interaction_schedule # Pre-generated pairs of (Agent ID 1, Agent ID 2)
        self.reputation_history_size = reputation_history_size
        self.enable_global_reputation = enable_global_reputation
        self.interactions_per_population_round = num_agents // 2
        
        self.interaction_number = 0 # Track sequential interactions
        self.initial_time = None
        self.current_population_round = None
        
        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'memory_size': memory_size,
            'interactions_per_run': 1,
            'mechanism_name': 'R-inspired Global Reputation Information',
            'enable_global_reputation': enable_global_reputation,
            'reputation_history_size': reputation_history_size,
            'reputation_snapshot_timing': 'start_of_population_round',
            'visible_public_information': (
                f'Each other agent ID and its last up to {reputation_history_size} '
                'completed actions; the current partner is marked.'
            )
        }
        
        self.game_stats = {'total_interactions': 0, 'cooperation_count': 0}
        self.game_logs = []
        self.choice_frequency = Counter()  # 统计每个选项的选择频率
        self.reset_public_action_history()
    
    def set_agents(self, agents: list):
        self.agents = agents

    def _get_agent_by_id(self, agent_id):
        """Helper to get Agent object from ID."""
        return next((a for a in self.agents if a._id == agent_id), None)

    def reset_public_action_history(self):
        """Clear all public records and round snapshots for an independent run."""
        self.public_action_history = {
            agent_id: []
            for agent_id in range(self.num_agents)
        }
        self.current_reputation_snapshot = {
            agent_id: []
            for agent_id in range(self.num_agents)
        }
        self.reputation_snapshots = {}
        self.current_population_round = None

    def create_reputation_snapshot(self) -> dict:
        """
        Freeze the last completed public actions before a population round.

        A copy is returned so actions completed later in the same population
        round cannot become visible until the next round.
        """
        return {
            agent_id: [
                dict(record)
                for record in history[-self.reputation_history_size:]
            ]
            for agent_id, history in self.public_action_history.items()
        }

    def _validate_reputation_snapshot(self, snapshot: dict, current_round: int):
        """Assert the R-group information boundary before decisions are made."""
        assert set(snapshot.keys()) == set(range(self.num_agents))
        for records in snapshot.values():
            assert len(records) <= self.reputation_history_size
            assert all(record["round"] < current_round for record in records)
            assert all(record["action"] in {"Yes", "No"} for record in records)

    def start_population_round(self, population_round: int):
        """
        Freeze one shared snapshot for every interaction in this population
        round. Only actions from earlier population rounds are included.
        """
        if population_round < 1:
            raise ValueError("population_round must be at least 1")

        snapshot = self.create_reputation_snapshot()
        self._validate_reputation_snapshot(snapshot, population_round)
        self.current_population_round = population_round
        self.current_reputation_snapshot = snapshot
        self.reputation_snapshots[population_round] = {
            agent_id: [dict(record) for record in records]
            for agent_id, records in snapshot.items()
        }

    def record_public_actions(self, population_round: int, actions: dict):
        """Record completed actions without changing the frozen current snapshot."""
        if population_round != self.current_population_round:
            raise ValueError(
                "Public actions must be recorded in the active population round."
            )

        for agent_id, action in actions.items():
            normalized_action = str(action).capitalize()
            if agent_id not in self.public_action_history:
                raise KeyError(f"Unknown agent ID: {agent_id}")
            if normalized_action not in {"Yes", "No"}:
                raise ValueError(f"Invalid public action: {action}")
            self.public_action_history[agent_id].append({
                "round": population_round,
                "action": normalized_action
            })
    
    async def run_interaction(self, interaction_num: int) -> dict:
        """
        执行单次交互
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()
        
        self.interaction_number = interaction_num
        population_round = (
            (interaction_num - 1) // self.interactions_per_population_round
        ) + 1

        # Defensive fallback: main() normally starts each round explicitly.
        if self.current_population_round != population_round:
            self.start_population_round(population_round)
        
        if interaction_num > len(self.interaction_schedule):
            logging.error("Interaction number exceeds schedule length.")
            return {}

        # 1. Get Agent IDs from the pre-generated schedule
        # Schedule index is interaction_num - 1
        agent1_id, agent2_id = self.interaction_schedule[interaction_num - 1]
        
        agent1 = self._get_agent_by_id(agent1_id)
        agent2 = self._get_agent_by_id(agent2_id)
        
        if not agent1 or not agent2:
            logging.error(f"Agent pair {agent1_id}, {agent2_id} not found.")
            return {}
            
        interaction_summary = {
            "interaction": interaction_num,
            "population_round": population_round,
            "pair_ids": (agent1_id, agent2_id),
            "pair_names": (agent1.name, agent2.name),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 2. 两个Agent读取完全相同的轮首快照。即使API调用按顺序执行，
            #    Agent 2也无法看到Agent 1在当前互动中的选择。
            action1, explanation1 = await agent1.make_decision(
                agent2._id,
                agent2.name,
                self.memory_size,
                self.current_reputation_snapshot,
                self.reputation_history_size,
                self.enable_global_reputation
            )
            action2, explanation2 = await agent2.make_decision(
                agent1._id,
                agent1.name,
                self.memory_size,
                self.current_reputation_snapshot,
                self.reputation_history_size,
                self.enable_global_reputation
            )
            
            # 3. 计算收益
            payoff1, payoff2 = get_payoff(action1, action2)
            
            # 4. --- 自主状态管理：更新代理的内部状态 ---
            # Agent 1 更新状态
            agent1.update_state(agent2._id, action1, action2, payoff1, payoff2)
            
            # Agent 2 更新状态
            agent2.update_state(agent1._id, action2, action1, payoff2, payoff1)
            # -------------------------------------------------

            # 5. 当前行为写入正式公开历史，但本轮剩余互动仍读取轮首快照。
            self.record_public_actions(
                population_round,
                {
                    agent1_id: action1,
                    agent2_id: action2
                }
            )
            
            # 6. 更新环境统计数据
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
                "public_records_snapshot_round": population_round,
                "public_records_latest_visible_round": population_round - 1
            }
            
        except Exception as e:
            logging.error(f"Error in interaction {interaction_num}: {e}")
            return {}
            
        # 更新选项频率统计
        self.choice_frequency[action1] += 1
        self.choice_frequency[action2] += 1
        
        # 更新合作统计
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
            experiment_name = f"PrisonersDilemma_{datetime.now().strftime('%m%d%H%M')}"
        
        # 使用统一的主结果文件夹
        if main_result_dir:
            result_dir = main_result_dir
        else:
            result_dir = os.path.join(
                "result_prisoners_dilemma_group_reputation",
                experiment_name
            )
        
        data_dir = os.path.join(result_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        self._config['max_tick'] = self.total_interactions
        self._config['random_seed'] = RANDOM_SEED # Record the used seed
        
        # 仅在第一次实验时保存配置文件
        if experiment_num == 1:
            config_path = os.path.join(data_dir, "experiment_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
        # 保存每轮实验的统计数据
        stats_path = os.path.join(data_dir, f"game_stats{experiment_num}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            self.game_stats['total_interactions'] = len(self.game_logs)
            json.dump(self.game_stats, f, indent=4, ensure_ascii=False)
            
        # 保存每轮实验的智能体状态
        states_path = os.path.join(data_dir, f"agent_states{experiment_num}.json")
        with open(states_path, 'w', encoding='utf-8') as f:
            # 从每个代理获取状态而不是从环境字典
            agent_states = {}
            for agent in self.agents:
                agent_states[agent._id] = agent.get_state_summary()
            
            # 将agent_states保存到文件
            json.dump(agent_states, f, indent=4, ensure_ascii=False)

        # Save game logs with numbered filenames
        game_logs_path = os.path.join(data_dir, f"game_log{experiment_num}.json")
        with open(game_logs_path, "w", encoding="utf-8") as f:
            json.dump(self.game_logs, f, ensure_ascii=False, indent=2)

        # Save a separate audit file so the original result files and their
        # primary structures remain available for existing analysis scripts.
        reputation_audit_path = os.path.join(
            data_dir,
            f"reputation_audit{experiment_num}.json"
        )
        reputation_audit = {
            "snapshot_timing": "start_of_population_round",
            "history_size": self.reputation_history_size,
            "snapshots": self.reputation_snapshots,
            "final_public_action_history": self.public_action_history
        }
        with open(reputation_audit_path, "w", encoding="utf-8") as f:
            json.dump(reputation_audit, f, ensure_ascii=False, indent=2)
            
        print(f"Game {experiment_num} results saved to: {result_dir}")
        return result_dir

# 简化的收益统计函数
def print_payoff_statistics(env):
    """Calculate and print basic payoff statistics"""
    if not env.game_logs:
        print("Warning: No valid game data")
        return
    
    # 计算总收益
    total_payoffs = {agent.name: 0 for agent in env.agents}
    cooperation_count = 0
    
    # 累计总收益
    for interaction in env.game_logs:
        detail = interaction['detail']
        total_payoffs[detail['agent1_name']] += detail['payoff1']
        total_payoffs[detail['agent2_name']] += detail['payoff2']
        if detail['cooperation']:
            cooperation_count += 1
    
    # 仅打印基本信息
    print("\n===== Basic Payoff Information =====")
    print(f"Total interactions: {len(env.game_logs)}")
    print(f"Cooperation rate: {cooperation_count / len(env.game_logs) * 100:.1f}%")
    print("\nTotal payoffs per agent:")
    for agent_name, total in total_payoffs.items():
        print(f"  {agent_name}: {total} pts")
    print(f"\nAverage agent payoff: {sum(total_payoffs.values()) / len(total_payoffs):.1f} pts")
    print(f"Total group payoff: {sum(total_payoffs.values())} pts")

# 增强的记忆统计函数
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
    
    # --- 实验参数设置 ---
    NUM_AGENTS = 24  # 代理数量
    MEMORY_SIZE = 0  # 私人、当前对手特定的记忆；R组公开记录与其相互独立
    REPUTATION_HISTORY_SIZE = 5  # 每名其他Agent最多展示最近5次已完成行为
    ENABLE_GLOBAL_REPUTATION = True  # 启用R组全局个体行为历史可见机制
    TOTAL_POPULATION_ROUNDS = 10  # 总群体回合数
    TOTAL_INTERACTIONS = TOTAL_POPULATION_ROUNDS * (NUM_AGENTS // 2)  # 总交互次数
    
    # --- 用户自定义文件夹名称 ---
    base_result_dir = "result_prisoners_dilemma_group_reputation"
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PD_group_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = f"PrisonersDilemma_{datetime.now().strftime('%m%d%H%M')}"

    REPEAT_TIMES = 3
    
    # 创建统一的主结果文件夹
    main_result_dir = os.path.join(base_result_dir, base_experiment_name)
    os.makedirs(main_result_dir, exist_ok=True)
    
    # --- 导入LLMAgent ---
    try:
        from LLMAPI.zgc import LLMAgent
        llm_agent = LLMAgent(name="PrisonerDilemmaAgent")
    except ImportError:
        # 如果silicon模块不存在，则创建一个模拟对象
        class MockLLMAgent:
            def __init__(self, name): self.name = name
            def __getattr__(self, name): return lambda *args, **kwargs: None
        llm_agent = MockLLMAgent(name="PrisonerDilemmaAgent")

    # 设置LLMAgent以兼容AgentLLM接口
    llm_agent.router = llm_agent
    # 添加model_name属性以兼容AgentLLM接口
    llm_agent.model_name = getattr(llm_agent, 'model', 'mock_model') # 使用 getattr 避免 mock 失败
    
    # 直接使用LLMAgent作为AgentLLM接口的实现
    agent_llm = llm_agent
    
    # --- 运行多次独立实验 ---
    for experiment_index in range(REPEAT_TIMES):
        experiment_num = experiment_index + 1
        experiment_suffix = f"第{experiment_num}次"
        experiment_name = f"{base_experiment_name}_{experiment_suffix}"
        
        print("\n" + "=" * 70)
        print(f"Prisoner's Dilemma Group Experiment - {experiment_suffix}")
        print("=" * 70)
        print(f"Experiment number: {experiment_num}/{REPEAT_TIMES}")
        print(f"Number of agents (N): {NUM_AGENTS}")
        print(f"Private partner-specific memory size (H): {MEMORY_SIZE}")
        print(f"Global public-action records enabled: {ENABLE_GLOBAL_REPUTATION}")
        print(f"Public-action history size: {REPUTATION_HISTORY_SIZE}")
        print(f"Total population rounds: {TOTAL_POPULATION_ROUNDS}")
        print(f"Total interactions: {TOTAL_INTERACTIONS}")
        print(f"Interaction Schedule: Pre-generated using seed {RANDOM_SEED}.")
        print(f"Experiment folder: {experiment_name}")
        print("=" * 70)
        
        # --- 预生成互动时间表 ---
        # 使用固定随机种子确保序列确定性
        agent_ids = list(range(NUM_AGENTS))
        interaction_schedule = []
        
        # Generate the schedule for TOTAL_INTERACTIONS pairs
        for _ in range(TOTAL_INTERACTIONS):
            # Sample two *different* agents (p1 != p2 is implicit in random.sample(..., 2))
            p1_id, p2_id = random.sample(agent_ids, 2)
            interaction_schedule.append((p1_id, p2_id))
        
        # --- 初始化环境 ---
        env = PrisonersDilemmaEnvironment(
            num_agents=NUM_AGENTS,
            memory_size=MEMORY_SIZE,
            total_interactions=TOTAL_INTERACTIONS,
            interaction_schedule=interaction_schedule,
            reputation_history_size=REPUTATION_HISTORY_SIZE,
            enable_global_reputation=ENABLE_GLOBAL_REPUTATION
        )
        
        # 设置通用的理性人 Profile，强化自私目标
        rational_profile = (
            "You are playing a game that is played in multiple rounds. "
            "Your goal is to maximize your benefits. " 
            "In each interaction, you and your partner choose simultaneously, so neither "
            "player can observe the other's current choice before deciding. "
            "After each interaction, your action is added to your public action record "
            "and becomes visible to all agents starting from the next population round. "
            "You can also observe other agents' completed public actions, and agents may "
            "condition future decisions on these records. "
            "The following is the payoff matrix that determines your benefits: "
            "- If both choose Yes: You get 3 points, [OPPONENT_NAME] gets 3 points "
            "- If you choose Yes and [OPPONENT_NAME] chooses No: You get 0 points, [OPPONENT_NAME] gets 5 points "
            "- If you choose No and [OPPONENT_NAME] chooses Yes: You get 5 points, [OPPONENT_NAME] gets 0 points "
            "- If both choose No: You get 1 point, [OPPONENT_NAME] gets 1 point "
            "Use only the rules and information provided to make your decision. You must output 1 sentences to explain your decision. "
        )
        
        # --- 初始化代理 ---
        agents = []
        for i in range(NUM_AGENTS):
            agent = PrisonersDilemmaAgent(
                id=i,
                name=f"Agent_{i}",
                profile=rational_profile.replace("[OPPONENT_NAME]", "Partner")
            )
            await agent.init(agent_llm)
            agents.append(agent)
        
        # 设置代理到环境
        env.set_agents(agents)
        # 设置环境引用到每个智能体
        for agent in agents:
            agent.set_environment(env)
        
        # --- 运行模拟 ---
        start_time = time.time()
        
        for interaction_num in range(1, TOTAL_INTERACTIONS + 1):
            
            # 每个Population Round包含NUM_AGENTS//2次交互
            if interaction_num % (NUM_AGENTS // 2) == 1:
                current_round = (interaction_num - 1) // (NUM_AGENTS // 2) + 1
                env.start_population_round(current_round)
                print(f"\n--- Population Round {current_round} | Interaction {interaction_num} ---")
            
            await env.run_interaction(interaction_num)
                
        end_time = time.time()
        
        # --- 最终统计 ---
        print(f"\nSimulation {experiment_num} finished.")
        print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Total interactions: {len(env.game_logs)}")
        
        # 打印收益统计
        print_payoff_statistics(env)
        
        # 打印记忆统计
        print_memory_statistics(env)
        
        # 保存结果
        await env.save_game_results(experiment_name, experiment_num, main_result_dir)

if __name__ == "__main__":
    try:
        # 启动异步事件循环
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program error occurred: {e}")
        import traceback
        traceback.print_exc()
