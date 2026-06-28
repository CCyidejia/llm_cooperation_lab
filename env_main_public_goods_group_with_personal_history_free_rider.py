#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Public Goods Game Group Experiment with Personal History and Free Riders - agentsociety V2 Platform Implementation
Based on env_main_公共物品博弈群体实验_with_personal_history.py with 22 LLM agents and 2 rule-based free-rider agents
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
from dotenv import load_dotenv
from LLMAPI.LLM_API import LLMAgent

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_public_goods_group", exist_ok=True)

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

class PublicGoodsAgent(AgentBase):
    """
    Agent for Public Goods Game that manages its own state,
    compatible with 24-agent group architecture.
    """
    
    def __init__(self, id: int, name: str, profile: str = ""):
        """ Initialize PublicGoodsAgent """
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []  # Track previous choices for stability
        
        # 自主状态管理：代理独立维护交互历史和状态
        self._my_history = []        # Own contribution history
        self._outcome_history = []   # Outcomes history (payoffs)
        self.history = []  # Store complete round summaries for history display
        
        self.initial_endowment = 20  # Initial coins per round
        self.max_contribution = 20  # Maximum contribution amount
        self.min_contribution = 0  # Minimum contribution amount
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    # 自主状态管理：添加更新内部状态的方法
    def update_state(self, my_contribution: int, outcome: int):
        """ Update agent's internal state """
        # Add choice history and outcome
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)
        
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={outcome}")
    
    def update_history(self, round_summary: dict):
        """Update agent's history with complete round summary"""
        self.history.append(round_summary)
    
    def get_state_summary(self) -> dict:
        """ Get agent's current state summary """
        return {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history)
        }
    
    def _build_history_string(self, all_agent_names: list) -> str:
        """Build history string with only total contribution and public gain"""
        if not self.history:
            return "No previous rounds have been played."
        
        history_lines = []
        for round_summary in self.history:
            r = round_summary["round"]
            total_contrib = round_summary.get("total_contribution", 0)
            gain_per_agent = round_summary.get("gain_per_agent", 0)
            history_lines.append(f"Round {r}: Total contribution = {total_contrib} coins, Gain per agent = {gain_per_agent:.2f} coins")
        
        return "\n".join(history_lines)
    
    def _build_personal_history_string(self) -> str:
        """Build personal history string with individual contributions and payoffs"""
        if not self.history:
            return ""
        
        history_lines = []
        history_lines.append("Your history:")
        for round_summary in self.history:
            r = round_summary["round"]
            my_contribution = round_summary["contributions"].get(self.name, 0)
            # Calculate public gain (equal for all agents)
            public_gain = round_summary["gain_per_agent"]
            # Calculate net payoff: public gain minus personal contribution
            net_payoff = public_gain - my_contribution
            
            history_lines.append(f"Round {r}:")
            history_lines.append(f"  Your contribution: {my_contribution} coins.")
            history_lines.append(f"  Your payoff from public fund: {public_gain:.2f} coins.")
            history_lines.append(f"  Your net payoff this round: {net_payoff:.2f} coins.")
        
        return "\n".join(history_lines)
    
    async def init(self, llm: AgentLLM):
        self._llm = llm
    
    @property
    def name(self):
        return self._name
    
    async def dump(self) -> dict:
        return {"id": self._id, "name": self._name, "profile": self._profile}
    
    async def load(self, dump_data: dict):
        self._id = dump_data.get("id", self._id)
        self._name = dump_data.get("name", self._name)
        self._profile = dump_data.get("profile", self._profile)
    
    async def ask(self, message: str, readonly: bool = True) -> str:
        return f"[Agent {self.name}] Received message: {message}"
    
    async def step(self, tick: int, t: datetime) -> str:
        return ""
    
    async def act(self, interaction_num: int, role: str = "", initial_funds: int = 0, 
                  multiplication_factor: float = 0.0, num_rounds: int = 0, 
                  current_funds: int = 0, partner_name: str = "") -> tuple:
        """Agent action implementation with personal history tracking"""
        
        # 获取环境参数
        num_agents = 24  # 固定24个代理
        initial_endowment = 20
        public_pool_multiplier = 1.2
        current_round_true = interaction_num
        
        # 1. 构建个人历史记录
        personal_history = self._build_personal_history_string()
        
        # 2. 构建公共历史记录
        public_history = self._build_history_string([agent.name for agent in self._env.agents])
        
        # 3. 构建提示词
        prompt_template = f"""
        You are a participant in a game. Your goal is to maximize your own accumulated coins.
        
        Game Rules:
        - You have {initial_endowment} coins each round.
        - You can contribute any integer amount between 0 and {initial_endowment} (inclusive).
        - All contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.
        - Your total coins = (your initial coins - your contribution) + (total contributions * {public_pool_multiplier} / {num_agents}).
        
        Current Game State:
        This is round {current_round_true}.
        You have {initial_endowment} coins available to contribute.
        
        {personal_history}
        
        {public_history}
        
        ***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.
        Your decision must be an integer between 0 and {initial_endowment} (inclusive).
        State the integer amount first, followed by a brief 1-2 sentence explanation.
        """
        
        system_message = ""
        
        # 4. 调用LLM
        try:
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=prompt_template
            )
            
            # 5. 解析响应
            contribution = 0  # 默认贡献
            explanation = "LLM call or parsing failed"
            
            # 尝试在响应中找到贡献值
            # 首先尝试：查找数字后跟可选的解释
            match = re.search(r'(\d+)\s*[-–—]?\s*(.*)$', content, re.DOTALL)
            
            if match:
                # 提取贡献值
                parsed_contribution = int(match.group(1))
                
                # 验证范围
                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                else:
                    logging.warning(f"[{self.name}] Contribution value out of range: {parsed_contribution}, using default 0")
                    contribution = 0
                
                # 提取解释
                if match.group(2):
                    explanation = match.group(2).strip()
                else:
                    # 如果没有找到解释，尝试从响应中获取更多上下文
                    lines = content.split('\n')
                    for line in lines:
                        if str(contribution) in line:
                            # 获取数字后的行内容
                            num_idx = line.find(str(contribution))
                            if num_idx != -1:
                                line_explanation = line[num_idx + len(str(contribution)):].strip()
                                if line_explanation and not line_explanation.startswith((':', '-', '—')):
                                    explanation = line_explanation
                                    break
                    if explanation == "LLM call or parsing failed":
                        explanation = "No explanation provided"
            else:
                # 尝试在整个响应中搜索数字，作为最后的兜底方案
                keyword_match = re.search(r'\b(\d+)\b', content)
                if keyword_match:
                    parsed_contribution = int(keyword_match.group(1))
                    if self.min_contribution <= parsed_contribution <= self.max_contribution:
                        contribution = parsed_contribution
                    else:
                        logging.warning(f"[{self.name}] Keyword matched value out of range: {parsed_contribution}, using default 0")
                        contribution = 0
                    
                    # 尝试从上下文中提取更好的解释
                    num_start = keyword_match.start()
                    num_end = keyword_match.end()
                    
                    # 尝试从同一行获取解释
                    lines = content.split('\n')
                    for line in lines:
                        if keyword_match.group(1) in line:
                            # 获取数字后的行内容
                            line_explanation = line[line.find(keyword_match.group(1)) + len(keyword_match.group(1)):].strip()
                            if line_explanation:
                                explanation = line_explanation
                                break
                    
                    if explanation == "LLM call or parsing failed":
                        explanation = f"Extracted contribution: {contribution}"
                else:
                    # 最终默认策略：选择最小贡献量
                    raise ValueError(f"Failed to parse valid contribution, content:\n{content[:200]}")
            
        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            contribution = 0
            explanation = f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, using default selection: 0"
        
        # 添加当前选择到历史记录以保持稳定性
        self._previous_choices.append(contribution)
        
        return contribution, explanation
    
    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        """Call LLM with retry mechanism"""
        for attempt in range(max_retries):
            try:
                # Call LLM using asyncio.to_thread to handle synchronous LLM calls
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                if generated_text:
                    return generated_text
                else:
                    raise ValueError("LLM returned empty string")
            except Exception as e:
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise

class FreeRiderAgent(AgentBase):
    """
    Rule-based free-rider agent for Public Goods Game
    Always contributes 0 to maximize personal gain
    """

    
    def __init__(self, id: int, name: str, profile: str = ""):
        """ Initialize FreeRiderAgent """
        super().__init__(id, profile)
        self._name = name
        self._env = None
        self._previous_choices = []  # Track previous choices
        
        # 自主状态管理：代理独立维护交互历史和状态
        self._my_history = []        # Own contribution history
        self._outcome_history = []   # Outcomes history (payoffs)
        self.history = []  # Store complete round summaries for history display
        
        self.initial_endowment = 20  # Initial coins per round
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    # 自主状态管理：添加更新内部状态的方法
    def update_state(self, my_contribution: int, outcome: int):
        """ Update agent's internal state """
        # Add choice history and outcome
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)
        
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={outcome}")
    
    def update_history(self, round_summary: dict):
        """Update agent's history with complete round summary"""
        self.history.append(round_summary)
    
    def get_state_summary(self) -> dict:
        """ Get agent's current state summary """
        return {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history)
        }
    
    async def init(self, llm: AgentLLM):
        # Free rider doesn't need LLM
        pass
    
    @property
    def name(self):
        return self._name
    
    async def dump(self) -> dict:
        return {"id": self._id, "name": self._name, "profile": self._profile}
    
    async def load(self, dump_data: dict):
        self._id = dump_data.get("id", self._id)
        self._name = dump_data.get("name", self._name)
        self._profile = dump_data.get("profile", self._profile)
    
    async def ask(self, message: str, readonly: bool = True) -> str:
        return f"[Agent {self.name}] Received message: {message}"
    
    async def step(self, tick: int, t: datetime) -> str:
        return ""
    
    async def act(self, interaction_num: int, role: str = "", initial_funds: int = 0, 
                  multiplication_factor: float = 0.0, num_rounds: int = 0, 
                  current_funds: int = 0, partner_name: str = "") -> tuple:
        """ Always contribute 0 (free-ride strategy) """
        contribution = 0  # Always free-ride
        explanation = "Free-rider strategy: always contribute 0 to maximize personal gain"
        self._previous_choices.append(contribution)
        return contribution, explanation

class PublicGoodsEnvironment:
    """
    Environment for Public Goods Game Group Experiment with 24 agents
    where all agents participate simultaneously in each round.
    """
    
    def __init__(self, num_agents: int, initial_endowment: int, public_pool_multiplier: float,
                 total_interactions: int):
        
        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions
        
        self.agents = [] # List of Agent objects
        self.interaction_number = 0 # Track sequential interactions
        self.initial_time = None
        
        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'public_pool_multiplier': public_pool_multiplier,
            'random_seed': RANDOM_SEED
        }
        
        # 统计数据
        self.game_stats = {
            'total_interactions': 0,
            'success_count': 0,
            'success_rate': 0.0
        }
        
        self.game_logs = []
        self.choice_frequency = Counter()
    
    def set_agents(self, agents):
        """ Set the agents in the environment """
        self.agents = agents
        for agent in self.agents:
            agent.set_environment(self)
    
    async def run_interaction(self, interaction_num: int):
        """ Run a single interaction round with all agents """
        
        if not self.agents:
            logging.error("No agents set in environment")
            return
        
        # 1. 获取所有代理的选择
        agent_contributions = {}
        explanations = {}
        
        for agent in self.agents:
            try:
                # 调用代理的act方法获取选择
                contribution, explanation = await agent.act(interaction_num)
                agent_contributions[agent._id] = contribution
                explanations[agent._id] = explanation
                
                # 记录选择频率
                self.choice_frequency[contribution] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_contributions[agent._id] = 0  # Default to 0 contribution
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency[0] += 1
        
        # 2. 验证贡献金额
        for agent_id, contribution in agent_contributions.items():
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                logging.warning(f"Invalid contribution {contribution} from agent {agent_id}, setting to 0")
                agent_contributions[agent_id] = 0
                explanations[agent_id] += " [Corrected to 0 due to invalid value]"
        
        # 3. 计算总贡献和收益
        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents
        
        # 4. 计算所有代理的收益
        payoffs = {}
        for agent in self.agents:
            agent_id = agent._id
            contribution = agent_contributions[agent_id]
            private_savings = self.initial_endowment - contribution
            total_gain = private_savings + gain_per_agent
            payoffs[agent_id] = total_gain
        
        # 5. 更新代理状态
        # 构建轮次摘要用于历史记录
        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "gain_per_agent": gain_per_agent,
            "contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents}
        }
        
        for agent in self.agents:
            agent_contribution = agent_contributions[agent._id]
            agent.update_state(
                my_contribution=agent_contribution,
                outcome=payoffs[agent._id]
            )
            # 更新代理的历史记录
            agent.update_history(round_summary)
        
        # 6. 记录成功
        self.game_stats['total_interactions'] += 1
        
        # 7. 记录交互
        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "agent_contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
            "timestamp": datetime.now().isoformat()
        }
        
        self.game_logs.append(interaction_log)
        
        return interaction_log
    
    def get_game_summary(self) -> dict:
        """Get summary of the game statistics"""
        return {
            'total_interactions': self.game_stats['total_interactions'],
            'success_count': self.game_stats['success_count'],
            'success_rate': self.game_stats['success_count'] / self.game_stats['total_interactions'] if self.game_stats['total_interactions'] > 0 else 0,
            'average_contribution': sum(k * v for k, v in self.choice_frequency.items()) / sum(self.choice_frequency.values()) if sum(self.choice_frequency.values()) > 0 else 0,
            'choice_distribution': dict(self.choice_frequency)
        }

async def main():
    """Main function to run the experiment"""
    # ------- Experiment Settings -------
    EXPERIMENT_SETTINGS = {
    "num_agents": 24,  # Number of agents (24 for group experiment)
    "initial_endowment": 20,  # Initial coins per agent per round
    "public_pool_multiplier": 1.2,  # Multiplier for public fund
    "num_rounds": 10,  # Number of rounds per game
    "random_seed": RANDOM_SEED
}
    
    # ------- Create results directory -------
    base_result_dir = "result_public_goods_group"
    target_dir = os.path.join(base_result_dir, "llama3-70b_m=1.2_with_personal_history_free_rider")
    experiment_time = input("Please enter experiment folder name (e.g., 'PG_with_personal_history_free_rider'): ").strip()
    if not experiment_time:
        experiment_time = datetime.now().strftime("%m%d_%H%M%S_PG")
    experiment_result_dir = os.path.join(target_dir, f"result_{experiment_time}")
    os.makedirs(experiment_result_dir, exist_ok=True)
    
    # ------- Set up LLM -------
    # Create LLMAgent instance
    llm_agent = LLMAgent()
    
    # Create AgentLLM wrapper with required parameters
    model_name = getattr(llm_agent, 'model', 'default_model')
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)
    
    # ------- Create Agents -------
    agents = []
    
    # Create 8 LLM-based agents
    for i in range(8):
        # Create agent with simple profile
        profile = f"You are a participant in a game. Your goal is to maximize your own accumulated coins."
        agent = PublicGoodsAgent(id=i+1, name=f"LLM_Agent_{i+1}", profile=profile)
        await agent.init(agent_llm)
        agents.append(agent)
    
    # Create 16 rule-based free-rider agents
    for i in range(8, 24):
        profile = f"You are a free-rider in a game. Your strategy is to always contribute 0."
        agent = FreeRiderAgent(id=i+1, name=f"FreeRider_Agent_{i-7}", profile=profile)
        agents.append(agent)
    
    # ------- Create Environment -------
    env = PublicGoodsEnvironment(
        num_agents=EXPERIMENT_SETTINGS["num_agents"],
        initial_endowment=EXPERIMENT_SETTINGS["initial_endowment"],
        public_pool_multiplier=EXPERIMENT_SETTINGS["public_pool_multiplier"],
        total_interactions=EXPERIMENT_SETTINGS["num_rounds"]
    )
    env.set_agents(agents)
    
    # ------- Run Experiment -------
    print(f"Starting Public Goods Game Group Experiment with {EXPERIMENT_SETTINGS['num_agents']} agents...")
    print(f"  - 8 LLM-based agents")
    print(f"  - 16 rule-based free-rider agents")
    print(f"  - {EXPERIMENT_SETTINGS['num_rounds']} rounds")
    print(f"Results will be saved to: {experiment_result_dir}")
    
    start_time = time.time()
    
    for round_num in range(1, EXPERIMENT_SETTINGS["num_rounds"] + 1):
        print(f"\nRunning round {round_num}/{EXPERIMENT_SETTINGS['num_rounds']}...")
        await env.run_interaction(round_num)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\nExperiment completed in {total_duration:.2f} seconds!")
    
    # ------- Save Results -------
    print(f"\nSaving results to {experiment_result_dir}...")
    
    # 1. Save experiment configuration
    config_file = os.path.join(experiment_result_dir, "experiment_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(EXPERIMENT_SETTINGS, f, ensure_ascii=False, indent=2)
    
    # 2. Save game logs
    logs_file = os.path.join(experiment_result_dir, "game_logs.json")
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(env.game_logs, f, ensure_ascii=False, indent=2)
    
    # 3. Save game statistics
    stats_file = os.path.join(experiment_result_dir, "game_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(env.get_game_summary(), f, ensure_ascii=False, indent=2)
    
    # 4. Save agent states
    agent_states = []
    for agent in agents:
        agent_states.append({
            "id": agent._id,
            "name": agent.name,
            "profile": agent._profile,
            "history": agent.history,
            "state_summary": agent.get_state_summary()
        })
    
    agent_states_file = os.path.join(experiment_result_dir, "agent_states.json")
    with open(agent_states_file, "w", encoding="utf-8") as f:
        json.dump(agent_states, f, ensure_ascii=False, indent=2)
    
    # 5. Save interaction results
    interaction_results = []
    for log in env.game_logs:
        interaction_results.append({
            "round": log["interaction"],
            "total_contribution": log["total_contribution"],
            "public_pool_gain": log["public_pool_gain"],
            "gain_per_agent": log["gain_per_agent"],
            "agent_contributions": log["agent_contributions"],
            "explanations": log["explanations"],
            "payoffs": log["payoffs"],
            "timestamp": log["timestamp"]
        })
    
    interactions_file = os.path.join(experiment_result_dir, "interaction_results.json")
    with open(interactions_file, "w", encoding="utf-8") as f:
        json.dump(interaction_results, f, ensure_ascii=False, indent=2)
    
    print(f"All results saved successfully!")
    print(f"Experiment results directory: {experiment_result_dir}")

if __name__ == "__main__":
    asyncio.run(main())
