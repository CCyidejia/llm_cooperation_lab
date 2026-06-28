#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Game Group Experiment with Free Riders - agentsociety V2 Platform Implementation
Based on env_main_公共物品博弈群体实验.py with 22 LLM agents and 2 rule-based free-rider agents
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
from LLMAPI.淘宝新买的 import LLMAgent

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
    Agent for Game that manages its own state,
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
        history_lines.append("History of previous rounds:")
        
        for round_summary in self.history:
            r = round_summary["round"]
            total_contrib = round_summary["total_contribution"]
            public_gain = round_summary["public_pool_gain"]
            
            history_lines.append(f"Round {r}:")
            history_lines.append(f"  Total contributed to public fund: {total_contrib} coins.")
            history_lines.append(f"  Public fund gain: {public_gain:.2f} coins.")
        
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
        # Standard generic ask, not used in game logic
        return f"[Agent {self.name}] Received message: {message}"
    
    async def step(self, tick: int, t: datetime) -> str:
        # step method is not used in game logic as it's managed by environment
        return ""
    
    async def choose_action(self, initial_endowment: int, public_pool_multiplier: float, num_agents: int, total_rounds: int, memory_size: int = 5):
        """ Choose contribution amount using LLM, considering history in group mode """
        # 1. Build History using baseline-style format
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        
        # Current round information
        current_round_true = len(self.history) + 1
        
        # 2. Construct the Query - exactly matching baseline format
        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true}.\n"
            f"You have {initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.\n"
            f"Your decision must be an integer between {self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "State the integer amount first, followed by a brief 1-2 sentence explanation."
        )

        system_message = ""  # Follow baseline pattern for consistency

        try:
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=final_prompt
            )
            
            # 5. Parse Response
            contribution = 0  # Default contribution (free-riding strategy)
            explanation = "LLM call or parsing failed"
            
            # Try to find the contribution amount anywhere in the response
            # First try: look for number with optional explanation after it
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
                    # If no explanation found after the number, try to get more context from the response
                    # Look for the line containing the number
                    lines = content.split('\n')
                    for line in lines:
                        if str(contribution) in line:
                            # Get the rest of the line after the number
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
                    
                    # Try to extract better explanation from context
                    num_start = keyword_match.start()
                    num_end = keyword_match.end()
                    
                    # Try to get explanation from the same line
                    lines = content.split('\n')
                    for line in lines:
                        if keyword_match.group(1) in line:
                            # Get the rest of the line after the number
                            line_explanation = line[line.find(keyword_match.group(1)) + len(keyword_match.group(1)):].strip()
                            if line_explanation:
                                explanation = line_explanation
                                break
                    
                    if explanation == "LLM call or parsing failed":
                        explanation = f"Extracted contribution: {contribution}"
                else:
                    # 最终默认策略：选择最小贡献量0
                    raise ValueError(f"Failed to parse valid contribution, content:\n{content[:200]}")
            
        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            contribution = 0
            explanation = f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, using default selection: 0"
        
        # Add current choice to history for stability
        self._previous_choices.append(contribution)
        
        return contribution, explanation
    

    
    def _get_stable_choice(self) -> int:
        """
        Enhanced choice stability logic:
        1. Prefer most frequent previous choice
        2. If no consistent choice, use environment's most popular option
        3. Finally use random choice as last resort
        """
        # 1. Prefer most frequent previous choice
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            if count > len(self._previous_choices) / 2:
                logging.info(f"[{self.name}] Using most frequent previous choice: {most_common_choice}")
                return most_common_choice
        
        # 2. If no consistent choice, use environment's most popular option
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, count = self._env.choice_frequency.most_common(1)[0]
                if self.min_contribution <= most_popular_choice <= self.max_contribution:
                    logging.info(f"[{self.name}] Using most popular environment choice: {most_popular_choice}")
                    return most_popular_choice
        
        # 3. Finally use random choice as last resort
        logging.info(f"[{self.name}] Using random choice as fallback")
        return random.randint(self.min_contribution, self.max_contribution)
    
    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
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
    Rule-based free-rider agent for Game
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
    
    async def choose_action(self, initial_endowment: int, public_pool_multiplier: float, num_agents: int, total_rounds: int, memory_size: int = 5):
        """ Always contribute 0 (free-ride strategy) """
        contribution = 0  # Always free-ride
        explanation = "Free-rider strategy: always contribute 0 to maximize personal gain"
        self._previous_choices.append(contribution)
        return contribution, explanation

class PublicGoodsEnvironment:
    """
    Environment for Game Group Experiment with 24 agents
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
            'interactions_per_run': 1
        }
        
        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)  # Track choice frequency
        self.game_logs = []
    
    def set_agents(self, agents: list):
        self.agents = agents
        # Set environment reference for each agent
        for agent in agents:
            agent.set_environment(self)
    
    def _get_agent_by_id(self, agent_id):
        """Helper to get Agent object from ID."""
        return next((a for a in self.agents if a._id == agent_id), None)
    
    async def run_interaction(self, interaction_num: int) -> dict:
        """
        Executes a single interaction where all agents participate simultaneously.
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()
        
        self.interaction_number = interaction_num
        
        # 1. All agents make decisions concurrently
        choice_tasks = []
        for agent in self.agents:
            task = agent.choose_action(
                initial_endowment=self.initial_endowment,
                public_pool_multiplier=self.public_pool_multiplier,
                num_agents=self.num_agents,
                total_rounds=self.total_interactions,
                memory_size=5
            )
            choice_tasks.append((agent, task))
        
        agent_contributions = {}
        explanations = {}
        for agent, task in choice_tasks:
            try:
                contribution, explanation = await task
                agent_contributions[agent._id] = contribution
                explanations[agent._id] = explanation
                # Track choice frequency
                self.choice_frequency[contribution] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_contributions[agent._id] = 0  # Default to 0 contribution
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency[0] += 1
        
        # 2. Validate contribution amounts
        for agent_id, contribution in agent_contributions.items():
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                logging.warning(f"Invalid contribution {contribution} from agent {agent_id}, setting to 0")
                agent_contributions[agent_id] = 0
                explanations[agent_id] += " [Corrected to 0 due to invalid value]"
        
        # 3. Calculate total contribution and gains
        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents
        
        # 4. Calculate payoffs for all agents
        payoffs = {}
        for agent in self.agents:
            agent_id = agent._id
            contribution = agent_contributions[agent_id]
            private_savings = self.initial_endowment - contribution
            total_gain = private_savings + gain_per_agent
            payoffs[agent_id] = total_gain
        
        # 5. Update agent states
        # Build round summary for history
        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
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
            # Update agent's history with complete round summary
            agent.update_history(round_summary)
        
        # 6. Record success
        self.game_stats['total_interactions'] += 1
        
        # 7. Log the interaction
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
    "public_pool_multiplier": 9.6,  # Multiplier for public fund
    "num_rounds": 10,  # Number of rounds per game (changed to 10)
    "random_seed": RANDOM_SEED
}
    
    # ------- Create results directory -------
    base_result_dir = "result_public_goods_group"
    experiment_time = input("Please enter experiment folder name (e.g., 'PG_free_rider_test'): ").strip()
    if not experiment_time:
        experiment_time = datetime.now().strftime("%m%d_%H%M%S_PG_free_rider")
    experiment_result_dir = os.path.join(base_result_dir, f"result_{experiment_time}")
    os.makedirs(experiment_result_dir, exist_ok=True)
    
    # ------- Set up LLM -------
    # Create LLMAgent instance
    llm_agent = LLMAgent()
    
    # Create AgentLLM wrapper with required parameters
    model_name = getattr(llm_agent, 'model', 'default_model')
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)
    
    # ------- Create Agents -------
    agents = []
    
    # Create 12 LLM-based agents
    for i in range(12):
        # Create agent with simple profile
        profile = f"You are a participant in a game. Your goal is to maximize your own accumulated coins."
        agent = PublicGoodsAgent(id=i+1, name=f"LLM_Agent_{i+1}", profile=profile)
        await agent.init(agent_llm)
        agents.append(agent)
    
    # Create 12 rule-based free-rider agents
    for i in range(12, 24):
        profile = f"You are a free-rider in a game. Your strategy is to always contribute 0."
        agent = FreeRiderAgent(id=i+1, name=f"FreeRider_Agent_{i-11}", profile=profile)
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
    print(f"Starting Game Group Experiment with {EXPERIMENT_SETTINGS['num_agents']} agents...")
    print(f"  - 12 LLM-based agents")
    print(f"  - 12 rule-based free-rider agents")
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
