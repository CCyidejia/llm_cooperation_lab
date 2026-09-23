#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Public Goods Game Group Experiment - agentsociety V2 Platform Implementation
Based on env_main_公共物品博弈baseline.py and 自发涌现.py with 24-agent group architecture
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
from LLMAPI.wuwen import LLMAgent

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_public_goods_group_altruistic_punishment", exist_ok=True)

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
        
        # Agent state and history management
        self._my_history = []        # Own contribution history
        self._outcome_history = []   # Outcomes history (payoffs)
        self._punishment_sent_history = []
        self._punishment_received_history = []
        self.history = []  # Store complete round summaries for history display
        
        self.initial_endowment = 20  # Initial coins per round
        self.max_contribution = 20  # Maximum contribution amount
        self.min_contribution = 0  # Minimum contribution amount
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    def update_state(self, my_contribution: int, outcome: int, punishment_sent: int = 0, punishment_received: int = 0, final_outcome: float | None = None):
        """ Update agent's internal state """
        if not hasattr(self, '_punishment_sent_history'):
            self._punishment_sent_history = []
        if not hasattr(self, '_punishment_received_history'):
            self._punishment_received_history = []
        if final_outcome is None:
            final_outcome = outcome
        self._my_history.append(my_contribution)
        self._outcome_history.append(final_outcome)
        self._punishment_sent_history.append(punishment_sent)
        self._punishment_received_history.append(punishment_received)
        
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={final_outcome}, punishment_sent={punishment_sent}, punishment_received={punishment_received}")
    
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
            'punishment_sent_history': getattr(self, '_punishment_sent_history', []),
            'punishment_received_history': getattr(self, '_punishment_received_history', []),
            'num_interactions': len(self._my_history)
        }
    
    def _build_history_string(self, all_agent_names: list) -> str:
        """Return the experiment's history-confidentiality notice.

        Complete round summaries are retained internally for logging and state
        accounting, but no previous contribution, punishment, payoff, or
        identity information is disclosed to an agent's decision prompt.
        """
        return (
            "Previous-round information is confidential and unavailable. "
            "Do not infer or assume any previous player's behavior."
        )
    
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
        """Choose a contribution without access to any previous-round history."""
        # Historical outcomes are deliberately not exposed to the participant.
        confidentiality_notice = self._build_history_string([])
        
        # Current round information
        current_round_true = len(self.history) + 1
        
        # 2. Construct the Query - exactly matching baseline format
        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true}.\n"
            f"You have {initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n"
            "After all players choose contributions, you will be shown the other group members' current-round investments under temporary anonymous labels.\n"
            "The anonymous labels reveal no persistent identity and are reset for every round.\n"
            "Then every player may assign 0 to 10 punishment points to each other player.\n"
            "Each punishment point costs the punisher 1 coin and reduces the target player's payoff by 3 coins.\n"
            "Your final payoff equals the baseline public goods payoff minus punishment costs you send and punishment penalties you receive.\n\n"
            f"{confidentiality_notice}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and current game state, determine your contribution amount.\n"
            f"Your decision must be an integer between {self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "Return ONLY the following two lines:\n"
            "Contribution: <one integer from 0 to 20>\n"
            "Explanation: <brief 1-2 sentence explanation>\n"
            "Do not explain the rules.\n"
            "Do not add any text before Contribution.\n"
            "Do not include any extra lines."
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
            
            # Parse only the explicit decision field first. This avoids
            # misreading numbers such as "24 players" as the contribution.
            match = re.search(
                r'(?im)^\s*Contribution\s*:\s*(\d+)\s*$',
                content
            )
            
            if match:
                parsed_contribution = int(match.group(1))
                
                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                else:
                    logging.warning(f"[{self.name}] Contribution value out of range: {parsed_contribution}, using default 0")
                    contribution = 0
                
                explanation_match = re.search(
                    r'(?ims)^\s*Explanation\s*:\s*(.+?)\s*$',
                    content
                )
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            else:
                raise ValueError(
                    "Failed to parse strict contribution format. "
                    "Expected lines starting with Contribution: and Explanation:. "
                    f"Content starts with: {content[:200]}"
                )
            
        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            if self._previous_choices:
                contribution = Counter(self._previous_choices).most_common(1)[0][0]
                explanation = (
                    f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, "
                    f"using most frequent previous selection: {contribution}"
                )
            else:
                contribution = 0
                explanation = (
                    f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, "
                    "no previous selection available, using default selection: 0"
                )
        
        # Add current choice to history for stability
        self._previous_choices.append(contribution)
        
        return contribution, explanation

    async def choose_punishment_actions(self, round_contributions: dict, punishment_point_range=(0, 10)):
        """Choose punishment using temporary, current-round anonymous labels.

        Real agent names never enter the LLM prompt. The temporary labels are
        mapped back to real names only after the response has been parsed.
        """
        min_points, max_points = punishment_point_range

        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []

        # Create a fresh, deterministic permutation for this participant and
        # round. This prevents a temporary label from becoming a persistent
        # cross-round identity while keeping seeded runs reproducible.
        other_agent_names = [name for name in agent_names if name != self.name]
        current_round = len(self.history) + 1
        alias_rng = random.Random(f"{RANDOM_SEED}:{current_round}:{self._id}")
        alias_rng.shuffle(other_agent_names)
        alias_to_name = {
            f"Anonymous_Member_{index}": name
            for index, name in enumerate(other_agent_names, start=1)
        }

        # Reveal only this round's investments under temporary aliases.
        contribution_lines = [
            f"  {alias}: {round_contributions[real_name]} coins"
            for alias, real_name in alias_to_name.items()
        ]
        contributions_text = "\n".join(contribution_lines) if contribution_lines else "  No other contributions available."

        prompt_context = (
            f"Current round contributions:\n"
            f"{contributions_text}\n\n"
            f"Punishment decision:\n"
            f"Assign {min_points}-{max_points} punishment points to each temporary anonymous member.\n"
            "Cost: 1 coin per point you assign.\n"
            "Effect: the target loses 3 coins per point.\n"
            "Only current-round investments are available; no previous-round information is available.\n"
            "These labels apply only to this decision and contain no persistent identity information.\n\n"
            "Return ONLY a JSON object mapping the exact anonymous labels above to integer punishment points.\n"
            "Example: {\"Anonymous_Member_1\": 0, \"Anonymous_Member_2\": 3}"
        )
        try:
            content = await self._call_llm_with_retry("", prompt_context)

            # Extract JSON from response
            import json
            try:
                punishment_dict = json.loads(content.strip())
            except json.JSONDecodeError:
                # Fallback to regex extraction if JSON fails
                import re
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    try:
                        punishment_dict = json.loads(json_match.group(0))
                    except:
                        punishment_dict = {}
                else:
                    punishment_dict = {}

            # Validate and sanitize punishment decisions
            # Accept only aliases supplied in this prompt. Unknown keys,
            # including hallucinated real Agent_N names, are ignored.
            sanitized_punishment = {}
            for alias, real_name in alias_to_name.items():
                points = punishment_dict.get(alias, 0)
                if not isinstance(points, int) or isinstance(points, bool):
                    points = 0
                elif points < min_points or points > max_points:
                    points = 0
                sanitized_punishment[real_name] = points

            return sanitized_punishment
        except Exception as e:
            logging.error(f"[Punishment] Error in {self.name}'s punishment decision: {e}")
            # Return empty punishment dict on error
            return {name: 0 for name in agent_names if name != self.name}
    

    
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
                if not generated_text:
                    raise ValueError("LLM returned empty string")
                if generated_text.lstrip().startswith("API Call Failed"):
                    raise RuntimeError(generated_text)
                return generated_text
            except Exception as e:
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise

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

        # 1. All agents make contribution decisions concurrently
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

        # 4. Calculate base payoffs for all agents
        base_payoffs = {}
        for agent in self.agents:
            agent_id = agent._id
            contribution = agent_contributions[agent_id]
            private_savings = self.initial_endowment - contribution
            total_gain = private_savings + gain_per_agent
            base_payoffs[agent_id] = total_gain

        # 5. All agents make punishment decisions concurrently
        punishment_tasks = []
        for agent in self.agents:
            task = agent.choose_punishment_actions(
                round_contributions={agent.name: agent_contributions[agent._id] for agent in self.agents}
            )
            punishment_tasks.append((agent, task))

        # Collect all punishment decisions
        all_punishment_decisions = {}
        for agent, task in punishment_tasks:
            try:
                punishment_decision = await task
                all_punishment_decisions[agent._id] = punishment_decision
            except Exception as e:
                logging.error(f"Error in punishment decision for agent {agent.name}: {e}")
                all_punishment_decisions[agent._id] = {}

        # 6. Apply punishment costs and penalties
        punishment_costs = {}
        punishment_penalties = {}
        final_payoffs = {}

        # Calculate punishment costs and penalties
        agent_name_to_id = {agent.name: agent._id for agent in self.agents}
        for agent in self.agents:
            agent_id = agent._id
            punishment_sent = all_punishment_decisions.get(agent_id, {})
            punishment_costs[agent_id] = sum(punishment_sent.values())
            punishment_penalties[agent_id] = 0

        for punisher_id, punishment_sent in all_punishment_decisions.items():
            for target_name, points in punishment_sent.items():
                target_id = agent_name_to_id.get(target_name)
                if target_id is None or target_id == punisher_id:
                    continue
                punishment_penalties[target_id] += 3 * points

        for agent in self.agents:
            agent_id = agent._id
            final_payoffs[agent_id] = base_payoffs[agent_id] - punishment_costs[agent_id] - punishment_penalties[agent_id]
        # 7. Update agent states
        # Build round summary for history
        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "base_payoffs": base_payoffs,
            "punishment_costs": punishment_costs,
            "punishment_penalties": punishment_penalties,
            "final_payoffs": final_payoffs,
            "punishment_matrix": all_punishment_decisions
        }

        # Update agent histories with punishment info
        for agent in self.agents:
            agent_id = agent._id
            contribution = agent_contributions[agent_id]

            # Update agent state with punishment info
            agent.update_state(
                my_contribution=contribution,
                outcome=final_payoffs[agent_id],
                punishment_sent=punishment_costs[agent_id],
                punishment_received=punishment_penalties[agent_id],
                final_outcome=final_payoffs[agent_id]
            )

            # Update agent's history with complete round summary
            agent.update_history(round_summary)

        # 8. Record success
        self.game_stats['total_interactions'] += 1

        # 9. Log the interaction
        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "agent_contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "payoffs": {agent.name: final_payoffs[agent._id] for agent in self.agents},
            "base_payoffs": base_payoffs,
            "punishment_costs": punishment_costs,
            "punishment_penalties": punishment_penalties,
            "final_payoffs": final_payoffs,
            "punishment_matrix": all_punishment_decisions,
            "timestamp": datetime.now().isoformat()
        }

        self.game_logs.append(interaction_log)

        return interaction_log
    
    def get_game_summary(self) -> dict:
        """Get summary of the game statistics"""
        possible_punishments = 0
        positive_punishments = 0
        total_punishment_sent = 0
        total_punishment_received = 0
        final_payoff_values = []
        for log in self.game_logs:
            matrix = log.get("punishment_matrix", {})
            possible_punishments += max(0, len(self.agents) * (len(self.agents) - 1))
            for punishment_sent in matrix.values():
                for points in punishment_sent.values():
                    total_punishment_sent += points
                    if points > 0:
                        positive_punishments += 1
            total_punishment_received += sum(log.get("punishment_penalties", {}).values()) / 3
            final_payoff_values.extend(log.get("final_payoffs", {}).values())
        return {
            'total_interactions': self.game_stats['total_interactions'],
            'success_count': self.game_stats['success_count'],
            'success_rate': self.game_stats['success_count'] / self.game_stats['total_interactions'] if self.game_stats['total_interactions'] > 0 else 0,
            'average_contribution': sum(k * v for k, v in self.choice_frequency.items()) / sum(self.choice_frequency.values()) if sum(self.choice_frequency.values()) > 0 else 0,
            'choice_distribution': dict(self.choice_frequency),
            'punishment_frequency': positive_punishments / possible_punishments if possible_punishments > 0 else 0,
            'average_punishment_sent': total_punishment_sent / (len(self.game_logs) * len(self.agents)) if self.game_logs and self.agents else 0,
            'average_punishment_received': total_punishment_received / (len(self.game_logs) * len(self.agents)) if self.game_logs and self.agents else 0,
            'average_final_payoff': sum(final_payoff_values) / len(final_payoff_values) if final_payoff_values else 0
        }
async def main(base_experiment_name, experiment_index):
    """Main function to run the experiment with provided name and index"""
    # ------- Experiment Settings -------
    EXPERIMENT_SETTINGS = {
    "num_agents": 24,  # Number of agents (24 for group experiment)
    "initial_endowment": 20,  # Initial coins per agent per round
    "public_pool_multiplier": 9.6,  # Multiplier for public fund
    "num_rounds": 30,  # Number of rounds per game
    "random_seed": RANDOM_SEED
}
    
    # ------- Create results directory -------
    base_result_dir = "result_public_goods_group_altruistic_punishment"
    # Use the provided base name for all experiments
    experiment_result_dir = os.path.join(base_result_dir, f"result_{base_experiment_name}")
    os.makedirs(experiment_result_dir, exist_ok=True)
    
    # ------- Set up LLM -------
    # Create LLMAgent instance
    llm_agent = LLMAgent()
    
    # Create AgentLLM wrapper with required parameters
    model_name = getattr(llm_agent, 'model', 'default_model')
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)
    
    # ------- Create Agents -------
    agents = []
    for i in range(EXPERIMENT_SETTINGS["num_agents"]):
        # Create agent with simple profile
        profile = f"You are a participant in a game. Your goal is to maximize your own accumulated coins."
        agent = PublicGoodsAgent(id=i+1, name=f"Agent_{i+1}", profile=profile)
        await agent.init(agent_llm)
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
    
    # 1. Save experiment configuration (only once for the first experiment)
    if experiment_index == 1:
        config_file = os.path.join(experiment_result_dir, "experiment_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(EXPERIMENT_SETTINGS, f, ensure_ascii=False, indent=2)
    
    # 2. Save game logs with experiment index
    logs_file = os.path.join(experiment_result_dir, f"game_logs{experiment_index}.json")
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(env.game_logs, f, ensure_ascii=False, indent=2)
    
    # 3. Save game statistics with experiment index
    stats_file = os.path.join(experiment_result_dir, f"game_stats{experiment_index}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(env.get_game_summary(), f, ensure_ascii=False, indent=2)
    
    # 4. Save agent states with experiment index
    agent_states = []
    for agent in agents:
        agent_states.append({
            "id": agent._id,
            "name": agent.name,
            "profile": agent._profile,
            "history": agent.history,
            "state_summary": agent.get_state_summary()
        })
    
    agent_states_file = os.path.join(experiment_result_dir, f"agent_states{experiment_index}.json")
    with open(agent_states_file, "w", encoding="utf-8") as f:
        json.dump(agent_states, f, ensure_ascii=False, indent=2)
    
    # 5. Save interaction results with experiment index
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
    
    interactions_file = os.path.join(experiment_result_dir, f"interaction_results{experiment_index}.json")
    with open(interactions_file, "w", encoding="utf-8") as f:
        json.dump(interaction_results, f, ensure_ascii=False, indent=2)
    
    print(f"All results saved successfully!")
    print(f"Experiment results directory: {experiment_result_dir}")

async def run_experiment(experiment_name, experiment_index):
    """Run the experiment once with provided name and index"""
    await main(experiment_name, experiment_index)

if __name__ == "__main__":
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PG_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = datetime.now().strftime("%m%d_%H%M%S_PG")

    NUM_EXPERIMENT_RUNS = 1  # Run three independent experiments
    for i in range(NUM_EXPERIMENT_RUNS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} completed ===")
