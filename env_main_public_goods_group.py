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
from LLMAPI.zgc import LLMAgent

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


def parse_contribution_response(content: str, min_contribution: int = 0, max_contribution: int = 20):
    """
    Parse the single authoritative action from the model's structured response.

    The contribution is intentionally stored in a named JSON field rather than
    inferred from the first number in free-form text. This prevents numbers in
    the explanation, or a later self-correction, from being recorded as the
    action.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned an empty decision response")

    response_text = content.strip()

    # Tolerate a single JSON Markdown fence, but no surrounding prose.
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        response_text = fenced_match.group(1).strip()

    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Decision response is not valid JSON: {exc.msg}") from exc

    required_fields = {"explanation", "final_contribution"}
    if not isinstance(response, dict) or set(response) != required_fields:
        raise ValueError(
            "Decision response must contain exactly 'explanation' and "
            "'final_contribution'"
        )

    contribution = response["final_contribution"]
    if isinstance(contribution, bool) or not isinstance(contribution, int):
        raise ValueError("final_contribution must be an integer")
    if not min_contribution <= contribution <= max_contribution:
        raise ValueError(
            f"final_contribution must be between {min_contribution} "
            f"and {max_contribution}"
        )

    explanation = response["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be a non-empty string")

    return contribution, explanation.strip()


class LLMRetryExhaustedError(RuntimeError):
    """Raised after all API-level LLM call attempts have failed."""


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
        # Only successfully parsed LLM outputs are eligible for API-failure
        # fallback. Reused fallback decisions are deliberately not added here,
        # so one outage cannot reinforce itself in later rounds.
        self._valid_decision_history = []
        
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
        
        # 2. Construct the query with one unambiguous, machine-readable action.
        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true} of {total_rounds}.\n"
            f"You have {initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n\n"
            f"Your payoff this round is: {initial_endowment} - your contribution "
            f"+ ({public_pool_multiplier} / {num_agents}) * "
            f"(the total contribution of all players).\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, "
            "determine your contribution amount before producing the response.\n"
            f"Your final contribution must be an integer between "
            f"{self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "Return exactly one JSON object and no Markdown or surrounding text. "
            "Use this field order:\n"
            '{"explanation":"brief 1-2 sentence explanation",'
            '"final_contribution":0}\n'
            "Replace 0 with your final decision. The final_contribution field is "
            "your sole authoritative action. It must agree with the explanation; "
            "do not revise it or state an alternative decision in the explanation."
        )

        system_message = ""  # Follow baseline pattern for consistency

        try:
            # A transport/API failure is retried with the identical prompt up
            # to five times. A successful but malformed response is not sent
            # back to the model for repair, because a second decision call
            # could change the model's behavioral choice.
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=final_prompt,
                max_retries=5,
            )
            contribution, explanation = parse_contribution_response(
                content,
                min_contribution=self.min_contribution,
                max_contribution=self.max_contribution,
            )
            self._valid_decision_history.append((contribution, explanation))
        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            contribution, explanation = self._fallback_to_modal_previous_output(e)
        
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

    def _fallback_to_modal_previous_output(self, error: Exception):
        """
        Reuse the modal contribution from this agent's prior valid LLM outputs.

        Explanations are normally unique free text, so after selecting the
        modal contribution we reuse its most recent corresponding explanation.
        Ties between contribution counts are also resolved by recency.
        """
        error_summary = f"{type(error).__name__}: {error}"

        if not self._valid_decision_history:
            logging.warning(
                f"[{self.name}] No prior valid LLM output is available; "
                "falling back to contribution 0"
            )
            return (
                0,
                "[FALLBACK_NO_VALID_HISTORY] No prior valid LLM output was "
                f"available after decision failure ({error_summary}); using 0.",
            )

        contribution_counts = Counter(
            contribution
            for contribution, _ in self._valid_decision_history
        )
        highest_count = max(contribution_counts.values())
        modal_contributions = {
            contribution
            for contribution, count in contribution_counts.items()
            if count == highest_count
        }

        # The latest valid output wins count ties and supplies the explanation.
        for contribution, previous_explanation in reversed(
            self._valid_decision_history
        ):
            if contribution in modal_contributions:
                logging.warning(
                    f"[{self.name}] Reusing modal prior contribution "
                    f"{contribution} after decision failure"
                )
                return (
                    contribution,
                    "[FALLBACK_MODAL_PREVIOUS_OUTPUT] "
                    f"{previous_explanation} "
                    f"(Reused after decision failure: {error_summary})",
                )

        raise RuntimeError("Unable to select a modal previous LLM output")
    
    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                # Call LLM using asyncio.to_thread to handle synchronous LLM calls
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                if not isinstance(generated_text, str) or not generated_text.strip():
                    raise ValueError("LLM returned empty string")
                if generated_text.lstrip().startswith("API Call Failed"):
                    raise RuntimeError(generated_text.strip())
                return generated_text
            except Exception as e:
                last_error = e
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        raise LLMRetryExhaustedError(
            f"LLM API failed after {max_retries} attempts: {last_error}"
        ) from last_error

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
    base_result_dir = "result_public_goods_group"
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

    NUM_EXPERIMENT_RUNS = 3  # 完整实验运行次数
    for i in range(NUM_EXPERIMENT_RUNS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} completed ===")
