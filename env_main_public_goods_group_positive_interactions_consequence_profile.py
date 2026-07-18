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
os.makedirs("result_public_goods_group_positive_interactions_consequence_prompt", exist_ok=True)

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
        self._previous_choices = []

        # 自主状态管理：代理独立维护交互历史和状态
        self._my_history = []
        self._outcome_history = []
        self.history = []

        # Reward-mechanism histories; baseline fields above are preserved.
        self._rewards_given_history = []
        self._rewards_received_history = []
        self._baseline_payoff_history = []
        self._reward_cost_paid_history = []
        self._reward_benefit_received_history = []
        self._total_payoff_history = []

        self.initial_endowment = 20
        self.max_contribution = 20
        self.min_contribution = 0
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    # 自主状态管理：添加更新内部状态的方法
    def update_state(self, my_contribution: int, outcome: float, baseline_payoff: float = None,
                     rewards_given=None, rewards_received=None, reward_cost_paid: float = 0.0,
                     reward_benefit_received: float = 0.0):
        """Update agent's internal state, preserving baseline contribution/outcome behavior."""
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)

        if baseline_payoff is None:
            baseline_payoff = outcome
        if rewards_given is None:
            rewards_given = []
        if rewards_received is None:
            rewards_received = []

        self._baseline_payoff_history.append(baseline_payoff)
        self._rewards_given_history.append(list(rewards_given))
        self._rewards_received_history.append(list(rewards_received))
        self._reward_cost_paid_history.append(reward_cost_paid)
        self._reward_benefit_received_history.append(reward_benefit_received)
        self._total_payoff_history.append(outcome)

        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, baseline_payoff={baseline_payoff}, final_outcome={outcome}, rewards_given={rewards_given}, rewards_received={rewards_received}")
    
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
            'num_interactions': len(self._my_history),
            'baseline_payoff_history': self._baseline_payoff_history,
            'total_payoff_history': self._total_payoff_history,
            'rewards_given_history': self._rewards_given_history,
            'rewards_received_history': self._rewards_received_history,
            'reward_cost_paid_history': self._reward_cost_paid_history,
            'reward_benefit_received_history': self._reward_benefit_received_history,
            'total_reward_cost_paid': sum(self._reward_cost_paid_history),
            'total_reward_benefit_received': sum(self._reward_benefit_received_history)
        }
    
    def _build_history_string(self, all_agent_names: list) -> str:
        """Build compact history string with public-goods totals and prior reward summaries."""
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
            if "rewards_given_count_by_agent" in round_summary and self._name in round_summary["rewards_given_count_by_agent"]:
                given = round_summary["rewards_given_count_by_agent"].get(self._name, 0)
                received = round_summary.get("rewards_received_count_by_agent", {}).get(self._name, 0)
                history_lines.append(f"  Your prior reward stage: gave {given} rewards and received {received} rewards.")

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
        """Choose contribution amount using LLM, considering history in group mode."""
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        current_round_true = len(self.history) + 1
        group_text = ", ".join(agent_names) if agent_names else f"all {num_agents} players"

        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true}.\n"
            f"You have {initial_endowment} coins.\n"
            f"You are interacting with the same persistent named group members across rounds: {group_text}.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n"
            "After all contributions are observed, there will be a second stage where each player may choose reward or no reward for each other player.\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.\n"
            f"Your decision must be an integer between {self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "State the integer amount first, followed by a brief 1-2 sentence explanation."
        )

        try:
            content = await self._call_llm_with_retry(system_message="", user_prompt=final_prompt)
            contribution = 0
            explanation = "LLM call or parsing failed"
            match = re.search(r'(\d+)\s*[-–—:]?\s*(.*)$', content, re.DOTALL)
            if match:
                parsed_contribution = int(match.group(1))
                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                    self._previous_choices.append(contribution)
                else:
                    contribution = self._get_most_common_llm_choice()
                    logging.warning(
                        f"[{self.name}] Contribution value out of range: {parsed_contribution}, "
                        f"using most common previous LLM choice: {contribution}"
                    )
                explanation = match.group(2).strip() if match.group(2) else "No explanation provided"
            else:
                keyword_match = re.search(r'\b(\d+)\b', content)
                if keyword_match:
                    parsed_contribution = int(keyword_match.group(1))
                    if self.min_contribution <= parsed_contribution <= self.max_contribution:
                        contribution = parsed_contribution
                        self._previous_choices.append(contribution)
                        explanation = "Parsed contribution from response."
                    else:
                        contribution = self._get_most_common_llm_choice()
                        explanation = f"Parsed contribution out of range; used most common previous LLM choice: {contribution}."
                else:
                    contribution = 0
                    explanation = "No valid integer contribution found; defaulted to 0."
            return contribution, explanation
        except Exception as e:
            logging.error(f"[{self.name}] Contribution decision failed: {e}")
            return 0, f"Error: {str(e)}"

    async def choose_reward_actions(self, current_round: int, observed_contributions: dict, baseline_payoffs: dict,
                                    reward_cost: float, reward_benefit: float):
        """Choose directed binary reward/no-reward actions for every other agent after contributions are observed."""
        valid_targets = {}
        target_lines = []
        if hasattr(self, '_env') and self._env:
            for agent in self._env.agents:
                if agent._id == self._id:
                    continue
                valid_targets[agent.name] = agent._id
                valid_targets[str(agent._id)] = agent._id
                contribution = observed_contributions.get(agent.name, observed_contributions.get(agent._id, 0))
                payoff = baseline_payoffs.get(agent.name, baseline_payoffs.get(agent._id, 0))
                target_lines.append(f"- {agent.name} (id {agent._id}): contributed {contribution} coins; baseline public-goods payoff {payoff:.2f}")

        default_decisions = {target_id: 0 for target_id in set(valid_targets.values())}
        if not default_decisions:
            return default_decisions, "No valid reward targets."

        prompt = (
            f"{self._profile}\n"
            f"You are in round {current_round} with the same persistent named group members as previous rounds.\n"
            "The public-goods contribution stage has already ended, and all current-round contributions are now visible.\n"
            f"You may now choose whether to reward each other player. Each reward you give costs you {reward_cost} coins and gives the target {reward_benefit} coins.\n"
            "You cannot reward yourself. Choose independently for every other player.\n\n"
            "Current-round observed contributions:\n"
            + "\n".join(target_lines)
            + "\n\nRespond with ONLY a JSON object mapping each target name to 0 or 1, where 1 means reward and 0 means no reward. Example: {\"Agent_2\": 1, \"Agent_3\": 0}"
        )

        try:
            content = await self._call_llm_with_retry(system_message="", user_prompt=prompt)
        except Exception as e:
            logging.error(f"[{self.name}] Reward decision LLM failed: {e}")
            return default_decisions, f"Reward LLM call failed; defaulted to no rewards: {str(e)}"

        decisions = dict(default_decisions)
        parse_note = "Parsed reward response."
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in reward response")
            raw_obj = json.loads(json_match.group(0))
            if not isinstance(raw_obj, dict):
                raise ValueError("Reward response JSON is not an object")
            for raw_key, raw_value in raw_obj.items():
                key = str(raw_key).strip()
                if key not in valid_targets:
                    logging.warning(f"[{self.name}] Ignoring unknown or self reward target: {key}")
                    continue
                if isinstance(raw_value, bool):
                    action = 1 if raw_value else 0
                elif isinstance(raw_value, (int, float)) and int(raw_value) in (0, 1):
                    action = int(raw_value)
                elif isinstance(raw_value, str) and raw_value.strip().lower() in ('0', '1', 'no', 'no_reward', 'none', 'reward', 'yes'):
                    action = 1 if raw_value.strip().lower() in ('1', 'reward', 'yes') else 0
                else:
                    logging.warning(f"[{self.name}] Ignoring non-binary reward value for {key}: {raw_value}")
                    continue
                decisions[valid_targets[key]] = action
        except Exception as e:
            logging.warning(f"[{self.name}] Invalid reward response; defaulting to no rewards. Error: {e}; response={content}")
            decisions = dict(default_decisions)
            parse_note = f"Invalid reward parse; defaulted to no rewards: {str(e)}"

        return decisions, parse_note
    

    def _get_most_common_llm_choice(self) -> int:
        """Return the most frequent valid contribution already produced by LLM agents."""
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            valid_choices = {
                choice: count
                for choice, count in self._env.choice_frequency.items()
                if self.min_contribution <= choice <= self.max_contribution and count > 0
            }
            if valid_choices:
                most_common_choice, count = max(
                    valid_choices.items(),
                    key=lambda item: (item[1], item[0])
                )
                logging.info(
                    f"[{self.name}] Using most common previous LLM choice: "
                    f"{most_common_choice} (count={count})"
                )
                return most_common_choice

        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            logging.info(
                f"[{self.name}] Using most common personal previous LLM choice: "
                f"{most_common_choice} (count={count})"
            )
            return most_common_choice

        fallback_choice = random.randint(self.min_contribution, self.max_contribution)
        logging.info(f"[{self.name}] No previous LLM choice available; using random fallback: {fallback_choice}")
        return fallback_choice

    
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

class PublicGoodsEnvironment:
    """
    Environment for Public Goods Game Group Experiment with 24 agents
    where all agents participate simultaneously in each round.
    """
    
    def __init__(self, num_agents: int, initial_endowment: int, public_pool_multiplier: float,
                 total_interactions: int, targeted_reward_enabled: bool = True,
                 reward_cost: float = 1.0, reward_benefit: float = 3.0):

        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions

        # Mechanism-only configuration; does not alter baseline public-goods settings.
        self.targeted_reward_enabled = targeted_reward_enabled
        self.reward_cost = reward_cost
        self.reward_benefit = reward_benefit

        self.agents = []
        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'public_pool_multiplier': public_pool_multiplier,
            'interactions_per_run': 1,
            'reward_mechanism': {
                'mechanism_id': 'M1',
                'name': 'Costly targeted reward after each public-goods round',
                'targeted_reward_enabled': targeted_reward_enabled,
                'reward_cost': reward_cost,
                'reward_benefit': reward_benefit
            }
        }

        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)
        self.game_logs = []

        self.reward_actions_by_round = []
        self.baseline_payoff_records = []
        self.total_payoff_records = []
        self.total_reward_actions = 0
        self.possible_reward_actions = 0
        self.reward_cost_total = 0.0
        self.reward_benefit_total = 0.0
    
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
        Baseline public-goods contributions and payoffs are computed first; targeted rewards are applied afterward.
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()

        self.interaction_number = interaction_num

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
        choice_results = await asyncio.gather(*(task for _, task in choice_tasks), return_exceptions=True)
        for (agent, _), result in zip(choice_tasks, choice_results):
            try:
                if isinstance(result, Exception):
                    raise result
                contribution, explanation = result
                agent_contributions[agent._id] = contribution
                explanations[agent._id] = explanation
                self.choice_frequency[contribution] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_contributions[agent._id] = 0
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency[0] += 1

        for agent_id, contribution in agent_contributions.items():
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                agent = self._get_agent_by_id(agent_id)
                fallback_contribution = agent._get_most_common_llm_choice() if agent else 0
                logging.warning(
                    f"Invalid contribution {contribution} from agent {agent_id}, "
                    f"using most common previous LLM choice: {fallback_contribution}"
                )
                if self.choice_frequency.get(contribution, 0) > 0:
                    self.choice_frequency[contribution] -= 1
                self.choice_frequency[fallback_contribution] += 1
                agent_contributions[agent_id] = fallback_contribution
                explanations[agent_id] += (
                    f" [Corrected to most common previous LLM choice {fallback_contribution} "
                    "due to invalid value]"
                )

        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents

        baseline_payoffs = {}
        for agent in self.agents:
            contribution = agent_contributions[agent._id]
            private_savings = self.initial_endowment - contribution
            baseline_payoffs[agent._id] = private_savings + gain_per_agent
        logging.info(f"Round {interaction_num}: baseline public-goods payoffs computed before rewards.")

        final_payoffs = dict(baseline_payoffs)
        observed_contributions_by_name = {agent.name: agent_contributions[agent._id] for agent in self.agents}
        baseline_payoffs_by_name = {agent.name: baseline_payoffs[agent._id] for agent in self.agents}
        reward_actions = []
        reward_decision_explanations = {}
        rewards_given_count_by_agent = {agent._id: 0 for agent in self.agents}
        rewards_received_count_by_agent = {agent._id: 0 for agent in self.agents}
        reward_cost_paid_by_agent = {agent._id: 0.0 for agent in self.agents}
        reward_benefit_received_by_agent = {agent._id: 0.0 for agent in self.agents}

        possible_this_round = self.num_agents * (self.num_agents - 1) if self.targeted_reward_enabled else 0
        selected_this_round = 0

        if self.targeted_reward_enabled:
            reward_tasks = []
            for agent in self.agents:
                reward_tasks.append((agent, agent.choose_reward_actions(
                    current_round=interaction_num,
                    observed_contributions=observed_contributions_by_name,
                    baseline_payoffs=baseline_payoffs_by_name,
                    reward_cost=self.reward_cost,
                    reward_benefit=self.reward_benefit
                )))
            reward_results = await asyncio.gather(*(task for _, task in reward_tasks), return_exceptions=True)

            for (giver, _), result in zip(reward_tasks, reward_results):
                if isinstance(result, Exception):
                    logging.error(f"Error in reward choice for agent {giver.name}: {result}")
                    decisions = {}
                    reward_decision_explanations[giver.name] = f"Error: {str(result)}; defaulted to no rewards"
                elif isinstance(result, tuple) and len(result) == 2:
                    decisions, reward_note = result
                    reward_decision_explanations[giver.name] = reward_note
                elif isinstance(result, dict):
                    decisions = result
                    reward_decision_explanations[giver.name] = "Parsed reward decisions."
                else:
                    decisions = {}
                    reward_decision_explanations[giver.name] = "Invalid reward return; defaulted to no rewards"

                for target in self.agents:
                    if target._id == giver._id:
                        continue
                    action = 1 if decisions.get(target._id, 0) == 1 else 0
                    cost = self.reward_cost if action == 1 else 0.0
                    benefit = self.reward_benefit if action == 1 else 0.0
                    reward_actions.append({
                        'giver_id': giver._id,
                        'giver': giver.name,
                        'target_id': target._id,
                        'target': target.name,
                        'action': action,
                        'cost': cost,
                        'benefit': benefit
                    })
                    if action == 1:
                        selected_this_round += 1
                        rewards_given_count_by_agent[giver._id] += 1
                        rewards_received_count_by_agent[target._id] += 1
                        reward_cost_paid_by_agent[giver._id] += cost
                        reward_benefit_received_by_agent[target._id] += benefit
                        final_payoffs[giver._id] -= cost
                        final_payoffs[target._id] += benefit

        self.total_reward_actions += selected_this_round
        self.possible_reward_actions += possible_this_round
        self.reward_cost_total += sum(reward_cost_paid_by_agent.values())
        self.reward_benefit_total += sum(reward_benefit_received_by_agent.values())
        reward_frequency = selected_this_round / possible_this_round if possible_this_round > 0 else 0
        logging.info(f"Round {interaction_num}: applied targeted rewards after baseline payoffs; selected_rewards={selected_this_round}, possible_rewards={possible_this_round}.")

        baseline_payoffs_named = {agent.name: baseline_payoffs[agent._id] for agent in self.agents}
        final_payoffs_named = {agent.name: final_payoffs[agent._id] for agent in self.agents}
        rewards_given_count_named = {agent.name: rewards_given_count_by_agent[agent._id] for agent in self.agents}
        rewards_received_count_named = {agent.name: rewards_received_count_by_agent[agent._id] for agent in self.agents}
        reward_cost_paid_named = {agent.name: reward_cost_paid_by_agent[agent._id] for agent in self.agents}
        reward_benefit_received_named = {agent.name: reward_benefit_received_by_agent[agent._id] for agent in self.agents}

        self.baseline_payoff_records.append({'round': interaction_num, 'payoffs': baseline_payoffs_named})
        self.total_payoff_records.append({'round': interaction_num, 'payoffs': final_payoffs_named})
        self.reward_actions_by_round.append({'round': interaction_num, 'actions': reward_actions})

        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "contributions": observed_contributions_by_name,
            "baseline_payoffs": baseline_payoffs_named,
            "payoffs": final_payoffs_named,
            "final_payoffs": final_payoffs_named,
            "reward_actions": reward_actions,
            "rewards_given_count_by_agent": rewards_given_count_named,
            "rewards_received_count_by_agent": rewards_received_count_named,
            "reward_cost_paid_by_agent": reward_cost_paid_named,
            "reward_benefit_received_by_agent": reward_benefit_received_named,
            "reward_frequency": reward_frequency
        }

        for agent in self.agents:
            given_targets = [a['target'] for a in reward_actions if a['giver_id'] == agent._id and a['action'] == 1]
            received_from = [a['giver'] for a in reward_actions if a['target_id'] == agent._id and a['action'] == 1]
            agent.update_state(
                my_contribution=agent_contributions[agent._id],
                outcome=final_payoffs[agent._id],
                baseline_payoff=baseline_payoffs[agent._id],
                rewards_given=given_targets,
                rewards_received=received_from,
                reward_cost_paid=reward_cost_paid_by_agent[agent._id],
                reward_benefit_received=reward_benefit_received_by_agent[agent._id]
            )
            agent.update_history(round_summary)

        self.game_stats['total_interactions'] += 1

        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "agent_contributions": observed_contributions_by_name,
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "payoffs": final_payoffs_named,
            "baseline_payoffs": baseline_payoffs_named,
            "final_payoffs": final_payoffs_named,
            "reward_actions": reward_actions,
            "reward_decision_explanations": reward_decision_explanations,
            "rewards_given_count_by_agent": rewards_given_count_named,
            "rewards_received_count_by_agent": rewards_received_count_named,
            "reward_cost_paid_by_agent": reward_cost_paid_named,
            "reward_benefit_received_by_agent": reward_benefit_received_named,
            "total_reward_actions": selected_this_round,
            "possible_reward_actions": possible_this_round,
            "reward_frequency": reward_frequency,
            "timestamp": datetime.now().isoformat()
        }

        self.game_logs.append(interaction_log)
        return interaction_log
    
    def get_game_summary(self) -> dict:
        """Get summary of the game statistics, preserving baseline fields and adding reward metrics."""
        total_choices = sum(self.choice_frequency.values())
        completed_rounds = self.game_stats['total_interactions']
        agent_rounds = completed_rounds * self.num_agents

        baseline_values = []
        for record in self.baseline_payoff_records:
            baseline_values.extend(record.get('payoffs', {}).values())
        total_values = []
        for record in self.total_payoff_records:
            total_values.extend(record.get('payoffs', {}).values())

        return {
            'total_interactions': self.game_stats['total_interactions'],
            'success_count': self.game_stats['success_count'],
            'success_rate': self.game_stats['success_count'] / self.game_stats['total_interactions'] if self.game_stats['total_interactions'] > 0 else 0,
            'average_contribution': sum(k * v for k, v in self.choice_frequency.items()) / total_choices if total_choices > 0 else 0,
            'choice_distribution': dict(self.choice_frequency),
            'per_round_average_contribution': [log['total_contribution'] / log['num_agents'] for log in self.game_logs],
            'reward_frequency': self.total_reward_actions / self.possible_reward_actions if self.possible_reward_actions > 0 else 0,
            'average_baseline_payoff': sum(baseline_values) / len(baseline_values) if baseline_values else 0,
            'average_total_payoff': sum(total_values) / len(total_values) if total_values else 0,
            'average_reward_cost_paid': self.reward_cost_total / agent_rounds if agent_rounds > 0 else 0,
            'average_reward_benefit_received': self.reward_benefit_total / agent_rounds if agent_rounds > 0 else 0,
            'total_reward_actions': self.total_reward_actions,
            'possible_reward_actions': self.possible_reward_actions,
            'reward_cost_total': self.reward_cost_total,
            'reward_benefit_total': self.reward_benefit_total,
            'reward_mechanism': {
                'mechanism_id': 'M1',
                'name': 'Costly targeted reward after each public-goods round',
                'targeted_reward_enabled': self.targeted_reward_enabled,
                'reward_cost': self.reward_cost,
                'reward_benefit': self.reward_benefit
            }
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
    base_result_dir = "result_public_goods_group_positive_interactions_consequence_prompt"
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
        profile = f"You are a participant in a game. Make your decision based on the rules, previous outcomes, and the possible consequences of your action."
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

    NUM_EXPERIMENT_RUNS = 2  # 完整实验运行次数
    for i in range(NUM_EXPERIMENT_RUNS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} completed ===")



