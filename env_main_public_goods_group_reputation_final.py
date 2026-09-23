#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Public Goods Game Group Experiment - agentsociety V2 Platform Implementation
Based on env_main_鍏叡鐗╁搧鍗氬紙baseline.py and 鑷彂娑岀幇.py with 24-agent group architecture
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

# 鍔犺浇鐜鍙橀噺
load_dotenv()

# Ensure results directory exists
os.makedirs("result_public_goods_group_reputation_indirect_reciprocity", exist_ok=True)

# 閰嶇疆鏃ュ織璁板綍
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


class LLMRequestError(RuntimeError):
    """Raised when an LLM request still fails after all retry attempts."""


class PublicGoodsAgent(AgentBase):
    """
    Agent for Public Goods Game that manages its own state,
    compatible with 24-agent group architecture.
    """
    
    def __init__(self, id: int, name: str, profile: str = ""):
        """ Initialize PublicGoodsAgent """
        super().__init__(id, profile)
        self._name = name
        self.pseudonym = name
        self._llm = None
        self._env = None
        self._previous_choices = []  # Track previous choices for stability

        # 鑷富鐘舵€佺鐞嗭細浠ｇ悊鐙珛缁存姢浜や簰鍘嗗彶鍜岀姸鎬?
        self._my_history = []        # Own contribution history
        self._outcome_history = []   # Outcomes history (final payoffs after any mechanism adjustment)
        self.history = []  # Store complete round summaries for history display

        self.initial_endowment = 20  # Initial coins per round
        self.max_contribution = 20  # Maximum contribution amount
        self.min_contribution = 0  # Minimum contribution amount

        # Reputation-coupled indirect reciprocity state. Pseudonym is stable across rounds.
        self.pg_contribution_history = []
        self.ir_giving_history = []
        self.ir_receiving_history = []
        self.reputation_or_image_score = 0.0
        self.total_mechanism_payoff_adjustment = 0.0
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    # 鑷富鐘舵€佺鐞嗭細娣诲姞鏇存柊鍐呴儴鐘舵€佺殑鏂规硶
    def update_state(self, my_contribution: int, outcome: int, baseline_pg_payoff=None, ir_adjustment: float = 0.0, current_round=None):
        """ Update agent's internal state while preserving baseline contribution/outcome histories. """
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)
        if current_round is not None:
            self.pg_contribution_history.append({
                'round': current_round,
                'contribution': my_contribution,
                'baseline_pg_payoff': baseline_pg_payoff if baseline_pg_payoff is not None else outcome
            })
        self.total_mechanism_payoff_adjustment += ir_adjustment
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={outcome}, baseline_pg_payoff={baseline_pg_payoff}, ir_adjustment={ir_adjustment}")
    
    def update_history(self, round_summary: dict):
        """Update agent's history with complete round summary"""
        self.history.append(round_summary)
    
    def get_state_summary(self) -> dict:
        """ Get agent's current state summary """
        return {
            'id': self._id,
            'name': self._name,
            'pseudonym': self.pseudonym,
            'my_history': self._my_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history),
            'pg_contribution_history': self.pg_contribution_history,
            'ir_giving_history': self.ir_giving_history,
            'ir_receiving_history': self.ir_receiving_history,
            'reputation_or_image_score': self.reputation_or_image_score,
            'total_help_given': sum(1 for item in self.ir_giving_history if item.get('helped')),
            'total_help_received': sum(1 for item in self.ir_receiving_history if item.get('helped')),
            'total_mechanism_payoff_adjustment': self.total_mechanism_payoff_adjustment
        }
    
    def _build_history_string(self, all_agent_names: list) -> str:
        """Build history string with aggregate public-goods history plus bounded public reputation context."""
        history_lines = []
        if not self.history:
            history_lines.append("No previous rounds have been played.")
        else:
            history_lines.append("History of previous rounds:")
            for round_summary in self.history[-5:]:
                r = round_summary["round"]
                total_contrib = round_summary["total_contribution"]
                public_gain = round_summary["public_pool_gain"]
                history_lines.append(f"Round {r}:")
                history_lines.append(f"  Total contributed to public fund: {total_contrib} coins.")
                history_lines.append(f"  Public fund gain: {public_gain:.2f} coins.")
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'public_reputation_ledger'):
            ledger = self._env.public_reputation_ledger
            history_lines.append("")
            history_lines.append("Public reputation ledger (stable pseudonyms; recent visible history):")
            if not ledger:
                history_lines.append("  No public reputation entries yet.")
            else:
                for agent_id in sorted(ledger.keys()):
                    entry = ledger[agent_id]
                    pg_recent = entry.get('pg_contributions', [])[-3:]
                    ir_recent = entry.get('ir_given', [])[-3:]
                    pg_text = ", ".join(f"R{x.get('round')}={x.get('contribution')}" for x in pg_recent) if pg_recent else "none"
                    ir_text = ", ".join(f"R{x.get('round')}->{x.get('receiver_name')}: {'HELP' if x.get('helped') else 'NO_HELP'}" for x in ir_recent) if ir_recent else "none"
                    history_lines.append(f"  {entry.get('name')}: reputation={entry.get('reputation_score', 0.0):.3f}; PG recent [{pg_text}]; help-giving recent [{ir_text}]")
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
        """ Choose contribution amount using LLM, considering history in group mode and public reputation visibility. """
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        current_round_true = len(self.history) + 1
        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true}.\n"
            f"Your stable public pseudonym is {self.pseudonym}.\n"
            f"You have {initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n"
            f"Your contribution must be an integer between {self.min_contribution} and {self.max_contribution} inclusive.\n\n"
            f"Reputation information: public-goods contributions and later help decisions are linked to stable pseudonyms and may be visible to participants in later indirect-reciprocity help decisions. The public-goods payoff rule itself is unchanged.\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.\n"
            "State the integer amount first, followed by a brief 1-2 sentence explanation."
        )
        system_message = ""
        contribution = 0
        explanation = "LLM call or parsing failed"
        try:
            content = await self._call_llm_with_retry(system_message=system_message, user_prompt=final_prompt)
            match = re.search(r'^\s*(\d+)\b\s*[-鈥撯€?,]?\s*(.*)$', content, re.DOTALL)
            if not match:
                match = re.search(r'\b(\d+)\b\s*[-鈥撯€?,]?\s*(.*)$', content, re.DOTALL)
            if match:
                parsed_contribution = int(match.group(1))
                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                    self._previous_choices.append(contribution)
                else:
                    contribution = self._get_stable_choice()
                    logging.warning(f"[{self.name}] Contribution value out of range: {parsed_contribution}, using fallback choice {contribution}")
                explanation = match.group(2).strip() if match.group(2) else "No explanation provided"
            else:
                contribution = self._get_stable_choice()
                logging.warning(f"[{self.name}] Could not parse contribution from response, using fallback choice {contribution}. Response: {content}")
                explanation = f"Could not parse contribution; used fallback choice {contribution}"
        except LLMRequestError as e:
            contribution = self._get_stable_choice()
            logging.warning(
                f"[{self.name}] All LLM retries failed; using fallback choice "
                f"{contribution}. Last error: {e}"
            )
            explanation = (
                f"All LLM retries failed; used fallback choice {contribution}"
            )
        except Exception as e:
            logging.error(f"[{self.name}] Error during contribution decision: {e}")
            contribution = self._get_stable_choice()
            explanation = f"Error: {str(e)}; used fallback choice {contribution}"
        return contribution, explanation

    async def choose_ir_help_action(self, receiver_agent, receiver_visible_history: dict, ir_help_cost: float, ir_help_benefit: float, current_round: int):
        """Choose whether to help a receiver in the indirect-reciprocity mechanism stage."""
        import json
        receiver_history_text = json.dumps(receiver_visible_history, ensure_ascii=False, indent=2)
        final_prompt = (
            f"{self._profile}\n"
            f"Indirect-reciprocity help decision for round {current_round}.\n"
            f"Your stable public pseudonym is {self.pseudonym}.\n"
            f"Receiver pseudonym: {receiver_agent.pseudonym}.\n\n"
            f"Visible receiver history:\n{receiver_history_text}\n\n"
            f"If you choose HELP, you pay {ir_help_cost} mechanism coins and {receiver_agent.pseudonym} receives {ir_help_benefit} mechanism coins.\n"
            "If you choose NO_HELP, neither of you receives a mechanism cost or benefit from this help opportunity.\n"
            "This is a separate mechanism stage after the public-goods payoff has already been calculated; do not choose a public-goods contribution here.\n\n"
            "Respond with HELP or NO_HELP first, followed by a brief explanation."
        )
        try:
            content = await self._call_llm_with_retry(system_message="", user_prompt=final_prompt)
            normalized = content.strip().upper()
            first_token = re.split(r'\s+', normalized, maxsplit=1)[0] if normalized else ""
            if first_token in {"NO_HELP", "NO-HELP", "NOHELP"} or re.match(r"^(NO\b|0\b|DO NOT HELP|DON'T HELP)", normalized):
                return False, content.strip()
            if first_token in {"HELP", "YES", "1"} or re.match(r'^(HELP\b|YES\b|1\b)', normalized):
                return True, content.strip()
            logging.warning(f"[{self.name}] Could not parse IR help response, defaulting to NO_HELP. Response: {content}")
            return False, f"Unparsed response defaulted to NO_HELP: {content.strip()}"
        except LLMRequestError as e:
            logging.warning(
                f"[{self.name}] All IR help LLM retries failed; "
                f"defaulting to NO_HELP. Last error: {e}"
            )
            return False, f"All LLM retries failed; defaulted to NO_HELP: {e}"
        except Exception as e:
            logging.error(f"[{self.name}] Error during IR help decision: {e}")
            return False, f"Error defaulted to NO_HELP: {str(e)}"
    

    
    def _get_stable_choice(self) -> int:
        """
        Enhanced choice stability logic:
        1. Prefer this agent's most frequent previous valid LLM choice
        2. If unavailable, use the environment's most popular previous valid choice
        3. Finally use 0 when no previous LLM output exists yet
        """
        # 1. Prefer most frequent previous choice
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            logging.info(f"[{self.name}] Using most frequent previous choice: {most_common_choice}")
            return most_common_choice
        
        # 2. If no consistent choice, use environment's most popular option
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, count = Counter(self._env.choice_frequency).most_common(1)[0]
                if self.min_contribution <= most_popular_choice <= self.max_contribution:
                    logging.info(f"[{self.name}] Using most popular environment choice: {most_popular_choice}")
                    return most_popular_choice
        
        # 3. Finally use 0 when no previous LLM output exists yet
        logging.info(f"[{self.name}] No previous LLM choice available; using fallback 0")
        return 0
    
    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                # Call LLM using asyncio.to_thread to handle synchronous LLM calls
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                if not isinstance(generated_text, str) or not generated_text.strip():
                    raise LLMRequestError("LLM returned empty content")
                if generated_text.lstrip().startswith("API Call Failed"):
                    raise LLMRequestError(generated_text.strip())
                return generated_text
            except Exception as e:
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    if isinstance(e, LLMRequestError):
                        raise
                    raise LLMRequestError(
                        f"LLM request failed after {max_retries} attempts: {e}"
                    ) from e

class PublicGoodsEnvironment:
    """
    Environment for Public Goods Game Group Experiment with 24 agents
    where all agents participate simultaneously in each round.
    """
    
    def __init__(self, num_agents: int, initial_endowment: int, public_pool_multiplier: float,
                 total_interactions: int, mechanism_config: dict = None):

        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions

        self.agents = [] # List of Agent objects
        self.interaction_number = 0 # Track sequential interactions
        self.initial_time = None

        default_mechanism_config = {
            'enabled': True,
            'name': 'reputation_coupled_indirect_reciprocity',
            'ir_help_cost': 1.0,
            'ir_help_benefit': 3.0,
            'ir_pairing_rule': 'all_donors_once_same_group_no_self_shuffled_rotation',
            'reputation_history_visibility': 'public_pseudonym_linked_bounded_prompt_full_log',
            'reputation_score_formula': '0.7 * normalized_average_pg_contribution + 0.3 * help_giving_rate; informational only'
        }
        self.mechanism_config = default_mechanism_config
        if mechanism_config:
            self.mechanism_config.update(mechanism_config)
        for key in ('ir_help_cost', 'ir_help_benefit'):
            if not isinstance(self.mechanism_config.get(key), (int, float)):
                raise ValueError(f"mechanism_config[{key}] must be numeric")

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'public_pool_multiplier': public_pool_multiplier,
            'interactions_per_run': 1,
            'mechanism': self.mechanism_config
        }

        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)  # Track choice frequency
        self.game_logs = []
        self.public_reputation_ledger = {}
        self.ir_logs = []
        self.reputation_scores_by_round = []

    def _ensure_reputation_ledger(self):
        """Ensure every current agent has a stable public reputation-ledger entry."""
        for agent in self.agents:
            if agent._id not in self.public_reputation_ledger:
                self.public_reputation_ledger[agent._id] = {
                    'id': agent._id,
                    'name': agent.name,
                    'pseudonym': getattr(agent, 'pseudonym', agent.name),
                    'pg_contributions': [],
                    'ir_given': [],
                    'ir_received': [],
                    'reputation_score': 0.0
                }
    
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
        Executes one public-goods interaction, then the reputation-coupled indirect-reciprocity mechanism stage.
        Baseline public-goods payoff is computed unchanged before any mechanism adjustment.
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()
        self.interaction_number = interaction_num
        self._ensure_reputation_ledger()

        # 1. All agents make public-goods contribution decisions.
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
                self.choice_frequency[contribution] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_contributions[agent._id] = 0
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency[0] += 1

        # 2. Validate contribution amounts.
        for agent_id, contribution in agent_contributions.items():
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                logging.warning(f"Invalid contribution {contribution} from agent {agent_id}, setting to 0")
                agent_contributions[agent_id] = 0
                explanations[agent_id] += " [Corrected to 0 due to invalid value]"

        # 3. Calculate unchanged baseline public-goods gains.
        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents

        # 4. Calculate unchanged baseline public-goods payoffs for all agents.
        baseline_pg_payoffs = {}
        for agent in self.agents:
            agent_id = agent._id
            contribution = agent_contributions[agent_id]
            private_savings = self.initial_endowment - contribution
            total_gain = private_savings + gain_per_agent
            baseline_pg_payoffs[agent_id] = total_gain
        logging.info(f"Round {interaction_num}: baseline public-goods payoffs computed before IR adjustments.")

        # 5. Update public PG ledger before IR decisions so contributions are reputation-visible.
        for agent in self.agents:
            pg_entry = {
                'round': interaction_num,
                'contribution': agent_contributions[agent._id],
                'baseline_pg_payoff': baseline_pg_payoffs[agent._id]
            }
            self.public_reputation_ledger[agent._id]['pg_contributions'].append(pg_entry)
        for agent in self.agents:
            self._recompute_reputation_score(agent._id)

        # 6. Collect indirect-reciprocity help decisions, without applying payoff updates yet.
        ir_pairings = []
        ir_decisions = []
        ir_payoff_adjustments = {agent._id: 0.0 for agent in self.agents}
        mechanism_enabled = self.mechanism_config.get('enabled', True)
        ir_help_cost = float(self.mechanism_config.get('ir_help_cost', 1.0))
        ir_help_benefit = float(self.mechanism_config.get('ir_help_benefit', 3.0))
        pairings = self._create_ir_pairings(interaction_num) if mechanism_enabled else []
        for donor, receiver in pairings:
            ir_pairings.append({'donor_id': donor._id, 'donor_name': donor.name, 'receiver_id': receiver._id, 'receiver_name': receiver.name})
            receiver_visible_history = self._get_receiver_visible_history(receiver)
            helped, ir_explanation = await donor.choose_ir_help_action(
                receiver_agent=receiver,
                receiver_visible_history=receiver_visible_history,
                ir_help_cost=ir_help_cost,
                ir_help_benefit=ir_help_benefit,
                current_round=interaction_num
            )
            ir_decisions.append({
                'round': interaction_num,
                'donor_id': donor._id,
                'donor_name': donor.name,
                'receiver_id': receiver._id,
                'receiver_name': receiver.name,
                'helped': bool(helped),
                'cost': ir_help_cost if helped else 0.0,
                'benefit': ir_help_benefit if helped else 0.0,
                'explanation': ir_explanation,
                'receiver_visible_history': receiver_visible_history
            })

        # 7. Apply IR mechanism payoff adjustments after all IR decisions are collected.
        for decision in ir_decisions:
            if decision['helped']:
                ir_payoff_adjustments[decision['donor_id']] -= ir_help_cost
                ir_payoff_adjustments[decision['receiver_id']] += ir_help_benefit
        final_payoffs = {agent._id: baseline_pg_payoffs[agent._id] + ir_payoff_adjustments[agent._id] for agent in self.agents}
        logging.info(f"Round {interaction_num}: applied IR payoff adjustments after baseline payoffs: {ir_payoff_adjustments}")

        # 8. Update IR histories and reputation ledger.
        for decision in ir_decisions:
            donor_id = decision['donor_id']
            receiver_id = decision['receiver_id']
            donor_agent = next(agent for agent in self.agents if agent._id == donor_id)
            receiver_agent = next(agent for agent in self.agents if agent._id == receiver_id)
            giving_entry = {
                'round': interaction_num,
                'receiver_id': receiver_id,
                'receiver_name': decision['receiver_name'],
                'helped': decision['helped'],
                'cost': decision['cost'],
                'explanation': decision['explanation']
            }
            receiving_entry = {
                'round': interaction_num,
                'donor_id': donor_id,
                'donor_name': decision['donor_name'],
                'helped': decision['helped'],
                'benefit': decision['benefit']
            }
            donor_agent.ir_giving_history.append(giving_entry)
            receiver_agent.ir_receiving_history.append(receiving_entry)
            self.public_reputation_ledger[donor_id]['ir_given'].append(giving_entry)
            self.public_reputation_ledger[receiver_id]['ir_received'].append(receiving_entry)
        for agent in self.agents:
            self._recompute_reputation_score(agent._id)
        reputation_snapshot = self._public_reputation_snapshot()
        reputation_scores_snapshot = {agent.name: self.public_reputation_ledger[agent._id]['reputation_score'] for agent in self.agents}
        self.reputation_scores_by_round.append({'round': interaction_num, 'scores': reputation_scores_snapshot})
        self.ir_logs.append({'round': interaction_num, 'pairings': ir_pairings, 'decisions': ir_decisions, 'payoff_adjustments': {agent.name: ir_payoff_adjustments[agent._id] for agent in self.agents}})

        # 9. Update agent states and compact round histories.
        round_summary = {
            'round': interaction_num,
            'total_contribution': total_contribution,
            'public_pool_gain': public_pool_gain,
            'gain_per_agent': gain_per_agent,
            'contributions': {agent.name: agent_contributions[agent._id] for agent in self.agents},
            'payoffs': {agent.name: final_payoffs[agent._id] for agent in self.agents},
            'baseline_pg_payoffs': {agent.name: baseline_pg_payoffs[agent._id] for agent in self.agents},
            'ir_payoff_adjustments': {agent.name: ir_payoff_adjustments[agent._id] for agent in self.agents},
            'final_payoffs': {agent.name: final_payoffs[agent._id] for agent in self.agents}
        }
        for agent in self.agents:
            agent.update_state(
                my_contribution=agent_contributions[agent._id],
                outcome=final_payoffs[agent._id],
                baseline_pg_payoff=baseline_pg_payoffs[agent._id],
                ir_adjustment=ir_payoff_adjustments[agent._id],
                current_round=interaction_num
            )
            agent.update_history(round_summary)

        # 10. Record success and log interaction.
        self.game_stats['total_interactions'] += 1
        help_rate = sum(1 for d in ir_decisions if d.get('helped')) / len(ir_decisions) if ir_decisions else 0.0
        interaction_log = {
            'interaction': interaction_num,
            'num_agents': len(self.agents),
            'total_contribution': total_contribution,
            'public_pool_gain': public_pool_gain,
            'gain_per_agent': gain_per_agent,
            'agent_contributions': {agent.name: agent_contributions[agent._id] for agent in self.agents},
            'explanations': {agent.name: explanations[agent._id] for agent in self.agents},
            'payoffs': {agent.name: final_payoffs[agent._id] for agent in self.agents},
            'baseline_pg_payoffs': {agent.name: baseline_pg_payoffs[agent._id] for agent in self.agents},
            'ir_pairings': ir_pairings,
            'ir_decisions': ir_decisions,
            'ir_payoff_adjustments': {agent.name: ir_payoff_adjustments[agent._id] for agent in self.agents},
            'final_payoffs': {agent.name: final_payoffs[agent._id] for agent in self.agents},
            'public_reputation_snapshot': reputation_snapshot,
            'mean_public_goods_contribution': total_contribution / self.num_agents if self.num_agents else 0.0,
            'help_rate': help_rate,
            'help_received_by_reputation_inputs': [{'receiver_name': d['receiver_name'], 'receiver_reputation_score': d['receiver_visible_history'].get('reputation_or_image_score', 0.0), 'helped': d['helped']} for d in ir_decisions],
            'mechanism_config': self.mechanism_config,
            'timestamp': datetime.now().isoformat()
        }
        self.game_logs.append(interaction_log)
        return interaction_log
    
    def get_game_summary(self) -> dict:
        """Get summary of the game statistics, preserving baseline fields and adding IR/reputation metrics."""
        baseline_summary = {
            'total_interactions': self.game_stats['total_interactions'],
            'success_count': self.game_stats['success_count'],
            'success_rate': self.game_stats['success_count'] / self.game_stats['total_interactions'] if self.game_stats['total_interactions'] > 0 else 0,
            'average_contribution': sum(k * v for k, v in self.choice_frequency.items()) / sum(self.choice_frequency.values()) if sum(self.choice_frequency.values()) > 0 else 0,
            'choice_distribution': dict(self.choice_frequency)
        }
        mean_by_round = [log.get('mean_public_goods_contribution', 0.0) for log in self.game_logs]
        contribution_decay = (mean_by_round[-1] - mean_by_round[0]) if len(mean_by_round) >= 2 else 0.0
        all_ir_decisions = []
        for log in self.game_logs:
            all_ir_decisions.extend(log.get('ir_decisions', []))
        overall_help_rate = sum(1 for d in all_ir_decisions if d.get('helped')) / len(all_ir_decisions) if all_ir_decisions else 0.0
        total_profit_by_agent = defaultdict(float)
        total_ir_adjustment = 0.0
        payoff_count = 0
        for log in self.game_logs:
            for name, payoff in log.get('final_payoffs', log.get('payoffs', {})).items():
                total_profit_by_agent[name] += payoff
                payoff_count += 1
            for adjustment in log.get('ir_payoff_adjustments', {}).values():
                total_ir_adjustment += adjustment
        average_final_payoff = sum(total_profit_by_agent.values()) / payoff_count if payoff_count else 0.0
        average_ir_adjustment = total_ir_adjustment / payoff_count if payoff_count else 0.0
        reputation_help_rows = []
        for d in all_ir_decisions:
            reputation_help_rows.append({
                'receiver_name': d.get('receiver_name'),
                'receiver_reputation_score': d.get('receiver_visible_history', {}).get('reputation_or_image_score', 0.0),
                'helped': d.get('helped', False)
            })
        baseline_summary.update({
            'mechanism': self.mechanism_config,
            'mean_public_goods_contribution_by_round': mean_by_round,
            'contribution_decay': contribution_decay,
            'overall_help_rate': overall_help_rate,
            'average_help_received_by_reputation_score': reputation_help_rows,
            'total_profit_by_agent': dict(total_profit_by_agent),
            'average_final_payoff': average_final_payoff,
            'average_ir_adjustment': average_ir_adjustment,
            'reputation_score_by_round': self.reputation_scores_by_round
        })
        return baseline_summary

    def _public_reputation_snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the public reputation ledger."""
        self._ensure_reputation_ledger()
        snapshot = {}
        for agent_id, entry in self.public_reputation_ledger.items():
            snapshot[entry.get('name', str(agent_id))] = {
                'id': entry.get('id', agent_id),
                'pseudonym': entry.get('pseudonym', entry.get('name', str(agent_id))),
                'pg_contributions': list(entry.get('pg_contributions', [])),
                'ir_given': list(entry.get('ir_given', [])),
                'ir_received': list(entry.get('ir_received', [])),
                'reputation_score': entry.get('reputation_score', 0.0)
            }
        return snapshot

    def _get_receiver_visible_history(self, receiver_agent) -> dict:
        """Return public prompt context for a receiver's pseudonym-linked visible history."""
        self._ensure_reputation_ledger()
        entry = self.public_reputation_ledger.get(receiver_agent._id, {})
        return {
            'id': receiver_agent._id,
            'pseudonym': getattr(receiver_agent, 'pseudonym', receiver_agent.name),
            'pg_contribution_history': entry.get('pg_contributions', []),
            'ir_giving_history': entry.get('ir_given', []),
            'ir_receiving_history_count': len(entry.get('ir_received', [])),
            'reputation_or_image_score': entry.get('reputation_score', 0.0)
        }

    def _recompute_reputation_score(self, agent_id: int) -> float:
        """Compute visible reputation score from public histories only; this never directly changes payoffs."""
        entry = self.public_reputation_ledger.get(agent_id, {})
        pg_entries = entry.get('pg_contributions', [])
        ir_given = entry.get('ir_given', [])
        avg_pg_norm = 0.0
        if pg_entries and self.initial_endowment:
            avg_pg_norm = sum(x.get('contribution', 0) for x in pg_entries) / (len(pg_entries) * float(self.initial_endowment))
        help_rate = 0.0
        if ir_given:
            help_rate = sum(1 for x in ir_given if x.get('helped')) / len(ir_given)
        score = round(0.7 * avg_pg_norm + 0.3 * help_rate, 6)
        entry['reputation_score'] = score
        for agent in self.agents:
            if agent._id == agent_id:
                agent.reputation_or_image_score = score
                break
        return score

    def _create_ir_pairings(self, interaction_num: int):
        """Create same-group donor-receiver opportunities; each existing agent is donor once and never paired with self."""
        import random
        if len(self.agents) < 2:
            return []
        seed_base = globals().get('RANDOM_SEED', 42)
        rng = random.Random(seed_base + interaction_num)
        donors = list(self.agents)
        receivers = list(self.agents)
        rng.shuffle(receivers)
        rotation = interaction_num % len(receivers)
        receivers = receivers[rotation:] + receivers[:rotation]
        pairings = []
        for idx, donor in enumerate(donors):
            receiver = receivers[idx % len(receivers)]
            if receiver._id == donor._id:
                receiver = receivers[(idx + 1) % len(receivers)]
            pairings.append((donor, receiver))
        return pairings

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
    base_result_dir = "result_public_goods_group_reputation_indirect_reciprocity"
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

    NUM_EXPERIMENT_RUNS = 3
    for i in range(NUM_EXPERIMENT_RUNS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_RUNS} completed ===")
