#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trust Game with Anticipated Third-Party Reward - agentsociety V2 implementation.

The baseline Trust Game parameters are preserved: 24 agents, 30 population
rounds, random roles in every interaction, an integer 0-10 investment, and an
integer return between 0 and the amount received.  The reward mechanism is
adapted from Zhurakhovska's "Strategic Trustworthiness via Unstrategic
Third-party Reward": after observing the investment and return, a costly
third-party Rewarder can transfer 0-10 coins, and every transferred coin gives
the Trustee 3 coins.  Trustors and Trustees know about this possible reward
before making their decisions (the paper's Anticipation treatment).

To fit the population code, the paper's later stranger-matched helping game is
implemented as a same-interaction third role.  This is a mechanism migration,
not an exact replication of the paper's original two-part matching protocol.
"""
import os
import json
from collections import defaultdict
import sys
import asyncio
import time
from datetime import datetime
import logging
import traceback
import re
import numpy as np
import random

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# V2 framework core components import
from llm_cooperation_lab.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv
from LLMAPI.zgc import LLMAgent

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_trust_game_population_reward", exist_ok=True)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_api_reward_log.txt"),
        logging.StreamHandler()
    ]
)

# 全局随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 基于V2框架的信任博弈代理实现 - 支持群体交互
class TrustGamePopulationAgent(AgentBase):
    """Agent for Trust Game Population Experiment based on V2 framework"""
    
    def __init__(self, id: int, name: str, profile: str):
        """Initialize TrustGamePopulationAgent"""
        # Initialize basic properties
        self._id = id
        self._name = name  # Agent name
        self._profile = profile  # Agent role description
        self._history = []  # Store interaction history with different partners
        self._reward_history = []  # Store decisions made as third-party Rewarder
        self._llm = None  # LLM will be initialized in init method
        self._env = None  # Explicitly set to None to avoid potential error references
        
        # Game parameters (default values)
        self._initial_funds = 10
        self._multiplication_factor = 3.0
        self._num_rounds = 100
        self._current_funds = self._initial_funds
        
        # Performance metrics
        self._total_payoff = 0.0
        self._decision_count = 0
        self._error_count = 0
        
        # Setup logger
        self.logger = logging.getLogger(f"TrustGamePopulationAgent.{self._name}")
        
        self.logger.info(f"TrustGamePopulationAgent initialized: {self._name}")
        self.logger.debug(f"Agent profile length: {len(self._profile)} characters")
        
        # Enhanced history tracking data structures
        self._interaction_partners = defaultdict(lambda: {
            "interaction_count": 0,
            "self_as_trustor": [],  # 自身作为Trustor的记录
            "self_as_trustee": [],  # 自身作为Trustee的记录
            "partner_as_trustor": [],  # 对方作为Trustor的记录
            "partner_as_trustee": []   # 对方作为Trustee的记录
        })
        self._interaction_sequence = []  # 记录交互的顺序
    
    @property
    def name(self) -> str:
        """Return agent name"""
        return self._name
    
    async def init(self, llm: AgentLLM):
        """Initialize LLM"""
        try:
            if not isinstance(llm, AgentLLM):
                raise TypeError(f"Invalid llm type: expected AgentLLM, got {type(llm).__name__}")
            
            self._llm = llm
            
            self.logger.info(f"Agent {self._name} successfully initialized with LLM: {getattr(llm, 'model_name', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Agent initialization failed: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(f"Init error traceback: {traceback.format_exc()}")
            self._error_count += 1
            raise Exception(f"Agent initialization failed for {self._name}") from e
    
    async def dump(self) -> dict:
        """Dump agent's profile and state to dict - V2 framework abstract method implementation"""
        try:
            dump_data = {
                "id": self._id,
                "name": self._name,
                "agent_type": "TrustGamePopulationAgent",
                "profile": self._profile,
                "history": self._history,
                "reward_history": self._reward_history,
                "interaction_partners": dict(self._interaction_partners),
                "interaction_sequence": self._interaction_sequence,
                "initial_funds": self._initial_funds,
                "multiplication_factor": self._multiplication_factor,
                "num_rounds": self._num_rounds,
                "total_payoff": self._total_payoff,
                "decision_count": self._decision_count,
                "error_count": self._error_count,
                "dump_timestamp": datetime.now().isoformat()
            }
            
            self.logger.debug(f"Agent {self._name} state dumped successfully")
            return dump_data
            
        except Exception as e:
            error_msg = f"Failed to dump agent state: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(f"Dump error traceback: {traceback.format_exc()}")
            self._error_count += 1
            
            # Return minimal state on error - compatible with V2 framework
            return {
                "id": self._id,
                "name": self._name,
                "agent_type": "TrustGamePopulationAgent",
                "error": error_msg,
                "dump_failed": True
            }
    
    async def load(self, dump_data: dict):
        """Load agent from dump data - V2 framework abstract method implementation"""
        try:
            if not isinstance(dump_data, dict):
                raise TypeError("dump_data must be a dictionary")
            
            # Essential fields
            self._id = dump_data.get("id", self._id)
            self._name = dump_data.get("name", self._name)
            self._profile = dump_data.get("profile", self._profile)
            self._history = dump_data.get("history", [])
            self._reward_history = dump_data.get("reward_history", [])
            
            # Enhanced history fields
            self._interaction_partners = defaultdict(lambda: {
                "interaction_count": 0,
                "self_as_trustor": [],
                "self_as_trustee": [],
                "partner_as_trustor": [],
                "partner_as_trustee": []
            })
            
            # Convert dict back to defaultdict
            saved_partners = dump_data.get("interaction_partners", {})
            for partner_name, partner_data in saved_partners.items():
                self._interaction_partners[partner_name] = partner_data
                
            self._interaction_sequence = dump_data.get("interaction_sequence", [])
            
            # Game parameters
            self._initial_funds = dump_data.get("initial_funds", self._initial_funds)
            self._multiplication_factor = dump_data.get("multiplication_factor", self._multiplication_factor)
            self._num_rounds = dump_data.get("num_rounds", self._num_rounds)
            
            # Performance metrics
            self._total_payoff = dump_data.get("total_payoff", self._total_payoff)
            self._decision_count = dump_data.get("decision_count", self._decision_count)
            self._error_count = dump_data.get("error_count", self._error_count)
            
            self.logger.info(f"Agent {self._name} state loaded successfully")
            self.logger.debug(f"Loaded {len(self._history)} interaction records")
            self.logger.debug(f"Loaded interaction history for {len(self._interaction_partners)} partners")
            
        except Exception as e:
            error_msg = f"Failed to load agent state: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(f"Load error traceback: {traceback.format_exc()}")
            self._error_count += 1
    
    async def ask(self, message: str, readonly: bool = True) -> str:
        """Answer questions - V2 framework abstract method implementation"""
        try:
            if not isinstance(message, str):
                raise TypeError("message must be a string")
            
            prompt = f"{self._profile}\n\n{message}"
            
            # Use retry mechanism for LLM calls
            response = await self._call_llm_with_retry(
                system_message="",
                user_prompt=prompt
            )
            
            self.logger.debug(f"Agent {self._name} successfully answered question")
            return response
            
        except Exception as e:
            error_msg = f"Failed to answer question: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(f"Ask error traceback: {traceback.format_exc()}")
            self._error_count += 1
            return f"[错误] {error_msg}"
    
    async def step(self, tick: int, t: datetime) -> str:
        """Execute one step - V2 framework abstract method implementation"""
        try:
            if not isinstance(tick, int) or tick < 0:
                raise ValueError("tick must be a non-negative integer")
            if not isinstance(t, datetime):
                raise TypeError("t must be a datetime object")
            
            result = f"{self._name} step executed at tick {tick}"
            self.logger.debug(result)
            return result
            
        except Exception as e:
            error_msg = f"Step execution failed: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(f"Step error traceback: {traceback.format_exc()}")
            self._error_count += 1
            return f"[错误] {error_msg}"
    
    async def act(self, interaction_num: int, role: str, initial_funds: int,
                 multiplication_factor: int, num_rounds: int, current_funds: int,
                 partner_name: str, reward_budget: int,
                 reward_multiplier: int) -> tuple[int, str]:
        """Execute action - make investment or return decision based on current role"""
        try:
            # Validate input parameters
            if role not in ["Trustor", "Trustee"]:
                raise ValueError(f"Invalid role: {role}. Must be 'Trustor' or 'Trustee'")
            
            # Update agent's current state
            self._current_funds = current_funds
            
            # Build history string for this partner
            history_str = self._build_partner_history_string(partner_name)
            
            # Build prompt based on role
            if role == "Trustor":
                prompt = self._build_trustor_prompt(interaction_num, initial_funds,
                                                   multiplication_factor, num_rounds,
                                                   current_funds, history_str, partner_name,
                                                   reward_budget, reward_multiplier)
            else:  # Trustee
                prompt = self._build_trustee_prompt(interaction_num, initial_funds,
                                                   multiplication_factor, num_rounds,
                                                   current_funds, history_str, partner_name,
                                                   reward_budget, reward_multiplier)
            
            # Call LLM with retry mechanism - use role-specific profile
            role_profile = (
                f"You are {self._name}, Trustor in a game.\n"
                "Goal: Maximize cumulative coins.\n"
                f"Rules: Start with {self._initial_funds} coins/round.\n"
                f"Send an integer from 0-{self._initial_funds} coins "
                f"(×{self._multiplication_factor} for Trustee).\n"
                f"Trustee returns some coins. Your payoff: {self._initial_funds} - sent + returned.\n"
                f"A third-party Rewarder observes the exchange and can spend 0-{reward_budget} "
                f"coins to reward the Trustee; each coin gives the Trustee {reward_multiplier} coins.\n"
                "The possible reward does not directly change your payoff.\n"
                "Consider past Trustee behavior."
            ) if role == "Trustor" else (
                f"You are {self._name}, Trustee in a game.\n"
                "Goal: Maximize cumulative coins.\n"
                f"Rules: Receive (sent amount)×{self._multiplication_factor} coins from Trustor.\n"
                "Return an integer amount from 0 to the received coins.\n"
                f"A third-party Rewarder observes your return and can spend 0-{reward_budget} "
                f"coins; every coin spent adds {reward_multiplier} coins to your payoff.\n"
                "You know this reward opportunity before choosing your return.\n"
                "Your payoff: received - returned + reward bonus.\n"
                "Consider past Trustor behavior."
            )
            
            content = await self._call_llm_with_retry(
                system_message=role_profile,  # Use role-specific profile as system message
                user_prompt=prompt
            )
            
            # Log raw response content for debugging
            self.logger.debug(f"[{self._name}] Raw LLM response: {content[:200]}..." if len(content) > 200 else f"[{self._name}] Raw LLM response: {content}")
            
            if not content or content.isspace():
                raise ValueError("LLM returned empty response")
            
            # Parse numerical value
            amount = self._parse_amount(content, role, initial_funds if role == "Trustor" else current_funds)
            
            # Extract explanation
            explanation = self._extract_explanation(content)
            
            self.logger.info(f"[{self._name}] [{role}] Decision: {amount} coins{', explanation: ' + explanation[:50] + '...' if len(explanation) > 50 else ''}")
            self.logger.debug(f"[{self._name}] Full explanation: {explanation}")
            
            # Update performance metrics
            self._decision_count += 1
            
            return amount, explanation
            
        except Exception as e:
            error_msg = f"Decision making failed: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] [{role}] {error_msg}")
            self.logger.debug(f"[{self._name}] Decision error traceback: {traceback.format_exc()}")
            
            # Update error metrics
            self._error_count += 1
            
            # Use default value on critical failure
            default_amount = 0
            default_explanation = f"[CRITICAL FAILURE] {error_msg}, using default amount: {default_amount}"
            self.logger.warning(f"[{self._name}] [{role}] Using default decision: {default_amount}")
            
            return default_amount, default_explanation

    async def act_as_rewarder(self, interaction_num: int, reward_budget: int,
                              reward_multiplier: int, trustor_name: str,
                              trustee_name: str, sent_amount: int,
                              received_amount: int,
                              returned_amount: int) -> tuple[int, str]:
        """Choose a costly third-party reward after observing the Trust Game."""
        try:
            return_rate = returned_amount / received_amount if received_amount > 0 else 0.0
            return_multiple = returned_amount / sent_amount if sent_amount > 0 else 0.0
            history_str = self._build_reward_history_string(trustee_name)
            role_profile = (
                f"You are {self._name}, an impartial third-party Rewarder.\n"
                f"You receive {reward_budget} coins for this interaction.\n"
                f"You may spend any integer amount from 0 to {reward_budget}. "
                f"Each coin you spend gives the Trustee {reward_multiplier} coins.\n"
                "A positive reward is personally costly and cannot increase your later material payoff.\n"
                "Decide whether and how strongly to reward the Trustee's observed trustworthiness."
            )
            prompt = (
                f"Interaction {interaction_num}\n"
                f"Observed Trustor: {trustor_name}\n"
                f"Observed Trustee: {trustee_name}\n"
                f"Trustor sent: {sent_amount} coins\n"
                f"Trustee received after multiplication: {received_amount} coins\n"
                f"Trustee returned: {returned_amount} coins\n"
                f"Return rate: {return_rate:.4f} of the received amount\n"
                f"Return multiple: {return_multiple:.4f} times the Trustor's original transfer\n"
                f"Reward budget: {reward_budget} coins\n"
                f"If you spend q coins, your payoff is {reward_budget} - q and "
                f"the Trustee receives a bonus of {reward_multiplier} × q.\n"
                "The Trustee knew before returning that this reward opportunity existed.\n\n"
                f"{history_str}\n\n"
                "Decision format:\n"
                f"Amount: [an integer from 0 to {reward_budget}; decimals are not allowed]\n"
                "Explanation: [brief reason for the reward decision]"
            )

            content = await self._call_llm_with_retry(
                system_message=role_profile,
                user_prompt=prompt
            )
            if not content or content.isspace():
                raise ValueError("LLM returned empty response")

            amount = self._parse_amount(content, "Rewarder", reward_budget)
            explanation = self._extract_explanation(content)
            self._decision_count += 1
            self.logger.info(
                f"[{self._name}] [Rewarder] Decision: {amount} coins; "
                f"Trustee bonus: {amount * reward_multiplier} coins"
            )
            return amount, explanation

        except Exception as e:
            error_msg = f"Reward decision failed: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] [Rewarder] {error_msg}")
            self.logger.debug(f"[{self._name}] Reward error traceback: {traceback.format_exc()}")
            self._error_count += 1
            return 0, f"[CRITICAL FAILURE] {error_msg}, using default amount: 0"

    def update_reward_history(self, interaction_summary: dict):
        """Record an interaction in which this agent acted as Rewarder."""
        self._reward_history.append(interaction_summary)
        self._interaction_sequence.append({
            "interaction": interaction_summary.get("interaction"),
            "partner_name": interaction_summary.get("trustee_name"),
            "role": "Rewarder",
            "timestamp": interaction_summary.get("timestamp")
        })

    def _build_reward_history_string(self, trustee_name: str) -> str:
        """Build Rewarder history concerning a previously observed Trustee."""
        records = [
            record for record in self._reward_history
            if record.get("trustee_name") == trustee_name
        ]
        if not records:
            return f"No previous reward decisions involving {trustee_name}."

        lines = [f"Previous reward decisions involving {trustee_name}:"]
        for index, record in enumerate(records, 1):
            lines.append(
                f"Observation {index}: sent={record.get('sent_amount', 0)}, "
                f"received={record.get('received_amount', 0)}, "
                f"returned={record.get('returned_amount', 0)}, "
                f"you spent={record.get('reward_cost', 0)}."
            )
        return "\n".join(lines)
    
    def update_history(self, interaction_summary: dict):
        """Update agent's history with current interaction summary"""
        self._history.append(interaction_summary)
        
        # Enhanced history tracking
        partner_name = interaction_summary.get("partner_name")
        role = interaction_summary.get("role")
        interaction_num = interaction_summary.get("interaction")
        
        # Update interaction sequence
        self._interaction_sequence.append({
            "interaction": interaction_num,
            "partner_name": partner_name,
            "role": role,
            "timestamp": interaction_summary.get("timestamp")
        })
        
        # Update partner-specific data
        partner_data = self._interaction_partners[partner_name]
        partner_data["interaction_count"] += 1
        
        if role == "Trustor":
            self_as_trustor_entry = {
                "interaction": interaction_num,
                "sent_amount": interaction_summary.get("sent_amount", 0),
                "returned_amount": interaction_summary.get("returned_amount", 0),
                "payoff": interaction_summary.get("payoff", 0),
                "timestamp": interaction_summary.get("timestamp")
            }
            partner_data["self_as_trustor"].append(self_as_trustor_entry)
            
            # For the partner, this was a Trustee role
            partner_as_trustee_entry = {
                "interaction": interaction_num,
                "sent_amount": interaction_summary.get("sent_amount", 0),
                "returned_amount": interaction_summary.get("returned_amount", 0),
                "payoff": interaction_summary.get("partner_payoff", 0),  # Note: This field might need to be added in the main interaction logic
                "timestamp": interaction_summary.get("timestamp")
            }
            partner_data["partner_as_trustee"].append(partner_as_trustee_entry)
            
        elif role == "Trustee":
            self_as_trustee_entry = {
                "interaction": interaction_num,
                "sent_amount": interaction_summary.get("sent_amount", 0),  # Amount received from Trustor
                "returned_amount": interaction_summary.get("returned_amount", 0),  # Amount returned
                "payoff": interaction_summary.get("payoff", 0),
                "timestamp": interaction_summary.get("timestamp")
            }
            partner_data["self_as_trustee"].append(self_as_trustee_entry)
            
            # For the partner, this was a Trustor role
            partner_as_trustor_entry = {
                "interaction": interaction_num,
                "sent_amount": interaction_summary.get("sent_amount", 0),  # Amount sent by partner
                "returned_amount": interaction_summary.get("returned_amount", 0),  # Amount returned by self
                "payoff": interaction_summary.get("partner_payoff", 0),  # Note: This field might need to be added in the main interaction logic
                "timestamp": interaction_summary.get("timestamp")
            }
            partner_data["partner_as_trustor"].append(partner_as_trustor_entry)
    
    def _build_partner_history_string(self, partner_name: str) -> str:
        """Build history string for specific partner"""
        partner_history = [record for record in self._history if record.get("partner_name") == partner_name]
        
        if not partner_history:
            return f"No previous interactions with {partner_name}."
        
        history_lines = []
        history_lines.append(f"History of previous interactions with {partner_name}:")
        
        for i, record in enumerate(partner_history, 1):
            role = record.get("role")
            if role == "Trustor":
                sent_amount = record.get("sent_amount", 0)
                returned_amount = record.get("returned_amount", 0)
                payoff = record.get("payoff", 0)
                reward_bonus = record.get("reward_bonus_to_trustee", 0)
                history_lines.append(
                    f"Round {i}: You sent {sent_amount} coins. {partner_name} returned "
                    f"{returned_amount} coins and then received a third-party reward bonus "
                    f"of {reward_bonus} coins. Your payoff: {payoff} coins."
                )
            else:  # Trustee
                sent_amount = record.get("sent_amount", 0)
                received_amount = record.get("received_amount", 0)
                returned_amount = record.get("returned_amount", 0)
                payoff = record.get("payoff", 0)
                reward_bonus = record.get("reward_bonus", 0)
                history_lines.append(
                    f"Round {i}: You received {received_amount} coins (from {partner_name}'s "
                    f"{sent_amount} coins × {record.get('multiplication_factor', 3)}). "
                    f"You returned {returned_amount} coins and received a third-party reward "
                    f"bonus of {reward_bonus} coins. Your payoff: {payoff} coins."
                )
        
        return "\n".join(history_lines)
    
    def _build_trustor_prompt(self, interaction_num: int, initial_funds: int,
                             multiplication_factor: int, num_rounds: int,
                             current_funds: int, history_str: str, partner_name: str,
                             reward_budget: int, reward_multiplier: int) -> str:
        """Build prompt for Trustor role"""
        return (
            f"Round {interaction_num}\n"
            f"Current funds: {current_funds}\n"
            f"Send an integer from 0-{initial_funds} coins to {partner_name} "
            f"(amount × {multiplication_factor} for them)\n"
            f"Your goal: Maximize total coins (payoff = {initial_funds} - sent + returned)\n\n"
            f"After {partner_name} returns coins, an impartial third party will observe both "
            f"decisions and may spend 0-{reward_budget} coins to reward {partner_name}. "
            f"Every reward coin becomes {reward_multiplier} coins for {partner_name}. "
            "This reward does not directly change your payoff.\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            f"Amount: [an integer from 0 to {initial_funds}; decimals are not allowed]\n"
            "Explanation: [brief strategic reason]"
        )
    
    def _build_trustee_prompt(self, interaction_num: int, initial_funds: int,
                             multiplication_factor: int, num_rounds: int,
                             current_funds: int, history_str: str, partner_name: str,
                             reward_budget: int, reward_multiplier: int) -> str:
        """Build prompt for Trustee role"""
        return (
            f"Round {interaction_num}\n"
            f"Received: {current_funds} coins "
            f"({partner_name} sent {current_funds // multiplication_factor})\n"
            f"Return an integer from 0-{current_funds} coins to {partner_name}\n"
            f"After your return, an impartial third party will observe both decisions and may "
            f"spend 0-{reward_budget} coins to reward you. Every reward coin gives you "
            f"{reward_multiplier} coins. You know this before choosing your return.\n"
            f"Your payoff = {current_funds} - returned + reward bonus.\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            f"Amount: [an integer from 0 to {current_funds}; decimals are not allowed]\n"
            "Explanation: [brief strategic reason]"
        )
    
    def _parse_amount(self, content: str, role: str, max_amount: int) -> int:
        """Parse a strict integer decision and enforce the feasible interval."""
        number_pattern = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
        labelled_match = re.search(
            rf"(?im)^\s*Amount\s*:\s*({number_pattern})",
            content
        )
        match = labelled_match or re.search(number_pattern, content)

        if not match:
            raise ValueError(
                f"Failed to parse valid numerical amount from response: {content[:200]}..."
            )

        raw_value = match.group(1) if labelled_match else match.group(0)
        if not re.fullmatch(r"-?\d+", raw_value):
            self.logger.warning(
                f"[{self._name}] Invalid {role.lower()} amount ({raw_value}); "
                "decimals are not allowed"
            )
            return 0

        parsed_val = int(raw_value)
        if not (0 <= parsed_val <= int(max_amount)):
            self.logger.warning(
                f"[{self._name}] Invalid {role.lower()} amount ({parsed_val}) detected; "
                f"required range is 0-{max_amount}"
            )
            return 0

        return parsed_val
    
    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from LLM response"""
        labelled_match = re.search(r"(?is)Explanation\s*:\s*(.+)$", content)
        if labelled_match:
            explanation = labelled_match.group(1).strip()
            return explanation if explanation else "No explanation provided"

        # Fallback: look for text after the first number.
        match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", content)
        if match:
            explanation = content[match.end():].strip()
            return explanation if explanation else "No explanation provided"
        
        # If no amount found, return entire content as explanation
        return content.strip() if content.strip() else "No explanation provided"
    
    async def _call_llm_with_retry(self, system_message, user_prompt, max_retries=4):
        """Call LLM with retry mechanism"""
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                if retries > 0:
                    print(f"[{self.name}] [RETRY] 第{retries}次重试LLM调用...")
                    await asyncio.sleep(0.5)
                
                # 调用LLM
                content = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                
                if not content or content.isspace():
                    raise ValueError("LLM returned empty response")
                
                # 检查返回内容是否包含错误信息
                error_keywords = ["API调用失败", "Error code", "Insufficient Balance", "invalid_request_error"]
                if any(keyword in content for keyword in error_keywords):
                    raise ValueError(f"LLM API returned error: {content}")
                
                return content
                
            except Exception as e:
                last_exception = e
                print(f"[{self.name}] [ERROR] LLM调用失败 (尝试 {retries+1}/{max_retries}): {type(e).__name__} - {str(e)}")
                retries += 1
        
        raise last_exception or Exception("All LLM call retries failed")
    
    def set_environment(self, env):
        """Set the environment for the agent"""
        self._env = env
    
    # Enhanced history query methods
    def get_partner_history(self, partner_name: str) -> dict:
        """Get detailed interaction history with a specific partner"""
        return {
            "interaction_count": self._interaction_partners[partner_name]["interaction_count"],
            "self_as_trustor": self._interaction_partners[partner_name]["self_as_trustor"],
            "self_as_trustee": self._interaction_partners[partner_name]["self_as_trustee"],
            "partner_as_trustor": self._interaction_partners[partner_name]["partner_as_trustor"],
            "partner_as_trustee": self._interaction_partners[partner_name]["partner_as_trustee"]
        }
    
    def get_all_partners(self) -> list:
        """Get list of all agents that this agent has interacted with"""
        return list(self._interaction_partners.keys())
    
    def get_role_statistics(self) -> dict:
        """Get statistics about roles played by self and interaction partners"""
        statistics = {
            "self_as_trustor": {
                "count": 0,
                "total_sent": 0,
                "total_returned": 0,
                "average_sent": 0.0,
                "average_returned": 0.0
            },
            "self_as_trustee": {
                "count": 0,
                "total_received": 0,
                "total_returned": 0,
                "average_received": 0.0,
                "average_returned": 0.0
            },
            "self_as_rewarder": {
                "count": len(self._reward_history),
                "total_reward_cost": sum(
                    entry.get("reward_cost", 0) for entry in self._reward_history
                ),
                "total_reward_bonus": sum(
                    entry.get("reward_bonus", 0) for entry in self._reward_history
                ),
                "average_reward_cost": 0.0,
                "average_reward_bonus": 0.0
            },
            "partner_as_trustor": {
                "count": 0,
                "total_sent": 0,
                "total_returned": 0,
                "average_sent": 0.0,
                "average_returned": 0.0
            },
            "partner_as_trustee": {
                "count": 0,
                "total_sent": 0,
                "total_returned": 0,
                "average_sent": 0.0,
                "average_returned": 0.0
            }
        }
        
        for partner_name, partner_data in self._interaction_partners.items():
            # Self as Trustor
            self_as_trustor = partner_data["self_as_trustor"]
            statistics["self_as_trustor"]["count"] += len(self_as_trustor)
            statistics["self_as_trustor"]["total_sent"] += sum(entry["sent_amount"] for entry in self_as_trustor)
            statistics["self_as_trustor"]["total_returned"] += sum(entry["returned_amount"] for entry in self_as_trustor)
            
            # Self as Trustee
            self_as_trustee = partner_data["self_as_trustee"]
            statistics["self_as_trustee"]["count"] += len(self_as_trustee)
            statistics["self_as_trustee"]["total_received"] += sum(
                entry.get("received_amount", 0) for entry in self_as_trustee
            )
            statistics["self_as_trustee"]["total_returned"] += sum(entry["returned_amount"] for entry in self_as_trustee)
            
            # Partner as Trustor
            partner_as_trustor = partner_data["partner_as_trustor"]
            statistics["partner_as_trustor"]["count"] += len(partner_as_trustor)
            statistics["partner_as_trustor"]["total_sent"] += sum(entry["sent_amount"] for entry in partner_as_trustor)
            statistics["partner_as_trustor"]["total_returned"] += sum(entry["returned_amount"] for entry in partner_as_trustor)
            
            # Partner as Trustee
            partner_as_trustee = partner_data["partner_as_trustee"]
            statistics["partner_as_trustee"]["count"] += len(partner_as_trustee)
            statistics["partner_as_trustee"]["total_sent"] += sum(entry["sent_amount"] for entry in partner_as_trustee)
            statistics["partner_as_trustee"]["total_returned"] += sum(entry["returned_amount"] for entry in partner_as_trustee)
        
        # Calculate averages
        if statistics["self_as_trustor"]["count"] > 0:
            statistics["self_as_trustor"]["average_sent"] = statistics["self_as_trustor"]["total_sent"] / statistics["self_as_trustor"]["count"]
            statistics["self_as_trustor"]["average_returned"] = statistics["self_as_trustor"]["total_returned"] / statistics["self_as_trustor"]["count"]
        
        if statistics["self_as_trustee"]["count"] > 0:
            statistics["self_as_trustee"]["average_received"] = statistics["self_as_trustee"]["total_received"] / statistics["self_as_trustee"]["count"]
            statistics["self_as_trustee"]["average_returned"] = statistics["self_as_trustee"]["total_returned"] / statistics["self_as_trustee"]["count"]

        if statistics["self_as_rewarder"]["count"] > 0:
            statistics["self_as_rewarder"]["average_reward_cost"] = statistics["self_as_rewarder"]["total_reward_cost"] / statistics["self_as_rewarder"]["count"]
            statistics["self_as_rewarder"]["average_reward_bonus"] = statistics["self_as_rewarder"]["total_reward_bonus"] / statistics["self_as_rewarder"]["count"]
        
        if statistics["partner_as_trustor"]["count"] > 0:
            statistics["partner_as_trustor"]["average_sent"] = statistics["partner_as_trustor"]["total_sent"] / statistics["partner_as_trustor"]["count"]
            statistics["partner_as_trustor"]["average_returned"] = statistics["partner_as_trustor"]["total_returned"] / statistics["partner_as_trustor"]["count"]
        
        if statistics["partner_as_trustee"]["count"] > 0:
            statistics["partner_as_trustee"]["average_sent"] = statistics["partner_as_trustee"]["total_sent"] / statistics["partner_as_trustee"]["count"]
            statistics["partner_as_trustee"]["average_returned"] = statistics["partner_as_trustee"]["total_returned"] / statistics["partner_as_trustee"]["count"]
        
        return statistics

# 信任博弈群体实验环境类 - 兼容V2框架
class TrustGamePopulationEnvironment:
    """Environment for Trust Game Population Experiment based on V2 framework"""
    
    def __init__(self, num_agents: int, initial_funds: int,
                 multiplication_factor: int, reward_budget: int,
                 reward_multiplier: int, num_rounds: int,
                 total_population_rounds: int, interaction_schedule: list):
        """Initialize environment with game settings"""
        self.num_agents = num_agents
        self.initial_funds = initial_funds
        self.multiplication_factor = multiplication_factor
        self.reward_budget = reward_budget
        self.reward_multiplier = reward_multiplier
        self.num_rounds = num_rounds
        self.total_population_rounds = total_population_rounds
        if num_agents % 3 != 0:
            raise ValueError("num_agents must be divisible by 3 for triadic interactions")
        self.interactions_per_population_round = num_agents // 3
        self.total_interactions = (
            total_population_rounds * self.interactions_per_population_round
        )
        self.interaction_schedule = interaction_schedule
        
        # Runtime state
        self.agents = []
        self.interaction_number = 0
        self.initial_time = None
        self.total_games = 0
        
        # Game statistics
        self.game_stats = defaultdict(list)
        self.game_logs = []
    
    def reset(self):
        """Reset environment for a new game"""
        self.interaction_number = 0
        self.initial_time = None
        
        # Reset history records for all agents
        for agent in self.agents:
            if hasattr(agent, '_history'):
                agent._history = []
            if hasattr(agent, '_reward_history'):
                agent._reward_history = []
            if hasattr(agent, '_interaction_partners'):
                agent._interaction_partners = defaultdict(lambda: {
                    "interaction_count": 0,
                    "self_as_trustor": [],
                    "self_as_trustee": [],
                    "partner_as_trustor": [],
                    "partner_as_trustee": []
                })
            if hasattr(agent, '_interaction_sequence'):
                agent._interaction_sequence = []
        
        # Increment game counter
        self.total_games += 1
    
    def set_agents(self, agents: list):
        """Set agents in the environment"""
        self.agents = agents
        # Set environment reference for each agent
        for agent in self.agents:
            if hasattr(agent, 'set_environment'):
                agent.set_environment(self)
    
    def _get_agent_by_id(self, agent_id):
        """Helper to get Agent object from ID."""
        return next((a for a in self.agents if a._id == agent_id), None)
    
    async def run_interaction(self, interaction_num: int) -> dict:
        """Execute one Trustor-Trustee-Rewarder interaction."""
        if self.initial_time is None:
            self.initial_time = datetime.now()
        
        self.interaction_number = interaction_num
        
        if interaction_num > len(self.interaction_schedule):
            logging.error("Interaction number exceeds schedule length.")
            return {}

        # Every schedule item is a trio. Each agent appears exactly once in a
        # population round, and the three roles are reshuffled every interaction.
        agent_ids = self.interaction_schedule[interaction_num - 1]
        if len(agent_ids) != 3:
            logging.error(f"Invalid trio in schedule: {agent_ids}")
            return {}

        trio = [self._get_agent_by_id(agent_id) for agent_id in agent_ids]
        if any(agent is None for agent in trio):
            logging.error(f"Agent trio {agent_ids} not found.")
            return {}

        # Randomly assign roles for this interaction
        roles = ["Trustor", "Trustee", "Rewarder"]
        random.shuffle(roles)
        role_to_agent = {role: agent for role, agent in zip(roles, trio)}
        trustor = role_to_agent["Trustor"]
        trustee = role_to_agent["Trustee"]
        rewarder = role_to_agent["Rewarder"]
        population_round = (
            (interaction_num - 1) // self.interactions_per_population_round + 1
        )

        interaction_summary = {
            "interaction": interaction_num,
            "population_round": population_round,
            "trio_ids": tuple(agent_ids),
            "trio_names": tuple(agent.name for agent in trio),
            "role_assignment": {
                "Trustor": trustor.name,
                "Trustee": trustee.name,
                "Rewarder": rewarder.name
            },
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Step 1: Trustor makes investment decision
            trustor_amount, trustor_explanation = await trustor.act(
                interaction_num=interaction_num,
                role="Trustor",
                initial_funds=self.initial_funds,
                multiplication_factor=self.multiplication_factor,
                num_rounds=self.num_rounds,
                current_funds=self.initial_funds,
                partner_name=trustee.name,
                reward_budget=self.reward_budget,
                reward_multiplier=self.reward_multiplier
            )

            # Step 2: Calculate received amount for Trustee
            trustee_received = trustor_amount * self.multiplication_factor

            # Step 3: Trustee makes return decision
            trustee_amount, trustee_explanation = await trustee.act(
                interaction_num=interaction_num,
                role="Trustee",
                initial_funds=self.initial_funds,
                multiplication_factor=self.multiplication_factor,
                num_rounds=self.num_rounds,
                current_funds=trustee_received,
                partner_name=trustor.name,
                reward_budget=self.reward_budget,
                reward_multiplier=self.reward_multiplier
            )

            # Step 4: Rewarder observes both decisions and chooses a costly reward.
            reward_cost, rewarder_explanation = await rewarder.act_as_rewarder(
                interaction_num=interaction_num,
                reward_budget=self.reward_budget,
                reward_multiplier=self.reward_multiplier,
                trustor_name=trustor.name,
                trustee_name=trustee.name,
                sent_amount=trustor_amount,
                received_amount=trustee_received,
                returned_amount=trustee_amount
            )
            reward_bonus = reward_cost * self.reward_multiplier

            # Step 5: Calculate final payoffs.
            trustor_payoff = self.initial_funds - trustor_amount + trustee_amount
            trustee_payoff_before_reward = trustee_received - trustee_amount
            trustee_payoff = trustee_payoff_before_reward + reward_bonus
            rewarder_payoff = self.reward_budget - reward_cost

            # Step 6: Update histories for all three roles.
            trustor_summary = {
                "interaction": interaction_num,
                "population_round": population_round,
                "role": "Trustor",
                "partner_id": trustee._id,
                "partner_name": trustee.name,
                "rewarder_id": rewarder._id,
                "rewarder_name": rewarder.name,
                "sent_amount": trustor_amount,
                "returned_amount": trustee_amount,
                "return_rate": (
                    trustee_amount / trustee_received
                    if trustee_received > 0 else 0.0
                ),
                "return_multiple": (
                    trustee_amount / trustor_amount
                    if trustor_amount > 0 else 0.0
                ),
                "reward_cost": reward_cost,
                "reward_bonus_to_trustee": reward_bonus,
                "payoff": trustor_payoff,
                "partner_payoff": trustee_payoff,
                "explanation": trustor_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat()
            }
            
            trustee_summary = {
                "interaction": interaction_num,
                "population_round": population_round,
                "role": "Trustee",
                "partner_id": trustor._id,
                "partner_name": trustor.name,
                "rewarder_id": rewarder._id,
                "rewarder_name": rewarder.name,
                "sent_amount": trustor_amount,
                "received_amount": trustee_received,
                "returned_amount": trustee_amount,
                "reward_cost": reward_cost,
                "reward_bonus": reward_bonus,
                "payoff_before_reward": trustee_payoff_before_reward,
                "payoff": trustee_payoff,
                "partner_payoff": trustor_payoff,
                "explanation": trustee_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat()
            }
            rewarder_summary = {
                "interaction": interaction_num,
                "population_round": population_round,
                "role": "Rewarder",
                "trustor_id": trustor._id,
                "trustor_name": trustor.name,
                "trustee_id": trustee._id,
                "trustee_name": trustee.name,
                "sent_amount": trustor_amount,
                "received_amount": trustee_received,
                "returned_amount": trustee_amount,
                "reward_cost": reward_cost,
                "reward_bonus": reward_bonus,
                "payoff": rewarder_payoff,
                "explanation": rewarder_explanation,
                "reward_multiplier": self.reward_multiplier,
                "timestamp": datetime.now().isoformat()
            }

            trustor.update_history(trustor_summary)
            trustee.update_history(trustee_summary)
            rewarder.update_reward_history(rewarder_summary)

            # Update total payoffs
            trustor._total_payoff += trustor_payoff
            trustee._total_payoff += trustee_payoff
            rewarder._total_payoff += rewarder_payoff

            # Step 7: Update environment statistics
            interaction_detail = {
                "trustor_id": trustor._id,
                "trustor_name": trustor.name,
                "trustee_id": trustee._id,
                "trustee_name": trustee.name,
                "rewarder_id": rewarder._id,
                "rewarder_name": rewarder.name,
                "sent_amount": trustor_amount,
                "returned_amount": trustee_amount,
                "trustee_received": trustee_received,
                "return_rate": (
                    trustee_amount / trustee_received
                    if trustee_received > 0 else 0.0
                ),
                "return_multiple": (
                    trustee_amount / trustor_amount
                    if trustor_amount > 0 else 0.0
                ),
                "reward_cost": reward_cost,
                "reward_bonus": reward_bonus,
                "trustor_payoff": trustor_payoff,
                "trustee_payoff_before_reward": trustee_payoff_before_reward,
                "trustee_payoff": trustee_payoff,
                "rewarder_payoff": rewarder_payoff,
                "trustor_explanation": trustor_explanation,
                "trustee_explanation": trustee_explanation,
                "rewarder_explanation": rewarder_explanation
            }

        except Exception as e:
            logging.error(
                f"Error in interaction {interaction_num}: {e}\n{traceback.format_exc()}"
            )
            return {}

        interaction_summary['detail'] = interaction_detail
        self.game_logs.append(interaction_summary)

        return interaction_summary


def _generate_triadic_schedule(num_agents: int,
                               total_population_rounds: int) -> list:
    """Create rounds of disjoint trios so every agent acts once per round."""
    if num_agents <= 0 or num_agents % 3 != 0:
        raise ValueError("num_agents must be a positive multiple of 3")
    if total_population_rounds <= 0:
        raise ValueError("total_population_rounds must be positive")

    schedule = []
    agent_ids = list(range(num_agents))
    for _ in range(total_population_rounds):
        round_ids = agent_ids.copy()
        random.shuffle(round_ids)
        schedule.extend(
            tuple(round_ids[index:index + 3])
            for index in range(0, num_agents, 3)
        )
    return schedule

async def main():
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # Experiment Configuration
    EXPERIMENT_CONFIG = {
        "experiment_name": "TrustGame_Anticipated_ThirdParty_Reward",
        "description": (
            "24-agent Trust Game with anticipated, costly third-party reward "
            "adapted from Zhurakhovska"
        ),
        "game_settings": {
            "num_agents": 24,  # Total number of agents
            "initial_funds": 10,  # Initial coins per Trustor per interaction
            "multiplication_factor": 3,  # Investment multiplication factor
            "reward_budget": 10,  # Normalized from the paper's 100 ECU helper budget
            "reward_multiplier": 3,  # Every reward coin gives Trustee 3 coins
            "reward_timing": "anticipated",  # Announced before Trustor/Trustee decisions
            "num_rounds": 30,  # Compatibility field; equals population rounds
            "total_population_rounds": 30,  # Total population rounds

        },
        "agent_settings": {
            "decision_timeout": 30.0,  # Timeout for agent decisions (seconds)
            "max_retries": 4,  # Maximum LLM call retries
            "retry_delay": 0.5  # Delay between retries (seconds)
        },
        "output_settings": {
            "save_raw_logs": True,
            "save_agent_histories": True,  # Changed to True to save enhanced histories
            "generate_summary_stats": True  # Changed to True to generate role statistics
        }
    }
    
    try:
        # User-defined folder name
        experiment_name = input("Please enter experiment folder name (e.g., 'TG_reward_test1'): ").strip()
        if not experiment_name:
            experiment_name = f"TrustGame_Reward_{datetime.now().strftime('%m%d%H%M')}"
        
        # 可配置的实验参数
        NUM_GAMES = 1  # 运行次数
        ROUNDS_PER_GAME = 30  # 每局游戏轮数
        
        # 更新配置
        EXPERIMENT_CONFIG["game_settings"]["total_population_rounds"] = ROUNDS_PER_GAME
        EXPERIMENT_CONFIG["game_settings"]["num_rounds"] = ROUNDS_PER_GAME
        
        # Initialize LLM System
        print("Initializing LLM system...")
        llm_agent = await _initialize_llm_system(EXPERIMENT_CONFIG)
        EXPERIMENT_CONFIG["game_settings"]["llm_model"] = getattr(llm_agent, 'model', None)
        
        # Setup Result Directory
        print("Setting up result directory...")
        result_dir = await _setup_result_directory(EXPERIMENT_CONFIG, experiment_name)
        await _save_experiment_config(EXPERIMENT_CONFIG, result_dir)
        
        # Initialize Statistics Collection
        experiment_stats = {
            "per_game_results": [],
            "overall_metrics": {
                "average_sent_per_interaction": [],
                "average_returned_per_interaction": [],
                "average_return_rate_per_interaction": [],
                "average_reward_cost_per_interaction": [],
                "average_reward_bonus_per_interaction": [],
                "total_payoffs_per_agent": defaultdict(float),
                "interaction_metrics_time_series": [],
                "role_statistics_per_agent": {}  # Added for enhanced role statistics
            },
            "experiment_start_time": datetime.now().isoformat(),
            "experiment_end_time": None
        }
        
        num_agents = EXPERIMENT_CONFIG["game_settings"]["num_agents"]
        total_population_rounds = EXPERIMENT_CONFIG["game_settings"]["total_population_rounds"]
        if num_agents % 3 != 0:
            raise ValueError("The reward treatment requires num_agents divisible by 3")
        interactions_per_population_round = num_agents // 3
        total_interactions = (
            total_population_rounds * interactions_per_population_round
        )
        
        # Run specified number of independent games
        for game_num in range(NUM_GAMES):
            print(f"\n" + "="*70)
            print(f"Starting Game {game_num + 1}")
            print("="*70)
            
            # Pre-generate Interaction Schedule for this game
            print("Generating interaction schedule...")
            interaction_schedule = _generate_triadic_schedule(
                num_agents=num_agents,
                total_population_rounds=total_population_rounds
            )
            
            # Create environment
            print("Creating environment...")
            env = TrustGamePopulationEnvironment(
                num_agents=EXPERIMENT_CONFIG["game_settings"]["num_agents"],
                initial_funds=EXPERIMENT_CONFIG["game_settings"]["initial_funds"],
                multiplication_factor=EXPERIMENT_CONFIG["game_settings"]["multiplication_factor"],
                reward_budget=EXPERIMENT_CONFIG["game_settings"]["reward_budget"],
                reward_multiplier=EXPERIMENT_CONFIG["game_settings"]["reward_multiplier"],
                num_rounds=EXPERIMENT_CONFIG["game_settings"]["num_rounds"],
                total_population_rounds=EXPERIMENT_CONFIG["game_settings"]["total_population_rounds"],
                interaction_schedule=interaction_schedule
            )
            
            # Create agents
            print("Creating agents...")
            agents = await _create_agents(EXPERIMENT_CONFIG, llm_agent)
            env.set_agents(agents)
            env.reset()
            
            # Run experiment
            print("\n" + "="*60)
            print(f"Trust Game Population Experiment - Game {game_num + 1}")
            print("="*60)
            print(f"Number of agents: {EXPERIMENT_CONFIG['game_settings']['num_agents']}")
            print(f"LLM Model: {EXPERIMENT_CONFIG['game_settings'].get('llm_model', 'None')}")
            print(f"Total population rounds: {EXPERIMENT_CONFIG['game_settings']['total_population_rounds']}")
            print(f"Triads per population round: {interactions_per_population_round}")
            print(f"Total interactions: {total_interactions}")
            print(
                "Reward mechanism: anticipated third-party reward, "
                f"budget={EXPERIMENT_CONFIG['game_settings']['reward_budget']}, "
                f"multiplier={EXPERIMENT_CONFIG['game_settings']['reward_multiplier']}"
            )
            print(f"Experiment folder: {experiment_name}")
            print(f"Results will be saved to: {result_dir}")
            print("="*60)
            
            # Initialize game-specific statistics
            game_stats = {
                "game_number": game_num + 1,
                "interactions": [],
                "agent_payoffs": defaultdict(float),
                "start_time": datetime.now().isoformat(),
                "end_time": None
            }
            
            # Run all interactions
            start_time = time.time()
            
            for interaction_num in range(1, total_interactions + 1):
                # Calculate current population round
                current_population_round = (
                    (interaction_num - 1) // interactions_per_population_round + 1
                )
                
                if interaction_num % interactions_per_population_round == 1:
                    print(f"\n--- Population Round {current_population_round}/{total_population_rounds} ---")
                    print(f"Starting interaction {interaction_num}...")
                
                try:
                    # Execute interaction
                    interaction_result = await env.run_interaction(interaction_num)
                    
                    if interaction_result:
                        # Update experiment statistics
                        await _update_experiment_statistics(interaction_num, interaction_result, experiment_stats)
                        game_stats["interactions"].append(interaction_result)
                        
                        # Update game-specific agent payoffs
                        if "trustor_name" in interaction_result.get("detail", {}):
                            trustor_name = interaction_result["detail"]["trustor_name"]
                            trustee_name = interaction_result["detail"]["trustee_name"]
                            rewarder_name = interaction_result["detail"]["rewarder_name"]
                            game_stats["agent_payoffs"][trustor_name] += interaction_result["detail"].get("trustor_payoff", 0)
                            game_stats["agent_payoffs"][trustee_name] += interaction_result["detail"].get("trustee_payoff", 0)
                            game_stats["agent_payoffs"][rewarder_name] += interaction_result["detail"].get("rewarder_payoff", 0)
                        
                        # Print interaction summary
                        if interaction_num % interactions_per_population_round == 0:
                            await _print_population_round_summary(interaction_num, env)
                            
                except Exception as e:
                    logging.error(f"Interaction {interaction_num} failed: {str(e)}")
                    continue
            
            end_time = time.time()
            duration = end_time - start_time
            game_stats["end_time"] = datetime.now().isoformat()
            
            # Collect enhanced role statistics for each agent
            game_role_stats = {}
            for agent in agents:
                if hasattr(agent, 'get_role_statistics'):
                    role_stats = agent.get_role_statistics()
                    game_role_stats[agent.name] = role_stats
                    experiment_stats["overall_metrics"]["role_statistics_per_agent"][agent.name] = role_stats
            
            game_stats["role_statistics"] = game_role_stats
            experiment_stats["per_game_results"].append(game_stats)
            
            # Save results for this game
            await _save_game_results(EXPERIMENT_CONFIG, env, agents, result_dir, experiment_stats, game_num + 1)
            
            print(f"\n{'='*60}")
            print(f"Game {game_num + 1} completed!")
            print(f"Elapsed time: {duration:.2f} seconds")
            print(f"{'='*60}")
        
        # Generate summary
        experiment_stats["experiment_end_time"] = datetime.now().isoformat()
        
        print(f"\n" + "="*70)
        print("All Games Completed Successfully!")
        print("="*70)
        print(f"Total games: {NUM_GAMES}")
        print(f"Results directory: {result_dir}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        return
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        print(f"\nExperiment failed with error: {str(e)}")
        return

async def _initialize_llm_system(config: dict) -> AgentLLM:
    """Initialize LLM system for agents"""
    try:
        # Initialize LLM agent
        llm_agent = LLMAgent(name="TrustGamePopulation_LLM")
        
        # Create AgentLLM wrapper with LLMAgent as router
        from llm_cooperation_lab.agent.base import AgentLLM
        agent_llm = AgentLLM(
            router=llm_agent,
            model_name=getattr(llm_agent, 'model', None)
        )
        
        logging.info(f"LLM system initialized with model: {agent_llm.model_name}")
        return agent_llm
        
    except Exception as e:
        logging.error(f"LLM initialization failed: {str(e)}")
        raise Exception(f"Failed to initialize LLM system: {str(e)}") from e

async def _setup_result_directory(config: dict, experiment_name: str) -> str:
    """Create and setup result directory structure"""
    base_dir = "result_trust_game_population_reward"
    
    # Create main result directory
    result_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create data subdirectory
    data_dir = os.path.join(result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    logging.info(f"Result directory structure created at: {result_dir}")
    return result_dir

async def _save_experiment_config(config: dict, result_dir: str):
    """Save experiment configuration to JSON file"""
    config_path = os.path.join(result_dir, "experiment_config.json")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"Experiment configuration saved to: {config_path}")

async def _create_agents(config: dict, llm_agent: AgentLLM) -> list:
    """Create trust game population agents"""
    agents = []
    num_agents = config["game_settings"]["num_agents"]
    
    # Create agents - random role assignment will happen during interactions
    for i in range(num_agents):
        agent_name = f"Agent_{i+1}"
        # Use a neutral profile initially, roles will be assigned per interaction
        neutral_profile = (
            f"You are {agent_name}, a decision-maker participating in a Trust Game "
            "with anticipated third-party reward.\n"
            "In each population round you are randomly assigned as Trustor, Trustee, or Rewarder.\n"
            "Make each decision from the information and material incentives shown in that role's prompt."
        )
        
        agent = TrustGamePopulationAgent(
            id=i,
            name=agent_name,
            profile=neutral_profile
        )
        
        # Initialize with LLM
        await agent.init(llm_agent)
        agents.append(agent)
    
    logging.info(f"Created {len(agents)} population agents")
    return agents

async def _update_experiment_statistics(interaction_num: int, interaction_result: dict,
                                       experiment_stats: dict):
    """Update experiment statistics with interaction results"""
    detail = interaction_result.get("detail", {})
    
    sent_amount = detail.get("sent_amount", 0)
    returned_amount = detail.get("returned_amount", 0)
    trustee_received = detail.get("trustee_received", 0)
    reward_cost = detail.get("reward_cost", 0)
    reward_bonus = detail.get("reward_bonus", 0)
    
    # Update basic statistics
    experiment_stats["overall_metrics"]["average_sent_per_interaction"].append(sent_amount)
    experiment_stats["overall_metrics"]["average_returned_per_interaction"].append(returned_amount)
    experiment_stats["overall_metrics"]["average_reward_cost_per_interaction"].append(reward_cost)
    experiment_stats["overall_metrics"]["average_reward_bonus_per_interaction"].append(reward_bonus)
    
    # Calculate return rate if sent_amount > 0
    if trustee_received > 0:
        return_rate = returned_amount / trustee_received
    else:
        return_rate = 0.0
    
    experiment_stats["overall_metrics"]["average_return_rate_per_interaction"].append(return_rate)
    
    # Update total payoffs per agent
    trustor_name = detail.get("trustor_name", "")
    trustee_name = detail.get("trustee_name", "")
    rewarder_name = detail.get("rewarder_name", "")
    trustor_payoff = detail.get("trustor_payoff", 0)
    trustee_payoff = detail.get("trustee_payoff", 0)
    rewarder_payoff = detail.get("rewarder_payoff", 0)
    
    experiment_stats["overall_metrics"]["total_payoffs_per_agent"][trustor_name] += trustor_payoff
    experiment_stats["overall_metrics"]["total_payoffs_per_agent"][trustee_name] += trustee_payoff
    experiment_stats["overall_metrics"]["total_payoffs_per_agent"][rewarder_name] += rewarder_payoff

async def _print_population_round_summary(interaction_num: int, env: TrustGamePopulationEnvironment):
    """Print summary of completed population round"""
    current_round = (
        (interaction_num - 1) // env.interactions_per_population_round + 1
    )
    
    # Calculate total sent and returned for this round
    round_interactions = env.game_logs[-env.interactions_per_population_round:]
    total_sent = sum(interaction.get("detail", {}).get("sent_amount", 0) for interaction in round_interactions)
    total_returned = sum(interaction.get("detail", {}).get("returned_amount", 0) for interaction in round_interactions)
    total_reward_cost = sum(interaction.get("detail", {}).get("reward_cost", 0) for interaction in round_interactions)
    total_reward_bonus = sum(interaction.get("detail", {}).get("reward_bonus", 0) for interaction in round_interactions)
    
    avg_sent = total_sent / len(round_interactions) if round_interactions else 0
    avg_returned = total_returned / len(round_interactions) if round_interactions else 0
    avg_reward_cost = total_reward_cost / len(round_interactions) if round_interactions else 0
    
    print(f"Population Round {current_round} completed!")
    print(f"  Total sent across all interactions: {total_sent}")
    print(f"  Total returned across all interactions: {total_returned}")
    print(f"  Average sent per interaction: {avg_sent:.2f}")
    print(f"  Average returned per interaction: {avg_returned:.2f}")
    print(f"  Total Rewarder spending: {total_reward_cost:.2f}")
    print(f"  Total reward bonus received by Trustees: {total_reward_bonus:.2f}")
    print(f"  Average Rewarder spending per interaction: {avg_reward_cost:.2f}")

async def _save_game_results(config: dict, env: TrustGamePopulationEnvironment, agents: list,
                            result_dir: str, experiment_stats: dict, game_num: int = 1):
    """Save game results to output files"""
    # Save game logs with game number
    if config["output_settings"]["save_raw_logs"]:
        logs_path = os.path.join(result_dir, "data", f"game_log{game_num}.json")
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(env.game_logs, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"Game {game_num} logs saved to: {logs_path}")
    
    # Save agent histories
    if config["output_settings"]["save_agent_histories"]:
        agent_histories_dir = os.path.join(result_dir, "data", "agent_histories")
        os.makedirs(agent_histories_dir, exist_ok=True)
        
        for agent in agents:
            try:
                agent_data = await agent.dump()
                agent_file_path = os.path.join(agent_histories_dir, f"{agent.name}_game{game_num}.json")
                with open(agent_file_path, "w", encoding="utf-8") as f:
                    json.dump(agent_data, f, indent=2, ensure_ascii=False, default=str)
                
                logging.info(f"Agent {agent.name} history for game {game_num} saved to: {agent_file_path}")
            except Exception as e:
                logging.error(f"Failed to save history for agent {agent.name}: {str(e)}")
    
    # Save experiment statistics
    stats_path = os.path.join(result_dir, "data", "experiment_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(experiment_stats, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"Experiment statistics saved to: {stats_path}")
    
    # Generate summary statistics file
    if config["output_settings"]["generate_summary_stats"]:
        summary_stats = {
            "average_sent_per_interaction": np.mean(experiment_stats["overall_metrics"]["average_sent_per_interaction"]),
            "average_returned_per_interaction": np.mean(experiment_stats["overall_metrics"]["average_returned_per_interaction"]),
            "average_return_rate_per_interaction": np.mean(experiment_stats["overall_metrics"]["average_return_rate_per_interaction"]),
            "average_reward_cost_per_interaction": np.mean(experiment_stats["overall_metrics"]["average_reward_cost_per_interaction"]),
            "average_reward_bonus_per_interaction": np.mean(experiment_stats["overall_metrics"]["average_reward_bonus_per_interaction"]),
            "total_interactions": len(experiment_stats["overall_metrics"]["average_sent_per_interaction"]),
            "total_payoffs_per_agent": experiment_stats["overall_metrics"]["total_payoffs_per_agent"],
            "role_statistics_per_agent": experiment_stats["overall_metrics"]["role_statistics_per_agent"]  # Added for enhanced role statistics
        }
        
        summary_path = os.path.join(result_dir, "data", "summary_stats.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"Summary statistics saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
