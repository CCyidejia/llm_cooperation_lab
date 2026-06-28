#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trust Game Population + HR 名誉机制（Trustor 在全局第 N≥2 次交互时先选择是否查看对方作为 Trustee 的名誉分）。

后台变量（每名 Agent）：history_trustee_count、history_total_received、history_total_returned；
名誉分 Reputation = history_total_returned / history_total_received（展示时计算；分母为 0 则无记录）。

Prompt：在 env_main_trust_game_group.py 基础上，第 N≥2 轮 Trustor 增加「是否查看」与可选的名誉追加段；Trustee 与原文件一致。
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
from agentsociety2.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv
from LLMAPI.wuwen import LLMAgent

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_trust_game_population_reputation", exist_ok=True)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_api_log.txt"),
        logging.StreamHandler()
    ]
)

# 全局随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 基于V2框架的信任博弈代理实现 - 支持群体交互 + HR 名誉
class TrustGamePopulationReputationAgent(AgentBase):
    """Trust Game Population Agent + HR reputation aggregates (Trustee-role history)."""
    
    def __init__(self, id: int, name: str, profile: str):
        """Initialize TrustGamePopulationReputationAgent"""
        # Initialize basic properties
        self._id = id
        self._name = name  # Agent name
        self._profile = profile  # Agent role description
        self._history = []  # Store interaction history with different partners
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
        self.logger = logging.getLogger(f"TrustGamePopulationReputationAgent.{self._name}")
        
        self.logger.info(f"TrustGamePopulationReputationAgent initialized: {self._name}")
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

        # HR 名誉底层变量（仅统计作为 Trustee 时；由环境在每轮交互后更新）
        self.history_trustee_count = 0
        self.history_total_received = 0  # 作为 Trustee 时收到的金币累计（= Trustor 发送额 × m）
        self.history_total_returned = 0  # 作为 Trustee 时返还给 Trustor 的金币累计
    
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
                "agent_type": "TrustGamePopulationReputationAgent",
                "profile": self._profile,
                "history": self._history,
                "interaction_partners": dict(self._interaction_partners),
                "interaction_sequence": self._interaction_sequence,
                "initial_funds": self._initial_funds,
                "multiplication_factor": self._multiplication_factor,
                "num_rounds": self._num_rounds,
                "total_payoff": self._total_payoff,
                "decision_count": self._decision_count,
                "error_count": self._error_count,
                "history_trustee_count": self.history_trustee_count,
                "history_total_received": self.history_total_received,
                "history_total_returned": self.history_total_returned,
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
                "agent_type": "TrustGamePopulationReputationAgent",
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
            self.history_trustee_count = dump_data.get("history_trustee_count", getattr(self, "history_trustee_count", 0))
            self.history_total_received = dump_data.get("history_total_received", getattr(self, "history_total_received", 0))
            self.history_total_returned = dump_data.get("history_total_returned", getattr(self, "history_total_returned", 0))
            
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
                 partner_name: str) -> tuple[int, str]:
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
                                                   current_funds, history_str, partner_name)
            else:  # Trustee
                prompt = self._build_trustee_prompt(interaction_num, initial_funds,
                                                   multiplication_factor, num_rounds,
                                                   current_funds, history_str, partner_name)
            
            # Call LLM with retry mechanism - use role-specific profile
            role_profile = (
                f"You are {self._name}, Trustor in a game.\n"
                "Goal: Maximize cumulative coins.\n"
                f"Rules: Start with {self._initial_funds} coins/round.\n"
                f"Send 0-{self._initial_funds} coins (×{self._multiplication_factor} for Trustee).\n"
                f"Trustee returns some coins. Your payoff: {self._initial_funds} - sent + returned.\n"
                "Consider past Trustee behavior."
            ) if role == "Trustor" else (
                f"You are {self._name}, Trustee in a game.\n"
                "Goal: Maximize cumulative coins.\n"
                f"Rules: Receive (sent amount)×{self._multiplication_factor} coins from Trustor.\n"
                "Return 0-received coins. Your payoff: received - returned.\n"
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

    def _parse_yes_no(self, content: str) -> bool:
        """解析 Yes/No；无法解析时视为 No。"""
        if not content or not str(content).strip():
            return False
        m = re.search(r"\b(yes|no)\b", content.strip(), re.IGNORECASE)
        if not m:
            return False
        return m.group(1).lower() == "yes"

    async def choose_view_reputation(self, interaction_num: int, partner_name: str) -> bool:
        """阶段1：是否查看 Trustee 名誉（仅回答 Yes 或 No）。"""
        prompt = (
            f"{self._profile}\n"
            f"Round {interaction_num}\n"
            "---\n"
            f"You will next decide how many coins to send to {partner_name} (they will be Trustee).\n"
            "Do you choose to view this partner's reputation score (based on their past total returned and total received when they were Trustee)? "
            "Answer with exactly one word: Yes or No.\n"
        )
        role_profile = (
            f"You are {self._name}, Trustor in a game.\n"
            "Goal: Maximize cumulative coins.\n"
            f"Rules: Start with {self._initial_funds} coins/round.\n"
            f"Send 0-{self._initial_funds} coins (×{self._multiplication_factor} for Trustee).\n"
            f"Trustee returns some coins. Your payoff: {self._initial_funds} - sent + returned.\n"
            "Consider past Trustee behavior."
        )
        try:
            content = await self._call_llm_with_retry(
                system_message=role_profile,
                user_prompt=prompt,
            )
            return self._parse_yes_no(content)
        except Exception as e:
            self.logger.error(f"[{self._name}] choose_view_reputation failed: {e}")
            self._error_count += 1
            return False

    async def act_trustor_investment(
        self,
        interaction_num: int,
        initial_funds: int,
        multiplication_factor: int,
        num_rounds: int,
        current_funds: int,
        partner_name: str,
        reputation_addon: str | None,
    ) -> tuple[int, str]:
        """阶段2：Trustor 投资；user prompt 与原版 _build_trustor_prompt 一致，可追加名誉说明。"""
        try:
            self._current_funds = current_funds
            history_str = self._build_partner_history_string(partner_name)
            prompt = self._build_trustor_prompt(
                interaction_num, initial_funds, multiplication_factor, num_rounds,
                current_funds, history_str, partner_name
            )
            if reputation_addon:
                prompt = prompt + "\n---\n" + reputation_addon.strip()
            role_profile = (
                f"You are {self._name}, Trustor in a game.\n"
                "Goal: Maximize cumulative coins.\n"
                f"Rules: Start with {self._initial_funds} coins/round.\n"
                f"Send 0-{self._initial_funds} coins (×{self._multiplication_factor} for Trustee).\n"
                f"Trustee returns some coins. Your payoff: {self._initial_funds} - sent + returned.\n"
                "Consider past Trustee behavior."
            )
            content = await self._call_llm_with_retry(
                system_message=role_profile,
                user_prompt=prompt,
            )
            if not content or content.isspace():
                raise ValueError("LLM returned empty response")
            amount = self._parse_amount(content, "Trustor", initial_funds)
            explanation = self._extract_explanation(content)
            self._decision_count += 1
            self.logger.info(
                f"[{self._name}] [Trustor] Decision: {amount} coins"
            )
            return amount, explanation
        except Exception as e:
            error_msg = f"Decision making failed: {type(e).__name__} - {str(e)}"
            self.logger.error(f"[{self._name}] [Trustor investment] {error_msg}")
            self._error_count += 1
            return 0, f"[CRITICAL FAILURE] {error_msg}, using default amount: 0"
    
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
                history_lines.append(f"Round {i}: You sent {sent_amount} coins. {partner_name} returned {returned_amount} coins. Your payoff: {payoff:.2f} coins.")
            else:  # Trustee
                sent_amount = record.get("sent_amount", 0)
                received_amount = record.get("received_amount", 0)
                returned_amount = record.get("returned_amount", 0)
                payoff = record.get("payoff", 0)
                history_lines.append(f"Round {i}: You received {received_amount} coins (from {partner_name}'s {sent_amount} coins * {record.get('multiplication_factor', 3)}). You returned {returned_amount} coins. Your payoff: {payoff:.2f} coins.")
        
        return "\n".join(history_lines)
    
    def _build_trustor_prompt(self, interaction_num: int, initial_funds: int,
                             multiplication_factor: int, num_rounds: int,
                             current_funds: int, history_str: str, partner_name: str) -> str:
        """Build prompt for Trustor role"""
        return (
            f"Round {interaction_num}\n"
            f"Current funds: {current_funds}\n"
            f"Send 0-{initial_funds} coins to {partner_name} (amount × {multiplication_factor} for them)\n"
            f"Your goal: Maximize total coins (payoff = {initial_funds} - sent + returned)\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            "Amount: [0-10]\n"
            "Explanation: [brief strategic reason]"
        )
    
    def _build_trustee_prompt(self, interaction_num: int, initial_funds: int,
                             multiplication_factor: int, num_rounds: int,
                             current_funds: int, history_str: str, partner_name: str) -> str:
        """Build prompt for Trustee role"""
        return (
            f"Round {interaction_num}\n"
            f"Received: {current_funds} coins ({partner_name} sent {current_funds/multiplication_factor:.1f})\n"
            f"Return 0-{current_funds} coins to {partner_name}\n"
            f"Your goal: Maximize total coins (payoff = {current_funds} - returned)\n\n"
            f"{history_str}\n\n"
            "Decision format:\n"
            f"Amount: [0-{current_funds}]\n"
            "Explanation: [brief strategic reason]"
        )
    
    def _parse_amount(self, content: str, role: str, max_amount: int) -> int:
        """Parse numerical amount from LLM response"""
        # 只匹配在合理范围内的数字，避免匹配错误信息中的数字（如402错误码）
        # 对于Trustor，max_amount通常是10；对于Trustee，max_amount是收到的金额
        if max_amount <= 10:
            # Trustor角色：只匹配0-10的数字
            match = re.search(r'\b([0-9]|10)\b', content)
        else:
            # Trustee角色：匹配0到max_amount的数字
            match = re.search(r'\b(\d+)\b', content)
        
        if not match:
            raise ValueError(f"Failed to parse valid numerical amount from response: {content[:200]}...")
        
        parsed_val = int(match.group(0))
        
        # Validate amount range based on role
        if not (0 <= parsed_val <= max_amount):
            self.logger.warning(f"[{self._name}] Invalid {role.lower()} amount ({parsed_val}) detected, must be between 0 and {max_amount}")
            return 0  # Use 0 as safe default
        
        return parsed_val
    
    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from LLM response"""
        # Look for explanation after the amount
        match = re.search(r'\b\d+\b', content)
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
            statistics["self_as_trustee"]["total_received"] += sum(entry["sent_amount"] for entry in self_as_trustee)
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
        
        if statistics["partner_as_trustor"]["count"] > 0:
            statistics["partner_as_trustor"]["average_sent"] = statistics["partner_as_trustor"]["total_sent"] / statistics["partner_as_trustor"]["count"]
            statistics["partner_as_trustor"]["average_returned"] = statistics["partner_as_trustor"]["total_returned"] / statistics["partner_as_trustor"]["count"]
        
        if statistics["partner_as_trustee"]["count"] > 0:
            statistics["partner_as_trustee"]["average_sent"] = statistics["partner_as_trustee"]["total_sent"] / statistics["partner_as_trustee"]["count"]
            statistics["partner_as_trustee"]["average_returned"] = statistics["partner_as_trustee"]["total_returned"] / statistics["partner_as_trustee"]["count"]
        
        return statistics


def format_trustee_reputation_addon(trustee: "TrustGamePopulationReputationAgent") -> str:
    """供 Trustor 选择查看时追加到 user prompt；Reputation = returned/received（作为 Trustee 时累计）。"""
    if trustee.history_total_received <= 0:
        return (
            f"Reputation of {trustee.name} (total returned / total received as Trustee): "
            "not available (no prior Trustee record)."
        )
    rep = trustee.history_total_returned / trustee.history_total_received
    return (
        f"Reputation of {trustee.name} (total returned / total received as Trustee): "
        f"{trustee.history_total_returned} / {trustee.history_total_received} = {rep:.4f}."
    )


# 信任博弈群体实验环境类 - 兼容V2框架 + 名誉更新
class TrustGamePopulationReputationEnvironment:
    """Environment for Trust Game Population + HR reputation."""
    
    def __init__(self, num_agents: int, initial_funds: int,
                 multiplication_factor: int, num_rounds: int,
                 total_population_rounds: int, interaction_schedule: list):
        """Initialize environment with game settings"""
        self.num_agents = num_agents
        self.initial_funds = initial_funds
        self.multiplication_factor = multiplication_factor
        self.num_rounds = num_rounds
        self.total_population_rounds = total_population_rounds
        self.total_interactions = total_population_rounds * (num_agents // 2)
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
            if hasattr(agent, "history_trustee_count"):
                agent.history_trustee_count = 0
                agent.history_total_received = 0
                agent.history_total_returned = 0
        
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
        """Execute a single interaction using the pre-generated schedule"""
        if self.initial_time is None:
            self.initial_time = datetime.now()
        
        self.interaction_number = interaction_num
        
        if interaction_num > len(self.interaction_schedule):
            logging.error("Interaction number exceeds schedule length.")
            return {}

        # Get Agent IDs from the pre-generated schedule
        agent1_id, agent2_id = self.interaction_schedule[interaction_num - 1]
        
        agent1 = self._get_agent_by_id(agent1_id)
        agent2 = self._get_agent_by_id(agent2_id)
        
        if not agent1 or not agent2:
            logging.error(f"Agent pair {agent1_id}, {agent2_id} not found.")
            return {}
        
        # Randomly assign roles for this interaction
        roles = ["Trustor", "Trustee"]
        random.shuffle(roles)
        agent1_role, agent2_role = roles
        
        interaction_summary = {
            "interaction": interaction_num,
            "pair_ids": (agent1_id, agent2_id),
            "pair_names": (agent1.name, agent2.name),
            "roles": (agent1_role, agent2_role),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Trustor makes investment decision
            trustor = agent1 if agent1_role == "Trustor" else agent2
            trustee = agent1 if agent1_role == "Trustee" else agent2
            
            trustor_partner = trustee.name
            trustee_partner = trustor.name
            
            # Trustor：第 1 次全局交互为单阶段（与原版一致）；第 N≥2 轮先选是否查看名誉，再投资
            view_reputation: bool | None = None
            reputation_addon: str | None = None
            if interaction_num >= 2:
                view_reputation = await trustor.choose_view_reputation(
                    interaction_num, trustor_partner
                )
                reputation_addon = (
                    format_trustee_reputation_addon(trustee) if view_reputation else None
                )
                trustor_amount, trustor_explanation = await trustor.act_trustor_investment(
                    interaction_num=interaction_num,
                    initial_funds=self.initial_funds,
                    multiplication_factor=self.multiplication_factor,
                    num_rounds=self.num_rounds,
                    current_funds=self.initial_funds,
                    partner_name=trustor_partner,
                    reputation_addon=reputation_addon,
                )
            else:
                trustor_amount, trustor_explanation = await trustor.act(
                    interaction_num=interaction_num,
                    role="Trustor",
                    initial_funds=self.initial_funds,
                    multiplication_factor=self.multiplication_factor,
                    num_rounds=self.num_rounds,
                    current_funds=self.initial_funds,
                    partner_name=trustor_partner
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
                partner_name=trustee_partner
            )
            
            # Step 4: Calculate payoffs
            trustor_payoff = self.initial_funds - trustor_amount + trustee_amount
            trustee_payoff = trustee_received - trustee_amount
            
            # HR：本交互决策前 Trustee 的名誉比（累计返还/累计收到，作为 Trustee 时）
            trustee_reputation_before = (
                trustee.history_total_returned / trustee.history_total_received
                if trustee.history_total_received > 0
                else None
            )
            
            # Step 5: Update agent histories with partner payoff for enhanced tracking
            trustor_summary = {
                "interaction": interaction_num,
                "role": "Trustor",
                "partner_id": trustee._id,
                "partner_name": trustee.name,
                "sent_amount": trustor_amount,
                "returned_amount": trustee_amount,
                "payoff": trustor_payoff,
                "partner_payoff": trustee_payoff,  # Added for enhanced history tracking
                "explanation": trustor_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat()
            }
            
            trustee_summary = {
                "interaction": interaction_num,
                "role": "Trustee",
                "partner_id": trustor._id,
                "partner_name": trustor.name,
                "sent_amount": trustor_amount,  # Amount sent by trustor
                "received_amount": trustee_received,
                "returned_amount": trustee_amount,
                "payoff": trustee_payoff,
                "partner_payoff": trustor_payoff,  # Added for enhanced history tracking
                "explanation": trustee_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat()
            }
            
            trustor.update_history(trustor_summary)
            trustee.update_history(trustee_summary)
            
            # HR：本轮结束后更新 Trustee 的全局累计（供后续交互计算名誉）
            trustee.history_trustee_count += 1
            trustee.history_total_received += int(trustee_received)
            trustee.history_total_returned += int(trustee_amount)
            
            # Update total payoffs
            trustor._total_payoff += trustor_payoff
            trustee._total_payoff += trustee_payoff
            
            # Step 6: Update environment statistics
            interaction_detail = {
                "trustor_id": trustor._id,
                "trustor_name": trustor.name,
                "trustee_id": trustee._id,
                "trustee_name": trustee.name,
                "sent_amount": trustor_amount,
                "returned_amount": trustee_amount,
                "trustee_received": trustee_received,
                "trustor_payoff": trustor_payoff,
                "trustee_payoff": trustee_payoff,
                "trustor_explanation": trustor_explanation,
                "trustee_explanation": trustee_explanation,
                "view_reputation": view_reputation,
                "trustee_reputation_before_decision": trustee_reputation_before,
            }
            
        except Exception as e:
            logging.error(f"Error in interaction {interaction_num}: {e}")
            return {}
            
        interaction_summary['detail'] = interaction_detail
        self.game_logs.append(interaction_summary)
        
        return interaction_summary

async def main():
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # Experiment Configuration
    EXPERIMENT_CONFIG = {
        "experiment_name": "TrustGame_Population_Reputation_Experiment",
        "description": "Trust Game population + HR reputation (view choice from interaction 2; Reputation = returned/received as Trustee)",
        "game_settings": {
            "num_agents": 24,  # Total number of agents
            "initial_funds": 10,  # Initial coins per Trustor per interaction
            "multiplication_factor": 3,  # Investment multiplication factor
            "num_rounds": 10,  # Rounds per agent interaction
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
        base_result_dir = "result_trust_game_population_reputation"
        experiment_name = input("Please enter experiment folder name (e.g., 'TG_population_test1'): ").strip()
        if not experiment_name:
            experiment_name = f"TrustGame_{datetime.now().strftime('%m%d%H%M')}"
        
        # 可配置的实验参数
        NUM_GAMES = 1  # independent full runs per script execution (was 3)
        ROUNDS_PER_GAME = 30  # 每局游戏轮数
        
        # 更新配置
        EXPERIMENT_CONFIG["game_settings"]["total_population_rounds"] = ROUNDS_PER_GAME
        
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
                "total_payoffs_per_agent": defaultdict(float),
                "interaction_metrics_time_series": [],
                "role_statistics_per_agent": {}  # Added for enhanced role statistics
            },
            "experiment_start_time": datetime.now().isoformat(),
            "experiment_end_time": None
        }
        
        num_agents = EXPERIMENT_CONFIG["game_settings"]["num_agents"]
        total_population_rounds = EXPERIMENT_CONFIG["game_settings"]["total_population_rounds"]
        total_interactions = total_population_rounds * (num_agents // 2)
        
        # Run specified number of independent games
        for game_num in range(NUM_GAMES):
            print(f"\n" + "="*70)
            print(f"Starting Game {game_num + 1}")
            print("="*70)
            
            # Pre-generate Interaction Schedule for this game
            print("Generating interaction schedule...")
            agent_ids = list(range(num_agents))
            interaction_schedule = []
            
            # Generate the schedule for TOTAL_INTERACTIONS pairs
            for _ in range(total_interactions):
                # Sample two different agents
                p1_id, p2_id = random.sample(agent_ids, 2)
                interaction_schedule.append((p1_id, p2_id))
            
            # Create environment
            print("Creating environment...")
            env = TrustGamePopulationReputationEnvironment(
                num_agents=EXPERIMENT_CONFIG["game_settings"]["num_agents"],
                initial_funds=EXPERIMENT_CONFIG["game_settings"]["initial_funds"],
                multiplication_factor=EXPERIMENT_CONFIG["game_settings"]["multiplication_factor"],
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
            print(f"Total interactions: {total_interactions}")
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
                current_population_round = (interaction_num - 1) // (num_agents // 2) + 1
                
                if interaction_num % (num_agents // 2) == 1:
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
                            game_stats["agent_payoffs"][trustor_name] += interaction_result["detail"].get("trustor_payoff", 0)
                            game_stats["agent_payoffs"][trustee_name] += interaction_result["detail"].get("trustee_payoff", 0)
                        
                        # Print interaction summary
                        if interaction_num % (num_agents // 2) == 0:
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
        from agentsociety2.agent.base import AgentLLM
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
    base_dir = "result_trust_game_population_reputation"
    
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
    
    # Agent profiles - aligned with baseline
    trustor_profile = (
        "You are {agent_name}, Trustor in a {num_rounds}-round Trust Game.\n"
        "Goal: Maximize cumulative coins.\n"
        f"Rules: Start with {config['game_settings']['initial_funds']} coins/round.\n"
        f"Send 0-{config['game_settings']['initial_funds']} coins (×{config['game_settings']['multiplication_factor']} for Trustee).\n"
        f"Trustee returns some coins. Your payoff: {config['game_settings']['initial_funds']} - sent + returned.\n"
        "Consider past Trustee behavior."
    )
    
    trustee_profile = (
        "You are {agent_name}, Trustee in a {num_rounds}-round Trust Game.\n"
        "Goal: Maximize cumulative coins.\n"
        f"Rules: Receive (sent amount)×{config['game_settings']['multiplication_factor']} coins from Trustor.\n"
        "Return 0-received coins. Your payoff: received - returned.\n"
        "Consider past Trustor behavior."
    )
    
    # Create agents - random role assignment will happen during interactions
    for i in range(num_agents):
        agent_name = f"Agent_{i+1}"
        # Use a neutral profile initially, roles will be assigned per interaction
        neutral_profile = (
            f"You are {agent_name}, a rational decision-maker participating in a Trust Game.\n"
            "You will interact with different partners in multiple rounds.\n"
            "Your goal is to maximize your cumulative coins over all interactions."
        )
        
        agent = TrustGamePopulationReputationAgent(
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
    
    # Update basic statistics
    experiment_stats["overall_metrics"]["average_sent_per_interaction"].append(sent_amount)
    experiment_stats["overall_metrics"]["average_returned_per_interaction"].append(returned_amount)
    
    # Calculate return rate if sent_amount > 0
    if sent_amount > 0:
        return_rate = returned_amount / (sent_amount * detail.get("trustee_received", sent_amount) / sent_amount)  # trustee_received should be sent_amount * multiplication_factor
    else:
        return_rate = 0.0
    
    experiment_stats["overall_metrics"]["average_return_rate_per_interaction"].append(return_rate)
    
    # Update total payoffs per agent
    trustor_name = detail.get("trustor_name", "")
    trustee_name = detail.get("trustee_name", "")
    trustor_payoff = detail.get("trustor_payoff", 0)
    trustee_payoff = detail.get("trustee_payoff", 0)
    
    experiment_stats["overall_metrics"]["total_payoffs_per_agent"][trustor_name] += trustor_payoff
    experiment_stats["overall_metrics"]["total_payoffs_per_agent"][trustee_name] += trustee_payoff

async def _print_population_round_summary(interaction_num: int, env: TrustGamePopulationReputationEnvironment):
    """Print summary of completed population round"""
    current_round = (interaction_num - 1) // (env.num_agents // 2) + 1
    
    # Calculate total sent and returned for this round
    round_interactions = env.game_logs[-(env.num_agents // 2):]
    total_sent = sum(interaction.get("detail", {}).get("sent_amount", 0) for interaction in round_interactions)
    total_returned = sum(interaction.get("detail", {}).get("returned_amount", 0) for interaction in round_interactions)
    
    avg_sent = total_sent / len(round_interactions) if round_interactions else 0
    avg_returned = total_returned / len(round_interactions) if round_interactions else 0
    
    print(f"Population Round {current_round} completed!")
    print(f"  Total sent across all interactions: {total_sent}")
    print(f"  Total returned across all interactions: {total_returned}")
    print(f"  Average sent per interaction: {avg_sent:.2f}")
    print(f"  Average returned per interaction: {avg_returned:.2f}")

async def _save_game_results(config: dict, env: TrustGamePopulationReputationEnvironment, agents: list,
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
