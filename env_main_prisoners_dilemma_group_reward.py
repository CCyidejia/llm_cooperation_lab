#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
囚徒困境（群体）+ 边际付款（Side Payment / 奖励承诺）机制
基于 env_main_prisoners_dilemma_group_enhanced.py

两阶段博弈：
  阶段一（承诺）：双方同时、独立宣布 s，且 s 只能从 {0,1,2,3,4,5} 中取值。
    - Agent A 的 s_A 表示：若 B 在阶段二选 Yes，则 A 向 B 支付 s_A（有约束力）。
    - Agent B 的 s_B 表示：若 A 在阶段二选 Yes，则 B 向 A 支付 s_B。
  阶段二（行动）：双方在看到对方承诺金额后，同时选 Yes 或 No。

结算（与标准 PD 基收益 T=5,R=3,P=1,S=0 一致，再叠加条件转移）：
  (Yes,Yes):   A: 3 - s_A + s_B,  B: 3 + s_A - s_B
  (Yes,No):    A: 0 + s_B,      B: 5 - s_B
  (No,Yes):    A: 5 - s_A,      B: 0 + s_A
  (No,No):     A: 1,            B: 1
仅当对方实际选 Yes 时，己方承诺的付款才会被扣除并转给对方。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_cooperation_lab.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv

load_dotenv()

os.makedirs("result_prisoners_dilemma_group_enhanced_side_payment", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_api_log.txt"),
        logging.StreamHandler()
    ]
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 承诺金额仅允许以下 6 个离散取值
SIDE_PAYMENT_CHOICES = (0, 1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# 与 env_main_prisoners_dilemma_group.py 中 rational_profile 保持一致的基础描述；
# reward 实验在此外再追加「两阶段 + 边际付款」说明。若修改 baseline 文案，请同步此处或改为公共模块。
# ---------------------------------------------------------------------------
PD_GROUP_BASE_PROFILE_TEMPLATE = (
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


def build_reward_rational_profile(opponent_label: str = "Partner") -> str:
    """group 的基础 profile + reward 机制追加（与 group 对照实验时基础一致）。"""
    base = PD_GROUP_BASE_PROFILE_TEMPLATE.replace("[OPPONENT_NAME]", opponent_label)
    choices_str = ", ".join(str(x) for x in SIDE_PAYMENT_CHOICES)
    addon = (
        "\n\n"
        "ADDITIONAL MECHANISM (this experiment only): Each time you are paired with an opponent, "
        "that interaction has TWO stages in order. "
        f"(1) COMMITMENT PHASE: You and your opponent each simultaneously announce an integer S; "
        f"S must be exactly one of: {choices_str}. "
        "Your S is what you must pay the opponent if they choose Yes in stage (2); if they choose No, you pay nothing from this promise. "
        "Their S is what they must pay you if you choose Yes. "
        "(2) ACTION PHASE: After both commitments are revealed, you and your opponent simultaneously choose Yes or No. "
        "Final points use the payoff matrix above as the base, then apply the side-payment rules given in each task prompt."
    )
    return base + addon


def clamp_side_payment(v) -> int:
    """将任意输入映射到最近的合法离散承诺金额。"""
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    if x in SIDE_PAYMENT_CHOICES:
        return x
    return min(SIDE_PAYMENT_CHOICES, key=lambda t: abs(t - x))


def compute_payoff_side_payment(
    action_a: str,
    action_b: str,
    s_a_to_b: int,
    s_b_to_a: int,
) -> tuple[int, int]:
    """
    A 为 agent1，B 为 agent2。
    s_a_to_b: A 承诺若 B 选 Yes 则付给 B 的金额。
    s_b_to_a: B 承诺若 A 选 Yes 则付给 A 的金额。
    """
    a = action_a.capitalize()
    b = action_b.capitalize()
    s_a_to_b = clamp_side_payment(s_a_to_b)
    s_b_to_a = clamp_side_payment(s_b_to_a)

    if a == "Yes" and b == "Yes":
        return 3 - s_a_to_b + s_b_to_a, 3 + s_a_to_b - s_b_to_a
    if a == "Yes" and b == "No":
        return 0 + s_b_to_a, 5 - s_b_to_a
    if a == "No" and b == "Yes":
        return 5 - s_a_to_b, 0 + s_a_to_b
    if a == "No" and b == "No":
        return 1, 1
    return 0, 0


class PrisonersDilemmaSidePaymentAgent(AgentBase):
    """囚徒困境 + 边际付款：维护历史与按对手维度的记忆。"""

    def __init__(self, id: int, name: str, profile: str = ""):
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []

        self._my_history = []
        self._partner_history = []
        self._outcome_history = []
        self._my_side_payment_history = []
        self._their_side_payment_history = []

        self._agent_memory = defaultdict(
            lambda: {
                "interaction_count": 0,
                "my_choices": [],
                "their_choices": [],
                "my_payoffs": [],
                "their_payoffs": [],
                "my_side_payments": [],
                "their_side_payments": [],
            }
        )

    def set_environment(self, env):
        self._env = env

    def update_state(
        self,
        partner_id: int,
        my_choice: str,
        partner_choice: str,
        my_payoff: int,
        their_payoff: int,
        my_side_payment: int = 0,
        their_side_payment: int = 0,
    ):
        self._my_history.append(my_choice)
        self._partner_history.append(partner_choice)
        self._outcome_history.append(my_payoff)
        self._my_side_payment_history.append(my_side_payment)
        self._their_side_payment_history.append(their_side_payment)

        m = self._agent_memory[partner_id]
        m["interaction_count"] += 1
        m["my_choices"].append(my_choice)
        m["their_choices"].append(partner_choice)
        m["my_payoffs"].append(my_payoff)
        m["their_payoffs"].append(their_payoff)
        m["my_side_payments"].append(my_side_payment)
        m["their_side_payments"].append(their_side_payment)

        logging.debug(
            f"[{self.name}] partner={partner_id} my={my_choice} their={partner_choice} "
            f"payoffs=({my_payoff},{their_payoff}) side=({my_side_payment},{their_side_payment})"
        )

    def get_state_summary(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "my_history": self._my_history,
            "partner_history": self._partner_history,
            "outcome_history": self._outcome_history,
            "my_side_payment_history": self._my_side_payment_history,
            "their_side_payment_history": self._their_side_payment_history,
            "num_interactions": len(self._my_history),
            "agent_memory": dict(self._agent_memory),
        }

    async def init(self, llm: AgentLLM):
        self._llm = llm

    @property
    def name(self):
        return self._name

    async def dump(self) -> dict:
        return {
            "profile": self._profile,
            "my_history": self._my_history,
            "partner_history": self._partner_history,
            "outcome_history": self._outcome_history,
            "my_side_payment_history": self._my_side_payment_history,
            "their_side_payment_history": self._their_side_payment_history,
            "previous_choices": self._previous_choices,
            "agent_memory": dict(self._agent_memory),
        }

    async def load(self, dump_data: dict):
        self._profile = dump_data.get("profile", "")
        self._my_history = dump_data.get("my_history", [])
        self._partner_history = dump_data.get("partner_history", [])
        self._outcome_history = dump_data.get("outcome_history", [])
        self._my_side_payment_history = dump_data.get("my_side_payment_history", [])
        self._their_side_payment_history = dump_data.get("their_side_payment_history", [])
        self._previous_choices = dump_data.get("previous_choices", [])
        self._agent_memory = defaultdict(
            lambda: {
                "interaction_count": 0,
                "my_choices": [],
                "their_choices": [],
                "my_payoffs": [],
                "their_payoffs": [],
                "my_side_payments": [],
                "their_side_payments": [],
            }
        )
        for agent_id, memory_data in dump_data.get("agent_memory", {}).items():
            try:
                self._agent_memory[int(agent_id)] = memory_data
            except (ValueError, TypeError):
                continue

    async def ask(self, message: str, readonly: bool = True) -> str:
        prompt = f"{self._profile}\n\n{message}"
        response = await asyncio.to_thread(
            self._llm.get_llm_response,
            "",
            prompt,
        )
        return response

    async def step(self, tick: int, t: datetime) -> str:
        return f"{self.name} step executed at tick {tick}"

    def _build_history_str(self, partner_id: int, partner_name: str) -> str:
        history_str = "History:\n"
        for i, (my_a, op_a) in enumerate(zip(self._my_history, self._partner_history)):
            history_str += (
                f"Round {i+1}: Your choice={my_a.capitalize()}, {partner_name}'s choice={op_a.capitalize()}\n"
            )
        if partner_id in self._agent_memory:
            am = self._agent_memory[partner_id]
            history_str += f"\nEnhanced Memory with {partner_name}:\n"
            history_str += f"Interaction count: {am['interaction_count']}\n"
            if am["interaction_count"] > 0:
                history_str += "All interactions:\n"
                for i in range(am["interaction_count"]):
                    msp = am["my_side_payments"][i] if i < len(am["my_side_payments"]) else 0
                    tsp = am["their_side_payments"][i] if i < len(am["their_side_payments"]) else 0
                    history_str += (
                        f"  Round {i+1}: Your commitment (pay them if they cooperate)={msp}, "
                        f"their commitment={tsp}, "
                        f"choices=({am['my_choices'][i].capitalize()},{am['their_choices'][i].capitalize()}), "
                        f"payoffs=({am['my_payoffs'][i]},{am['their_payoffs'][i]})\n"
                    )
        return history_str

    def _parse_side_payment(self, content: str) -> tuple[int, str]:
        if not content or not content.strip():
            return 0, "Empty response"
        m = re.search(r"<side_payment>\s*(\d+)\s*</side_payment>", content, re.IGNORECASE | re.DOTALL)
        if m:
            v = clamp_side_payment(int(m.group(1)))
            expl = ""
            em = re.search(r"<explanation>(.*?)</explanation>", content, re.DOTALL | re.IGNORECASE)
            if em:
                expl = em.group(1).strip()
            return v, expl or "parsed from XML"
        first_int = None
        for n in re.finditer(r"\b(\d+)\b", content):
            v = int(n.group(1))
            if v in SIDE_PAYMENT_CHOICES:
                return v, content.strip()[:500]
            if first_int is None:
                first_int = v
        if first_int is not None:
            return clamp_side_payment(first_int), content.strip()[:500]
        logging.warning(
            f"[{self.name}] No valid side payment in {SIDE_PAYMENT_CHOICES}; default 0"
        )
        return 0, content.strip()[:500]

    def _parse_action(self, content: str) -> tuple[str, str]:
        action = "No"
        explanation = "parse failed"
        m = re.search(r"<action>\s*(Yes|No)\s*</action>", content, re.IGNORECASE)
        if m:
            action = m.group(1).capitalize()
            em = re.search(r"<explanation>(.*?)</explanation>", content, re.DOTALL | re.IGNORECASE)
            explanation = em.group(1).strip() if em else "no explanation"
            return action, explanation
        lines = content.strip().split("\n")
        if lines:
            fm = re.search(r"^\s*(yes|no)\s*[.!]?$", lines[0].strip(), re.IGNORECASE)
            if fm:
                return fm.group(1).capitalize(), " ".join(lines[1:]).strip() or "single line"
        return action, explanation

    async def _call_llm_retry(self, user_prompt: str, max_retries: int = 5, base_delay: float = 2.0) -> str:
        last_err = None
        for attempt in range(max_retries):
            try:
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    "",
                    user_prompt,
                )
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    raise ConnectionError(f"API error: {content[:120]!r}")
                return content
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt) + random.uniform(0.1, 1.0))
                else:
                    logging.error(f"[{self.name}] LLM failed after retries: {e}")
        raise last_err if last_err else RuntimeError("LLM call failed")

    @staticmethod
    def _clip_explanation(text: str, max_len: int = 500) -> str:
        if not text:
            return ""
        t = text.strip().replace("\n", " ")
        return t if len(t) <= max_len else t[: max_len - 3] + "..."

    async def commit_side_payment(self, partner_id: int, partner_name: str, memory_size: int) -> tuple[int, str]:
        """阶段一：宣布若对方选 Yes 则愿意支付对方的非负整数金额（有约束力）。"""
        history_str = self._build_history_str(partner_id, partner_name)
        choices_csv = ", ".join(str(x) for x in SIDE_PAYMENT_CHOICES)
        prompt = (
            f"{self._profile}\n"
            f"{history_str}\n"
            "--- COMMITMENT PHASE (before action phase) ---\n"
            f"You are about to play one interaction with {partner_name} in TWO stages (see ADDITIONAL MECHANISM in your profile).\n"
            "STAGE 1 (now): You simultaneously announce ONE integer, call it **S**.\n"
            f"Meaning: **If {partner_name} chooses Yes (Cooperate) in Stage 2, you MUST pay them exactly S points.** "
            "If they choose No, you pay nothing related to this promise.\n"
            f"Similarly, {partner_name} will announce their own integer: what they pay YOU if YOU choose Yes.\n"
            f"S must be EXACTLY one of these six values: {choices_csv}.\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules, the payoff matrix in your profile, and history, "
            "decide your commitment S.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n"
            "\n"
            "<output>\n"
            f"  <side_payment>one of: {choices_csv}</side_payment>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>"
            "\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure. "
            "Your entire output must consist of exactly the XML format shown above."
        )
        try:
            content = await self._call_llm_retry(prompt)
            s_val, expl = self._parse_side_payment(content)
            logging.info(
                f"[{self.name}] [COMMIT vs {partner_name}] S={s_val}, "
                f"explanation: {self._clip_explanation(expl)}"
            )
            return s_val, expl
        except Exception as e:
            logging.error(f"[{self.name}] commit_side_payment error: {e}")
            return 0, f"default 0 due to error: {e}"

    async def make_decision(
        self,
        partner_id: int,
        partner_name: str,
        memory_size: int,
        my_side_payment: int,
        their_side_payment: int,
    ) -> tuple[str, str]:
        """阶段二：在看到双方承诺后选择 Yes 或 No。"""
        history_str = self._build_history_str(partner_id, partner_name)
        my_s = clamp_side_payment(my_side_payment)
        their_s = clamp_side_payment(their_side_payment)

        prompt = (
            f"{self._profile}\n"
            f"{history_str}\n"
            "--- ACTION PHASE ---\n"
            f"Opponent: {partner_name}.\n"
            f"Your commitment: If {partner_name} chooses Yes, you pay them **{my_s}** points (binding).\n"
            f"Their commitment: If you choose Yes, they pay you **{their_s}** points (binding).\n\n"
            "Base game (before side payments; same as the matrix in your profile):\n"
            "- Both Yes: 3 each\n"
            "- You Yes, they No: you get 0, they get 5\n"
            "- You No, they Yes: you get 5, they get 0\n"
            "- Both No: 1 each\n\n"
            "After accounting for side payments (only transfers when the recipient actually chose Yes):\n"
            f"- Both Yes: your payoff = 3 - {my_s} + {their_s}, their payoff = 3 + {my_s} - {their_s}\n"
            f"- You Yes, they No: your payoff = 0 + {their_s}, their payoff = 5 - {their_s}\n"
            f"- You No, they Yes: your payoff = 5 - {my_s}, their payoff = 0 + {my_s}\n"
            "- Both No: 1 each (no transfers)\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules, the payoff matrix in your profile, and history, "
            "determine your action.\n"
            "Your decision must be either **Yes (Cooperate)** or **No (Defect/Betray)**.\n"
            "YOU MUST FOLLOW THIS OUTPUT FORMAT EXACTLY:\n"
            "\n"
            "<output>\n"
            "  <action>Yes or No</action>\n"
            "  <explanation>Your 1-2 sentence reasoning here.</explanation>\n"
            "</output>"
            "\n"
            "IMPORTANT: Do not write any additional text before or after the XML structure. "
            "Your entire output must consist of exactly the XML format shown above."
        )
        action = "No"
        explanation = "LLM failure"
        for attempt in range(5):
            try:
                content = await self._call_llm_retry(prompt)
                action, explanation = self._parse_action(content)
                self._previous_choices.append(action)
                logging.info(
                    f"[{self.name}] [ACTION vs {partner_name}] choice={action}, "
                    f"explanation: {self._clip_explanation(explanation)}"
                )
                return action, explanation
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 * (2**attempt) + random.uniform(0.1, 1.0))
                else:
                    logging.error(f"[{self.name}] make_decision error: {e}")
        return action, explanation

    async def report_token_usage(self):
        pass

    def get_agent_memory(self, agent_id: int) -> dict:
        return self._agent_memory.get(
            agent_id,
            {
                "interaction_count": 0,
                "my_choices": [],
                "their_choices": [],
                "my_payoffs": [],
                "their_payoffs": [],
                "my_side_payments": [],
                "their_side_payments": [],
            },
        )


class PrisonersDilemmaSidePaymentEnvironment:
    """囚徒困境 + 边际付款：每轮交互 = 承诺阶段 + 行动阶段。"""

    def __init__(
        self,
        num_agents: int,
        memory_size: int,
        total_interactions: int,
        interaction_schedule: list,
    ):
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.total_interactions = total_interactions
        self.agents = []
        self.interaction_schedule = interaction_schedule
        self.interaction_number = 0
        self.initial_time = None
        self._config = {
            "max_tick": total_interactions,
            "num_agents": num_agents,
            "memory_size": memory_size,
            "interactions_per_run": 1,
            "mechanism": "side_payment_two_stage",
            "side_payment_choices": list(SIDE_PAYMENT_CHOICES),
        }
        self.game_stats = {"total_interactions": 0, "cooperation_count": 0}
        self.game_logs = []
        self.choice_frequency = Counter()

    def set_agents(self, agents: list):
        self.agents = agents

    def _get_agent_by_id(self, agent_id):
        return next((a for a in self.agents if a._id == agent_id), None)

    async def run_interaction(self, interaction_num: int) -> dict:
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
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # 阶段一：同时承诺
            c1, e1 = await agent1.commit_side_payment(agent2._id, agent2.name, self.memory_size)
            c2, e2 = await agent2.commit_side_payment(agent1._id, agent1.name, self.memory_size)
            # agent1 承诺给 agent2 的是 c1；agent2 承诺给 agent1 的是 c2
            s1_to_2, s2_to_1 = c1, c2

            # 阶段二：同时行动（双方已知承诺）
            a1, x1 = await agent1.make_decision(
                agent2._id, agent2.name, self.memory_size, s1_to_2, s2_to_1
            )
            a2, x2 = await agent2.make_decision(
                agent1._id, agent1.name, self.memory_size, s2_to_1, s1_to_2
            )

            p1, p2 = compute_payoff_side_payment(a1, a2, s1_to_2, s2_to_1)

            agent1.update_state(
                agent2._id, a1, a2, p1, p2, my_side_payment=s1_to_2, their_side_payment=s2_to_1
            )
            agent2.update_state(
                agent1._id, a2, a1, p2, p1, my_side_payment=s2_to_1, their_side_payment=s1_to_2
            )

            interaction_detail = {
                "agent1_id": agent1_id,
                "agent1_name": agent1.name,
                "agent2_id": agent2_id,
                "agent2_name": agent2.name,
                "side_payment_1_to_2": s1_to_2,
                "side_payment_2_to_1": s2_to_1,
                "commit_explanation1": e1,
                "commit_explanation2": e2,
                "choice1": a1,
                "choice2": a2,
                "explanation1": x1,
                "explanation2": x2,
                "payoff1": p1,
                "payoff2": p2,
                "cooperation": (a1 == "Yes" and a2 == "Yes"),
            }
            logging.info(
                f"[interaction {interaction_num}] {agent1.name} vs {agent2.name} | "
                f"commit: S_1→2={s1_to_2} ({PrisonersDilemmaSidePaymentAgent._clip_explanation(e1, 240)}) | "
                f"S_2→1={s2_to_1} ({PrisonersDilemmaSidePaymentAgent._clip_explanation(e2, 240)}) | "
                f"action: {a1}/{a2} | "
                f"expl1: {PrisonersDilemmaSidePaymentAgent._clip_explanation(x1, 240)} | "
                f"expl2: {PrisonersDilemmaSidePaymentAgent._clip_explanation(x2, 240)}"
            )
        except Exception as e:
            logging.error(f"Error in interaction {interaction_num}: {e}")
            return {}

        self.choice_frequency[a1] += 1
        self.choice_frequency[a2] += 1
        if interaction_detail["cooperation"]:
            self.game_stats["cooperation_count"] += 1

        interaction_summary["detail"] = interaction_detail
        self.game_logs.append(interaction_summary)
        self.game_stats["total_interactions"] += 1
        return interaction_summary

    async def save_game_results(
        self,
        experiment_name: str = None,
        experiment_num: int = 1,
        main_result_dir: str = None,
    ):
        if experiment_name is None:
            experiment_name = f"PD_side_payment_{datetime.now().strftime('%m%d%H%M')}"
        if main_result_dir:
            result_dir = main_result_dir
        else:
            result_dir = os.path.join("result_prisoners_dilemma_group_enhanced_side_payment", experiment_name)
        data_dir = os.path.join(result_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        self._config["max_tick"] = self.total_interactions
        self._config["random_seed"] = RANDOM_SEED

        if experiment_num == 1:
            with open(os.path.join(data_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)

        with open(os.path.join(data_dir, f"game_stats{experiment_num}.json"), "w", encoding="utf-8") as f:
            self.game_stats["total_interactions"] = len(self.game_logs)
            json.dump(self.game_stats, f, indent=4, ensure_ascii=False)

        with open(os.path.join(data_dir, f"agent_states{experiment_num}.json"), "w", encoding="utf-8") as f:
            agent_states = {a._id: a.get_state_summary() for a in self.agents}
            json.dump(agent_states, f, indent=4, ensure_ascii=False)

        with open(os.path.join(data_dir, f"game_log{experiment_num}.json"), "w", encoding="utf-8") as f:
            json.dump(self.game_logs, f, ensure_ascii=False, indent=2)

        print(f"Game {experiment_num} results saved to: {result_dir}")
        return result_dir


def print_payoff_statistics(env):
    if not env.game_logs:
        print("Warning: No valid game data")
        return
    total_payoffs = {agent.name: 0 for agent in env.agents}
    cooperation_count = 0
    for interaction in env.game_logs:
        detail = interaction["detail"]
        total_payoffs[detail["agent1_name"]] += detail["payoff1"]
        total_payoffs[detail["agent2_name"]] += detail["payoff2"]
        if detail["cooperation"]:
            cooperation_count += 1
    print("\n===== Basic Payoff Information =====")
    print(f"Total interactions: {len(env.game_logs)}")
    print(f"Cooperation rate: {cooperation_count / len(env.game_logs) * 100:.1f}%")
    print("\nTotal payoffs per agent:")
    for agent_name, total in total_payoffs.items():
        print(f"  {agent_name}: {total} pts")
    print(f"\nAverage agent payoff: {sum(total_payoffs.values()) / len(total_payoffs):.1f} pts")
    print(f"Total group payoff: {sum(total_payoffs.values())} pts")


def print_memory_statistics(env):
    print("\n===== Memory Statistics =====")
    print(f"Total agents: {len(env.agents)}")
    for agent in env.agents:
        ms = agent.get_memory_summary() if hasattr(agent, "get_memory_summary") else {}
        print(f"\nAgent {agent.name}: interactions in log={len(agent._my_history)}")


async def main():
    print(f"Random seed set to: {RANDOM_SEED}")

    NUM_AGENTS = 24
    MEMORY_SIZE = 0
    TOTAL_POPULATION_ROUNDS = 10
    TOTAL_INTERACTIONS = TOTAL_POPULATION_ROUNDS * (NUM_AGENTS // 2)

    base_result_dir = "result_prisoners_dilemma_group_enhanced_side_payment"
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PD_side_payment_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = f"PD_side_payment_{datetime.now().strftime('%m%d%H%M')}"

    main_result_dir = os.path.join(base_result_dir, base_experiment_name)
    os.makedirs(main_result_dir, exist_ok=True)

    try:
        from LLMAPI.zgc import LLMAgent

        llm_agent = LLMAgent(name="PrisonerDilemmaSidePaymentAgent")
    except ImportError:

        class MockLLMAgent:
            def __init__(self, name):
                self.name = name

            def get_llm_response(self, *args, **kwargs):
                return (
                    "<output><side_payment>0</side_payment>"
                    "<explanation>mock</explanation></output>"
                )

        llm_agent = MockLLMAgent(name="mock")

    llm_agent.router = llm_agent
    llm_agent.model_name = getattr(llm_agent, "model", "mock_model")
    agent_llm = llm_agent

    rational_profile = build_reward_rational_profile("Partner")

    NUM_EXPERIMENT_REPEATS = 3
    for experiment_index in range(NUM_EXPERIMENT_REPEATS):
        experiment_num = experiment_index + 1
        experiment_suffix = f"第{experiment_num}次"
        experiment_name = f"{base_experiment_name}_{experiment_suffix}"

        print("\n" + "=" * 70)
        print(f"Prisoner's Dilemma + Side Payment — {experiment_suffix} ({experiment_num}/{NUM_EXPERIMENT_REPEATS})")
        print("=" * 70)

        agent_ids = list(range(NUM_AGENTS))
        interaction_schedule = []
        for _ in range(TOTAL_INTERACTIONS):
            interaction_schedule.append(tuple(random.sample(agent_ids, 2)))

        env = PrisonersDilemmaSidePaymentEnvironment(
            num_agents=NUM_AGENTS,
            memory_size=MEMORY_SIZE,
            total_interactions=TOTAL_INTERACTIONS,
            interaction_schedule=interaction_schedule,
        )

        agents = []
        for i in range(NUM_AGENTS):
            agent = PrisonersDilemmaSidePaymentAgent(
                id=i,
                name=f"Agent_{i}",
                profile=rational_profile,
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

        print(f"\nSimulation {experiment_num} finished. Elapsed: {end_time - start_time:.2f}s")
        print_payoff_statistics(env)
        print_memory_statistics(env)
        await env.save_game_results(experiment_name, experiment_num, main_result_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
        import traceback

        traceback.print_exc()
