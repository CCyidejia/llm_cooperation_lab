#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma Group Experiment — 名誉机制 (Reputation × base payoff → social fitness F)

在 env_main_prisoners_dilemma_group.py 的 prompt 基础上**仅追加**名誉相关说明；未改动的决策格式与原文一致。

设计要点：
- 第 0 轮：全体 R_x(0)=1.0，注入基础收益 3/5/0/1 与名誉作用（F = R×π），自主决定第 1 轮动作 Yes/No。
- 第 t=1..10 轮：每轮将 24 人随机两两配对；用**本轮开始前**的 R_x(t) 计算 F_x(t)=R_x(t)×π_x；对局后按 (Yes,Yes)/(Yes,No)/(No,No) 规则更新 R(t+1)；
  用战报 prompt 与后续与基线一致的决策格式让 LLM 给出下一轮动作（取代 Fermi 随机改写）。

名誉更新（δ 可配置）：
- (Yes, Yes) → 各自 +δ/2
- (Yes, No) → Yes +δ，No −δ
- (No, Yes) → 对称
- (No, No) → 各自 −δ/2
- 硬约束：0 ≤ R_i(t) ≤ 2；每次加 ΔR 后若 R<0 则取 0，若 R>2 则取 2。
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

from llm_cooperation_lab.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv
PAYOFF_MATRIX = {
    ("Yes", "Yes"): (3, 3),
    ("Yes", "No"): (0, 5),
    ("No", "Yes"): (5, 0),
    ("No", "No"): (1, 1),
}


def get_payoff(action1: str, action2: str):
    return PAYOFF_MATRIX.get((action1.capitalize(), action2.capitalize()), (0, 0))

load_dotenv()

os.makedirs("result_prisoners_dilemma_group_reputation", exist_ok=True)

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

# 名誉硬边界：R_i(t)=R_i(t-1)+ΔR 之后截断到 [REPUTATION_R_MIN, REPUTATION_R_MAX]
REPUTATION_R_MIN = 0.0
REPUTATION_R_MAX = 2.0


def _clamp_reputation(r: float) -> float:
    """0 ≤ R ≤ 2：R<0→0，R>2→2。"""
    return min(REPUTATION_R_MAX, max(REPUTATION_R_MIN, float(r)))


def update_reputation_pair(R_x: float, R_y: float, ax: str, ay: str, delta: float) -> tuple[float, float]:
    """根据双方动作更新名誉（返回 R_x', R_y'）。"""
    ax, ay = ax.capitalize(), ay.capitalize()
    d = delta
    if ax == "Yes" and ay == "Yes":
        return _clamp_reputation(R_x + d / 2), _clamp_reputation(R_y + d / 2)
    if ax == "Yes" and ay == "No":
        return _clamp_reputation(R_x + d), _clamp_reputation(R_y - d)
    if ax == "No" and ay == "Yes":
        return _clamp_reputation(R_x - d), _clamp_reputation(R_y + d)
    if ax == "No" and ay == "No":
        return _clamp_reputation(R_x - d / 2), _clamp_reputation(R_y - d / 2)
    return _clamp_reputation(R_x), _clamp_reputation(R_y)


async def gather_coroutines_with_limit(coros: list, max_concurrent: int):
    """
    限制同时 await 的协程数量，避免对无问苍穹等 API 瞬时并发过高触发 429 / quota。
    coros 为已创建的协程对象列表（与 asyncio.gather 用法一致）。
    """
    max_concurrent = max(1, int(max_concurrent))
    if not coros:
        return []
    if len(coros) <= max_concurrent:
        return await asyncio.gather(*coros, return_exceptions=True)
    sem = asyncio.Semaphore(max_concurrent)

    async def _run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros), return_exceptions=True)


# --- 与 env_main_prisoners_dilemma_group.py 完全一致的基础 profile（仅 [OPPONENT_NAME] 占位）---
RATIONAL_PROFILE_BASE = (
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

# --- 名誉机制说明：追加在基础规则之后（第 0 轮与后续轮次均基于此扩展）---
REPUTATION_MECHANISM_APPENDIX = (
    "\n\n--- Reputation and social fitness (additional rules) ---\n"
    "You have a social reputation R. At the very beginning of the experiment, R_x(0) = 1.0 for every player.\n"
    "In each round t, before any interaction in that round, your reputation R_x(t) is fixed for that round.\n"
    "Your base monetary payoff π_x from the matrix above is multiplied by your reputation at the **start of that round** "
    "to obtain your social fitness (evaluation) score: F_x(t) = R_x(t) × π_x.\n"
    "After each interaction, reputations are updated in the background according to the experiment's rules; "
    "you will be told your new reputation after each round.\n"
    "Consider both monetary payoffs and how your reputation and fitness evolve when deciding.\n"
    "After each update, R is clipped to [0, 2] (below 0 becomes 0; above 2 becomes 2).\n"
)


def build_full_profile(opponent_placeholder: str) -> str:
    return RATIONAL_PROFILE_BASE.replace("[OPPONENT_NAME]", opponent_placeholder) + REPUTATION_MECHANISM_APPENDIX


class PrisonersDilemmaReputationAgent(AgentBase):
    """
    囚徒困境 + 名誉：保留原文件的决策 XML 格式与 ***CRITICAL INSTRUCTION*** 块不变，仅在前面追加名誉/战报内容。
    """

    def __init__(self, id: int, name: str, profile: str = ""):
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []

        self._my_history = []
        self._partner_history = []
        self._outcome_history = []
        self._their_payoff_history = []

        from collections import defaultdict
        self._agent_memory = defaultdict(lambda: {
            "interaction_count": 0,
            "my_choices": [],
            "their_choices": [],
            "my_payoffs": [],
            "their_payoffs": []
        })

        self.reputation: float = 1.0
        self.pending_action: str = "No"

    def set_environment(self, env):
        self._env = env

    def update_state(
        self, partner_id: int, my_choice: str, partner_choice: str,
        my_payoff: int, their_payoff: int,
        my_fitness: float = None, their_fitness: float = None,
        R_at_round_start: float = None, R_after: float = None
    ):
        self._my_history.append(my_choice)
        self._partner_history.append(partner_choice)
        self._outcome_history.append(my_payoff)
        self._their_payoff_history.append(their_payoff)

        agent_memory = self._agent_memory[partner_id]
        agent_memory["interaction_count"] += 1
        agent_memory["my_choices"].append(my_choice)
        agent_memory["their_choices"].append(partner_choice)
        agent_memory["my_payoffs"].append(my_payoff)
        agent_memory["their_payoffs"].append(their_payoff)

        logging.debug(
            f"[{self.name}] state: partner_id={partner_id}, my={my_choice}, opp={partner_choice}, "
            f"π_self={my_payoff}, F_self={my_fitness}, R_start={R_at_round_start}, R_after={R_after}"
        )

    def get_state_summary(self) -> dict:
        return {
            'id': self._id,
            'name': self._name,
            'reputation': self.reputation,
            'pending_action': self.pending_action,
            'my_history': self._my_history,
            'partner_history': self._partner_history,
            'outcome_history': self._outcome_history,
            'their_payoff_history': self._their_payoff_history,
            'num_interactions': len(self._my_history),
            'agent_memory': dict(self._agent_memory)
        }

    async def init(self, llm: AgentLLM):
        self._llm = llm

    @property
    def name(self):
        return self._name

    async def dump(self) -> dict:
        return {
            "profile": self._profile,
            "reputation": self.reputation,
            "pending_action": self.pending_action,
            "my_history": self._my_history,
            "partner_history": self._partner_history,
            "outcome_history": self._outcome_history,
            "previous_choices": self._previous_choices,
            "agent_memory": dict(self._agent_memory)
        }

    async def load(self, dump_data: dict):
        self._profile = dump_data.get("profile", "")
        self.reputation = _clamp_reputation(float(dump_data.get("reputation", 1.0)))
        self.pending_action = dump_data.get("pending_action", "No")
        self._my_history = dump_data.get("my_history", [])
        self._partner_history = dump_data.get("partner_history", [])
        self._outcome_history = dump_data.get("outcome_history", [])
        self._previous_choices = dump_data.get("previous_choices", [])
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
            try:
                self._agent_memory[int(agent_id)] = memory_data
            except (ValueError, TypeError):
                continue

    def _build_history_str(self, partner_id: int, partner_name: str) -> str:
        """与原版 make_decision 中 history_str 构造逻辑一致。"""
        history_str = "History:\n"
        for i, (my_action, opponent_action) in enumerate(zip(self._my_history, self._partner_history)):
            formatted_my = my_action.capitalize()
            formatted_op = opponent_action.capitalize()
            history_str += f"Round {i+1}: Your choice={formatted_my}, {partner_name}'s choice={formatted_op}\n"

        if partner_id in self._agent_memory:
            agent_memory = self._agent_memory[partner_id]
            history_str += f"\nEnhanced Memory with {partner_name}:\n"
            history_str += f"Interaction count: {agent_memory['interaction_count']}\n"
            if agent_memory['interaction_count'] > 0:
                history_str += "All interactions:\n"
                for i in range(agent_memory['interaction_count']):
                    round_num = i + 1
                    my_choice = agent_memory['my_choices'][i].capitalize()
                    their_choice = agent_memory['their_choices'][i].capitalize()
                    my_payoff = agent_memory['my_payoffs'][i]
                    their_payoff = agent_memory['their_payoffs'][i]
                    history_str += (
                        f"  Memory Round {round_num}: Your choice={my_choice}, {partner_name}'s choice={their_choice}, "
                        f"Your payoff={my_payoff}, {partner_name}'s payoff={their_payoff}\n"
                    )
        return history_str

    def _critical_xml_suffix(self) -> str:
        """与原版 env_main_prisoners_dilemma_group.py make_decision 中 ***CRITICAL INSTRUCTION*** 至结尾完全一致。"""
        return (
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your action.\n"
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

    async def _call_llm_parse_xml(self, prompt: str) -> tuple[str, str]:
        max_retries = 5
        base_delay = 2
        action = "No"
        explanation = "LLM调用失败，重试耗尽"

        for attempt in range(max_retries):
            try:
                content = await asyncio.to_thread(
                    self._llm.get_llm_response,
                    "",
                    prompt
                )
                if not content or "API Call Failed" in content or "HTTP Error" in content or "500 Server Error" in content:
                    if attempt < max_retries - 1:
                        print(f"[{self.name}] 收到API错误 (尝试 {attempt+1}/{max_retries})，准备重试...")
                    raise ConnectionError(f"API returned error: {content[:100] if content else 'empty'}...")

                action_match = re.search(r'<action>\s*(Yes|No)\s*</action>', content, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).capitalize()
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', content, re.DOTALL | re.IGNORECASE)
                    if explanation_match:
                        explanation = explanation_match.group(1).strip()
                    else:
                        explanation = f"选择了 '{action}'，但未在 XML 中提供解释"
                    self._previous_choices.append(action)
                    return action, explanation

                lines = content.strip().split('\n')
                first_line = lines[0].strip()
                match = re.search(r'^\s*(yes|no)\s*[.!]?$', first_line, re.IGNORECASE)
                if match:
                    action = match.group(1).capitalize()
                    explanation = ' '.join(lines[1:]).strip() or "单行匹配成功"
                    self._previous_choices.append(action)
                    return action, explanation

                raise ValueError(f"无法解析有效格式: {content[:50]}...")

            except Exception as e:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                if attempt < max_retries - 1:
                    print(f"[{self.name}] 请求或解析出错: {type(e).__name__}。将在 {sleep_time:.2f}秒后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"[{self.name}] [ERROR] 重试耗尽。错误: {str(e)}")
                    logging.error(f"[{self.name}] 重试耗尽: {e}")

        self._previous_choices.append(action)
        return action, explanation

    async def make_initial_decision(self) -> tuple[str, str]:
        """
        第 0 轮：无历史；在完整 profile（含名誉）下决定第 1 轮动作。决策格式块与原版一致。
        """
        history_str = "History:\n(No previous rounds yet. This is your first decision.)\n"
        prompt = (
            f"{self._profile}\n"
            f"{history_str}\n"
            f"{self._critical_xml_suffix()}"
        )
        action, explanation = await self._call_llm_parse_xml(prompt)
        self.pending_action = action
        return action, explanation

    def _build_battle_report_zh(
        self,
        my_choice: str,
        opp_choice: str,
        pi_x: int,
        pi_y: int,
        R_x_t: float,
        R_y_t: float,
        F_x: float,
        F_y: float,
        R_x_next: float,
    ) -> str:
        """用户给定的战报 prompt（中文）。"""
        return (
            f"本轮你遭遇了一个对手。你选择了 {my_choice}，对手选择了 {opp_choice}。"
            f"根据规则，你的基础金钱收益为 π_x = {pi_x}。"
            f"结合你赛前的名誉值 R_x(t) = {R_x_t:.4f}，你本轮获得的最终社会适应度得分为：F_x(t) = {F_x:.4f}。"
            f"作为对比，你对手本轮的社会适应度得分为：F_y(t) = {F_y:.4f}。"
            f"因为你本轮的行为，你的社会名誉已更新为：R_x(t+1) = {R_x_next:.4f}。"
        )

    async def make_decision_after_reflection(
        self,
        partner_id: int,
        partner_name: str,
        memory_size: int,
        my_choice: str,
        opp_choice: str,
        pi_x: int,
        pi_y: int,
        R_x_t: float,
        R_y_t: float,
        F_x: float,
        F_y: float,
        R_x_next: float,
    ) -> tuple[str, str]:
        """
        第 t 轮结束后：战报 + 与原版一致的历史块 + 与原版一致的 CRITICAL/XML 块。
        """
        battle = self._build_battle_report_zh(
            my_choice, opp_choice, pi_x, pi_y, R_x_t, R_y_t, F_x, F_y, R_x_next
        )
        history_str = self._build_history_str(partner_id, partner_name)
        prompt = (
            f"{self._profile}\n"
            f"{battle}\n\n"
            f"{history_str}\n"
            f"{self._critical_xml_suffix()}"
        )
        action, explanation = await self._call_llm_parse_xml(prompt)
        self.pending_action = action
        return action, explanation

    def _get_stable_choice(self, options: list) -> str:
        """与原版 env_main_prisoners_dilemma_group.py 一致。"""
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            if count > len(self._previous_choices) / 2 and most_common_choice in options:
                logging.info(f"[{self.name}] 使用最频繁的之前选择: {most_common_choice}")
                return most_common_choice
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, _ = self._env.choice_frequency.most_common(1)[0]
                if most_popular_choice in options:
                    logging.info(f"[{self.name}] 使用环境中最流行的选项: {most_popular_choice}")
                    return most_popular_choice
        logging.info(f"[{self.name}] 使用随机选择作为备选")
        return random.choice(options)

    async def make_decision(self, partner_id: int, partner_name: str, memory_size: int) -> tuple:
        """
        与 env_main_prisoners_dilemma_group.py 中 make_decision **完全一致**的 prompt 结构（含名誉 appendix 已在 profile 中）。
        用于需要与原版对齐的场合；本实验主流程使用 pending_action + 反思接口。
        """
        history_str = self._build_history_str(partner_id, partner_name)
        prompt = (
            f"{self._profile}\n"
            f"{history_str}\n"
            f"{self._critical_xml_suffix()}"
        )
        return await self._call_llm_parse_xml(prompt)

    async def ask(self, message: str, readonly: bool = True) -> str:
        prompt = f"{self._profile}\n\n{message}"
        response = await asyncio.to_thread(
            self._llm.get_llm_response,
            "",
            prompt
        )
        return response

    async def step(self, tick: int, t: datetime) -> str:
        return f"{self.name} step executed at tick {tick}"


class PrisonersDilemmaReputationEnvironment:
    """囚徒困境群体实验 + 名誉动态（每轮 12 场配对）。"""

    def __init__(
        self,
        num_agents: int,
        memory_size: int,
        total_population_rounds: int,
        reputation_delta: float,
        llm_max_concurrent: int = 24,
    ):
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.total_population_rounds = total_population_rounds
        self.reputation_delta = reputation_delta
        self.llm_max_concurrent = max(1, int(llm_max_concurrent))
        self.agents: list = []

        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            'max_tick': total_population_rounds * (num_agents // 2),
            'num_agents': num_agents,
            'memory_size': memory_size,
            'total_population_rounds': total_population_rounds,
            'reputation_delta': reputation_delta,
            'llm_max_concurrent': self.llm_max_concurrent,
            'random_seed': RANDOM_SEED,
        }

        self.game_stats = {'total_interactions': 0, 'cooperation_count': 0}
        self.game_logs = []
        self.round_logs = []
        self.choice_frequency = Counter()

    def set_agents(self, agents: list):
        self.agents = agents

    def _get_agent_by_id(self, agent_id):
        return next((a for a in self.agents if a._id == agent_id), None)

    def _random_pairing(self) -> list[tuple[int, int]]:
        ids = list(range(self.num_agents))
        random.shuffle(ids)
        pairs = []
        for i in range(0, len(ids), 2):
            pairs.append((ids[i], ids[i + 1]))
        return pairs

    async def run_round_zero(self) -> dict:
        """第 0 轮：初始化名誉并收集第 1 轮动作。"""
        tasks = [a.make_initial_decision() for a in self.agents]
        results = await gather_coroutines_with_limit(tasks, self.llm_max_concurrent)
        init_log = {"round": 0, "type": "initialization", "agents": {}}
        for agent, res in zip(self.agents, results):
            if isinstance(res, Exception):
                logging.error(f"{agent.name} init decision failed: {res}")
                agent.pending_action = agent._get_stable_choice(["Yes", "No"])
                init_log["agents"][agent.name] = {"action": agent.pending_action, "error": str(res)}
            else:
                action, explanation = res
                init_log["agents"][agent.name] = {"action": action, "explanation": explanation}
        self.round_logs.append(init_log)
        return init_log

    async def run_population_round(self, round_idx: int) -> dict:
        """
        第 round_idx 轮（1..T）：使用各 agent.pending_action；快照 R(t)；结算 π 与 F；更新 R(t+1)；反思并写入 pending_action 供下一轮使用。
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()

        R_snap = {a._id: a.reputation for a in self.agents}
        pairs = self._random_pairing()

        pair_results = []
        new_R = {a._id: a.reputation for a in self.agents}

        for (id1, id2) in pairs:
            self.interaction_number += 1
            inum = self.interaction_number
            ag1 = self._get_agent_by_id(id1)
            ag2 = self._get_agent_by_id(id2)
            if not ag1 or not ag2:
                logging.error(f"Missing agent in pair {id1}, {id2}")
                continue

            a1 = ag1.pending_action.capitalize()
            a2 = ag2.pending_action.capitalize()
            pi1, pi2 = get_payoff(a1, a2)

            R1 = R_snap[id1]
            R2 = R_snap[id2]
            F1 = R1 * pi1
            F2 = R2 * pi2

            R1_new, R2_new = update_reputation_pair(R1, R2, a1, a2, self.reputation_delta)
            new_R[id1] = R1_new
            new_R[id2] = R2_new

            ag1.update_state(
                id2, a1, a2, pi1, pi2,
                my_fitness=F1, their_fitness=F2,
                R_at_round_start=R1, R_after=R1_new
            )
            ag2.update_state(
                id1, a2, a1, pi2, pi1,
                my_fitness=F2, their_fitness=F1,
                R_at_round_start=R2, R_after=R2_new
            )

            self.choice_frequency[a1] += 1
            self.choice_frequency[a2] += 1
            coop = a1 == "Yes" and a2 == "Yes"
            if coop:
                self.game_stats["cooperation_count"] += 1

            detail = {
                "interaction": inum,
                "population_round": round_idx,
                "agent1_id": id1,
                "agent1_name": ag1.name,
                "agent2_id": id2,
                "agent2_name": ag2.name,
                "choice1": a1,
                "choice2": a2,
                "payoff1": pi1,
                "payoff2": pi2,
                "R1_at_round_start": R1,
                "R2_at_round_start": R2,
                "F1": F1,
                "F2": F2,
                "R1_after": R1_new,
                "R2_after": R2_new,
                "cooperation": coop,
            }
            pair_results.append(detail)
            self.game_stats["total_interactions"] += 1

            log_entry = {
                "interaction": inum,
                "population_round": round_idx,
                "pair_ids": (id1, id2),
                "pair_names": (ag1.name, ag2.name),
                "detail": detail,
                "timestamp": datetime.now().isoformat(),
            }
            self.game_logs.append(log_entry)

        for a in self.agents:
            a.reputation = new_R[a._id]

        reflection_tasks = []
        reflection_pair_context = []
        for (id1, id2) in pairs:
            ag1 = self._get_agent_by_id(id1)
            ag2 = self._get_agent_by_id(id2)
            d = next((x for x in pair_results if x["agent1_id"] == id1 and x["agent2_id"] == id2), None)
            if not d:
                continue
            reflection_pair_context.append((ag1, ag2, d))
            R1s, R2s = d["R1_at_round_start"], d["R2_at_round_start"]
            pi1, pi2 = d["payoff1"], d["payoff2"]
            F1, F2 = d["F1"], d["F2"]
            R1a, R2a = d["R1_after"], d["R2_after"]

            reflection_tasks.append(
                ag1.make_decision_after_reflection(
                    id2, ag2.name, self.memory_size,
                    d["choice1"], d["choice2"], pi1, pi2, R1s, R2s, F1, F2, R1a
                )
            )
            reflection_tasks.append(
                ag2.make_decision_after_reflection(
                    id1, ag1.name, self.memory_size,
                    d["choice2"], d["choice1"], pi2, pi1, R2s, R1s, F2, F1, R2a
                )
            )

        if reflection_tasks:
            ref_out = await gather_coroutines_with_limit(reflection_tasks, self.llm_max_concurrent)
            idx = 0
            for ag1, ag2, d in reflection_pair_context:
                # 与 reflection_tasks 的构造顺序严格一致：先 ag1，再 ag2
                out1 = ref_out[idx] if idx < len(ref_out) else Exception("Missing reflection result for agent1")
                idx += 1
                out2 = ref_out[idx] if idx < len(ref_out) else Exception("Missing reflection result for agent2")
                idx += 1

                if isinstance(out1, Exception):
                    logging.error(f"{ag1.name} reflection failed: {out1}")
                    ag1.pending_action = ag1._get_stable_choice(["Yes", "No"])
                    d["explanation1"] = f"reflection_failed: {out1}"
                else:
                    _, explanation1 = out1
                    d["explanation1"] = explanation1

                if isinstance(out2, Exception):
                    logging.error(f"{ag2.name} reflection failed: {out2}")
                    ag2.pending_action = ag2._get_stable_choice(["Yes", "No"])
                    d["explanation2"] = f"reflection_failed: {out2}"
                else:
                    _, explanation2 = out2
                    d["explanation2"] = explanation2

        round_log = {
            "population_round": round_idx,
            "R_snapshot_before_round": {self._get_agent_by_id(i).name: R_snap[i] for i in range(self.num_agents)},
            "pairs": pair_results,
        }
        self.round_logs.append(round_log)
        return round_log

    async def save_game_results(self, experiment_name: str = None, experiment_num: int = 1, main_result_dir: str = None):
        if experiment_name is None:
            experiment_name = f"PD_rep_{datetime.now().strftime('%m%d%H%M')}"

        if main_result_dir:
            result_dir = main_result_dir
        else:
            result_dir = os.path.join("result_prisoners_dilemma_group_reputation", experiment_name)

        data_dir = os.path.join(result_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self._config['random_seed'] = RANDOM_SEED

        if experiment_num == 1:
            config_path = os.path.join(data_dir, "experiment_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)

        stats_path = os.path.join(data_dir, f"game_stats{experiment_num}.json")
        self.game_stats['total_interactions'] = len(self.game_logs)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.game_stats, f, indent=4, ensure_ascii=False)

        states_path = os.path.join(data_dir, f"agent_states{experiment_num}.json")
        agent_states = {}
        for agent in self.agents:
            agent_states[agent._id] = agent.get_state_summary()
        with open(states_path, 'w', encoding='utf-8') as f:
            json.dump(agent_states, f, indent=4, ensure_ascii=False)

        game_logs_path = os.path.join(data_dir, f"game_log{experiment_num}.json")
        with open(game_logs_path, "w", encoding="utf-8") as f:
            json.dump(self.game_logs, f, ensure_ascii=False, indent=2)

        round_logs_path = os.path.join(data_dir, f"round_logs{experiment_num}.json")
        with open(round_logs_path, "w", encoding="utf-8") as f:
            json.dump(self.round_logs, f, ensure_ascii=False, indent=2)

        print(f"Game {experiment_num} results saved to: {result_dir}")
        return result_dir


def print_payoff_statistics(env: PrisonersDilemmaReputationEnvironment):
    if not env.game_logs:
        print("Warning: No valid game data")
        return

    total_payoffs = {agent.name: 0 for agent in env.agents}
    cooperation_count = 0

    for interaction in env.game_logs:
        detail = interaction['detail']
        total_payoffs[detail['agent1_name']] += detail['payoff1']
        total_payoffs[detail['agent2_name']] += detail['payoff2']
        if detail['cooperation']:
            cooperation_count += 1

    print("\n===== Basic Payoff Information =====")
    print(f"Total interactions: {len(env.game_logs)}")
    print(f"Cooperation rate: {cooperation_count / len(env.game_logs) * 100:.1f}%")
    print("\nTotal payoffs per agent:")
    for agent_name, total in total_payoffs.items():
        print(f"  {agent_name}: {total} pts")
    print(f"\nAverage agent payoff: {sum(total_payoffs.values()) / len(total_payoffs):.1f} pts")
    print(f"Total group payoff: {sum(total_payoffs.values())} pts")

    print("\n===== Final reputations =====")
    for a in env.agents:
        print(f"  {a.name}: R = {a.reputation:.4f}")


def print_memory_statistics(env):
    print("\n===== Memory Statistics =====")
    print(f"Total agents: {len(env.agents)}")
    for agent in env.agents:
        ms = agent.get_state_summary()
        print(f"\nAgent {agent.name}: interactions recorded = {ms['num_interactions']}")


def _parse_positive_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return max(1, default)
    try:
        return max(1, int(v))
    except ValueError:
        return max(1, default)


async def main():
    print(f"Random seed set to: {RANDOM_SEED}")

    NUM_AGENTS = 24
    MEMORY_SIZE = 0
    TOTAL_POPULATION_ROUNDS = 10
    REPUTATION_DELTA = 0.1

    base_result_dir = "result_prisoners_dilemma_group_reputation"
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PD_rep_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = f"PD_rep_{datetime.now().strftime('%m%d%H%M')}"

    main_result_dir = os.path.join(base_result_dir, base_experiment_name)
    os.makedirs(main_result_dir, exist_ok=True)

    # 强制使用 wuwen 后端（LLMAPI/wuwen.py）。
    from LLMAPI.zgc import LLMAgent
    _llm_backend_label = "LLMAPI.wuwen"
    llm_agent = LLMAgent(name="PrisonerDilemmaReputationAgent")

    llm_agent.router = llm_agent
    llm_agent.model_name = getattr(llm_agent, 'model', 'mock_model')
    agent_llm = llm_agent
    print(f"LLM backend: {_llm_backend_label} | model = {getattr(llm_agent, 'model', llm_agent.model_name)}")

    # 同时发起的 LLM 请求数上限（第 0 轮 24 人 + 每轮结束 24 人反思原为一键 gather，易触发 429）。
    # 环境变量 REPUTATION_LLM_MAX_CONCURRENT 可覆盖；未设置时若模型名含 qwen 则默认降为 6。
    llm_max_concurrent = _parse_positive_int_env("REPUTATION_LLM_MAX_CONCURRENT", 24)
    if os.getenv("REPUTATION_LLM_MAX_CONCURRENT") is None:
        _model_lc = (getattr(llm_agent, "model", "") or "").lower()
        if "qwen" in _model_lc:
            llm_max_concurrent = 6
            print(
                "提示: 检测到 Qwen 系列模型且未设置 REPUTATION_LLM_MAX_CONCURRENT，"
                f"已将并发上限设为 {llm_max_concurrent} 以减轻 429/配额压力。"
                "若仍报错可设为 2~4 或在控制台提升该模型配额。"
            )
    print(f"LLM concurrent cap: {llm_max_concurrent} (env REPUTATION_LLM_MAX_CONCURRENT)")

    NUM_EXPERIMENT_REPEATS = 1
    for experiment_index in range(NUM_EXPERIMENT_REPEATS):
        experiment_num = experiment_index + 1
        experiment_suffix = f"第{experiment_num}次"
        experiment_name = f"{base_experiment_name}_{experiment_suffix}"

        print("\n" + "=" * 70)
        print(f"Prisoner's Dilemma + Reputation — {experiment_suffix}")
        print("=" * 70)
        print(f"Experiment number: {experiment_num}/{NUM_EXPERIMENT_REPEATS}")
        print(f"N = {NUM_AGENTS}, population rounds = {TOTAL_POPULATION_ROUNDS}, δ = {REPUTATION_DELTA}")
        print(f"Experiment folder: {experiment_name}")
        print("=" * 70)

        env = PrisonersDilemmaReputationEnvironment(
            num_agents=NUM_AGENTS,
            memory_size=MEMORY_SIZE,
            total_population_rounds=TOTAL_POPULATION_ROUNDS,
            reputation_delta=REPUTATION_DELTA,
            llm_max_concurrent=llm_max_concurrent,
        )

        agents = []
        for i in range(NUM_AGENTS):
            prof = build_full_profile("Partner")
            agent = PrisonersDilemmaReputationAgent(id=i, name=f"Agent_{i}", profile=prof)
            await agent.init(agent_llm)
            agents.append(agent)

        env.set_agents(agents)
        for agent in agents:
            agent.set_environment(env)

        start_time = time.time()

        print("\n--- Round 0: initialization (R=1.0), first-round actions ---")
        await env.run_round_zero()

        for r in range(1, TOTAL_POPULATION_ROUNDS + 1):
            print(f"\n--- Population round {r}/{TOTAL_POPULATION_ROUNDS} ---")
            await env.run_population_round(r)

        end_time = time.time()

        print(f"\nSimulation {experiment_num} finished.")
        print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Total interactions: {len(env.game_logs)}")

        print_payoff_statistics(env)
        print_memory_statistics(env)

        await env.save_game_results(experiment_name, experiment_num, main_result_dir)

        exp_cfg_path = os.path.join(main_result_dir, "experiment_parameters.json")
        with open(exp_cfg_path, 'w', encoding='utf-8') as f:
            json.dump({
                "NUM_AGENTS": NUM_AGENTS,
                "TOTAL_POPULATION_ROUNDS": TOTAL_POPULATION_ROUNDS,
                "REPUTATION_DELTA": REPUTATION_DELTA,
                "RANDOM_SEED": RANDOM_SEED,
                "llm_max_concurrent": llm_max_concurrent,
            }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program error occurred: {e}")
        import traceback
        traceback.print_exc()
