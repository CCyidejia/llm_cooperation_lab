#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公共物品博弈（群体）+ 惩罚机制
基于 env_main_public_goods_group.py

阶段一：与原版相同，计算每人在公共池中的阶段一收益 stage1_payoff。
阶段二：每人对其余 23 人各分配 0–10 点「原始惩罚点」（LLM 决策）。
- 有效惩罚点 = 某人收到的原始惩罚点之和 × (3/23)，上限 10。
- 每 1 点有效惩罚点，扣除阶段一收益的 10%；有效点上限 10 → 最多扣光阶段一收益（不低于 0）。
- 对单一对象分配 k 点需支付的硬币成本为阶梯表（0→0，1→1，…，10→30）。
- 最终收益 = max(0, stage1×(1-扣除比例)) − 惩罚他人所付总成本；因惩罚支出可为负。

参数与原版一致处：24 人、初始禀赋、公共池乘数、轮数等由 EXPERIMENT_SETTINGS 配置。
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

from agentsociety2.agent.base import AgentBase, AgentLLM
from dotenv import load_dotenv
from LLMAPI.fiblab_http import LLMAgent

load_dotenv()

os.makedirs("result_public_goods_group_punishment", exist_ok=True)

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

# --- 惩罚机制常量（与需求表一致）---
# 对某一目标分配 k 点原始惩罚点时，惩罚者需支付的硬币成本
PUNISHMENT_COST_BY_POINTS = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 9, 6: 12, 7: 16, 8: 20, 9: 25, 10: 30
}
EFFECTIVE_PUNISHMENT_SCALE = 3.0 / 23.0  # 收到的原始点之和 × 3/23 → 有效惩罚点
MAX_EFFECTIVE_PUNISHMENT = 10
DEDUCTION_PER_EFFECTIVE_POINT = 0.10  # 每 1 有效点扣阶段一收益的 10%
MAX_PUNISHMENT_POINTS_PER_TARGET = 10


def _punishment_cost_for_points(k: int) -> int:
    k = max(0, min(MAX_PUNISHMENT_POINTS_PER_TARGET, int(k)))
    return PUNISHMENT_COST_BY_POINTS.get(k, 0)


class PublicGoodsPunishmentAgent(AgentBase):
    """公共物品博弈 + 惩罚：状态记录每轮最终收益。"""

    def __init__(self, id: int, name: str, profile: str = ""):
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []

        self._my_history = []
        self._outcome_history = []
        self.history = []

        self.initial_endowment = 20
        self.max_contribution = 20
        self.min_contribution = 0

    def set_environment(self, env):
        self._env = env

    def update_state(self, my_contribution: int, outcome: int):
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={outcome}")

    def update_history(self, round_summary: dict):
        self.history.append(round_summary)

    def get_state_summary(self) -> dict:
        return {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history)
        }

    def _build_history_string(self, all_agent_names: list) -> str:
        if not self.history:
            return "No previous rounds have been played."

        history_lines = ["History of previous rounds:"]
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
        return f"[Agent {self.name}] Received message: {message}"

    async def step(self, tick: int, t: datetime) -> str:
        return ""

    def _parse_contribution_response(self, content: str) -> tuple[int, str]:
        """
        在回复中按出现顺序查找第一个落在 [min_contribution, max_contribution] 的整数，
        避免把 23、96、9.6 中的片段等误当作贡献（旧逻辑只取「第一个任意整数」易错）。
        """
        if not content or not content.strip():
            return 0, "Empty response"

        for m in re.finditer(r"\b(\d+)\b", content):
            v = int(m.group(1))
            if self.min_contribution <= v <= self.max_contribution:
                rest = content[m.end() :].strip()
                explanation = rest if rest else "No explanation provided"
                return v, explanation

        logging.warning(
            f"[{self.name}] No integer in [{self.min_contribution},{self.max_contribution}] in LLM response; "
            f"using contribution 0. Snippet: {content[:120]!r}"
        )
        return 0, content.strip()[:500]

    async def choose_action(self, initial_endowment: int, public_pool_multiplier: float, num_agents: int, total_rounds: int, memory_size: int = 5):
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        current_round_true = len(self.history) + 1

        final_prompt = (
            f"{self._profile}\n"
            f"Current Game State:\n"
            f"This is round {current_round_true}.\n"
            f"You have {initial_endowment} coins.\n"
            f"Public fund contributions are multiplied by {public_pool_multiplier} and divided equally among all {num_agents} players.\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL INSTRUCTION***: Based ONLY on the rules and history, determine your contribution amount.\n"
            f"Your decision must be an integer between {self.min_contribution} and {self.max_contribution} (inclusive).\n"
            "Put that integer on the first line (you may prefix with Contribution: ), then 1-2 sentences of explanation."
        )
        system_message = ""

        try:
            content = await self._call_llm_with_retry(system_message=system_message, user_prompt=final_prompt)
            contribution, explanation = self._parse_contribution_response(content)
        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            contribution = 0
            explanation = f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, using default selection: 0"

        self._previous_choices.append(contribution)
        return contribution, explanation

    async def choose_punishment(
        self,
        other_agents_ordered: list,
        contributions_by_name: dict,
        my_stage1_payoff: float,
        round_label: int,
    ) -> tuple[dict, str]:
        """
        对其余每人分配 0–10 原始惩罚点。other_agents_ordered: 按固定顺序的 Agent 对象列表（不含自己）。
        返回 ({对方 name -> 点数}, explanation)
        """
        lines = []
        for a in other_agents_ordered:
            c = contributions_by_name.get(a.name, 0)
            lines.append(f"  - {a.name}: contributed {c} coins to the public fund this round.")
        others_block = "\n".join(lines)

        cost_lines = "\n".join(
            f"  Assigning {k} point(s) to one person costs you {PUNISHMENT_COST_BY_POINTS[k]} coins."
            for k in range(1, 11)
        )

        order_names = [a.name for a in other_agents_ordered]
        order_str = ", ".join(order_names)

        prompt = (
            f"{self._profile}\n\n"
            f"--- Punishment stage (after round {round_label} public goods) ---\n"
            f"You are {self.name}. Your earnings from this round BEFORE punishment (stage 1) are {my_stage1_payoff:.2f} coins.\n\n"
            f"Each other player's contribution to the public fund this round:\n{others_block}\n\n"
            "You may assign 0 to 10 **punishment points** to each of the other 23 players (integers only).\n"
            "Rules:\n"
            "- Points you assign to different people are chosen independently; costs add up.\n"
            f"- Cost to you for assigning k points to **one** specific person (k=0..10): "
            f"0→0, 1→1, 2→2, 3→4, 4→6, 5→9, 6→12, 7→16, 8→20, 9→25, 10→30 coins.\n"
            f"{cost_lines}\n"
            "- After everyone submits, each player's **received** punishment points are summed; "
            f"**effective punishment** = (sum received) × ({EFFECTIVE_PUNISHMENT_SCALE:.10f}… = 3/23), capped at {MAX_EFFECTIVE_PUNISHMENT}. "
            f"Each effective point removes {int(DEDUCTION_PER_EFFECTIVE_POINT * 100)}% of that player's **stage 1** earnings (max 100%).\n"
            "- Your **final** coins this round = (stage 1 earnings after being punished) minus (total cost of punishments you assigned). Final can be negative.\n\n"
            f"You MUST output exactly 23 integers separated by commas, in this exact order for players: {order_str}\n"
            "Example format: 0,0,1,0,10,3,...\n"
            "Then a short line starting with EXPLANATION: ...\n"
        )

        try:
            content = await self._call_llm_with_retry("", prompt)
            parsed, expl = self._parse_punishment_vector(content, order_names)
            return parsed, expl
        except Exception as e:
            logging.error(f"[{self.name}] Punishment LLM failed: {e}")
            return {n: 0 for n in order_names}, f"Punishment parse failed: {e}"

    def _parse_punishment_vector(self, content: str, order_names: list) -> tuple[dict, str]:
        expl = ""
        if "EXPLANATION:" in content.upper():
            idx = content.upper().find("EXPLANATION:")
            expl = content[idx:].split(":", 1)[-1].strip()
            content = content[:idx]
        nums = re.findall(r'\b(\d+)\b', content)
        values = []
        for s in nums[: len(order_names)]:
            v = int(s)
            if v < 0:
                v = 0
            if v > MAX_PUNISHMENT_POINTS_PER_TARGET:
                v = MAX_PUNISHMENT_POINTS_PER_TARGET
            values.append(v)
        while len(values) < len(order_names):
            values.append(0)
        if len(values) > len(order_names):
            values = values[: len(order_names)]
        return {name: values[i] for i, name in enumerate(order_names)}, expl or "parsed"

    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                if generated_text:
                    return generated_text
                raise ValueError("LLM returned empty string")
            except Exception as e:
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise


class PublicGoodsPunishmentEnvironment:
    """公共物品博弈 + 惩罚阶段。"""

    def __init__(self, num_agents: int, initial_endowment: int, public_pool_multiplier: float, total_interactions: int):
        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions

        self.agents = []
        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'initial_endowment': initial_endowment,
            'public_pool_multiplier': public_pool_multiplier,
            'interactions_per_run': 1,
            'punishment_mechanism': True,
            'effective_punishment_scale_3_23': EFFECTIVE_PUNISHMENT_SCALE,
            'max_effective_punishment': MAX_EFFECTIVE_PUNISHMENT,
            'deduction_per_effective_point': DEDUCTION_PER_EFFECTIVE_POINT,
            'punishment_cost_table': PUNISHMENT_COST_BY_POINTS,
        }

        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)
        self.game_logs = []

    def set_agents(self, agents: list):
        self.agents = agents
        for agent in agents:
            agent.set_environment(self)

    def _get_agent_by_id(self, agent_id):
        return next((a for a in self.agents if a._id == agent_id), None)

    def _ordered_others(self, agent):
        others = [a for a in self.agents if a._id != agent._id]
        others.sort(key=lambda x: x._id)
        return others

    async def run_interaction(self, interaction_num: int) -> dict:
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

        for agent_id, contribution in list(agent_contributions.items()):
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                logging.warning(f"Invalid contribution {contribution} from agent {agent_id}, setting to 0")
                agent_contributions[agent_id] = 0

        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents

        stage1_payoffs = {}
        for agent in self.agents:
            aid = agent._id
            contribution = agent_contributions[aid]
            private_savings = self.initial_endowment - contribution
            stage1_payoffs[aid] = float(private_savings + gain_per_agent)

        contributions_by_name = {self._get_agent_by_id(aid).name: agent_contributions[aid] for aid in agent_contributions}

        punish_tasks = []
        for agent in self.agents:
            others = self._ordered_others(agent)
            punish_tasks.append(
                agent.choose_punishment(
                    others,
                    contributions_by_name,
                    stage1_payoffs[agent._id],
                    interaction_num,
                )
            )
        punish_results = await asyncio.gather(*punish_tasks, return_exceptions=True)

        punishment_out = {}
        punishment_explanations = {}
        for agent, res in zip(self.agents, punish_results):
            if isinstance(res, Exception):
                logging.error(f"Punishment error {agent.name}: {res}")
                others = self._ordered_others(agent)
                punishment_out[agent._id] = {a.name: 0 for a in others}
                punishment_explanations[agent._id] = str(res)
            else:
                vec, expl = res
                punishment_out[agent._id] = vec
                punishment_explanations[agent._id] = expl

        n = self.num_agents
        matrix = [[0] * n for _ in range(n)]
        id_to_idx = {a._id: i for i, a in enumerate(sorted(self.agents, key=lambda x: x._id))}
        ids_sorted = sorted(id_to_idx.keys())

        for i in ids_sorted:
            for j in ids_sorted:
                if i == j:
                    continue
                target_agent = self._get_agent_by_id(j)
                punisher = self._get_agent_by_id(i)
                pts = punishment_out[i].get(target_agent.name, 0)
                try:
                    pts = int(pts)
                except (TypeError, ValueError):
                    pts = 0
                pts = max(0, min(MAX_PUNISHMENT_POINTS_PER_TARGET, pts))
                matrix[id_to_idx[i]][id_to_idx[j]] = pts

        received_raw = [0] * n
        for j_idx in range(n):
            for i_idx in range(n):
                if i_idx != j_idx:
                    received_raw[j_idx] += matrix[i_idx][j_idx]

        effective = []
        deduction_fraction = []
        after_punished_income = []
        punishment_costs = []

        for j_idx in range(n):
            r_sum = received_raw[j_idx]
            eff = min(r_sum * EFFECTIVE_PUNISHMENT_SCALE, MAX_EFFECTIVE_PUNISHMENT)
            effective.append(eff)
            ded = min(eff * DEDUCTION_PER_EFFECTIVE_POINT, 1.0)
            deduction_fraction.append(ded)
            s1 = stage1_payoffs[ids_sorted[j_idx]]
            after = max(0.0, s1 * (1.0 - ded))
            after_punished_income.append(after)

        for i_idx in range(n):
            cost = 0
            for j_idx in range(n):
                if i_idx == j_idx:
                    continue
                cost += _punishment_cost_for_points(matrix[i_idx][j_idx])
            punishment_costs.append(float(cost))

        final_payoffs = {}
        for j_idx, aid in enumerate(ids_sorted):
            final_payoffs[aid] = after_punished_income[j_idx] - punishment_costs[j_idx]

        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "contributions": {self._get_agent_by_id(i).name: agent_contributions[i] for i in ids_sorted},
            "stage1_payoffs": {self._get_agent_by_id(i).name: stage1_payoffs[i] for i in ids_sorted},
            "final_payoffs": {self._get_agent_by_id(i).name: final_payoffs[i] for i in ids_sorted},
            "punishment": {
                "received_raw_points": {self._get_agent_by_id(ids_sorted[j]).name: received_raw[j] for j in range(n)},
                "effective_punishment": {self._get_agent_by_id(ids_sorted[j]).name: effective[j] for j in range(n)},
                "deduction_fraction": {self._get_agent_by_id(ids_sorted[j]).name: deduction_fraction[j] for j in range(n)},
                "punishment_cost_paid": {self._get_agent_by_id(ids_sorted[j]).name: punishment_costs[j] for j in range(n)},
            },
        }

        for agent in self.agents:
            agent.update_state(
                my_contribution=agent_contributions[agent._id],
                outcome=int(round(final_payoffs[agent._id]))
            )
            agent.update_history(round_summary)

        self.game_stats['total_interactions'] += 1

        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "agent_contributions": {self._get_agent_by_id(i).name: agent_contributions[i] for i in ids_sorted},
            "explanations": {self._get_agent_by_id(i).name: explanations[i] for i in ids_sorted},
            "stage1_payoffs": {self._get_agent_by_id(i).name: stage1_payoffs[i] for i in ids_sorted},
            "punishment_assignments": {self._get_agent_by_id(i).name: dict(punishment_out[i]) for i in ids_sorted},
            "punishment_explanations": {self._get_agent_by_id(i).name: punishment_explanations[i] for i in ids_sorted},
            "punishment_received_raw_sum": {self._get_agent_by_id(ids_sorted[j]).name: received_raw[j] for j in range(n)},
            "effective_punishment": {self._get_agent_by_id(ids_sorted[j]).name: effective[j] for j in range(n)},
            "deduction_fraction": {self._get_agent_by_id(ids_sorted[j]).name: deduction_fraction[j] for j in range(n)},
            "punishment_cost_paid": {self._get_agent_by_id(ids_sorted[j]).name: punishment_costs[j] for j in range(n)},
            "payoffs": {self._get_agent_by_id(i).name: final_payoffs[i] for i in ids_sorted},
            "timestamp": datetime.now().isoformat()
        }

        self.game_logs.append(interaction_log)
        return interaction_log

    def get_game_summary(self) -> dict:
        return {
            'total_interactions': self.game_stats['total_interactions'],
            'success_count': self.game_stats['success_count'],
            'success_rate': self.game_stats['success_count'] / self.game_stats['total_interactions'] if self.game_stats['total_interactions'] > 0 else 0,
            'average_contribution': sum(k * v for k, v in self.choice_frequency.items()) / sum(self.choice_frequency.values()) if sum(self.choice_frequency.values()) > 0 else 0,
            'choice_distribution': dict(self.choice_frequency)
        }


async def main(base_experiment_name, experiment_index):
    EXPERIMENT_SETTINGS = {
        "num_agents": 24,
        "initial_endowment": 20,
        "public_pool_multiplier": 9.6,
        "num_rounds": 30,
        "random_seed": RANDOM_SEED,
        "punishment_mechanism": True,
    }

    base_result_dir = "result_public_goods_group_punishment"
    experiment_result_dir = os.path.join(base_result_dir, f"result_{base_experiment_name}")
    os.makedirs(experiment_result_dir, exist_ok=True)

    llm_agent = LLMAgent()
    model_name = getattr(llm_agent, 'model', 'default_model')
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)

    agents = []
    for i in range(EXPERIMENT_SETTINGS["num_agents"]):
        profile = (
            "You are a participant in a public goods game with a subsequent punishment stage. "
            "Your goal is to maximize your own accumulated coins across rounds."
        )
        agent = PublicGoodsPunishmentAgent(id=i + 1, name=f"Agent_{i+1}", profile=profile)
        await agent.init(agent_llm)
        agents.append(agent)

    env = PublicGoodsPunishmentEnvironment(
        num_agents=EXPERIMENT_SETTINGS["num_agents"],
        initial_endowment=EXPERIMENT_SETTINGS["initial_endowment"],
        public_pool_multiplier=EXPERIMENT_SETTINGS["public_pool_multiplier"],
        total_interactions=EXPERIMENT_SETTINGS["num_rounds"]
    )
    env.set_agents(agents)

    print(f"Starting Public Goods + Punishment ({EXPERIMENT_SETTINGS['num_agents']} agents)...")
    print(f"Results: {experiment_result_dir}")

    start_time = time.time()
    for round_num in range(1, EXPERIMENT_SETTINGS["num_rounds"] + 1):
        print(f"\nRound {round_num}/{EXPERIMENT_SETTINGS['num_rounds']}...")
        await env.run_interaction(round_num)

    end_time = time.time()
    print(f"\nDone in {end_time - start_time:.2f} s.")

    if experiment_index == 1:
        cfg = {**EXPERIMENT_SETTINGS, **env._config}
        with open(os.path.join(experiment_result_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(experiment_result_dir, f"game_logs{experiment_index}.json"), "w", encoding="utf-8") as f:
        json.dump(env.game_logs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(experiment_result_dir, f"game_stats{experiment_index}.json"), "w", encoding="utf-8") as f:
        json.dump(env.get_game_summary(), f, ensure_ascii=False, indent=2)

    agent_states = []
    for agent in agents:
        agent_states.append({
            "id": agent._id,
            "name": agent.name,
            "profile": agent._profile,
            "history": agent.history,
            "state_summary": agent.get_state_summary()
        })
    with open(os.path.join(experiment_result_dir, f"agent_states{experiment_index}.json"), "w", encoding="utf-8") as f:
        json.dump(agent_states, f, ensure_ascii=False, indent=2)

    print("Saved.")


async def run_experiment(experiment_name, experiment_index):
    await main(experiment_name, experiment_index)


if __name__ == "__main__":
    base_experiment_name = input("Please enter experiment folder name (e.g., 'PG_punish_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = datetime.now().strftime("%m%d_%H%M%S_PG_punishment")

    # 独立重复实验次数：仅运行 1 次；若需多次可改为 3 等，会生成 game_logs1/2/3…
    NUM_REPEATS = 1
    for experiment_index in range(1, NUM_REPEATS + 1):
        print(f"\n{'=' * 70}\n=== 公共物品+惩罚 实验 {experiment_index}/{NUM_REPEATS} ===\n{'=' * 70}")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== 实验 {experiment_index}/{NUM_REPEATS} 已结束 ===\n")
