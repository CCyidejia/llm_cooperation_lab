#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公共物品博弈（24 人）+ 间接互惠（IR）名誉机制。

阶段 A（PGG）：与 env_main_public_goods_group.py 相同；prompt 与原版一致。
阶段 B（IR）：每人被随机分配 k=2 个接收者（单向、无互指）；可见各接收者**自开始至今全部**合作二元记录（旧→新；1=贡献≥阈值，0=否）；
            对每人二元 Help(1)/Not Help(0)，Help 成本 5、对方 +8。
最终收益：π_i = PGG 部分 + Σ_m (8·I_i^m − 5·O_i^m)。

LLM 仅做决策；结算由程序完成。
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
from LLMAPI.wuwen import LLMAgent

load_dotenv()

os.makedirs("result_public_goods_group_reputation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_api_log.txt"),
        logging.StreamHandler(),
    ],
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --- IR / 合作判定参数 ---
COOPERATION_THRESHOLD = 10  # c >= 10 记为 1（合作）
K_RECEIVERS = 2
HELP_COST = 5
HELP_BENEFIT = 8


def build_donor_receivers_ring_shuffle(num_agents: int, rng: random.Random) -> dict[int, list[int]]:
    """
    每轮随机打乱顺序后，顺序中位置 p 的参与者向 (p+1)、(p+2) 捐赠（模 n）。
    保证每人出度 2、入度 2，且无 i→j 同时 j→i。
    返回：agent 索引（0..n-1） -> [receiver_idx, receiver_idx]
    """
    order = list(range(num_agents))
    rng.shuffle(order)
    out: dict[int, list[int]] = {}
    for p in range(num_agents):
        donor = order[p]
        r1 = order[(p + 1) % num_agents]
        r2 = order[(p + 2) % num_agents]
        out[donor] = [r1, r2]
    return out


def agent_index(agent) -> int:
    return int(agent._id) - 1


class PublicGoodsReputationAgent(AgentBase):
    """PGG 与原版一致；额外维护合作二元历史供 IR 展示。"""

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

        # 每轮 PGG 结束后追加 0/1（是否合作）
        self._cooperation_bits: list[int] = []

    def set_environment(self, env):
        self._env = env

    def update_state(self, my_contribution: int, outcome: float):
        self._my_history.append(my_contribution)
        self._outcome_history.append(outcome)
        logging.debug(f"[{self.name}] Updated state: my_contribution={my_contribution}, outcome={outcome}")

    def update_history(self, round_summary: dict):
        self.history.append(round_summary)

    def get_state_summary(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "my_history": self._my_history,
            "outcome_history": self._outcome_history,
            "num_interactions": len(self._my_history),
            "cooperation_bits": list(self._cooperation_bits),
        }

    def _build_history_string(self, all_agent_names: list) -> str:
        """与 env_main_public_goods_group 逐字同构（仅总贡献与公共池增益）。"""
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
        return f"[Agent {self.name}] Received message: {message}"

    async def step(self, tick: int, t: datetime) -> str:
        return ""

    async def choose_action(self, initial_endowment: int, public_pool_multiplier: float, num_agents: int, total_rounds: int, memory_size: int = 5):
        """阶段 A：与 env_main_public_goods_group.PublicGoodsAgent.choose_action 相同。"""
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, "_env") and self._env else []
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
            "State the integer amount first, followed by a brief 1-2 sentence explanation."
        )

        system_message = ""

        try:
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=final_prompt,
            )

            contribution = 0
            explanation = "LLM call or parsing failed"

            match = re.search(r"(\d+)\s*[-–—]?\s*(.*)$", content, re.DOTALL)

            if match:
                parsed_contribution = int(match.group(1))

                if self.min_contribution <= parsed_contribution <= self.max_contribution:
                    contribution = parsed_contribution
                else:
                    logging.warning(f"[{self.name}] Contribution value out of range: {parsed_contribution}, using default 0")
                    contribution = 0

                if match.group(2):
                    explanation = match.group(2).strip()
                else:
                    lines = content.split("\n")
                    for line in lines:
                        if str(contribution) in line:
                            num_idx = line.find(str(contribution))
                            if num_idx != -1:
                                line_explanation = line[num_idx + len(str(contribution)) :].strip()
                                if line_explanation and not line_explanation.startswith((":", "-", "—")):
                                    explanation = line_explanation
                                    break
                    if explanation == "LLM call or parsing failed":
                        explanation = "No explanation provided"
            else:
                keyword_match = re.search(r"\b(\d+)\b", content)
                if keyword_match:
                    parsed_contribution = int(keyword_match.group(1))
                    if self.min_contribution <= parsed_contribution <= self.max_contribution:
                        contribution = parsed_contribution
                    else:
                        logging.warning(f"[{self.name}] Keyword matched value out of range: {parsed_contribution}, using default 0")
                        contribution = 0

                    lines = content.split("\n")
                    for line in lines:
                        if keyword_match.group(1) in line:
                            line_explanation = line[line.find(keyword_match.group(1)) + len(keyword_match.group(1)) :].strip()
                            if line_explanation:
                                explanation = line_explanation
                                break

                    if explanation == "LLM call or parsing failed":
                        explanation = f"Extracted contribution: {contribution}"
                else:
                    raise ValueError(f"Failed to parse valid contribution, content:\n{content[:200]}")

        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            contribution = 0
            explanation = f"[CRITICAL FAILURE] {type(e).__name__} - {str(e)}, using default selection: 0"

        self._previous_choices.append(contribution)
        return contribution, explanation

    def _parse_ir_line(self, line: str, receiver_name: str) -> int | None:
        """解析 'Name: Help' / 'Name: Not Help' / 1 / 0。"""
        line = line.strip()
        if not line:
            return None
        if receiver_name in line:
            rest = line.split(":", 1)[-1].strip().lower()
            if rest.startswith("help") and "not" not in rest:
                return 1
            if "not" in rest and "help" in rest:
                return 0
        low = line.lower()
        if "not help" in low or low.endswith(" 0") or low == "0":
            return 0
        if "help" in low and "not" not in low:
            return 1
        m = re.search(r"\b([01])\b", line)
        if m:
            return int(m.group(1))
        return None

    async def choose_ir_actions(
        self,
        round_num: int,
        receiver_entries: list[tuple[str, list[int]]],
        help_cost: int,
        help_benefit: int,
    ) -> tuple[list[int], str]:
        """
        阶段 B：对每个接收者选择 Help(1) 或 Not Help(0)。
        receiver_entries: (receiver_name, history_bits 从旧到新，为截至目前全部轮次)
        """
        lines_desc = []
        for name, bits in receiver_entries:
            seq = ",".join(str(b) for b in bits) if bits else "(empty — no prior rounds)"
            lines_desc.append(f"- {name}: full past behavior sequence {seq} (1=cooperated in PGG, 0=did not)")

        block = "\n".join(lines_desc)

        ir_prompt = (
            f"{self._profile}\n"
            "---\n"
            f"Phase 2 (same round {round_num}, after public goods).\n"
            "You are matched to two different receivers. For each, choose Help or Not Help.\n"
            f"Help: you pay {help_cost} coins; that receiver gains {help_benefit} coins. Not Help: no cost or gain for this link.\n\n"
            "Receivers and their full observable cooperation history so far (oldest to newest):\n"
            f"{block}\n\n"
            "Output exactly two lines, one per receiver:\n"
            '"<ReceiverName>: Help" or "<ReceiverName>: Not Help"\n'
        )

        try:
            content = await self._call_llm_with_retry(system_message="", user_prompt=ir_prompt)
            decisions = [0, 0]
            for idx, (rname, _) in enumerate(receiver_entries):
                found = None
                for line in content.splitlines():
                    if rname in line:
                        found = self._parse_ir_line(line, rname)
                        if found is not None:
                            break
                if found is None:
                    for line in content.splitlines():
                        found = self._parse_ir_line(line, rname)
                        if found is not None:
                            break
                decisions[idx] = 1 if found == 1 else 0
            return decisions, content
        except Exception as e:
            logging.error(f"[{self.name}] IR phase failed: {e}")
            return [0, 0], str(e)

    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt,
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


class PublicGoodsReputationEnvironment:
    """PGG + IR 两阶段。"""

    def __init__(
        self,
        num_agents: int,
        initial_endowment: int,
        public_pool_multiplier: float,
        total_interactions: int,
        k_receivers: int,
        cooperation_threshold: int,
        rng: random.Random | None = None,
    ):
        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions
        self.k_receivers = k_receivers
        self.cooperation_threshold = cooperation_threshold
        self._rng = rng if rng is not None else random.Random(RANDOM_SEED)

        self.agents = []
        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            "max_tick": total_interactions,
            "num_agents": num_agents,
            "initial_endowment": initial_endowment,
            "public_pool_multiplier": public_pool_multiplier,
            "interactions_per_run": 1,
        }

        self.game_stats = {"success_count": 0, "total_interactions": 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)
        self.game_logs = []

    def set_agents(self, agents: list):
        self.agents = agents
        for agent in agents:
            agent.set_environment(self)

    def _get_agent_by_index(self, idx: int):
        for a in self.agents:
            if agent_index(a) == idx:
                return a
        return None

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
                memory_size=5,
            )
            choice_tasks.append((agent, task))

        agent_contributions = {}
        explanations = {}
        for agent, task in choice_tasks:
            try:
                contribution, explanation = await task
                agent_contributions[agent._id] = contribution
                explanations[agent.name] = explanation
                self.choice_frequency[contribution] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_contributions[agent._id] = 0
                explanations[agent.name] = f"Error: {str(e)}"
                self.choice_frequency[0] += 1

        for agent_id, contribution in list(agent_contributions.items()):
            if not isinstance(contribution, int) or contribution < 0 or contribution > self.initial_endowment:
                logging.warning(f"Invalid contribution {contribution} from agent {agent_id}, setting to 0")
                agent_contributions[agent_id] = 0
                ag = next((a for a in self.agents if a._id == agent_id), None)
                if ag:
                    explanations[ag.name] += " [Corrected to 0 due to invalid value]"

        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents

        pgg_payoffs: dict[int, float] = {}
        for agent in self.agents:
            aid = agent._id
            c = agent_contributions[aid]
            private_savings = self.initial_endowment - c
            pgg_payoffs[aid] = private_savings + gain_per_agent

        # 合作二元（本轮 PGG 之后写入历史，再供 IR 展示）
        coop_this_round: dict[int, int] = {}
        for agent in self.agents:
            c = agent_contributions[agent._id]
            bit = 1 if c >= self.cooperation_threshold else 0
            coop_this_round[agent._id] = bit
            agent._cooperation_bits.append(bit)

        n = self.num_agents
        donor_to_recv_idx = build_donor_receivers_ring_shuffle(n, self._rng)

        ir_tasks = []
        for agent in self.agents:
            di = agent_index(agent)
            ridxs = donor_to_recv_idx[di]
            receiver_entries: list[tuple[str, list[int]]] = []
            for rj in ridxs:
                recv_agent = self._get_agent_by_index(rj)
                assert recv_agent is not None
                # 含本轮 PGG 刚写入的合作记录；展示截至目前全部二元历史（旧→新）
                bits = recv_agent._cooperation_bits
                receiver_entries.append((recv_agent.name, list(bits)))

            t = agent.choose_ir_actions(
                interaction_num,
                receiver_entries,
                HELP_COST,
                HELP_BENEFIT,
            )
            ir_tasks.append((agent, t))

        donor_help: dict[int, list[int]] = {}
        ir_raw: dict[str, str] = {}
        for agent, task in ir_tasks:
            try:
                decisions, raw = await task
                di = agent_index(agent)
                donor_help[di] = decisions
                ir_raw[agent.name] = raw
            except Exception as e:
                logging.error(f"IR error {agent.name}: {e}")
                donor_help[agent_index(agent)] = [0, 0]
                ir_raw[agent.name] = str(e)

        # O[di][pos]：捐赠者 di 对其第 pos 个接收者的 Help 决策
        O: dict[int, list[int]] = {agent_index(a): donor_help.get(agent_index(a), [0, 0]) for a in self.agents}
        # I[j]：接收者 j 从每位指向自己的捐赠者处是否收到帮助（按捐赠者索引升序，与公式中 m=1..k 一致）
        I: dict[int, list[int]] = {}
        for j in range(n):
            pairs: list[tuple[int, int]] = []
            for di, rlist in donor_to_recv_idx.items():
                if j in rlist:
                    pos = rlist.index(j)
                    pairs.append((di, pos))
            pairs.sort(key=lambda x: x[0])
            helps_in = []
            for di, pos in pairs[: self.k_receivers]:
                od = O.get(di, [0, 0])
                h = od[pos] if pos < len(od) else 0
                helps_in.append(h)
            while len(helps_in) < self.k_receivers:
                helps_in.append(0)
            I[j] = helps_in[: self.k_receivers]

        final_payoffs: dict[int, float] = {}
        for agent in self.agents:
            idx = agent_index(agent)
            pi = pgg_payoffs[agent._id]
            for m in range(self.k_receivers):
                pi += HELP_BENEFIT * I[idx][m] - HELP_COST * O[idx][m]
            final_payoffs[agent._id] = pi

        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "payoffs_pgg": {agent.name: pgg_payoffs[agent._id] for agent in self.agents},
            "payoffs_total": {agent.name: final_payoffs[agent._id] for agent in self.agents},
            "cooperation_bits_this_round": {agent.name: coop_this_round[agent._id] for agent in self.agents},
        }

        for agent in self.agents:
            agent.update_state(
                my_contribution=agent_contributions[agent._id],
                outcome=final_payoffs[agent._id],
            )
            agent.update_history(round_summary)

        self.game_stats["total_interactions"] += 1

        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "agent_contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "explanations_pgg": {agent.name: explanations[agent.name] for agent in self.agents},
            "payoffs_pgg": {agent.name: pgg_payoffs[agent._id] for agent in self.agents},
            "payoffs_total": {agent.name: final_payoffs[agent._id] for agent in self.agents},
            "cooperation_bits": {agent.name: coop_this_round[agent._id] for agent in self.agents},
            "donor_to_receivers_index": {self._get_agent_by_index(k).name: [self._get_agent_by_index(v).name for v in vs] for k, vs in donor_to_recv_idx.items()},
            "donor_help_decisions": {self._get_agent_by_index(k).name: O[k] for k in O},
            "receiver_incoming_help": {self._get_agent_by_index(k).name: I[k] for k in I},
            "ir_raw": ir_raw,
            "timestamp": datetime.now().isoformat(),
        }

        self.game_logs.append(interaction_log)
        return interaction_log

    def get_game_summary(self) -> dict:
        return {
            "total_interactions": self.game_stats["total_interactions"],
            "success_count": self.game_stats["success_count"],
            "success_rate": self.game_stats["success_count"] / self.game_stats["total_interactions"]
            if self.game_stats["total_interactions"] > 0
            else 0,
            "average_contribution": sum(k * v for k, v in self.choice_frequency.items()) / sum(self.choice_frequency.values())
            if sum(self.choice_frequency.values()) > 0
            else 0,
            "choice_distribution": dict(self.choice_frequency),
        }


async def main(base_experiment_name, experiment_index):
    EXPERIMENT_SETTINGS = {
        "num_agents": 24,
        "initial_endowment": 20,
        "public_pool_multiplier": 9.6,
        "num_rounds": 30,
        "random_seed": RANDOM_SEED,
        "cooperation_threshold": COOPERATION_THRESHOLD,
        "ir_history_display": "full_sequence_all_rounds",
        "k_receivers": K_RECEIVERS,
        "help_cost": HELP_COST,
        "help_benefit": HELP_BENEFIT,
        "mechanism": "PGG + IR (binary cooperation record, full history per receiver)",
    }

    base_result_dir = "result_public_goods_group_reputation"
    experiment_result_dir = os.path.join(base_result_dir, f"result_{base_experiment_name}")
    os.makedirs(experiment_result_dir, exist_ok=True)

    llm_agent = LLMAgent()
    model_name = getattr(llm_agent, "model", "default_model")
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)

    agents = []
    for i in range(EXPERIMENT_SETTINGS["num_agents"]):
        profile = f"You are a participant in a game. Your goal is to maximize your own accumulated coins."
        agent = PublicGoodsReputationAgent(id=i + 1, name=f"Agent_{i + 1}", profile=profile)
        await agent.init(agent_llm)
        agents.append(agent)

    rng = random.Random(RANDOM_SEED + experiment_index * 9973)

    env = PublicGoodsReputationEnvironment(
        num_agents=EXPERIMENT_SETTINGS["num_agents"],
        initial_endowment=EXPERIMENT_SETTINGS["initial_endowment"],
        public_pool_multiplier=EXPERIMENT_SETTINGS["public_pool_multiplier"],
        total_interactions=EXPERIMENT_SETTINGS["num_rounds"],
        k_receivers=EXPERIMENT_SETTINGS["k_receivers"],
        cooperation_threshold=EXPERIMENT_SETTINGS["cooperation_threshold"],
        rng=rng,
    )
    env.set_agents(agents)

    print(f"Starting PGG + IR experiment with {EXPERIMENT_SETTINGS['num_agents']} agents...")
    print(f"Results will be saved to: {experiment_result_dir}")

    start_time = time.time()

    for round_num in range(1, EXPERIMENT_SETTINGS["num_rounds"] + 1):
        print(f"\nRunning round {round_num}/{EXPERIMENT_SETTINGS['num_rounds']}...")
        await env.run_interaction(round_num)

    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\nExperiment completed in {total_duration:.2f} seconds!")

    print(f"\nSaving results to {experiment_result_dir}...")

    if experiment_index == 1:
        config_file = os.path.join(experiment_result_dir, "experiment_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(EXPERIMENT_SETTINGS, f, ensure_ascii=False, indent=2)

    logs_file = os.path.join(experiment_result_dir, f"game_logs{experiment_index}.json")
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(env.game_logs, f, ensure_ascii=False, indent=2)

    stats_file = os.path.join(experiment_result_dir, f"game_stats{experiment_index}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(env.get_game_summary(), f, ensure_ascii=False, indent=2)

    agent_states = []
    for agent in agents:
        agent_states.append(
            {
                "id": agent._id,
                "name": agent.name,
                "profile": agent._profile,
                "history": agent.history,
                "state_summary": agent.get_state_summary(),
            }
        )

    agent_states_file = os.path.join(experiment_result_dir, f"agent_states{experiment_index}.json")
    with open(agent_states_file, "w", encoding="utf-8") as f:
        json.dump(agent_states, f, ensure_ascii=False, indent=2)

    interaction_results = []
    for log in env.game_logs:
        interaction_results.append(
            {
                "round": log["interaction"],
                "total_contribution": log["total_contribution"],
                "public_pool_gain": log["public_pool_gain"],
                "gain_per_agent": log["gain_per_agent"],
                "agent_contributions": log["agent_contributions"],
                "explanations_pgg": log.get("explanations_pgg"),
                "payoffs_pgg": log.get("payoffs_pgg"),
                "payoffs_total": log.get("payoffs_total"),
                "cooperation_bits": log.get("cooperation_bits"),
                "timestamp": log["timestamp"],
            }
        )

    interactions_file = os.path.join(experiment_result_dir, f"interaction_results{experiment_index}.json")
    with open(interactions_file, "w", encoding="utf-8") as f:
        json.dump(interaction_results, f, ensure_ascii=False, indent=2)

    print("All results saved successfully!")
    print(f"Experiment results directory: {experiment_result_dir}")


async def run_experiment(experiment_name, experiment_index):
    await main(experiment_name, experiment_index)


if __name__ == "__main__":
    import asyncio

    base_experiment_name = input("Please enter experiment folder name (e.g., 'PG_IR_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = datetime.now().strftime("%m%d_%H%M%S_PG_IR")

    NUM_EXPERIMENT_REPEATS = 1
    for i in range(NUM_EXPERIMENT_REPEATS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_REPEATS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_REPEATS} completed ===")
