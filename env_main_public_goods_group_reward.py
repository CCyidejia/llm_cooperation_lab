#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Public Goods Game (24 agents) + Targeted Reward (RN-style, Rand et al. 2009).

阶段一：与 env_main_public_goods_group.py 相同；π_1,i = (E - c_i) + m·Σc_j / n，E=20，m=9.6。
阶段二：公布全员贡献；每人对其余 23 人各选 Reward 或 Do nothing；Reward 成本 1、对方收益 3；
       总奖励支出 ≤ π_1,i，逐笔执行前剩余预算 ≥ 1（按目标 agent id 升序落实）。
阶段三：π_total,i = π_1,i + 3·R_received,i - R_given,i。

LLM 仅做决策：阶段一贡献额；阶段二对每个目标一行二元选择。结算由程序完成。
阶段一 user prompt（含历史段落）与 env_main_public_goods_group 逐字同构；阶段二在原 profile 后以分隔线追加最短说明。
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
from LLMAPI.wuwen import LLMAgent

load_dotenv()

os.makedirs("result_public_goods_group_reward", exist_ok=True)

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

REWARD_COST = 1.0
REWARD_BENEFIT = 3.0


class PublicGoodsRewardAgent(AgentBase):
    """公共物品 + 定向奖励：阶段一 prompt 与原版一致；阶段二单独调用。"""

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
        }

    def _build_history_string(self, all_agent_names: list) -> str:
        """与 env_main_public_goods_group.PublicGoodsAgent._build_history_string 逐字同构（仅总贡献与公共池增益）。"""
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
        """阶段一：与 env_main_public_goods_group.PublicGoodsAgent.choose_action 相同。"""
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

    def _parse_reward_line_for_target(self, content: str, target_name: str) -> bool | None:
        """解析 '<target>: Reward' 或 Do nothing；未匹配返回 None。"""
        esc = re.escape(target_name)
        pat = re.compile(
            rf"(?im)^\s*{esc}\s*:\s*(Reward|Do\s*nothing|Nothing)\s*$",
        )
        m = pat.search(content)
        if m:
            val = m.group(1).lower()
            return val == "reward"
        return None

    def parse_reward_actions(self, content: str, sorted_target_names: list[str]) -> dict[str, bool]:
        """每个目标默认 Do nothing；解析到的 Reward 为 True。"""
        out = {n: False for n in sorted_target_names}
        for name in sorted_target_names:
            v = self._parse_reward_line_for_target(content, name)
            if v is not None:
                out[name] = v
            else:
                for line in content.splitlines():
                    line_st = line.strip()
                    if not line_st.startswith(name):
                        continue
                    rest = line_st[len(name) :].strip()
                    if not rest.startswith(":"):
                        continue
                    rest = rest[1:].strip().lower()
                    if rest.startswith("reward") and "nothing" not in rest:
                        out[name] = True
                        break
                    if "nothing" in rest or rest == "nothing":
                        out[name] = False
                        break
        return out

    async def choose_reward_actions(
        self,
        pi_1: float,
        contributions_by_name: dict[str, int],
        sorted_target_names: list[str],
    ) -> tuple[dict[str, bool], str]:
        """阶段二：最短追加说明；返回 (目标名 -> 是否 Reward, 原始文本)。"""
        lines_contrib = "\n".join(
            f"{n}: {contributions_by_name[n]}" for n in sorted(contributions_by_name.keys(), key=lambda x: int(x.split("_")[1]))
        )
        lines_targets = "\n".join(sorted_target_names)

        reward_prompt = (
            f"{self._profile}\n"
            "---\n"
            "Phase 2 (same round, after public goods).\n"
            "Contributions this round:\n"
            f"{lines_contrib}\n\n"
            f"Your phase-1 payoff this round (budget for rewards): {pi_1:.2f}. "
            f"Each Reward costs {int(REWARD_COST)} from this budget.\n"
            "For each other player, output exactly one line: \"<name>: Reward\" or \"<name>: Do nothing\".\n\n"
            "Targets:\n"
            f"{lines_targets}\n"
        )

        try:
            content = await self._call_llm_with_retry(
                system_message="",
                user_prompt=reward_prompt,
            )
            parsed = self.parse_reward_actions(content, sorted_target_names)
            return parsed, content
        except Exception as e:
            logging.error(f"[{self.name}] Reward phase LLM failed: {e}")
            return {n: False for n in sorted_target_names}, f"[ERROR] {e}"

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


class PublicGoodsRewardEnvironment:
    """公共物品 + 阶段二奖励 + 阶段三结算。"""

    def __init__(
        self,
        num_agents: int,
        initial_endowment: int,
        public_pool_multiplier: float,
        total_interactions: int,
    ):
        self.num_agents = num_agents
        self.initial_endowment = initial_endowment
        self.public_pool_multiplier = public_pool_multiplier
        self.total_interactions = total_interactions

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

    def _get_agent_by_id(self, agent_id):
        return next((a for a in self.agents if a._id == agent_id), None)

    def _apply_reward_budget(
        self,
        pi_1_by_id: dict[int, float],
        intention_by_id: dict[int, dict[int, bool]],
    ) -> dict[int, dict[int, bool]]:
        """按目标 id 升序尝试落实 Reward；每次前剩余预算 >= 1。"""
        applied: dict[int, dict[int, bool]] = defaultdict(dict)
        sorted_agents = sorted(self.agents, key=lambda a: a._id)
        for agent in sorted_agents:
            aid = agent._id
            remaining = float(pi_1_by_id.get(aid, 0.0))
            others = [a for a in sorted_agents if a._id != aid]
            for target in others:
                tid = target._id
                want = intention_by_id.get(aid, {}).get(tid, False)
                if want and remaining >= 1.0 - 1e-9:
                    applied[aid][tid] = True
                    remaining -= REWARD_COST
                else:
                    applied[aid][tid] = False
        return applied

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
                ag = self._get_agent_by_id(agent_id)
                if ag:
                    explanations[ag.name] += " [Corrected to 0 due to invalid value]"

        total_contribution = sum(agent_contributions.values())
        public_pool_gain = total_contribution * self.public_pool_multiplier
        gain_per_agent = public_pool_gain / self.num_agents

        payoffs_pi1: dict[int, float] = {}
        for agent in self.agents:
            aid = agent._id
            c = agent_contributions[aid]
            private_savings = self.initial_endowment - c
            payoffs_pi1[aid] = private_savings + gain_per_agent

        contributions_by_name = {a.name: agent_contributions[a._id] for a in self.agents}
        sorted_by_id = sorted(self.agents, key=lambda a: a._id)

        reward_tasks = []
        for agent in self.agents:
            others = [a for a in sorted_by_id if a._id != agent._id]
            target_names = [a.name for a in others]
            t = agent.choose_reward_actions(
                pi_1=payoffs_pi1[agent._id],
                contributions_by_name=contributions_by_name,
                sorted_target_names=target_names,
            )
            reward_tasks.append((agent, t))

        reward_raw_text: dict[str, str] = {}
        intention_by_id: dict[int, dict[int, bool]] = defaultdict(dict)

        for agent, task in reward_tasks:
            try:
                parsed, raw = await task
                reward_raw_text[agent.name] = raw
                name_to_agent = {a.name: a for a in self.agents}
                for tname, wants in parsed.items():
                    if tname in name_to_agent:
                        intention_by_id[agent._id][name_to_agent[tname]._id] = wants
            except Exception as e:
                logging.error(f"Reward phase error for {agent.name}: {e}")
                reward_raw_text[agent.name] = str(e)

        applied = self._apply_reward_budget(payoffs_pi1, intention_by_id)

        R_given: dict[int, int] = {}
        R_received: dict[int, int] = defaultdict(int)
        for agent in self.agents:
            aid = agent._id
            rg = sum(1 for tid, ok in applied.get(aid, {}).items() if ok)
            R_given[aid] = rg
        for agent in self.agents:
            tid = agent._id
            rr = sum(1 for aid in applied if tid != aid and applied[aid].get(tid, False))
            R_received[tid] = rr

        payoff_total: dict[int, float] = {}
        for agent in self.agents:
            aid = agent._id
            pi1 = payoffs_pi1[aid]
            payoff_total[aid] = pi1 + REWARD_BENEFIT * R_received[aid] - REWARD_COST * R_given[aid]

        payoffs_pi1_name = {a.name: payoffs_pi1[a._id] for a in self.agents}
        payoff_total_name = {a.name: payoff_total[a._id] for a in self.agents}
        R_given_name = {a.name: R_given[a._id] for a in self.agents}
        R_received_name = {a.name: R_received[a._id] for a in self.agents}

        round_summary = {
            "round": interaction_num,
            "total_contribution": total_contribution,
            "public_pool_gain": public_pool_gain,
            "gain_per_agent": gain_per_agent,
            "contributions": {agent.name: agent_contributions[agent._id] for agent in self.agents},
            "payoffs": payoffs_pi1_name,
            "payoff_total": payoff_total_name,
            "R_given": R_given_name,
            "R_received": R_received_name,
        }

        for agent in self.agents:
            agent.update_state(
                my_contribution=agent_contributions[agent._id],
                outcome=payoff_total[agent._id],
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
            "explanations": explanations,
            "payoffs_pi1": payoffs_pi1_name,
            "payoff_total": payoff_total_name,
            "R_given": R_given_name,
            "R_received": R_received_name,
            "reward_phase_raw": reward_raw_text,
            "reward_applied": {
                self._get_agent_by_id(aid).name: {
                    self._get_agent_by_id(tid).name: v for tid, v in ad.items()
                }
                for aid, ad in applied.items()
            },
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
        "reward_cost": REWARD_COST,
        "reward_benefit": REWARD_BENEFIT,
        "mechanism": "PGG + RN targeted reward (Rand et al. 2009 style)",
    }

    base_result_dir = "result_public_goods_group_reward"
    experiment_result_dir = os.path.join(base_result_dir, f"result_{base_experiment_name}")
    os.makedirs(experiment_result_dir, exist_ok=True)

    llm_agent = LLMAgent()
    model_name = getattr(llm_agent, "model", "default_model")
    agent_llm = AgentLLM(router=llm_agent, model_name=model_name)

    agents = []
    for i in range(EXPERIMENT_SETTINGS["num_agents"]):
        profile = f"You are a participant in a game. Your goal is to maximize your own accumulated coins."
        agent = PublicGoodsRewardAgent(id=i + 1, name=f"Agent_{i + 1}", profile=profile)
        await agent.init(agent_llm)
        agents.append(agent)

    env = PublicGoodsRewardEnvironment(
        num_agents=EXPERIMENT_SETTINGS["num_agents"],
        initial_endowment=EXPERIMENT_SETTINGS["initial_endowment"],
        public_pool_multiplier=EXPERIMENT_SETTINGS["public_pool_multiplier"],
        total_interactions=EXPERIMENT_SETTINGS["num_rounds"],
    )
    env.set_agents(agents)

    print(f"Starting Public Goods + Reward experiment with {EXPERIMENT_SETTINGS['num_agents']} agents...")
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
                "explanations": log["explanations"],
                "payoffs_pi1": log["payoffs_pi1"],
                "payoff_total": log["payoff_total"],
                "R_given": log["R_given"],
                "R_received": log["R_received"],
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

    base_experiment_name = input("Please enter experiment folder name (e.g., 'PG_reward_test1'): ").strip()
    if not base_experiment_name:
        base_experiment_name = datetime.now().strftime("%m%d_%H%M%S_PG_reward")

    NUM_EXPERIMENT_REPEATS = 1
    for i in range(NUM_EXPERIMENT_REPEATS):
        experiment_index = i + 1
        print(f"\n=== Running experiment {experiment_index}/{NUM_EXPERIMENT_REPEATS} ===")
        asyncio.run(run_experiment(base_experiment_name, experiment_index))
        print(f"=== Experiment {experiment_index}/{NUM_EXPERIMENT_REPEATS} completed ===")
