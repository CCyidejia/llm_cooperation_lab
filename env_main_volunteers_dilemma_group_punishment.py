#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Volunteer's Dilemma Group + 二阶惩罚机制（agentsociety V2）

阶段一：24 人同时选 Volunteer (V) 或 Stand by (S)。
- 情形 A：全员 S → 公共品失败，全员收益 0；无阶段二。
- 情形 B：至少 1 人 V → 暂存基础收益（V=60，S=100）；若同时存在 S，进入阶段二；若全员 V，直接结算（全员 60）。

阶段二（仅 B 且存在 S）：仅向阶段一选 V 的 Agent 询问是否支付 K=10 惩罚搭便车者。
- 情形 C：至少 1 名 V 愿意支付 → 惩罚全局生效：执行惩罚的 V 得 60−K；未执行的 V 得 60；所有 S 得 100−P。
- 情形 D：所有 V 均不愿支付 → 惩罚失败：所有 V 得 60；所有 S 得 100。

基于 env_main_volunteers_dilemma_group.py。

阶段一给 LLM 的规则文本与 baseline（group）一致（B/C 表述 + 相同编号与 Payoff Structure），
仅在其后追加「阶段二 / 二阶惩罚」说明与同一目标句，便于对照实验时把「制度变化」与「表述差异」分开。
"""
import os
import json
import sys
import asyncio
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

os.makedirs("result_volunteer_dilemma_group_punishment", exist_ok=True)

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

# --- 收益与惩罚参数（与需求一致）---
PAYOFF_V_BASE = 60
PAYOFF_S_BASE = 100
K_PUNISH = 10
P_FREE_RIDER = 40


class VolunteerDilemmaPunishmentAgent(AgentBase):
    """志愿者博弈 + 二阶惩罚：阶段一选 V/S，阶段二（若触发）选是否支付 K。"""

    def __init__(self, id: int, name: str, profile: str = ""):
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []

        self._my_history = []
        self._partner_history = []
        self._outcome_history = []
        self._partner_map = defaultdict(list)
        self.history = []

        self.decision_options = ["Volunteer", "Stand by"]
        self.punish_options = ["Pay", "Decline"]
        self._profile = profile

    def set_environment(self, env):
        self._env = env

    def update_state(self, my_choice: str, partner_choice: str, partner_id: int, outcome: int):
        self._my_history.append(my_choice)
        self._partner_history.append(partner_choice)
        self._outcome_history.append(outcome)
        self._partner_map[partner_id].append((my_choice, partner_choice, outcome))
        logging.debug(
            f"[{self.name}] Updated state: my_choice={my_choice}, partner_choice={partner_choice}, "
            f"partner_id={partner_id}, outcome={outcome}"
        )

    def update_history(self, round_summary: dict):
        self.history.append(round_summary)

    def get_state_summary(self) -> dict:
        return {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'partner_history': self._partner_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history),
            'interaction_partners': dict(self._partner_map)
        }

    def _build_history_string(self, all_agent_names: list) -> str:
        if not self.history:
            return "No previous rounds have been played."

        history_lines = ["History of previous rounds:"]
        for round_summary in self.history:
            r = round_summary["round"]
            num_v = round_summary["num_volunteers"]
            scenario = round_summary.get("scenario", "")
            history_lines.append(f"Round {r}:")
            history_lines.append(f"  Volunteers: {num_v}, scenario: {scenario}")
            if round_summary.get("phase2_triggered"):
                history_lines.append("  Phase 2 (punishment) was offered to volunteers.")
                pw = round_summary.get("punish_willing_by_name", {})
                if pw:
                    history_lines.append(f"  Who paid to punish: {pw}")
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

    def _get_benefit_cost(self) -> tuple[int, int]:
        """与 env_main_volunteers_dilemma_group 中 benefit_b / cost_c 一致：B = payoff_s_base，C = B − V。"""
        if self._env is not None:
            sb = getattr(self._env, "payoff_s_base", PAYOFF_S_BASE)
            vb = getattr(self._env, "payoff_v_base", PAYOFF_V_BASE)
            return sb, sb - vb
        return PAYOFF_S_BASE, PAYOFF_S_BASE - PAYOFF_V_BASE

    def _get_volunteer_baseline_rules_text(self, benefit_b: int, cost_c: int) -> str:
        """
        与 env_main_volunteers_dilemma_group.VolunteerDilemmaAgent._get_volunteer_rules_text
        相同的阶段一规则（含 Context / Game Rules 1–5 / Payoff Structure），不含最后一句目标，
        以便在全文末尾统一写目标句，并在中间插入惩罚机制说明。
        """
        return f"""
Context: You are playing a game with a group of players.

Game Rules:
1. The game involves multiple rounds where all players participate simultaneously.
2. In each round, every player chooses either to "Volunteer" or "Stand by".
3. If at least one player volunteers, all players receive a benefit of {benefit_b} points.
4. However, volunteers pay a cost of {cost_c} points.
5. If no one volunteers, everyone receives 0 points.

Payoff Structure:
- If you volunteer: You get {benefit_b - cost_c} points (regardless of others' choices)
- If you stand by and someone volunteers: You get {benefit_b} points
- If no one volunteers: You get 0 points
"""

    def _get_punishment_addon_rules_text(self) -> str:
        """在 baseline 规则之后追加：仅本实验存在的阶段二（二阶惩罚），数值来自环境或模块常量。"""
        env = self._env
        k = getattr(env, "k_punish", K_PUNISH) if env else K_PUNISH
        p = getattr(env, "p_free_rider", P_FREE_RIDER) if env else P_FREE_RIDER
        vb = getattr(env, "payoff_v_base", PAYOFF_V_BASE) if env else PAYOFF_V_BASE
        sb = getattr(env, "payoff_s_base", PAYOFF_S_BASE) if env else PAYOFF_S_BASE
        return f"""
Additional rules (Phase 2 — second-order punishment, only in this experiment):
This phase occurs only after phase-1 choices in a round, and only if at least one player volunteered AND at least one player stood by.
Only players who chose Volunteer in phase 1 are asked whether they will pay {k} points to activate a punishment that reduces each stand-by player's payoff by {p} points (applied to the phase-1 stand-by payoff described above).
- If at least one volunteer pays {k}: punishment applies. Volunteers who paid get {vb - k} points this round; volunteers who did not pay get {vb} points; each stand-by player gets {sb - p} points.
- If no volunteer pays: punishment does not apply. Each volunteer gets {vb} points; each stand-by player gets {sb} points.

If everyone volunteers in phase 1 (no stand-by players), phase 2 does not occur; each volunteer gets {vb} points.
"""

    def _get_volunteer_rules_text(self) -> str:
        """baseline（与 group 一致）+ 惩罚追加说明 + 与 group 相同措辞的目标句。"""
        benefit_b, cost_c = self._get_benefit_cost()
        baseline = self._get_volunteer_baseline_rules_text(benefit_b, cost_c).strip()
        addon = self._get_punishment_addon_rules_text().strip()
        goal = "Your goal is to maximize your own accumulated points."
        return "\n".join([baseline, "", addon, "", goal])

    async def choose_action(self, partner_id: int = None, memory_size: int = 5):
        rules = self._get_volunteer_rules_text()
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        current_round_true = len(self.history) + 1
        current_score = sum(self._outcome_history) if hasattr(self, '_outcome_history') else 0
        history_intro = ""

        # 与 env_main_volunteers_dilemma_group.choose_action 中 new_query 保持一致，仅阶段一决策用语可比
        new_query = (
            f"It is now round {current_round_true} of your interactions.\n" +
            f"Your current total score is {current_score}.\n" +
            f"Please decide whether to 'Volunteer' or 'Stand by'.\n" +
            "\n" +
            "***CRITICAL DECISION INSTRUCTION***:\n" +
            "Please make your decision strictly following this format:\n" +
            "Decision: [Volunteer/Stand by]\n" +
            "Explanation: [Brief 1-2 sentence explanation]"
        )

        if len(self.history) == 0:
            final_prompt = "\n".join([rules, new_query])
        else:
            final_prompt = "\n".join([rules, history_intro, histories_text, new_query])

        system_message = self._profile
        try:
            content = await self._call_llm_with_retry(system_message=system_message, user_prompt=final_prompt)
            choice = self.decision_options[1]
            explanation = "LLM call or parsing failed"

            match_volunteer = re.search(r'Volunteer', content, re.IGNORECASE)
            match_standby = re.search(r'Stand\s*by', content, re.IGNORECASE)

            if match_volunteer and (not match_standby or match_volunteer.start() < match_standby.start()):
                choice = self.decision_options[0]
                if match_volunteer.end() < len(content):
                    explanation = content[match_volunteer.end():].strip()
            elif match_standby:
                choice = self.decision_options[1]
                if match_standby.end() < len(content):
                    explanation = content[match_standby.end():].strip()
            else:
                raise ValueError(f"Failed to parse valid choice, content:\n{content[:200]}")

            if choice == self.decision_options[1]:
                found_options = [opt for opt in self.decision_options if opt in content]
                if found_options:
                    choice = found_options[-1]

            if choice not in self.decision_options:
                logging.warning(f"[{self.name}] Hallucinated '{choice}', using stable choice logic.")
                choice = self._get_stable_choice()
                explanation = f"Using stable choice logic: {choice}"

        except Exception as e:
            logging.error(f"[{self.name}] LLM Interaction failed: {e}")
            choice = self._get_stable_choice()
            explanation = f"LLM call failed: {type(e).__name__}"

        self._previous_choices.append(choice)
        return choice, explanation

    async def choose_punish_decision(
        self,
        round_label: int,
        num_standby_players: int,
    ) -> tuple[str, str]:
        """
        阶段二：是否支付 K 以触发对全体 S 的惩罚。返回 ("Pay"|"Decline", explanation)。
        """
        rules = self._get_volunteer_rules_text()
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        current_score = sum(self._outcome_history) if hasattr(self, '_outcome_history') else 0
        env = self._env
        k = getattr(env, "k_punish", K_PUNISH) if env else K_PUNISH
        p = getattr(env, "p_free_rider", P_FREE_RIDER) if env else P_FREE_RIDER
        vb = getattr(env, "payoff_v_base", PAYOFF_V_BASE) if env else PAYOFF_V_BASE
        sb = getattr(env, "payoff_s_base", PAYOFF_S_BASE) if env else PAYOFF_S_BASE

        prompt = (
            f"{rules}\n\n"
            f"--- Phase 2 of round {round_label} ---\n"
            f"In phase 1 of this round you chose Volunteer. There are {num_standby_players} player(s) who chose Stand by (free riders).\n"
            f"Your current total score before this round's final settlement is based on prior rounds; your running total is {current_score}.\n\n"
            f"If you pay {k} points now and at least one volunteer pays, punishment applies globally: "
            f"each stand-by player loses {p} points from their {sb} base; "
            f"volunteers who pay get {vb - k} this round; volunteers who decline get {vb} if punishment still happens because someone else paid.\n"
            f"If NO volunteer pays, everyone keeps the base payoffs ({vb} for volunteers, {sb} for stand-by).\n\n"
            f"{histories_text}\n\n"
            "***CRITICAL DECISION INSTRUCTION***:\n"
            "Decide whether to PAY the cost to support punishment or DECLINE.\n"
            "Respond in this format:\n"
            "Decision: [Pay/Decline]\n"
            "Explanation: [Brief 1-2 sentence explanation]"
        )

        system_message = self._profile
        try:
            content = await self._call_llm_with_retry(system_message=system_message, user_prompt=prompt)
            decision = "Decline"
            explanation = "LLM call or parsing failed"

            match_pay = re.search(r'\bPay\b', content, re.IGNORECASE)
            match_decline = re.search(r'\bDecline\b', content, re.IGNORECASE)

            if match_pay and match_decline:
                decision = self.punish_options[0] if match_pay.start() < match_decline.start() else self.punish_options[1]
            elif match_pay:
                decision = self.punish_options[0]
            elif match_decline:
                decision = self.punish_options[1]
            else:
                if re.search(r'not\s+pay|refuse|won\'t|will not', content, re.IGNORECASE):
                    decision = "Decline"
                elif re.search(r'pay\s+the\s+cost|willing\s+to\s+pay|i\s+pay', content, re.IGNORECASE):
                    decision = "Pay"
                else:
                    raise ValueError(f"Failed to parse Pay/Decline, content:\n{content[:200]}")

            if decision == "Pay":
                if match_pay and match_pay.end() < len(content):
                    explanation = content[match_pay.end():].strip()[:500]
            else:
                if match_decline and match_decline.end() < len(content):
                    explanation = content[match_decline.end():].strip()[:500]

        except Exception as e:
            logging.error(f"[{self.name}] Punish decision LLM failed: {e}")
            decision = random.choice(self.punish_options)
            explanation = f"Fallback: {type(e).__name__}"

        return decision, explanation

    def _get_stable_choice(self) -> str:
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            if count > len(self._previous_choices) / 2:
                logging.info(f"[{self.name}] Using most frequent previous choice: {most_common_choice}")
                return most_common_choice

        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, count = self._env.choice_frequency.most_common(1)[0]
                if most_popular_choice in self.decision_options:
                    logging.info(f"[{self.name}] Using most popular environment choice: {most_popular_choice}")
                    return most_popular_choice

        logging.info(f"[{self.name}] Using random choice as fallback")
        return random.choice(self.decision_options)

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


class VolunteerDilemmaPunishmentEnvironment:
    """24 人志愿者博弈 + 二阶惩罚。"""

    def __init__(
        self,
        num_agents: int,
        total_interactions: int,
        memory_size: int = 100,
        interaction_schedule: list = None,
        payoff_v_base: int = PAYOFF_V_BASE,
        payoff_s_base: int = PAYOFF_S_BASE,
        k_punish: int = K_PUNISH,
        p_free_rider: int = P_FREE_RIDER,
    ):
        self.num_agents = num_agents
        self.total_interactions = total_interactions
        self.memory_size = memory_size
        self.interaction_schedule = interaction_schedule

        self.payoff_v_base = payoff_v_base
        self.payoff_s_base = payoff_s_base
        self.k_punish = k_punish
        self.p_free_rider = p_free_rider

        self.agents = []
        self.interaction_number = 0
        self.initial_time = None

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'payoff_v_base': payoff_v_base,
            'payoff_s_base': payoff_s_base,
            'k_punish': k_punish,
            'p_free_rider': p_free_rider,
            'memory_size': memory_size,
            'interactions_per_run': 1
        }

        self.game_stats = {
            'success_count': 0,
            'total_interactions': 0,
            'punishment_triggered_count': 0,
            'punishment_success_count': 0,
        }
        self.success_record = []
        self.choice_frequency = defaultdict(int)
        self.game_logs = []

    def set_agents(self, agents: list):
        self.agents = agents
        for agent in agents:
            agent.set_environment(self)

    def _get_agent_by_id(self, agent_id):
        return next((a for a in self.agents if a._id == agent_id), None)

    async def run_interaction(self, interaction_num: int) -> dict:
        if self.initial_time is None:
            self.initial_time = datetime.now()

        self.interaction_number = interaction_num

        choice_tasks = []
        for agent in self.agents:
            task = agent.choose_action(partner_id=None, memory_size=self.memory_size)
            choice_tasks.append((agent, task))

        agent_choices = {}
        explanations = {}
        for agent, task in choice_tasks:
            try:
                choice, explanation = await task
                agent_choices[agent._id] = choice
                explanations[agent._id] = explanation
                self.choice_frequency[choice] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_choices[agent._id] = "Stand by"
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency["Stand by"] += 1

        num_volunteers = sum(1 for c in agent_choices.values() if c == "Volunteer")
        num_standby = self.num_agents - num_volunteers
        is_someone_volunteering = num_volunteers > 0

        volunteer_agent_ids = [aid for aid, c in agent_choices.items() if c == "Volunteer"]
        punish_willing = {}
        punish_explanations = {}
        phase2_triggered = False
        scenario = ""

        payoffs = {}

        if not is_someone_volunteering:
            scenario = "A_all_standby"
            for aid in agent_choices:
                payoffs[aid] = 0
        elif num_standby == 0:
            scenario = "B_all_volunteer_no_phase2"
            for aid in agent_choices:
                payoffs[aid] = self.payoff_v_base
        else:
            phase2_triggered = True
            self.game_stats['punishment_triggered_count'] += 1

            punish_tasks = []
            for aid in volunteer_agent_ids:
                ag = self._get_agent_by_id(aid)
                t = ag.choose_punish_decision(
                    round_label=interaction_num,
                    num_standby_players=num_standby,
                )
                punish_tasks.append((ag, t))

            for ag, t in punish_tasks:
                try:
                    dec, expl = await t
                    punish_willing[ag._id] = dec
                    punish_explanations[ag._id] = expl
                except Exception as e:
                    logging.error(f"Punish decision error {ag.name}: {e}")
                    punish_willing[ag._id] = "Decline"
                    punish_explanations[ag._id] = str(e)

            at_least_one_pays = any(punish_willing.get(aid) == "Pay" for aid in volunteer_agent_ids)

            if at_least_one_pays:
                scenario = "C_punishment_success"
                self.game_stats['punishment_success_count'] += 1
                for aid, ch in agent_choices.items():
                    if ch == "Volunteer":
                        payoffs[aid] = (
                            self.payoff_v_base - self.k_punish
                            if punish_willing.get(aid) == "Pay"
                            else self.payoff_v_base
                        )
                    else:
                        payoffs[aid] = self.payoff_s_base - self.p_free_rider
            else:
                scenario = "D_punishment_failed"
                for aid, ch in agent_choices.items():
                    payoffs[aid] = self.payoff_v_base if ch == "Volunteer" else self.payoff_s_base

        round_summary = {
            "round": interaction_num,
            "choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "num_volunteers": num_volunteers,
            "num_standby": num_standby,
            "scenario": scenario,
            "phase2_triggered": phase2_triggered,
            "punish_willing_by_name": {
                self._get_agent_by_id(aid).name: punish_willing.get(aid)
                for aid in volunteer_agent_ids
            } if volunteer_agent_ids else {},
            "punish_explanations_by_name": {
                self._get_agent_by_id(aid).name: punish_explanations.get(aid, "")
                for aid in volunteer_agent_ids
            } if volunteer_agent_ids else {},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
        }

        for agent in self.agents:
            agent_choice = agent_choices[agent._id]
            agent.update_state(
                my_choice=agent_choice,
                partner_choice=None,
                partner_id=None,
                outcome=payoffs[agent._id]
            )
            agent.update_history(round_summary)

        self.game_stats['total_interactions'] += 1
        if is_someone_volunteering:
            self.game_stats['success_count'] += 1
            self.success_record.append(1)
        else:
            self.success_record.append(0)

        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "num_volunteers": num_volunteers,
            "num_standby": num_standby,
            "agent_choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "scenario": scenario,
            "phase2_triggered": phase2_triggered,
            "punish_willing": {self._get_agent_by_id(aid).name: punish_willing.get(aid) for aid in volunteer_agent_ids} if volunteer_agent_ids else {},
            "punish_explanations": {self._get_agent_by_id(aid).name: punish_explanations.get(aid, "") for aid in volunteer_agent_ids} if volunteer_agent_ids else {},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
            "is_someone_volunteering": is_someone_volunteering,
            "params": {
                "PAYOFF_V_BASE": self.payoff_v_base,
                "PAYOFF_S_BASE": self.payoff_s_base,
                "K_PUNISH": self.k_punish,
                "P_FREE_RIDER": self.p_free_rider,
            },
            "timestamp": datetime.now().isoformat()
        }

        self.game_logs.append(interaction_log)
        return interaction_log

    def get_game_summary(self) -> dict:
        success_rate = (
            self.game_stats['success_count'] / self.game_stats['total_interactions'] * 100
            if self.game_stats['total_interactions'] > 0 else 0
        )
        return {
            "total_interactions": self.game_stats['total_interactions'],
            "success_count": self.game_stats['success_count'],
            "success_rate": success_rate,
            "punishment_triggered_count": self.game_stats['punishment_triggered_count'],
            "punishment_success_count": self.game_stats['punishment_success_count'],
            "choice_frequency": dict(self.choice_frequency),
            "interaction_logs": self.game_logs
        }


async def main():
    EXPERIMENT_CONFIG = {
        "num_agents": 24,
        "payoff_v_base": PAYOFF_V_BASE,
        "payoff_s_base": PAYOFF_S_BASE,
        "k_punish": K_PUNISH,
        "p_free_rider": P_FREE_RIDER,
        "total_rounds": 30,
        "total_experiments": 1,  # 独立重复实验次数；仅运行 1 次完整实验
        "memory_size": 100,
        "random_seed": RANDOM_SEED
    }

    print("=== Volunteer's Dilemma Group + Punishment ===")
    print(f"Agents: {EXPERIMENT_CONFIG['num_agents']}")
    print(f"V base: {EXPERIMENT_CONFIG['payoff_v_base']}, S base: {EXPERIMENT_CONFIG['payoff_s_base']}")
    print(f"K: {EXPERIMENT_CONFIG['k_punish']}, P: {EXPERIMENT_CONFIG['p_free_rider']}")
    print(f"Rounds: {EXPERIMENT_CONFIG['total_rounds']}, Experiments: {EXPERIMENT_CONFIG['total_experiments']}")
    print("=" * 60)

    print("Initializing LLM...")
    from agentsociety2.agent.base import AgentLLM

    try:
        llm_agent = LLMAgent()
        agent_llm = AgentLLM(router=llm_agent, model_name=llm_agent.model)
        print(f"LLM Model: {agent_llm.model_name}")
    except Exception as e:
        logging.error(f"LLM initialization failed: {e}")
        print(f"[ERROR] LLM initialization failed: {e}")
        return

    base_result_dir = "result_volunteer_dilemma_group_punishment"
    experiment_time = input("Please enter experiment folder name (e.g., 'VD_punish_test1'): ").strip()
    if not experiment_time:
        experiment_time = datetime.now().strftime("%m%d_%H%M%S_VD_punish")
    experiment_result_dir = os.path.join(base_result_dir, experiment_time)
    os.makedirs(experiment_result_dir, exist_ok=True)

    data_dir = os.path.join(experiment_result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    config_path = os.path.join(experiment_result_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(EXPERIMENT_CONFIG, f, ensure_ascii=False, indent=2)

    total_start_time = datetime.now()
    all_experiment_summaries = []

    for experiment_num in range(1, EXPERIMENT_CONFIG['total_experiments'] + 1):
        print(f"\n=== Experiment {experiment_num}/{EXPERIMENT_CONFIG['total_experiments']} ===")

        exp_dir = os.path.join(experiment_result_dir, f"experiment_{experiment_num}")
        os.makedirs(exp_dir, exist_ok=True)

        print("Creating agents...")
        agents = []
        for i in range(EXPERIMENT_CONFIG['num_agents']):
            agent_name = f"Agent_{i+1}"
            agent_profile = (
                f"You are {agent_name}, a participant in a Volunteer's Dilemma game with an optional second-order punishment stage."
            )
            agent = VolunteerDilemmaPunishmentAgent(id=i, name=agent_name, profile=agent_profile)
            await agent.init(agent_llm)
            agents.append(agent)

        print("Creating environment...")
        env = VolunteerDilemmaPunishmentEnvironment(
            num_agents=EXPERIMENT_CONFIG['num_agents'],
            total_interactions=EXPERIMENT_CONFIG['total_rounds'],
            memory_size=EXPERIMENT_CONFIG['memory_size'],
            interaction_schedule=None,
            payoff_v_base=EXPERIMENT_CONFIG['payoff_v_base'],
            payoff_s_base=EXPERIMENT_CONFIG['payoff_s_base'],
            k_punish=EXPERIMENT_CONFIG['k_punish'],
            p_free_rider=EXPERIMENT_CONFIG['p_free_rider'],
        )
        env.set_agents(agents)

        print("Starting experiment...")
        start_time = datetime.now()
        all_interaction_results = []

        for interaction_num in range(1, EXPERIMENT_CONFIG['total_rounds'] + 1):
            print(f"\rRunning round {interaction_num}/{EXPERIMENT_CONFIG['total_rounds']}...", end="", flush=True)
            try:
                interaction_result = await env.run_interaction(interaction_num)
                if interaction_result:
                    all_interaction_results.append(interaction_result)
                    logging.info(
                        f"Experiment {experiment_num}: Completed round {interaction_num}/{EXPERIMENT_CONFIG['total_rounds']}"
                    )
            except Exception as e:
                logging.error(f"Error in experiment {experiment_num}, interaction {interaction_num}: {e}")
                print(f"\n[ERROR] Interaction {interaction_num} failed: {e}")

        print("\nExperiment completed!")
        print("Generating results...")

        end_time = datetime.now()
        experiment_duration = end_time - start_time
        game_summary = env.get_game_summary()

        agent_states = [agent.get_state_summary() for agent in agents]

        game_logs_path = os.path.join(data_dir, f"game_logs{experiment_num}.json")
        with open(game_logs_path, 'w', encoding='utf-8') as f:
            json.dump(env.game_logs, f, ensure_ascii=False, indent=2)

        stats_path = os.path.join(data_dir, f"game_stats{experiment_num}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(game_summary, f, ensure_ascii=False, indent=2)

        agent_states_path = os.path.join(data_dir, f"agent_states{experiment_num}.json")
        with open(agent_states_path, 'w', encoding='utf-8') as f:
            json.dump(agent_states, f, ensure_ascii=False, indent=2)

        interactions_path = os.path.join(data_dir, f"interaction_results{experiment_num}.json")
        with open(interactions_path, 'w', encoding='utf-8') as f:
            json.dump(all_interaction_results, f, ensure_ascii=False, indent=2)

        time_info = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "total_seconds": experiment_duration.total_seconds()
        }
        time_path = os.path.join(data_dir, f"experiment_time{experiment_num}.json")
        with open(time_path, 'w', encoding='utf-8') as f:
            json.dump(time_info, f, ensure_ascii=False, indent=2)

        exp_summary = {
            "experiment_number": experiment_num,
            "total_rounds": game_summary['total_interactions'],
            "success_count": game_summary['success_count'],
            "success_rate": game_summary['success_rate'],
            "punishment_triggered_count": game_summary['punishment_triggered_count'],
            "punishment_success_count": game_summary['punishment_success_count'],
            "choice_frequency": dict(game_summary['choice_frequency']),
            "experiment_time": experiment_duration.total_seconds(),
            "result_directory": exp_dir
        }
        all_experiment_summaries.append(exp_summary)

        print("\n=== Experiment Summary ===")
        print(f"Total Interactions: {game_summary['total_interactions']}")
        print(f"Rounds with ≥1 volunteer: {game_summary['success_count']}")
        print(f"Phase 2 offered (V and S both present): {game_summary['punishment_triggered_count']}")
        print(f"Punishment activated (≥1 V paid): {game_summary['punishment_success_count']}")
        print(f"Success Rate: {game_summary['success_rate']:.2f}%")
        print(f"Choice Frequency: {dict(game_summary['choice_frequency'])}")
        print(f"Experiment Time: {experiment_duration.total_seconds():.2f} seconds")
        print(f"Game logs: {game_logs_path}")

    total_end_time = datetime.now()
    overall_total_time = total_end_time - total_start_time

    overall_stats = {
        "total_experiments": EXPERIMENT_CONFIG['total_experiments'],
        "total_rounds": sum(s['total_rounds'] for s in all_experiment_summaries),
        "total_success_count": sum(s['success_count'] for s in all_experiment_summaries),
        "average_success_rate": sum(s['success_rate'] for s in all_experiment_summaries) / len(all_experiment_summaries),
        "total_experiment_time": overall_total_time.total_seconds(),
        "individual_experiment_times": [s['experiment_time'] for s in all_experiment_summaries]
    }

    overall_summary_path = os.path.join(experiment_result_dir, "overall_summary.json")
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)

    all_summaries_path = os.path.join(experiment_result_dir, "all_experiment_summaries.json")
    with open(all_summaries_path, 'w', encoding='utf-8') as f:
        json.dump(all_experiment_summaries, f, ensure_ascii=False, indent=2)

    print("\n=== Overall Summary ===")
    print(f"Results: {experiment_result_dir}")
    print(f"Overall summary: {overall_summary_path}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
