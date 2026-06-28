#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Volunteer's Dilemma Group Experiment with Reputation (LLM prompt only)

Based on env_main_volunteers_dilemma_group.py:
- Rules and payoffs unchanged
- Global per-agent reputation (does not change payoffs):
  Reputation = (volunteer_count + 1) / (participation_rounds + 2)
  volunteer_count: times the agent chose "Volunteer" in completed rounds
  participation_rounds: rounds completed (each agent participates every round)
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

os.makedirs("result_volunteer_dilemma_group_reputation", exist_ok=True)

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


def compute_reputation(volunteer_count: int, participation_rounds: int) -> float:
    """Reputation = (volunteer_count + 1) / (participation_rounds + 2); first round ~ 0.5."""
    return (volunteer_count + 1) / (participation_rounds + 2)


class VolunteerDilemmaReputationAgent(AgentBase):
    """
    Volunteer's dilemma agent with other players' volunteer reputation in the prompt.
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
        self._partner_map = defaultdict(list)
        self.history = []

        self.decision_options = ["Volunteer", "Stand by"]
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
        summary = {
            'id': self._id,
            'name': self._name,
            'my_history': self._my_history,
            'partner_history': self._partner_history,
            'outcome_history': self._outcome_history,
            'num_interactions': len(self._my_history),
            'interaction_partners': dict(self._partner_map)
        }
        if self._env and hasattr(self._env, "get_reputation"):
            summary["my_reputation"] = self._env.get_reputation(self._id)
        return summary

    def _build_history_string(self, all_agent_names: list) -> str:
        if not self.history:
            return "No previous rounds have been played."

        history_lines = ["History of previous rounds:"]
        for round_summary in self.history:
            r = round_summary["round"]
            num_volunteers = round_summary["num_volunteers"]
            has_volunteer = num_volunteers > 0
            history_lines.append(f"Round {r}:")
            history_lines.append(f"  Number of volunteers: {num_volunteers}")
            history_lines.append(f"  Volunteer status: {'Success' if has_volunteer else 'No volunteer'}")

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

    def _get_volunteer_rules_text(self, benefit_b, cost_c):
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

Your goal is to maximize your own accumulated points.
"""

    async def choose_action(
        self,
        benefit_b: int,
        cost_c: int,
        partner_id: int = None,
        memory_size: int = 5,
        others_reputation_block: str = ""
    ):
        """Choose Volunteer or Stand by; optional block lists other players' reputations (pre-round snapshot)."""
        rules = self._get_volunteer_rules_text(benefit_b, cost_c)
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)

        current_round_true = len(self.history) + 1
        current_score = sum(self._outcome_history) if hasattr(self, '_outcome_history') else 0

        history_intro = ""

        new_query = (
            f"It is now round {current_round_true} of your interactions.\n"
            f"Your current total score is {current_score}.\n"
            f"Please decide whether to 'Volunteer' or 'Stand by'.\n"
            "\n"
            "***CRITICAL DECISION INSTRUCTION***:\n"
            "Please make your decision strictly following this format:\n"
            "Decision: [Volunteer/Stand by]\n"
            "Explanation: [Brief 1-2 sentence explanation]"
        )

        rep_section = ""
        if others_reputation_block:
            rep_section = others_reputation_block.strip() + "\n\n"

        if len(self.history) == 0:
            final_prompt = "\n".join([rules, rep_section, new_query]).strip()
        else:
            final_prompt = "\n".join([rules, rep_section, history_intro, histories_text, new_query]).strip()

        system_message = self._profile

        try:
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=final_prompt
            )

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


class VolunteerDilemmaReputationEnvironment:
    """
    Same as VolunteerDilemmaEnvironment, plus global volunteer reputation per agent_id.
    """

    def __init__(
        self,
        num_agents: int,
        benefit_b: int,
        cost_c: int,
        total_interactions: int,
        memory_size: int = 100,
        interaction_schedule: list = None
    ):
        self.num_agents = num_agents
        self.benefit_b = benefit_b
        self.cost_c = cost_c
        self.total_interactions = total_interactions
        self.memory_size = memory_size
        self.interaction_schedule = interaction_schedule

        self.agents = []
        self.interaction_number = 0
        self.initial_time = None

        self._volunteer_count = defaultdict(int)
        self._participation_count = defaultdict(int)

        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'benefit_b': benefit_b,
            'cost_c': cost_c,
            'memory_size': memory_size,
            'interactions_per_run': 1,
            'reputation_formula': '(volunteer_count + 1) / (participation_rounds + 2)',
        }

        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)
        self.game_logs = []

    def set_agents(self, agents: list):
        self.agents = agents
        for agent in agents:
            agent.set_environment(self)

    def get_reputation(self, agent_id: int) -> float:
        return compute_reputation(
            self._volunteer_count[agent_id],
            self._participation_count[agent_id]
        )

    def build_others_reputation_block(self, for_agent) -> str:
        """English-only: other players' volunteer reputation before this round's decisions."""
        lines = [
            "Other players' Volunteer Reputation (0-1; higher means they volunteered more often in past rounds).",
            "Reputation formula: (volunteer_count + 1) / (participation_rounds + 2). "
            "Values use only completed rounds before this round.",
        ]
        for other in sorted(self.agents, key=lambda a: a.name):
            if other._id == for_agent._id:
                continue
            r = compute_reputation(
                self._volunteer_count[other._id],
                self._participation_count[other._id]
            )
            lines.append(f"  {other.name}: {r:.3f}")
        return "\n".join(lines)

    def _snapshot_all_reputations(self) -> dict:
        return {a.name: self.get_reputation(a._id) for a in self.agents}

    async def run_interaction(self, interaction_num: int) -> dict:
        if self.initial_time is None:
            self.initial_time = datetime.now()

        self.interaction_number = interaction_num

        pre_round_reputation = self._snapshot_all_reputations()

        choice_tasks = []
        for agent in self.agents:
            rep_block = self.build_others_reputation_block(agent)
            task = agent.choose_action(
                benefit_b=self.benefit_b,
                cost_c=self.cost_c,
                partner_id=None,
                memory_size=self.memory_size,
                others_reputation_block=rep_block
            )
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

        num_volunteers = sum(1 for choice in agent_choices.values() if choice == "Volunteer")
        is_someone_volunteering = num_volunteers > 0

        payoffs = {}
        for agent_id, choice in agent_choices.items():
            if is_someone_volunteering:
                if choice == "Volunteer":
                    payoffs[agent_id] = self.benefit_b - self.cost_c
                else:
                    payoffs[agent_id] = self.benefit_b
            else:
                payoffs[agent_id] = 0

        round_summary = {
            "round": interaction_num,
            "choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "num_volunteers": num_volunteers,
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
            "pre_round_reputation": pre_round_reputation,
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

        for agent in self.agents:
            aid = agent._id
            self._participation_count[aid] += 1
            if agent_choices[aid] == "Volunteer":
                self._volunteer_count[aid] += 1

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
            "agent_choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
            "is_someone_volunteering": is_someone_volunteering,
            "pre_round_reputation": pre_round_reputation,
            "timestamp": datetime.now().isoformat()
        }

        self.game_logs.append(interaction_log)

        return interaction_log

    def get_game_summary(self) -> dict:
        success_rate = (
            self.game_stats['success_count'] / self.game_stats['total_interactions'] * 100
        ) if self.game_stats['total_interactions'] > 0 else 0

        return {
            "total_interactions": self.game_stats['total_interactions'],
            "success_count": self.game_stats['success_count'],
            "success_rate": success_rate,
            "choice_frequency": dict(self.choice_frequency),
            "interaction_logs": self.game_logs,
            "reputation_formula": "(volunteer_count + 1) / (participation_rounds + 2)",
        }


async def main():
    EXPERIMENT_CONFIG = {
        "num_agents": 24,
        "benefit_b": 100,
        "cost_c": 40,
        "total_rounds": 30,
        "total_experiments": 2,
        "memory_size": 100,
        "random_seed": RANDOM_SEED
    }

    print(f"=== Volunteer's Dilemma Group Experiment (Reputation) ===")
    print(f"Number of Agents: {EXPERIMENT_CONFIG['num_agents']}")
    print(f"Benefit (B): {EXPERIMENT_CONFIG['benefit_b']}")
    print(f"Cost (C): {EXPERIMENT_CONFIG['cost_c']}")
    print(f"Total Rounds per Experiment: {EXPERIMENT_CONFIG['total_rounds']}")
    print(f"Reputation: (volunteer_count + 1) / (participation_rounds + 2)")
    print("=" * 60)

    print("Initializing LLM...")
    try:
        llm_agent = LLMAgent()
        agent_llm = AgentLLM(router=llm_agent, model_name=llm_agent.model)
        print(f"LLM Model: {agent_llm.model_name}")
    except Exception as e:
        logging.error(f"LLM initialization failed: {e}")
        print(f"[ERROR] LLM initialization failed: {e}")
        return

    base_result_dir = "result_volunteer_dilemma_group_reputation"
    experiment_time = input("Please enter experiment folder name (e.g., 'VD_rep_test1'): ").strip()
    if not experiment_time:
        experiment_time = datetime.now().strftime("%m%d_%H%M%S_VD_rep")
    experiment_result_dir = os.path.join(base_result_dir, experiment_time)
    os.makedirs(experiment_result_dir, exist_ok=True)

    data_dir = os.path.join(experiment_result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    config_path = os.path.join(experiment_result_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(EXPERIMENT_CONFIG, f, ensure_ascii=False, indent=2)

    rep_config_path = os.path.join(experiment_result_dir, "reputation_config_resolved.json")
    with open(rep_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "reputation_formula": "(volunteer_count + 1) / (participation_rounds + 2)",
            "volunteer_count": "times chose Volunteer in completed rounds",
            "participation_rounds": "rounds completed before current update",
        }, f, ensure_ascii=False, indent=2)

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
                f"You are {agent_name}, a participant in a Volunteer's Dilemma game. "
                "You may see other players' Volunteer Reputation summarizing how often they volunteered in past rounds; "
                "this information does not change payoff rules."
            )
            agent = VolunteerDilemmaReputationAgent(id=i, name=agent_name, profile=agent_profile)
            await agent.init(agent_llm)
            agents.append(agent)

        print("Creating environment...")
        env = VolunteerDilemmaReputationEnvironment(
            num_agents=EXPERIMENT_CONFIG['num_agents'],
            benefit_b=EXPERIMENT_CONFIG['benefit_b'],
            cost_c=EXPERIMENT_CONFIG['cost_c'],
            total_interactions=EXPERIMENT_CONFIG['total_rounds'],
            memory_size=EXPERIMENT_CONFIG['memory_size'],
            interaction_schedule=None
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

        agent_states = []
        for agent in agents:
            agent_states.append(agent.get_state_summary())

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
            "choice_frequency": dict(game_summary['choice_frequency']),
            "experiment_time": experiment_duration.total_seconds(),
            "result_directory": exp_dir
        }
        all_experiment_summaries.append(exp_summary)

        print("\n=== Experiment Summary ===")
        print(f"Total Interactions: {game_summary['total_interactions']}")
        print(f"Successful Interactions (someone volunteered): {game_summary['success_count']}")
        print(f"Success Rate: {game_summary['success_rate']:.2f}%")
        print(f"Choice Frequency: {dict(game_summary['choice_frequency'])}")
        print(f"Experiment Time: {experiment_duration.total_seconds():.2f} seconds")
        print(f"Game logs saved to: {game_logs_path}")

    total_end_time = datetime.now()
    overall_total_time = total_end_time - total_start_time

    overall_stats = {
        "total_experiments": EXPERIMENT_CONFIG['total_experiments'],
        "total_rounds": sum(summary['total_rounds'] for summary in all_experiment_summaries),
        "total_success_count": sum(summary['success_count'] for summary in all_experiment_summaries),
        "average_success_rate": sum(summary['success_rate'] for summary in all_experiment_summaries) / len(all_experiment_summaries),
        "total_experiment_time": overall_total_time.total_seconds(),
        "individual_experiment_times": [summary['experiment_time'] for summary in all_experiment_summaries]
    }

    overall_summary_path = os.path.join(experiment_result_dir, "overall_summary.json")
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)

    all_summaries_path = os.path.join(experiment_result_dir, "all_experiment_summaries.json")
    with open(all_summaries_path, 'w', encoding='utf-8') as f:
        json.dump(all_experiment_summaries, f, ensure_ascii=False, indent=2)

    print("\n=== Overall Experiment Summary ===")
    print(f"Total Experiments: {overall_stats['total_experiments']}")
    print(f"Total Rounds Across All Experiments: {overall_stats['total_rounds']}")
    print(f"Total Successful Rounds: {overall_stats['total_success_count']}")
    print(f"Average Success Rate: {overall_stats['average_success_rate']:.2f}%")
    print(f"Total Experiment Time: {overall_stats['total_experiment_time']:.2f} seconds")
    print(f"Results saved to: {experiment_result_dir}")
    print(f"- Overall Summary: {overall_summary_path}")
    print("=" * 60)
    print("Experiment completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
