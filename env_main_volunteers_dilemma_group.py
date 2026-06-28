#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Volunteer's Dilemma Group Experiment - agentsociety V2 Platform Implementation
Based on env_main_志愿者博弈baseline.py and spontaneous emergence architecture with 24 agents
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

# 加载环境变量
load_dotenv()

# Ensure results directory exists
os.makedirs("result_volunteer_dilemma_group", exist_ok=True)

# 配置日志记录
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
# 完整实验重复运行次数（每次从新建智能体与环境开始）
NUM_EXPERIMENT_RUNS = 3
random.seed(RANDOM_SEED)

class VolunteerDilemmaAgent(AgentBase):
    """
    Agent for Volunteer's Dilemma Game that manages its own state,
    compatible with 24-agent group architecture.
    """
    
    def __init__(self, id: int, name: str, profile: str = ""):
        """ Initialize VolunteerDilemmaAgent """
        super().__init__(id, profile)
        self._name = name
        self._llm = None
        self._env = None
        self._previous_choices = []  # Track previous choices for stability
        
        # 自主状态管理：代理独立维护交互历史和状态
        self._my_history = []        # Own choices history
        self._partner_history = []   # Partners' choices history
        self._outcome_history = []   # Outcomes history (payoffs)
        self._partner_map = defaultdict(list)  # Track interactions per partner
        self.history = []  # Store complete round summaries for history display
        
        self.decision_options = ["Volunteer", "Stand by"]  # Possible choices
        self._profile = profile  # Store profile for system message use
    
    def set_environment(self, env):
        """ Set the environment for the agent """
        self._env = env
    
    # 自主状态管理：添加更新内部状态的方法
    def update_state(self, my_choice: str, partner_choice: str, partner_id: int, outcome: int):
        """ Update agent's internal state """
        # Add choice history and outcome
        self._my_history.append(my_choice)
        self._partner_history.append(partner_choice)
        self._outcome_history.append(outcome)
        self._partner_map[partner_id].append((my_choice, partner_choice, outcome))
        
        logging.debug(f"[{self.name}] Updated state: my_choice={my_choice}, partner_choice={partner_choice}, partner_id={partner_id}, outcome={outcome}")
    
    def update_history(self, round_summary: dict):
        """Update agent's history with complete round summary"""
        self.history.append(round_summary)
    
    def get_state_summary(self) -> dict:
        """ Get agent's current state summary """
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
        """Build history string with only volunteer information"""
        if not self.history:
            return "No previous rounds have been played."
        
        history_lines = []
        history_lines.append("History of previous rounds:")
        
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
        # Standard generic ask, not used in game logic
        return f"[Agent {self.name}] Received message: {message}"
    
    async def step(self, tick: int, t: datetime) -> str:
        # step method is not used in game logic as it's managed by environment
        return ""
    
    def _get_volunteer_rules_text(self, benefit_b, cost_c):
        """ Get the volunteer dilemma rules text """
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
    
    async def choose_action(self, benefit_b: int, cost_c: int, partner_id: int = None, memory_size: int = 5):
        """ Choose action using LLM, considering history in group mode """
        # 1. Get Rules
        rules = self._get_volunteer_rules_text(benefit_b, cost_c)
        
        # 2. Build History using baseline-style format
        agent_names = [agent.name for agent in self._env.agents] if hasattr(self, '_env') and self._env else []
        histories_text = self._build_history_string(agent_names)
        
        # Current round information
        current_round_true = len(self.history) + 1
        current_score = sum(self._outcome_history) if hasattr(self, '_outcome_history') else 0
        
        history_intro = ""
        
        # 3. Construct the Query
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
        
        # 4. Assemble Final Prompt
        if len(self.history) == 0:
            final_prompt = "\n".join([rules, new_query])
        else:
            final_prompt = "\n".join([rules, history_intro, histories_text, new_query])

        system_message = self._profile  # Use agent profile as system message

        try:
            content = await self._call_llm_with_retry(
                system_message=system_message,
                user_prompt=final_prompt
            )
            
            # 5. Parse Response
            choice = self.decision_options[1]  # Default to 'Stand by'
            explanation = "LLM call or parsing failed"
            
            # Use regex to find decision
            match_volunteer = re.search(r'Volunteer', content, re.IGNORECASE)
            match_standby = re.search(r'Stand\s*by', content, re.IGNORECASE)
            
            if match_volunteer and (not match_standby or match_volunteer.start() < match_standby.start()):
                choice = self.decision_options[0]  # 'Volunteer'
                # Extract explanation from response
                if match_volunteer.end() < len(content):
                    explanation = content[match_volunteer.end():].strip()
            elif match_standby:
                choice = self.decision_options[1]  # 'Stand by'
                # Extract explanation from response
                if match_standby.end() < len(content):
                    explanation = content[match_standby.end():].strip()
            else:
                raise ValueError(f"Failed to parse valid choice, content:\n{content[:200]}")
            
            # Fallback: find any valid option in text if decision format not found
            if choice == self.decision_options[1]:  # Still default
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
        
        # Add current choice to history for stability
        self._previous_choices.append(choice)
        
        return choice, explanation
    
    def _get_stable_choice(self) -> str:
        """
        Enhanced choice stability logic:
        1. Prefer most frequent previous choice
        2. If no consistent choice, use environment's most popular option
        3. Finally use random choice as last resort
        """
        # 1. Prefer most frequent previous choice
        if self._previous_choices:
            choice_counter = Counter(self._previous_choices)
            most_common_choice, count = choice_counter.most_common(1)[0]
            if count > len(self._previous_choices) / 2:
                logging.info(f"[{self.name}] Using most frequent previous choice: {most_common_choice}")
                return most_common_choice
        
        # 2. If no consistent choice, use environment's most popular option
        if hasattr(self, '_env') and self._env and hasattr(self._env, 'choice_frequency'):
            if self._env.choice_frequency:
                most_popular_choice, count = self._env.choice_frequency.most_common(1)[0]
                if most_popular_choice in self.decision_options:
                    logging.info(f"[{self.name}] Using most popular environment choice: {most_popular_choice}")
                    return most_popular_choice
        
        # 3. Finally use random choice as last resort
        logging.info(f"[{self.name}] Using random choice as fallback")
        return random.choice(self.decision_options)

    async def _call_llm_with_retry(self, system_message: str, user_prompt: str, max_retries: int = 5, retry_delay: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                # Call LLM using asyncio.to_thread to handle sync call
                generated_text = await asyncio.to_thread(
                    self._llm.router.get_llm_response,
                    system_message,
                    user_prompt
                )
                if generated_text:
                    return generated_text
                else:
                    raise ValueError("LLM returned empty string")
            except Exception as e:
                logging.error(f"[{self.name}] LLM Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise

class VolunteerDilemmaEnvironment:
    """
    Environment for Volunteer's Dilemma Group Experiment with 24 agents
    where all agents participate simultaneously in each round.
    """
    
    def __init__(self, num_agents: int, benefit_b: int, cost_c: int, 
                 total_interactions: int, memory_size: int = 100, interaction_schedule: list = None):
        
        self.num_agents = num_agents
        self.benefit_b = benefit_b  # Benefit for everyone if someone volunteers
        self.cost_c = cost_c  # Cost for a volunteer
        self.total_interactions = total_interactions
        self.memory_size = memory_size  # Memory size for history consideration
        self.interaction_schedule = interaction_schedule # Not used in group mode
        
        self.agents = [] # List of Agent objects
        self.interaction_number = 0 # Track sequential interactions
        self.initial_time = None
        
        self._config = {
            'max_tick': total_interactions,
            'num_agents': num_agents,
            'benefit_b': benefit_b,
            'cost_c': cost_c,
            'memory_size': memory_size,
            'interactions_per_run': 1
        }
        
        self.game_stats = {'success_count': 0, 'total_interactions': 0}
        self.success_record = []
        self.choice_frequency = defaultdict(int)  # Track choice frequency
        self.game_logs = []
    
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
        Executes a single interaction where all agents participate simultaneously.
        """
        if self.initial_time is None:
            self.initial_time = datetime.now()
        
        self.interaction_number = interaction_num
        
        # 1. All agents make decisions concurrently
        choice_tasks = []
        for agent in self.agents:
            task = agent.choose_action(
                benefit_b=self.benefit_b,
                cost_c=self.cost_c,
                partner_id=None,  # No specific partner in group mode
                memory_size=self.memory_size
            )
            choice_tasks.append((agent, task))
        
        agent_choices = {}
        explanations = {}
        for agent, task in choice_tasks:
            try:
                choice, explanation = await task
                agent_choices[agent._id] = choice
                explanations[agent._id] = explanation
                # Track choice frequency
                self.choice_frequency[choice] += 1
            except Exception as e:
                logging.error(f"Error in choice for agent {agent.name}: {e}")
                agent_choices[agent._id] = "Stand by"  # Default to 'Stand by'
                explanations[agent._id] = f"Error: {str(e)}"
                self.choice_frequency["Stand by"] += 1
        
        # 2. Calculate number of volunteers
        num_volunteers = sum(1 for choice in agent_choices.values() if choice == "Volunteer")
        is_someone_volunteering = num_volunteers > 0
        
        # 3. Calculate payoffs for all agents
        payoffs = {}
        for agent_id, choice in agent_choices.items():
            if is_someone_volunteering:
                if choice == "Volunteer":
                    payoffs[agent_id] = self.benefit_b - self.cost_c
                else:  # "Stand by"
                    payoffs[agent_id] = self.benefit_b
            else:
                payoffs[agent_id] = 0
        
        # 4. Update agent states
        # Build round summary for history
        round_summary = {
            "round": interaction_num,
            "choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "num_volunteers": num_volunteers,
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents}
        }
        
        for agent in self.agents:
            agent_choice = agent_choices[agent._id]
            # In group mode, partner choice is None since all agents act together
            agent.update_state(
                my_choice=agent_choice,
                partner_choice=None,  # No specific partner
                partner_id=None,  # No specific partner
                outcome=payoffs[agent._id]
            )
            # Update agent's history with complete round summary
            agent.update_history(round_summary)
        
        # 5. Record success (if at least one volunteer)
        self.game_stats['total_interactions'] += 1
        if is_someone_volunteering:
            self.game_stats['success_count'] += 1
            self.success_record.append(1)
        else:
            self.success_record.append(0)
        
        # 6. Log the interaction
        interaction_log = {
            "interaction": interaction_num,
            "num_agents": len(self.agents),
            "num_volunteers": num_volunteers,
            "agent_choices": {agent.name: agent_choices[agent._id] for agent in self.agents},
            "explanations": {agent.name: explanations[agent._id] for agent in self.agents},
            "payoffs": {agent.name: payoffs[agent._id] for agent in self.agents},
            "is_someone_volunteering": is_someone_volunteering,
            "timestamp": datetime.now().isoformat()
        }
        
        self.game_logs.append(interaction_log)
        
        return interaction_log
    
    def get_game_summary(self) -> dict:
        """Get summary of the game statistics"""
        success_rate = (self.game_stats['success_count'] / self.game_stats['total_interactions'] * 100) \
                      if self.game_stats['total_interactions'] > 0 else 0
        
        return {
            "total_interactions": self.game_stats['total_interactions'],
            "success_count": self.game_stats['success_count'],
            "success_rate": success_rate,
            "choice_frequency": dict(self.choice_frequency),
            "interaction_logs": self.game_logs
        }

async def main():
    """Main function to run the 24-agent volunteer dilemma group experiment"""
    # ------- Experiment Configuration -------
    EXPERIMENT_CONFIG = {
        "num_agents": 24,  # Number of agents in the group
        "benefit_b": 100,  # Benefit for everyone if someone volunteers
        "cost_c": 40,  # Cost for a volunteer
        "total_rounds": 30, # Total group interactions (rounds) per experiment
        "total_experiments": NUM_EXPERIMENT_RUNS,  # 完整实验运行次数
        "memory_size": 100,  # Memory size for history consideration (set to 100 to remember all rounds)
        "random_seed": RANDOM_SEED
    }
    
    print(f"=== Volunteer's Dilemma Group Experiment ===")
    print(f"Number of Agents: {EXPERIMENT_CONFIG['num_agents']}")
    print(f"Benefit (B): {EXPERIMENT_CONFIG['benefit_b']}")
    print(f"Cost (C): {EXPERIMENT_CONFIG['cost_c']}")
    print(f"Total Rounds per Experiment: {EXPERIMENT_CONFIG['total_rounds']}")
    print(f"Total Experiments: {EXPERIMENT_CONFIG['total_experiments']}")
    print(f"Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    print("=" * 60)
    
    # ------- Initialize LLM -------
    print("Initializing LLM...")
    from llm_cooperation_lab.agent.base import AgentLLM
    
    try:
        llm_agent = LLMAgent()
        agent_llm = AgentLLM(router=llm_agent, model_name=llm_agent.model)
        print(f"LLM Model: {agent_llm.model_name}")
    except Exception as e:
        logging.error(f"LLM initialization failed: {e}")
        print(f"[ERROR] LLM initialization failed: {e}")
        return
    
    # ------- Create Results Directory -------
    base_result_dir = "result_volunteer_dilemma_group"
    experiment_time = input("Please enter experiment folder name (e.g., 'VD_group_test1'): ").strip()
    if not experiment_time:
        experiment_time = datetime.now().strftime("%m%d_%H%M%S_VD_group")
    experiment_result_dir = os.path.join(base_result_dir, experiment_time)
    os.makedirs(experiment_result_dir, exist_ok=True)

    # Create data directory for all logs
    data_dir = os.path.join(experiment_result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save experiment configuration
    config_path = os.path.join(experiment_result_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(EXPERIMENT_CONFIG, f, ensure_ascii=False, indent=2)

    # ------- Run Multiple Experiments -------
    total_start_time = datetime.now()
    all_experiment_summaries = []
    
    for experiment_num in range(1, EXPERIMENT_CONFIG['total_experiments'] + 1):
        print(f"\n=== Experiment {experiment_num}/{EXPERIMENT_CONFIG['total_experiments']} ===")
        
        # Create experiment-specific directory if needed
        exp_dir = os.path.join(experiment_result_dir, f"experiment_{experiment_num}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # ------- Create Agents -------
        print("Creating agents...")
        agents = []
        for i in range(EXPERIMENT_CONFIG['num_agents']):
            agent_name = f"Agent_{i+1}"
            agent_profile = f"You are {agent_name}, a participant in a Volunteer's Dilemma game."
            agent = VolunteerDilemmaAgent(id=i, name=agent_name, profile=agent_profile)
            await agent.init(agent_llm)
            agents.append(agent)
        
        # ------- Create Environment -------
        print("Creating environment...")
        interaction_schedule = None  # No interaction schedule in group mode
        env = VolunteerDilemmaEnvironment(
            num_agents=EXPERIMENT_CONFIG['num_agents'],
            benefit_b=EXPERIMENT_CONFIG['benefit_b'],
            cost_c=EXPERIMENT_CONFIG['cost_c'],
            total_interactions=EXPERIMENT_CONFIG['total_rounds'],
            memory_size=EXPERIMENT_CONFIG['memory_size'],
            interaction_schedule=interaction_schedule
        )
        env.set_agents(agents)
        
        # ------- Run Experiment -------
        print("Starting experiment...")
        start_time = datetime.now()
        
        all_interaction_results = []
        
        for interaction_num in range(1, EXPERIMENT_CONFIG['total_rounds'] + 1):
            print(f"\rRunning round {interaction_num}/{EXPERIMENT_CONFIG['total_rounds']}...", end="", flush=True)
            
            try:
                interaction_result = await env.run_interaction(interaction_num)
                if interaction_result:
                    all_interaction_results.append(interaction_result)
                    
                    # Log every 1 round
                    logging.info(f"Experiment {experiment_num}: Completed round {interaction_num}/{EXPERIMENT_CONFIG['total_rounds']}")
                        
            except Exception as e:
                logging.error(f"Error in experiment {experiment_num}, interaction {interaction_num}: {e}")
                print(f"\n[ERROR] Interaction {interaction_num} failed: {e}")
        
        print("\nExperiment completed!")
        
        # ------- Generate Results for This Experiment -------
        print("Generating results...")
        
        # Calculate experiment time
        end_time = datetime.now()
        experiment_time = end_time - start_time
        
        # Get game summary
        game_summary = env.get_game_summary()
        
        # Collect agent states
        agent_states = []
        for agent in agents:
            agent_states.append(agent.get_state_summary())
        
        # Save game logs
        game_logs_path = os.path.join(data_dir, f"game_logs{experiment_num}.json")
        with open(game_logs_path, 'w', encoding='utf-8') as f:
            json.dump(env.game_logs, f, ensure_ascii=False, indent=2)
        
        # Save game statistics
        stats_path = os.path.join(data_dir, f"game_stats{experiment_num}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(game_summary, f, ensure_ascii=False, indent=2)
        
        # Save agent states
        agent_states_path = os.path.join(data_dir, f"agent_states{experiment_num}.json")
        with open(agent_states_path, 'w', encoding='utf-8') as f:
            json.dump(agent_states, f, ensure_ascii=False, indent=2)
        
        # Save interaction results
        interactions_path = os.path.join(data_dir, f"interaction_results{experiment_num}.json")
        with open(interactions_path, 'w', encoding='utf-8') as f:
            json.dump(all_interaction_results, f, ensure_ascii=False, indent=2)
        
        # Save experiment time information
        time_info = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "total_seconds": experiment_time.total_seconds()
        }
        time_path = os.path.join(data_dir, f"experiment_time{experiment_num}.json")
        with open(time_path, 'w', encoding='utf-8') as f:
            json.dump(time_info, f, ensure_ascii=False, indent=2)
        
        # Store experiment summary
        exp_summary = {
            "experiment_number": experiment_num,
            "total_rounds": game_summary['total_interactions'],
            "success_count": game_summary['success_count'],
            "success_rate": game_summary['success_rate'],
            "choice_frequency": dict(game_summary['choice_frequency']),
            "experiment_time": experiment_time.total_seconds(),
            "result_directory": exp_dir
        }
        all_experiment_summaries.append(exp_summary)
        
        # ------- Print Experiment Summary -------
        print("\n=== Experiment Summary ===")
        print(f"Total Interactions: {game_summary['total_interactions']}")
        print(f"Successful Interactions (someone volunteered): {game_summary['success_count']}")
        print(f"Success Rate: {game_summary['success_rate']:.2f}%")
        print(f"Choice Frequency: {dict(game_summary['choice_frequency'])}")
        print(f"Experiment Time: {experiment_time.total_seconds():.2f} seconds")
        print(f"Game logs saved to: {game_logs_path}")
    
    # ------- Calculate Overall Summary -------
    total_end_time = datetime.now()
    overall_total_time = total_end_time - total_start_time
    
    # Calculate overall statistics
    overall_stats = {
        "total_experiments": EXPERIMENT_CONFIG['total_experiments'],
        "total_rounds": sum(summary['total_rounds'] for summary in all_experiment_summaries),
        "total_success_count": sum(summary['success_count'] for summary in all_experiment_summaries),
        "average_success_rate": sum(summary['success_rate'] for summary in all_experiment_summaries) / len(all_experiment_summaries),
        "total_experiment_time": overall_total_time.total_seconds(),
        "individual_experiment_times": [summary['experiment_time'] for summary in all_experiment_summaries]
    }
    
    # Save overall summary
    overall_summary_path = os.path.join(experiment_result_dir, "overall_summary.json")
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)
    
    # Save all experiment summaries
    all_summaries_path = os.path.join(experiment_result_dir, "all_experiment_summaries.json")
    with open(all_summaries_path, 'w', encoding='utf-8') as f:
        json.dump(all_experiment_summaries, f, ensure_ascii=False, indent=2)
    
    # ------- Print Overall Summary -------
    print("\n=== Overall Experiment Summary ===")
    print(f"Total Experiments: {overall_stats['total_experiments']}")
    print(f"Total Rounds Across All Experiments: {overall_stats['total_rounds']}")
    print(f"Total Successful Rounds: {overall_stats['total_success_count']}")
    print(f"Average Success Rate: {overall_stats['average_success_rate']:.2f}%")
    print(f"Total Experiment Time: {overall_stats['total_experiment_time']:.2f} seconds")
    print(f"Results saved to: {experiment_result_dir}")
    print(f"- Overall Summary: {overall_summary_path}")
    print(f"- All Experiment Summaries: {all_summaries_path}")
    print("=" * 60)
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
