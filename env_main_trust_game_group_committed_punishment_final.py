#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trust Game with binding, pre-committed costly punishment.

This is a separate treatment built on
``env_main_trust_game_group_costly_punishment.py``. The original file is not
modified. All settings outside the decision timing are inherited:

1. The Trustor chooses an investment.
2. Before seeing any return, the Trustor commits to a punishment level for
   every possible integer return.
3. The complete binding schedule is shown to the Trustee.
4. The Trustee chooses a return.
5. The environment automatically applies the committed level corresponding
   to the actual return. The Trustor cannot revise it.
"""

import asyncio
import json
import logging
import re
import traceback
from collections import Counter
from datetime import datetime

import env_main_trust_game_group_costly_punishment as base


EXPERIMENT_SLUG = "trust_game_group_committed_punishment"
RESULTS_BASE_DIR = f"result_{EXPERIMENT_SLUG}"
NUM_GAMES = 3


class CommittedPunishmentAgent(base.TrustGamePopulationAgent):
    """Agent with investment, commitment, and informed-return decisions."""

    def _parse_amount(
        self,
        content: str,
        role: str,
        max_amount: int,
    ) -> int:
        """Parse exactly one explicit XML amount and ignore all other numbers."""
        matches = re.findall(
            r"<amount>\s*(\d+)\s*</amount>",
            content,
            re.IGNORECASE,
        )
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one <amount>integer</amount> tag, "
                f"found {len(matches)}"
            )

        amount = int(matches[0])
        if not 0 <= amount <= max_amount:
            raise ValueError(
                f"{role} amount {amount} must be between 0 and {max_amount}"
            )
        return amount

    @staticmethod
    def _extract_xml_explanation(content: str) -> str:
        """Extract only the explanation field paired with an XML decision."""
        match = re.search(
            r"<explanation>(.*?)</explanation>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip() or "No explanation provided"
        return "No explanation provided"

    def _get_historical_mode_decision(
        self,
        decision_label: str,
        max_amount: int,
    ) -> tuple[int, str] | None:
        """Return the most frequent prior valid decision and its explanation."""
        if decision_label == "Investment":
            required_role = "Trustor"
            amount_key = "sent_amount"
            valid_key = "investment_decision_valid"
        elif decision_label == "ReturnAfterCommitment":
            required_role = "Trustee"
            amount_key = "returned_amount"
            valid_key = "return_decision_valid"
        else:
            return None

        candidates = []
        for record in self._history:
            if record.get("role") != required_role:
                continue
            if record.get(valid_key) is not True:
                continue
            amount = record.get(amount_key)
            if not isinstance(amount, int) or not 0 <= amount <= max_amount:
                continue
            explanation = record.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                explanation = "No explanation provided"
            candidates.append((amount, explanation))

        if not candidates:
            return None

        counts = Counter(amount for amount, _ in candidates)
        highest_count = max(counts.values())
        modal_amounts = {
            amount
            for amount, count in counts.items()
            if count == highest_count
        }

        # Resolve a tied mode with the most recent valid occurrence and reuse
        # the explanation attached to that same occurrence.
        for amount, explanation in reversed(candidates):
            if amount in modal_amounts:
                return amount, explanation
        return None

    async def _request_xml_amount_decision(
        self,
        system_message: str,
        user_prompt: str,
        role: str,
        max_amount: int,
        decision_label: str,
    ) -> tuple[int, str, bool, str, str]:
        """Request a strict XML amount, retrying API or format failures."""
        prompt = user_prompt
        raw_response = ""
        last_error = None

        for format_attempt in range(3):
            try:
                raw_response = await self._call_llm_with_retry(
                    system_message=system_message,
                    user_prompt=prompt,
                )
                amount = self._parse_amount(
                    raw_response,
                    role,
                    max_amount,
                )
                explanation = self._extract_xml_explanation(raw_response)
                self._decision_count += 1
                return (
                    amount,
                    explanation,
                    True,
                    raw_response,
                    "current_response",
                )
            except Exception as error:
                last_error = error
                self.logger.warning(
                    f"[{self._name}] [{decision_label}] Decision attempt "
                    f"{format_attempt + 1}/3 failed: {error}"
                )
                if format_attempt < 2:
                    prompt += (
                        "\n\nYour previous response could not be parsed. "
                        "Return exactly one decision block with exactly one "
                        "<amount>integer</amount> tag. Do not use any other "
                        "<amount> tag. Numbers in your reasoning will be "
                        "ignored."
                    )

        self._error_count += 1
        historical_fallback = self._get_historical_mode_decision(
            decision_label=decision_label,
            max_amount=max_amount,
        )
        if historical_fallback is not None:
            amount, explanation = historical_fallback
            self.logger.error(
                f"[{self._name}] [{decision_label}] Decision retries "
                f"exhausted ({last_error}); using historical modal amount "
                f"{amount}."
            )
            return (
                amount,
                explanation,
                False,
                raw_response,
                "historical_mode",
            )

        explanation = (
            "[CRITICAL FAILURE] Decision retries exhausted and no "
            f"feasible prior valid {decision_label} decision exists: "
            f"{last_error}. Using amount 0."
        )
        self.logger.error(f"[{self._name}] [{decision_label}] {explanation}")
        return (
            0,
            explanation,
            False,
            raw_response,
            "no_valid_history_default",
        )

    async def choose_investment_with_commitment(
        self,
        interaction_num: int,
        initial_funds: int,
        multiplication_factor: int,
        partner_name: str,
    ) -> tuple[int, str, bool, str, str]:
        """Stage 1: choose an investment knowing commitment comes next."""
        try:
            self._current_funds = initial_funds
            history_str = self._build_partner_history_string(partner_name)
            role_profile = (
                f"You are {self._name}, the Trustor in a Trust Game.\n"
                "Goal: Maximize cumulative coins over all interactions.\n"
                f"You start this interaction with {initial_funds} coins and "
                f"may send an integer from 0 to {initial_funds}. The sent "
                f"amount is multiplied by {multiplication_factor} for the "
                "Trustee.\n"
                "After investing, but before the Trustee chooses a return, "
                "you must submit a binding punishment schedule. The Trustee "
                "will see the complete schedule before choosing a return. "
                "The environment will automatically enforce the schedule."
            )
            prompt = (
                f"Interaction {interaction_num}: Stage 1 of 5 - Investment\n"
                f"You have {initial_funds} coins.\n"
                f"Choose an integer amount from 0 to {initial_funds} to send "
                f"to {partner_name}. It will be multiplied by "
                f"{multiplication_factor}.\n"
                "After this investment, you will commit to a punishment level "
                "for every possible return before the Trustee makes that "
                "return decision.\n\n"
                f"{history_str}\n\n"
                "Return exactly:\n"
                "<decision>\n"
                f"  <amount>[integer 0-{initial_funds}]</amount>\n"
                "  <explanation>[brief strategic reason]</explanation>\n"
                "</decision>"
            )
            amount, explanation, valid, raw_response, decision_source = (
                await self._request_xml_amount_decision(
                    system_message=role_profile,
                    user_prompt=prompt,
                    role="Trustor",
                    max_amount=initial_funds,
                    decision_label="Investment",
                )
            )
            self.logger.info(
                f"[{self._name}] [Investment] Decision: {amount} coins; "
                f"valid={valid}; source={decision_source}"
            )
            return (
                amount,
                explanation,
                valid,
                raw_response,
                decision_source,
            )
        except Exception as error:
            error_msg = (
                f"Investment decision failed: {type(error).__name__} - "
                f"{error}"
            )
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(traceback.format_exc())
            self._error_count += 1
            return (
                0,
                f"[CRITICAL FAILURE] {error_msg}; using amount 0",
                False,
                "",
                "unhandled_error_default",
            )

    @staticmethod
    def _extract_schedule_block(content: str) -> str:
        """Extract the machine-readable schedule block from an LLM response."""
        tagged_match = re.search(
            r"<punishment_schedule>(.*?)</punishment_schedule>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        block = tagged_match.group(1) if tagged_match else content
        return re.sub(r"```(?:json)?|```", "", block, flags=re.IGNORECASE).strip()

    @classmethod
    def _parse_committed_schedule(
        cls,
        content: str,
        maximum_return: int,
    ) -> dict[int, int]:
        """Parse and strictly validate one level for every possible return."""
        block = cls._extract_schedule_block(content)
        raw_schedule = None

        try:
            json_match = re.search(r"\{.*\}", block, re.DOTALL)
            if json_match:
                raw_schedule = json.loads(json_match.group(0))
        except (json.JSONDecodeError, TypeError):
            raw_schedule = None

        if raw_schedule is None:
            pairs = re.findall(
                r'["\']?(\d+)["\']?\s*[:=]\s*(10|[0-9])\b',
                block,
            )
            raw_schedule = {return_value: level for return_value, level in pairs}

        schedule = {}
        for raw_return, raw_level in raw_schedule.items():
            try:
                return_value = int(raw_return)
                level = int(raw_level)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Non-integer schedule entry: {raw_return}: {raw_level}"
                ) from error

            if 0 <= return_value <= maximum_return:
                if not (
                    base.MIN_PUNISHMENT_LEVEL
                    <= level
                    <= base.MAX_PUNISHMENT_LEVEL
                ):
                    raise ValueError(
                        f"Punishment level {level} for return {return_value} "
                        "is outside the allowed range"
                    )
                schedule[return_value] = level

        expected_returns = set(range(maximum_return + 1))
        missing_returns = sorted(expected_returns - set(schedule))
        if missing_returns:
            raise ValueError(
                "Schedule is incomplete; missing return amounts: "
                + ", ".join(str(value) for value in missing_returns)
            )
        return {value: schedule[value] for value in range(maximum_return + 1)}

    @staticmethod
    def _extract_commitment_explanation(content: str) -> str:
        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if explanation_match:
            return explanation_match.group(1).strip()

        labelled_match = re.search(
            r"Explanation\s*:\s*(.*)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if labelled_match:
            return labelled_match.group(1).strip()
        return "No explanation provided"

    async def choose_committed_punishment_schedule(
        self,
        interaction_num: int,
        initial_funds: int,
        sent_amount: int,
        received_amount: int,
        partner_name: str,
    ) -> tuple[dict[int, int], str, bool]:
        """Stage 2: commit to punishment for every possible return."""
        if received_amount == 0:
            return (
                {0: 0},
                "No investment occurred; return 0 is the only possible "
                "outcome and its committed punishment level is 0.",
                True,
            )

        history_str = self._build_partner_history_string(partner_name)
        possible_returns = ", ".join(
            str(value) for value in range(received_amount + 1)
        )
        role_profile = (
            f"You are {self._name}, the Trustor at the binding commitment "
            "stage of a Trust Game.\n"
            "Goal: Maximize cumulative coins over all interactions.\n"
            "You must now choose a punishment level for every possible return. "
            "The schedule will be shown to the Trustee before their return "
            "decision and will then be enforced automatically. You cannot "
            "revise it after observing the return.\n"
            f"Each punishment level costs you "
            f"{base.PUNISHER_COST_PER_LEVEL} coins and reduces the Trustee's "
            f"payoff by {base.TARGET_LOSS_PER_LEVEL} coins."
        )
        prompt = (
            f"Interaction {interaction_num}: Stage 2 of 5 - Binding "
            "punishment commitment\n"
            f"You invested {sent_amount} of your {initial_funds} coins.\n"
            f"{partner_name} received {received_amount} coins after "
            "multiplication.\n"
            f"Possible integer returns are: {possible_returns}.\n\n"
            "Submit one punishment level for every possible return. Each level "
            f"must be an integer from {base.MIN_PUNISHMENT_LEVEL} to "
            f"{base.MAX_PUNISHMENT_LEVEL}.\n"
            "If the actual return is r and your committed level is p:\n"
            f"- Your final payoff = {initial_funds} - {sent_amount} + r - "
            f"{base.PUNISHER_COST_PER_LEVEL} * p\n"
            f"- {partner_name}'s final payoff = {received_amount} - r - "
            f"{base.TARGET_LOSS_PER_LEVEL} * p\n"
            "The Trustee will see this complete schedule before choosing r. "
            "The environment, not you, will apply the corresponding level.\n\n"
            f"{history_str}\n\n"
            "Return exactly this structure, using every possible return "
            "amount once:\n"
            "<punishment_schedule>\n"
            '{"0": level, "1": level, "...": level}\n'
            "</punishment_schedule>\n"
            "<explanation>brief strategic reason</explanation>"
        )

        last_error = None
        for format_attempt in range(3):
            try:
                content = await self._call_llm_with_retry(
                    system_message=role_profile,
                    user_prompt=prompt,
                )
                schedule = self._parse_committed_schedule(
                    content,
                    maximum_return=received_amount,
                )
                explanation = self._extract_commitment_explanation(content)
                self._decision_count += 1
                self.logger.info(
                    f"[{self._name}] [Commitment] Valid schedule with "
                    f"{len(schedule)} entries"
                )
                return schedule, explanation, True
            except Exception as error:
                last_error = error
                self.logger.warning(
                    f"[{self._name}] Commitment format attempt "
                    f"{format_attempt + 1}/3 failed: {error}"
                )
                prompt += (
                    "\n\nYour previous response was invalid. Include every "
                    f"integer key from 0 through {received_amount}, with no "
                    "missing keys."
                )

        self._error_count += 1
        fallback_schedule = {
            return_value: 0
            for return_value in range(received_amount + 1)
        }
        explanation = (
            "[CRITICAL FAILURE] Could not parse a complete commitment "
            f"schedule: {last_error}. Using an all-zero schedule."
        )
        self.logger.error(f"[{self._name}] {explanation}")
        return fallback_schedule, explanation, False

    @staticmethod
    def format_schedule_for_trustee(
        schedule: dict[int, int],
        received_amount: int,
    ) -> str:
        """Display the complete schedule and payoff consequences."""
        rows = []
        for return_value in range(received_amount + 1):
            level = schedule[return_value]
            trustee_payoff = (
                received_amount
                - return_value
                - base.TARGET_LOSS_PER_LEVEL * level
            )
            rows.append(
                f"- Return {return_value}: punishment level {level}; "
                f"your final payoff {trustee_payoff:.2f}"
            )
        return "\n".join(rows)

    async def choose_return_after_commitment(
        self,
        interaction_num: int,
        sent_amount: int,
        received_amount: int,
        partner_name: str,
        committed_schedule: dict[int, int],
    ) -> tuple[int, str, bool, str, str]:
        """Stages 3-4: observe the schedule, then choose the return."""
        try:
            self._current_funds = received_amount
            history_str = self._build_partner_history_string(partner_name)
            schedule_text = self.format_schedule_for_trustee(
                committed_schedule,
                received_amount,
            )
            role_profile = (
                f"You are {self._name}, the Trustee in a Trust Game.\n"
                "Goal: Maximize cumulative coins over all interactions.\n"
                f"You received {received_amount} coins after the Trustor's "
                f"investment of {sent_amount} was multiplied.\n"
                "Before your return decision, the Trustor submitted a binding "
                "punishment schedule. You can see the complete schedule. The "
                "environment will automatically apply it, and the Trustor "
                "cannot revise it."
            )
            prompt = (
                f"Interaction {interaction_num}: Stages 3-4 of 5 - Observe "
                "commitment and choose return\n"
                f"{partner_name} sent {sent_amount} coins; you received "
                f"{received_amount} coins.\n"
                f"Choose an integer return from 0 to {received_amount}.\n\n"
                "Binding punishment schedule:\n"
                f"{schedule_text}\n\n"
                "For return r with committed punishment level p, your final "
                f"payoff is {received_amount} - r - "
                f"{base.TARGET_LOSS_PER_LEVEL} * p.\n\n"
                f"{history_str}\n\n"
                "Return exactly:\n"
                "<decision>\n"
                f"  <amount>[integer 0-{received_amount}]</amount>\n"
                "  <explanation>[brief strategic reason]</explanation>\n"
                "</decision>"
            )
            (
                returned_amount,
                explanation,
                valid,
                raw_response,
                decision_source,
            ) = (
                await self._request_xml_amount_decision(
                    system_message=role_profile,
                    user_prompt=prompt,
                    role="Trustee",
                    max_amount=received_amount,
                    decision_label="ReturnAfterCommitment",
                )
            )
            self.logger.info(
                f"[{self._name}] [ReturnAfterCommitment] Decision: "
                f"{returned_amount} coins; valid={valid}; "
                f"source={decision_source}"
            )
            return (
                returned_amount,
                explanation,
                valid,
                raw_response,
                decision_source,
            )
        except Exception as error:
            error_msg = (
                f"Return decision failed: {type(error).__name__} - {error}"
            )
            self.logger.error(f"[{self._name}] {error_msg}")
            self.logger.debug(traceback.format_exc())
            self._error_count += 1
            return (
                0,
                f"[CRITICAL FAILURE] {error_msg}; using return 0",
                False,
                "",
                "unhandled_error_default",
            )


class CommittedPunishmentEnvironment(base.TrustGamePopulationEnvironment):
    """Environment that enforces a punishment schedule committed ex ante."""

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

        roles = ["Trustor", "Trustee"]
        base.random.shuffle(roles)
        agent1_role, agent2_role = roles
        trustor = agent1 if agent1_role == "Trustor" else agent2
        trustee = agent1 if agent1_role == "Trustee" else agent2

        interaction_summary = {
            "interaction": interaction_num,
            "pair_ids": (agent1_id, agent2_id),
            "pair_names": (agent1.name, agent2.name),
            "roles": (agent1_role, agent2_role),
            "mechanism": "binding_precommitted_punishment",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Stage 1: Trustor chooses the investment.
            (
                sent_amount,
                trustor_explanation,
                investment_decision_valid,
                raw_investment_response,
                investment_decision_source,
            ) = (
                await trustor.choose_investment_with_commitment(
                    interaction_num=interaction_num,
                    initial_funds=self.initial_funds,
                    multiplication_factor=self.multiplication_factor,
                    partner_name=trustee.name,
                )
            )
            received_amount = sent_amount * self.multiplication_factor

            # Stage 2: Trustor commits before the Trustee chooses a return.
            committed_schedule, commitment_explanation, commitment_valid = (
                await trustor.choose_committed_punishment_schedule(
                    interaction_num=interaction_num,
                    initial_funds=self.initial_funds,
                    sent_amount=sent_amount,
                    received_amount=received_amount,
                    partner_name=trustee.name,
                )
            )

            # Stages 3-4: show the complete schedule, then request the return.
            (
                returned_amount,
                trustee_explanation,
                return_decision_valid,
                raw_return_response,
                return_decision_source,
            ) = (
                await trustee.choose_return_after_commitment(
                    interaction_num=interaction_num,
                    sent_amount=sent_amount,
                    received_amount=received_amount,
                    partner_name=trustor.name,
                    committed_schedule=committed_schedule,
                )
            )

            # Stage 5: automatically enforce the previously committed level.
            punishment_level = committed_schedule[returned_amount]
            payoff_result = base.calculate_payoffs_with_costly_punishment(
                initial_funds=self.initial_funds,
                sent_amount=sent_amount,
                received_amount=received_amount,
                returned_amount=returned_amount,
                punishment_level=punishment_level,
            )
            trustor_base_payoff = payoff_result["trustor_base_payoff"]
            trustee_base_payoff = payoff_result["trustee_base_payoff"]
            punishment_cost = payoff_result["punishment_cost"]
            punishment_loss = payoff_result["punishment_loss"]
            trustor_payoff = payoff_result["trustor_final_payoff"]
            trustee_payoff = payoff_result["trustee_final_payoff"]

            trustor_summary = {
                "interaction": interaction_num,
                "role": "Trustor",
                "partner_id": trustee._id,
                "partner_name": trustee.name,
                "sent_amount": sent_amount,
                "returned_amount": returned_amount,
                "investment_decision_valid": investment_decision_valid,
                "return_decision_valid": return_decision_valid,
                "investment_decision_source": investment_decision_source,
                "return_decision_source": return_decision_source,
                "raw_investment_response": raw_investment_response,
                "raw_return_response": raw_return_response,
                "base_payoff": trustor_base_payoff,
                "partner_base_payoff": trustee_base_payoff,
                "committed_punishment_schedule": committed_schedule,
                "commitment_valid": commitment_valid,
                "punishment_level": punishment_level,
                "punishment_cost": punishment_cost,
                "punishment_loss": punishment_loss,
                "payoff": trustor_payoff,
                "partner_payoff": trustee_payoff,
                "explanation": trustor_explanation,
                "commitment_explanation": commitment_explanation,
                "punishment_explanation": commitment_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat(),
            }
            trustee_summary = {
                "interaction": interaction_num,
                "role": "Trustee",
                "partner_id": trustor._id,
                "partner_name": trustor.name,
                "sent_amount": sent_amount,
                "received_amount": received_amount,
                "returned_amount": returned_amount,
                "investment_decision_valid": investment_decision_valid,
                "return_decision_valid": return_decision_valid,
                "investment_decision_source": investment_decision_source,
                "return_decision_source": return_decision_source,
                "raw_investment_response": raw_investment_response,
                "raw_return_response": raw_return_response,
                "base_payoff": trustee_base_payoff,
                "partner_base_payoff": trustor_base_payoff,
                "committed_punishment_schedule": committed_schedule,
                "commitment_valid": commitment_valid,
                "punishment_level": punishment_level,
                "punishment_cost": punishment_cost,
                "punishment_loss": punishment_loss,
                "payoff": trustee_payoff,
                "partner_payoff": trustor_payoff,
                "explanation": trustee_explanation,
                "commitment_explanation": commitment_explanation,
                "punishment_explanation": commitment_explanation,
                "multiplication_factor": self.multiplication_factor,
                "timestamp": datetime.now().isoformat(),
            }
            trustor.update_history(trustor_summary)
            trustee.update_history(trustee_summary)
            trustor._total_payoff += trustor_payoff
            trustee._total_payoff += trustee_payoff

            interaction_detail = {
                "trustor_id": trustor._id,
                "trustor_name": trustor.name,
                "trustee_id": trustee._id,
                "trustee_name": trustee.name,
                "sent_amount": sent_amount,
                "returned_amount": returned_amount,
                "trustee_received": received_amount,
                "investment_decision_valid": investment_decision_valid,
                "return_decision_valid": return_decision_valid,
                "investment_decision_source": investment_decision_source,
                "return_decision_source": return_decision_source,
                "raw_investment_response": raw_investment_response,
                "raw_return_response": raw_return_response,
                "trustor_base_payoff": trustor_base_payoff,
                "trustee_base_payoff": trustee_base_payoff,
                "committed_punishment_schedule": committed_schedule,
                "commitment_valid": commitment_valid,
                "commitment_explanation": commitment_explanation,
                "schedule_shown_to_trustee": True,
                "punishment_automatically_enforced": True,
                "punishment_level": punishment_level,
                "punishment_cost": punishment_cost,
                "punishment_loss": punishment_loss,
                "trustor_payoff": trustor_payoff,
                "trustee_payoff": trustee_payoff,
                "trustor_explanation": trustor_explanation,
                "trustee_explanation": trustee_explanation,
                "punishment_explanation": commitment_explanation,
            }
        except Exception as error:
            logging.error(
                f"Error in committed-punishment interaction "
                f"{interaction_num}: {error}"
            )
            logging.debug(traceback.format_exc())
            return {}

        interaction_summary["detail"] = interaction_detail
        self.game_logs.append(interaction_summary)
        return interaction_summary


_ORIGINAL_SETUP_RESULT_DIRECTORY = base._setup_result_directory
_ORIGINAL_INPUT = input


async def _setup_committed_result_directory(
    config: dict,
    experiment_name: str,
) -> str:
    """Add commitment metadata before the inherited config is saved."""
    config["experiment_name"] = (
        "TrustGame_Group_Committed_Punishment_Experiment"
    )
    config["description"] = (
        "Population Trust Game with a binding punishment schedule committed "
        "before the Trustee chooses a return"
    )
    config["game_settings"].update({
        "num_games": NUM_GAMES,
        "mechanism_name": "binding_precommitted_punishment",
        "decision_timing": [
            "Trustor chooses investment",
            "Trustor commits punishment schedule",
            "Schedule is shown to Trustee",
            "Trustee chooses return",
            "Environment automatically enforces committed punishment",
        ],
        "commitment_is_binding": True,
        "schedule_shown_before_return": True,
    })
    config["agent_settings"].update({
        "amount_decision_format": "strict_xml",
        "amount_decision_outer_attempts": 3,
        "save_raw_amount_responses": True,
        "amount_failure_fallback": (
            "most_frequent_prior_valid_value_for_same_decision_type; "
            "most_recent_occurrence_breaks_ties; zero only when no feasible "
            "valid history exists"
        ),
    })
    return await _ORIGINAL_SETUP_RESULT_DIRECTORY(config, experiment_name)


def _committed_experiment_input(_prompt: str) -> str:
    """Use a treatment-specific folder prompt and default name."""
    experiment_name = _ORIGINAL_INPUT(
        "Please enter experiment folder name "
        "(e.g., 'TG_group_committed_punishment_test1'): "
    ).strip()
    if experiment_name:
        return experiment_name
    return (
        "TrustGame_Group_Committed_Punishment_"
        f"{datetime.now().strftime('%m%d%H%M')}"
    )


async def main():
    """Run the inherited experiment shell with only the treatment replaced."""
    base.EXPERIMENT_SLUG = EXPERIMENT_SLUG
    base.RESULTS_BASE_DIR = RESULTS_BASE_DIR
    base.TrustGamePopulationAgent = CommittedPunishmentAgent
    base.TrustGamePopulationEnvironment = CommittedPunishmentEnvironment
    base._setup_result_directory = _setup_committed_result_directory
    base.input = _committed_experiment_input
    await base.main(num_games=NUM_GAMES)


if __name__ == "__main__":
    asyncio.run(main())
