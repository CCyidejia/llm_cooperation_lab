import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from env_main_public_goods_group import (
    LLMRetryExhaustedError,
    PublicGoodsAgent,
    parse_contribution_response,
)


class PublicGoodsResponseParsingTests(unittest.TestCase):
    def test_uses_named_final_contribution_not_numbers_in_explanation(self):
        content = (
            '{"explanation":"With 20 coins and a 9.6 / 24 = 0.4 return, '
            'contributing would reduce my payoff.",'
            '"final_contribution":0}'
        )

        contribution, explanation = parse_contribution_response(content)

        self.assertEqual(contribution, 0)
        self.assertIn("20 coins", explanation)

    def test_accepts_a_single_json_code_fence(self):
        content = (
            "```json\n"
            '{"explanation":"I choose the final action shown below.",'
            '"final_contribution":20}\n'
            "```"
        )

        contribution, _ = parse_contribution_response(content)

        self.assertEqual(contribution, 20)

    def test_rejects_legacy_self_correcting_free_text(self):
        content = (
            "20\n"
            "After calculating the payoff, I should contribute 0.\n"
            "0"
        )

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            parse_contribution_response(content)

    def test_rejects_out_of_range_contribution(self):
        content = (
            '{"explanation":"Invalid amount for this game.",'
            '"final_contribution":21}'
        )

        with self.assertRaisesRegex(ValueError, "between 0 and 20"):
            parse_contribution_response(content)

    def test_rejects_boolean_contribution(self):
        content = (
            '{"explanation":"A Boolean is not an integer action.",'
            '"final_contribution":true}'
        )

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            parse_contribution_response(content)

    def test_fallback_uses_modal_contribution_and_latest_matching_explanation(self):
        agent = PublicGoodsAgent(1, "Agent_1")
        agent._valid_decision_history = [
            (20, "First full-contribution explanation."),
            (0, "A zero-contribution explanation."),
            (20, "Latest full-contribution explanation."),
        ]

        contribution, explanation = agent._fallback_to_modal_previous_output(
            LLMRetryExhaustedError("five failures")
        )

        self.assertEqual(contribution, 20)
        self.assertIn("Latest full-contribution explanation.", explanation)
        self.assertTrue(explanation.startswith("[FALLBACK_MODAL_PREVIOUS_OUTPUT]"))

    def test_fallback_count_tie_is_resolved_by_recency(self):
        agent = PublicGoodsAgent(1, "Agent_1")
        agent._valid_decision_history = [
            (0, "Earlier zero explanation."),
            (20, "More recent full-contribution explanation."),
        ]

        contribution, explanation = agent._fallback_to_modal_previous_output(
            ValueError("invalid structured response")
        )

        self.assertEqual(contribution, 20)
        self.assertIn("More recent full-contribution explanation.", explanation)

    def test_fallback_without_valid_history_uses_marked_zero(self):
        agent = PublicGoodsAgent(1, "Agent_1")

        contribution, explanation = agent._fallback_to_modal_previous_output(
            LLMRetryExhaustedError("five failures")
        )

        self.assertEqual(contribution, 0)
        self.assertTrue(explanation.startswith("[FALLBACK_NO_VALID_HISTORY]"))

class _SequenceRouter:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def get_llm_response(self, system_message, user_prompt):
        output = self.outputs[self.call_count]
        self.call_count += 1
        if isinstance(output, Exception):
            raise output
        return output


class PublicGoodsAPIRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_failure_text_is_retried_up_to_success(self):
        router = _SequenceRouter(
            ["API Call Failed: temporary outage"] * 4
            + ['{"explanation":"Valid fifth response.","final_contribution":0}']
        )
        agent = PublicGoodsAgent(1, "Agent_1")
        agent._llm = SimpleNamespace(router=router)

        result = await agent._call_llm_with_retry(
            system_message="",
            user_prompt="prompt",
            max_retries=5,
            retry_delay=0,
        )

        self.assertIn("Valid fifth response.", result)
        self.assertEqual(router.call_count, 5)

    async def test_five_api_failures_raise_retry_exhausted(self):
        router = _SequenceRouter(
            [RuntimeError("temporary outage")] * 5
        )
        agent = PublicGoodsAgent(1, "Agent_1")
        agent._llm = SimpleNamespace(router=router)

        with self.assertRaisesRegex(
            LLMRetryExhaustedError,
            "failed after 5 attempts",
        ):
            await agent._call_llm_with_retry(
                system_message="",
                user_prompt="prompt",
                max_retries=5,
                retry_delay=0,
            )

        self.assertEqual(router.call_count, 5)

    async def test_choose_action_reuses_modal_history_after_retry_exhaustion(self):
        agent = PublicGoodsAgent(1, "Agent_1")
        agent._valid_decision_history = [
            (0, "Zero explanation."),
            (20, "First full explanation."),
            (20, "Latest full explanation."),
        ]
        agent._call_llm_with_retry = AsyncMock(
            side_effect=LLMRetryExhaustedError("five failures")
        )

        contribution, explanation = await agent.choose_action(
            initial_endowment=20,
            public_pool_multiplier=9.6,
            num_agents=24,
            total_rounds=30,
        )

        self.assertEqual(contribution, 20)
        self.assertIn("Latest full explanation.", explanation)
        self.assertEqual(
            agent._valid_decision_history,
            [
                (0, "Zero explanation."),
                (20, "First full explanation."),
                (20, "Latest full explanation."),
            ],
        )


if __name__ == "__main__":
    unittest.main()
