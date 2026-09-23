from __future__ import annotations

import ast
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch as mock_patch

from LLMAPI.zgc import LLMAgent as ZGCLLMAgent
from skills.baseline_mapper.mapper import (
    build_profiled_baseline_excerpt,
    build_user_prompt as build_mapper_user_prompt,
    call_llm_with_length_retry as call_mapper_with_length_retry,
    ensure_required_context,
)
from skills.baseline_mapper.plan_schema import validate_implementation_plan
from skills.code_implementation.implementer import apply_edits, implement_code, validate_generated_code
from skills.consistency_checker.checker import enforce_compatibility_gate
from skills.game_adapters import evaluate_compatibility, normalize_game_type, profile_baseline
from skills.literature_mechanism_extractor.prompts import build_user_prompt as build_extractor_user_prompt
from skills.literature_mechanism_extractor.schema import validate_mechanism_document


ROOT = Path(__file__).resolve().parents[1]


class ZGCConfigurationTests(unittest.TestCase):
    def test_zero_limits_omit_max_tokens_and_disable_client_timeout(self) -> None:
        response = Mock()
        response.text = '{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
        response.json.return_value = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }
        response.raise_for_status.return_value = None

        with mock_patch("LLMAPI.zgc.requests.post", return_value=response) as post:
            result = ZGCLLMAgent(
                api_key="test-key",
                max_tokens=0,
                request_timeout=0,
            ).get_llm_response("system", "user")

        self.assertEqual(result, "{}")
        self.assertNotIn("max_tokens", post.call_args.kwargs["json"])
        self.assertIsNone(post.call_args.kwargs["timeout"])


def mechanism_document(*, network: bool = False) -> dict:
    topology = "dynamic_network" if network else "dyadic_random_matching"
    mechanism_type = "network_formation" if network else "punishment"
    stage_type = "matching_or_network" if network else "same_stage_action"
    return {
        "schema_version": "cooperation-mechanism-v2",
        "source": {"input_type": "pdf", "file_name": "paper.pdf", "extraction_method": "llm_structured_extraction"},
        "paper": {"title": "Test paper", "authors": [], "year": 2020, "research_question": "", "study_type": "experiment", "sample": "", "setting": ""},
        "game_context": {
            "game_family": "prisoners_dilemma", "topology": topology, "rounds": 10,
            "group_size": 2, "endowment": None, "native_actions": ["C", "D", "P"] if not network else ["C", "D"],
            "payoff_structure": "matrix", "real_incentives": True, "baseline_condition": "B",
            "decision_stages": [{"stage_id": "S1", "stage_type": stage_type, "actions": [], "observed_before_decision": []}],
        },
        "treatment_matrix": [
            {"treatment_id": item, "label": item, "active_mechanism_ids": (["M1"] if item != "B" else []), "information_available": [], "network_rule": ""}
            for item in (["B", "R", "N", "RN"] if network else ["B", "P"])
        ],
        "mechanisms": [{
            "mechanism_id": "M1", "name": "Network formation" if network else "Punishment action",
            "type": mechanism_type, "summary": "",
            "actors": {"decision_maker": "player", "target": "partner", "observer": "partner"},
            "timing": {"stage": "S1", "stage_type": stage_type, "frequency": "each round", "simultaneous_with_primary_action": not network},
            "information_structure": {"anonymity": "", "public_information": [], "private_information": [], "history_visibility": ""},
            "action_space_changes": [], "payoff_changes": [], "state_variables": [], "prompt_requirements": [], "metrics": [],
            "effect_profile": {"cooperation_direction": "mixed", "conditions": [], "heterogeneity": [], "reported_evidence": ""},
            "transfer_requirements": {
                "required_game_features": ["dyadic decisions"],
                "required_topology_features": ["persistent links"] if network else [],
                "required_information_features": [], "requires_network": network,
                "requires_new_decision_stage": False,
            },
            "implementation_notes": {"candidate_surfaces": [], "minimal_code_changes": [], "risks_or_assumptions": []},
            "confidence": 0.9,
        }],
        "experiment_design": {"baseline_condition": "B", "treatment_conditions": [], "control_variables": [], "suggested_rounds": None, "suggested_runs": None},
        "evidence": {"cooperation_outcomes": [], "reported_effect_direction": "", "reported_effect_size": "", "key_passages": []},
        "uncertainties": [],
    }


def trust_mechanism_document() -> dict:
    document = deepcopy(mechanism_document())
    document["paper"]["title"] = "On the psychology and economics of antisocial personality"
    document["game_context"].update({
        "game_family": "trust_game",
        "decision_protocol": "strategy_method",
        "native_actions": ["Trustor binary transfer", "Trustee conditional return", "Trustor conditional punishment"],
        "payoff_structure": "sequential transfer, multiplied investment, return, and optional costly punishment",
        "decision_stages": [
            {"stage_id": "S1", "stage_type": "primary_action", "actions": ["transfer"], "observed_before_decision": []},
            {"stage_id": "S2", "stage_type": "post_action_response", "actions": ["return schedule"], "observed_before_decision": []},
            {"stage_id": "S3", "stage_type": "post_action_response", "actions": ["punishment schedule"], "observed_before_decision": []},
        ],
    })
    document["treatment_matrix"] = [
        {"treatment_id": "NPT", "label": "No punishment", "active_mechanism_ids": [], "information_available": [], "network_rule": ""},
        {"treatment_id": "PT", "label": "Punishment", "active_mechanism_ids": ["M1"], "information_available": ["return schedule"], "network_rule": ""},
    ]
    mechanism = document["mechanisms"][0]
    mechanism["name"] = "Costly punishment after return"
    mechanism["actors"] = {"decision_maker": "Trustor", "target": "Trustee", "observer": "Trustee"}
    mechanism["timing"] = {
        "stage": "S3", "stage_type": "post_action_response",
        "frequency": "each interaction in PT", "simultaneous_with_primary_action": False,
    }
    mechanism["payoff_changes"] = [{
        "who_pays": "Trustor", "who_receives_or_loses": "Trustee",
        "amount_or_formula": "Trustor pays 1 to reduce Trustee payoff by 5", "condition": "PT",
    }]
    mechanism["effect_profile"]["heterogeneity"] = ["antisociality moderates trust, return, beliefs, and punishment"]
    mechanism["transfer_requirements"]["requires_new_decision_stage"] = True
    return document


class GameAdapterTests(unittest.TestCase):
    def test_profiles_all_supported_baselines(self) -> None:
        pg = profile_baseline(ROOT / "env_main_public_goods_group.py")
        pd = profile_baseline(ROOT / "env_main_prisoners_dilemma_group.py")
        trust = profile_baseline(ROOT / "env_main_trust_game_group.py")
        self.assertEqual(pg["game_family"], "public_goods")
        self.assertEqual(pd["game_family"], "prisoners_dilemma")
        self.assertEqual(trust["game_family"], "trust_game")
        self.assertIn("PrisonersDilemmaAgent.make_decision", pd["decision_method_candidates"])
        self.assertIn("get_payoff", pd["functions"])
        self.assertIn("TrustGamePopulationAgent.act", trust["decision_method_candidates"])
        self.assertIn("TrustGamePopulationEnvironment.run_interaction", trust["environment_round_method_candidates"])
        self.assertIn("TrustGamePopulationEnvironment.run_interaction", trust["payoff_location_candidates"])
        self.assertEqual(trust["decision_stage_model"], "sequential_dyadic")
        self.assertEqual(trust["agent_profile_policy"], "preserve_exact")
        self.assertEqual(len(trust["agent_profile_invariants"]), 6)
        self.assertTrue(all(item["matches"] for item in trust["agent_profile_invariants"]))
        self.assertEqual([stage["role"] for stage in trust["native_decision_stages"]], ["Trustor", "Trustee"])
        self.assertIn("trustor_payoff = self.initial_funds - trustor_amount + trustee_amount", trust["available_core_invariants"])

    def test_trust_game_aliases_normalize(self) -> None:
        self.assertEqual(normalize_game_type("Trust Game"), "trust_game")
        self.assertEqual(normalize_game_type("investment game"), "trust_game")

    def test_trust_punishment_keeps_three_sequential_stages(self) -> None:
        document = trust_mechanism_document()
        validate_mechanism_document(document)
        self.assertEqual(document["game_context"]["decision_protocol"], "strategy_method")
        self.assertEqual(
            [stage["stage_type"] for stage in document["game_context"]["decision_stages"]],
            ["primary_action", "post_action_response", "post_action_response"],
        )
        self.assertEqual([item["treatment_id"] for item in document["treatment_matrix"]], ["NPT", "PT"])
        self.assertTrue(document["mechanisms"][0]["transfer_requirements"]["requires_new_decision_stage"])

    def test_trust_profile_can_populate_a_valid_plan(self) -> None:
        document = trust_mechanism_document()
        baseline = ROOT / "env_main_trust_game_group.py"
        profile = profile_baseline(baseline)
        compatibility = evaluate_compatibility(document, profile, "preserve_baseline", False)
        plan = {
            "required_code_changes": [{
                "change_id": "C1", "surface": "action_space",
                "target_location": "TrustGamePopulationEnvironment.run_interaction",
                "operation": "replace_method", "description": "Add PT punishment after return",
                "details": {}, "depends_on": [], "acceptance_criteria": [],
            }],
            "new_state_variables": [], "new_logs": [], "new_metrics": [],
            "implementation_order": ["C1"], "tests_or_checks": [], "risks": [], "out_of_scope": [],
        }
        mapped = ensure_required_context(
            plan, document, Path("mechanism.json"), baseline, profile, compatibility,
            "preserve_baseline", False,
        )
        validate_implementation_plan(mapped)
        self.assertEqual(mapped["game_adapter"]["game_type"], "trust_game")
        self.assertEqual(mapped["game_adapter"]["decision_stage_model"], "sequential_dyadic")
        self.assertEqual(mapped["game_adapter"]["agent_profile_policy"], "preserve_exact")
        self.assertIn("TrustGamePopulationEnvironment.run_interaction", mapped["game_adapter"]["payoff_locations"])
        self.assertTrue(mapped["baseline_invariants"]["preserve_agent_profiles"])
        self.assertEqual(
            mapped["baseline_invariants"]["agent_profile_invariants"],
            profile["agent_profile_invariants"],
        )

    def test_trust_mapper_excerpt_keeps_late_stage_and_payoff_code(self) -> None:
        baseline = ROOT / "env_main_trust_game_group.py"
        source = baseline.read_text(encoding="utf-8-sig")
        profile = profile_baseline(baseline)
        excerpt = build_profiled_baseline_excerpt(source, profile, 24000)
        self.assertLess(len(excerpt), len(source))
        self.assertIn("async def act", excerpt)
        self.assertIn("async def run_interaction", excerpt)
        self.assertIn("trustor_payoff = self.initial_funds - trustor_amount + trustee_amount", excerpt)
        self.assertIn("async def _create_agents", excerpt)

    def test_extractor_prompt_contains_trust_game_rules(self) -> None:
        prompt = build_extractor_user_prompt("trust game source", "paper.pdf")
        self.assertIn("public_goods|prisoners_dilemma|trust_game", prompt)
        self.assertIn("NPT/PT", prompt)
        self.assertIn("strategy-method elicitation", prompt)

    def test_mapper_retries_length_failures_with_smaller_profiled_context(self) -> None:
        mechanism = trust_mechanism_document()
        mechanism_path = Path("mechanism.json")
        baseline_path = ROOT / "env_main_trust_game_group.py"
        baseline_source = baseline_path.read_text(encoding="utf-8-sig")
        profile = profile_baseline(baseline_path)
        compatibility = evaluate_compatibility(mechanism, profile, "preserve_baseline", False)
        prompts: list[str] = []

        def fake_call(
            _system: str,
            prompt: str,
            _provider: str,
            _max_output_tokens: int | None,
            _request_timeout: float | None,
        ) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return "API Call Failed: Model output was truncated (finish_reason=length); retrying is required"
            return '{"schema_version":"implementation-plan-v2"}'

        with mock_patch("skills.baseline_mapper.mapper.call_llm", side_effect=fake_call):
            raw = call_mapper_with_length_retry(
                mechanism, mechanism_path, baseline_path, baseline_source, profile,
                compatibility, "preserve_baseline", False, "", "zgc", 50000,
            )
        self.assertEqual(raw, '{"schema_version":"implementation-plan-v2"}')
        self.assertEqual(len(prompts), 2)
        self.assertLess(len(prompts[1]), len(prompts[0]))
        self.assertNotIn('"value_ast"', prompts[0])

    def test_mapper_retries_transient_timeout_without_shrinking_prompt(self) -> None:
        mechanism = trust_mechanism_document()
        mechanism_path = Path("mechanism.json")
        baseline_path = ROOT / "env_main_trust_game_group.py"
        baseline_source = baseline_path.read_text(encoding="utf-8-sig")
        profile = profile_baseline(baseline_path)
        compatibility = evaluate_compatibility(mechanism, profile, "preserve_baseline", False)
        prompts: list[str] = []

        def fake_call(
            _system: str,
            prompt: str,
            _provider: str,
            _max_output_tokens: int | None,
            _request_timeout: float | None,
        ) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return "API Call Failed: HTTPSConnectionPool: Read timed out. (read timeout=300)"
            return '{"schema_version":"implementation-plan-v2"}'

        with mock_patch("skills.baseline_mapper.mapper.call_llm", side_effect=fake_call), mock_patch(
            "skills.baseline_mapper.mapper.time.sleep",
        ):
            raw = call_mapper_with_length_retry(
                mechanism, mechanism_path, baseline_path, baseline_source, profile,
                compatibility, "preserve_baseline", False, "", "zgc", 24000,
                16384, 600, 2,
            )
        self.assertEqual(raw, '{"schema_version":"implementation-plan-v2"}')
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0], prompts[1])

    def test_mapper_prompt_summarizes_profile_fingerprints(self) -> None:
        mechanism = trust_mechanism_document()
        baseline_path = ROOT / "env_main_trust_game_group.py"
        profile = profile_baseline(baseline_path)
        compatibility = evaluate_compatibility(mechanism, profile, "preserve_baseline", False)
        prompt = build_mapper_user_prompt(
            mechanism, Path("mechanism.json"), baseline_path, "baseline excerpt",
            profile, compatibility, "preserve_baseline", False,
        )
        self.assertNotIn('"value_ast"', prompt)
        self.assertIn('"match_count":', prompt)

    def test_winners_same_stage_is_compatible_without_second_stage(self) -> None:
        document = mechanism_document(network=False)
        validate_mechanism_document(document)
        pd = profile_baseline(ROOT / "env_main_prisoners_dilemma_group.py")
        result = evaluate_compatibility(document, pd, "preserve_baseline", False)
        self.assertEqual(result["status"], "compatible")
        self.assertTrue(result["code_implementation_allowed"])
        self.assertEqual(document["mechanisms"][0]["timing"]["stage_type"], "same_stage_action")

    def test_network_treatments_are_gated(self) -> None:
        document = mechanism_document(network=True)
        validate_mechanism_document(document)
        pd = profile_baseline(ROOT / "env_main_prisoners_dilemma_group.py")
        blocked = evaluate_compatibility(document, pd, "preserve_baseline", False)
        approval_needed = evaluate_compatibility(document, pd, "extend_topology", False)
        allowed = evaluate_compatibility(document, pd, "extend_topology", True)
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(approval_needed["status"], "requires_approval")
        self.assertEqual(allowed["status"], "compatible_with_extension")
        self.assertEqual([item["treatment_id"] for item in document["treatment_matrix"]], ["B", "R", "N", "RN"])

    def test_mapper_context_cannot_override_gate(self) -> None:
        document = mechanism_document(network=True)
        pd = profile_baseline(ROOT / "env_main_prisoners_dilemma_group.py")
        compatibility = evaluate_compatibility(document, pd, "preserve_baseline", False)
        plan = {
            "compatibility": {"status": "compatible", "code_implementation_allowed": True},
            "required_code_changes": [{
                "change_id": "C1", "surface": "prompt_context", "target_location": "PrisonersDilemmaAgent.make_decision",
                "operation": "replace_method", "description": "x", "details": {}, "depends_on": [], "acceptance_criteria": [],
            }],
            "new_state_variables": [], "new_logs": [], "new_metrics": [], "implementation_order": [],
            "tests_or_checks": [], "risks": [], "out_of_scope": [],
        }
        mapped = ensure_required_context(
            plan, document, Path("mechanism.json"), ROOT / "env_main_prisoners_dilemma_group.py",
            pd, compatibility, "preserve_baseline", False,
        )
        self.assertFalse(mapped["compatibility"]["code_implementation_allowed"])


class StructuredEditTests(unittest.TestCase):
    def test_methods_functions_and_assignments_can_be_edited(self) -> None:
        source = "SETTING = 1\n\nclass Agent:\n    def decide(self):\n        return 'C'\n\ndef run():\n    return SETTING\n"
        patch = {
            "schema_version": "code-implementation-patch-v2",
            "implemented_change_ids": ["C1", "C2", "C3"],
            "edits": [
                {"change_id": "C1", "operation": "replace_method", "class_name": "Agent", "method_name": "decide", "code": "def decide(self):\n    return 'P'"},
                {"change_id": "C2", "operation": "replace_function", "function_name": "run", "code": "def run():\n    return SETTING + 1"},
                {"change_id": "C3", "operation": "replace_module_assignment", "assignment_name": "SETTING", "code": "SETTING = 2"},
            ],
        }
        output, applied = apply_edits(source, patch)
        compile(output, "toy.py", "exec")
        self.assertIn("return 'P'", output)
        self.assertIn("SETTING = 2", output)
        self.assertEqual(len(applied), 3)

    def test_required_change_coverage_is_deterministic(self) -> None:
        plan = {
            "compatibility": {"code_implementation_allowed": True},
            "baseline_invariants": {"required_source_snippets": ["SETTING = 1"]},
            "required_code_changes": [{"change_id": "C1"}, {"change_id": "C2"}],
            "validation_requirements": [],
        }
        patch = {"edits": [{"change_id": "C1"}]}
        issues = validate_generated_code("SETTING = 1\n", "SETTING = 1\n", plan, patch)
        self.assertTrue(any("C2" in issue for issue in issues))


class AgentProfileInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = ROOT / "env_main_trust_game_group.py"
        self.source = self.baseline.read_text(encoding="utf-8-sig")
        profile = profile_baseline(self.baseline)
        self.plan = {
            "compatibility": {"code_implementation_allowed": True},
            "baseline_invariants": {
                "required_source_snippets": [],
                "preserve_agent_profiles": True,
                "agent_profile_invariants": profile["agent_profile_invariants"],
            },
            "required_code_changes": [],
            "validation_requirements": [],
        }

    def test_unchanged_trust_profiles_pass(self) -> None:
        issues = validate_generated_code(self.source, self.source, self.plan)
        self.assertFalse(any("agent profile" in issue for issue in issues))

    def test_changed_neutral_profile_fails(self) -> None:
        changed = self.source.replace(
            "Your goal is to maximize your cumulative coins over all interactions.",
            "Your goal is to maximize your own accumulated coins.",
            1,
        )
        issues = validate_generated_code(changed, self.source, self.plan)
        self.assertTrue(any("agent profile invariant changed" in issue for issue in issues))

    def test_changed_constructor_profile_binding_fails(self) -> None:
        changed = self.source.replace("profile=neutral_profile", "profile=trustor_profile", 1)
        issues = validate_generated_code(changed, self.source, self.plan)
        self.assertTrue(any("agent profile invariant changed" in issue for issue in issues))

    def test_added_profile_override_fails(self) -> None:
        changed = self.source.replace(
            "        agents.append(agent)",
            "        agent._profile = 'changed'\n        agents.append(agent)",
            1,
        )
        issues = validate_generated_code(changed, self.source, self.plan)
        self.assertTrue(any("agent profile invariant changed" in issue for issue in issues))

    def test_failed_profile_validation_does_not_write_output(self) -> None:
        document = trust_mechanism_document()
        profile = profile_baseline(self.baseline)
        compatibility = evaluate_compatibility(document, profile, "preserve_baseline", False)
        plan = {
            "required_code_changes": [{
                "change_id": "C1", "surface": "prompt_context",
                "target_location": "_create_agents", "operation": "replace_function",
                "description": "Attempt a forbidden profile rewrite", "details": {},
                "depends_on": [], "acceptance_criteria": [],
            }],
            "new_state_variables": [], "new_logs": [], "new_metrics": [],
            "implementation_order": ["C1"], "tests_or_checks": [], "risks": [], "out_of_scope": [],
        }
        plan = ensure_required_context(
            plan, document, Path("mechanism.json"), self.baseline, profile,
            compatibility, "preserve_baseline", False,
        )

        function_node = next(
            node for node in ast.parse(self.source).body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_create_agents"
        )
        changed_function = ast.get_source_segment(self.source, function_node).replace(
            "Your goal is to maximize your cumulative coins over all interactions.",
            "Your goal is to maximize your own accumulated coins.",
            1,
        )
        generated_patch = {
            "schema_version": "code-implementation-patch-v2",
            "summary": "forbidden profile rewrite",
            "implemented_change_ids": ["C1"],
            "edits": [{
                "change_id": "C1", "operation": "replace_function",
                "function_name": "_create_agents", "code": changed_function,
            }],
            "notes": [],
        }

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            temp = Path(temp_dir)
            plan_path = temp / "plan.json"
            output_path = temp / "generated.py"
            report_path = temp / "report.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with mock_patch(
                "skills.code_implementation.implementer.generate_patch",
                return_value=generated_patch,
            ):
                report = implement_code(
                    plan_path=plan_path,
                    baseline_path=self.baseline,
                    output_path=output_path,
                    report_path=report_path,
                    review_path=None,
                    provider="zgc",
                    repair_attempts=0,
                    api_retries=1,
                )
            self.assertEqual(report["validation_status"], "fail")
            self.assertFalse(report["output_written"])
            self.assertFalse(output_path.exists())
            self.assertTrue(report_path.exists())


class GateAndHistoryTests(unittest.TestCase):
    def test_checker_forces_blocked_review_to_fail(self) -> None:
        review = {"overall_status": "pass", "required_fixes": [], "approval_for_next_step": {"ready_for_code_implementation": True, "reason": "ok"}}
        plan = {"compatibility": {"status": "blocked", "code_implementation_allowed": False, "reason": "network required"}}
        gated = enforce_compatibility_gate(review, plan)
        self.assertEqual(gated["overall_status"], "fail")
        self.assertFalse(gated["approval_for_next_step"]["ready_for_code_implementation"])

    def test_pd_history_is_partner_specific_and_memory_zero_is_private(self) -> None:
        source = (ROOT / "env_main_prisoners_dilemma_group.py").read_text(encoding="utf-8-sig")
        method = source[source.index("    async def make_decision"):source.index("    def get_memory_summary")]
        self.assertIn("if memory_size > 0 and partner_id in self._agent_memory", method)
        self.assertIn("No previous interactions with this partner are visible", method)
        self.assertNotIn("zip(self._my_history, self._partner_history)", method)


if __name__ == "__main__":
    unittest.main()
