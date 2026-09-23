from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a careful coding agent implementing an approved game-mechanism transfer plan.
Return only compact valid JSON containing structured Python edits. Preserve unrelated baseline behavior.
Support public-goods, prisoner's-dilemma, and trust-game adapters without assuming class or method names.
"""


def _compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(plan: dict[str, Any], review: dict[str, Any] | None, baseline_text: str, baseline_name: str, output_name: str) -> str:
    review_text = _compact_json(review) if review else "null"
    return f"""Implement the approved plan by generating structured edits.

Baseline: {baseline_name}
Output: {output_name}

Constraints:
- Do not edit the original baseline; the host applies edits to a new output file.
- Preserve all source snippets in baseline_invariants.required_source_snippets.
- Preserve native population, rounds, payoff constants, repeat count, actions, and topology except changes explicitly authorized in the plan.
- For public goods, compute the original contribution/public-pool payoff before mechanism adjustments.
- For prisoner's dilemma, retain the original payoff matrix as the base before mechanism adjustments.
- For trust games, retain the original transfer multiplier and Trustor/Trustee payoff formulas as the base; preserve transfer-before-return ordering and add any later Trustor sanction as a distinct response.
- Preserve role-specific prompts, action bounds, history, and result fields. A shared role-aware decision method may implement more than one trust-game stage.
- When baseline_invariants.preserve_agent_profiles is true, do not modify, replace, reorder, bypass, or add alternatives to any protected agent-profile definition, `_profile` assignment, or constructor profile binding.
- Add mechanism instructions through separate decision-context variables or stage prompts; never encode them by changing a protected baseline profile.
- Preserve baseline-native parameters while implementing separately identified mechanism-specific cost/effect parameters.
- Respect timing_stage_type: same_stage_action modifies one decision; post_action_response creates a later stage.
- Never create network state or rewiring when compatibility does not explicitly allow it.
- Preserve existing result keys and add required mechanism/treatment fields.
- Every required_code_changes change_id must be covered by at least one edit.
- Code strings must contain a complete standalone method, function, or assignment statement and use escaped newlines in JSON.

Return this shape:
{{
  "schema_version": "code-implementation-patch-v2",
  "summary": "",
  "implemented_change_ids": ["C1"],
  "edits": [
    {{"change_id": "C1", "operation": "replace_method", "class_name": "ActualClass", "method_name": "actual_method", "code": "async def actual_method(...):\\n    ...\\n"}},
    {{"change_id": "C2", "operation": "insert_method_in_class", "class_name": "ActualClass", "method_name": "new_method", "after_method": "existing_method", "code": "def new_method(...):\\n    ...\\n"}},
    {{"change_id": "C3", "operation": "replace_function", "function_name": "actual_function", "code": "def actual_function(...):\\n    ...\\n"}},
    {{"change_id": "C4", "operation": "insert_function", "function_name": "new_function", "after_function": "actual_function", "code": "def new_function(...):\\n    ...\\n"}},
    {{"change_id": "C5", "operation": "replace_module_assignment", "assignment_name": "SETTING", "code": "SETTING = ...\\n"}}
  ],
  "notes": []
}}

Use only needed edits and actual locations from game_adapter/baseline_summary.

Implementation plan:
{_compact_json(plan)}

Consistency review:
{review_text}

Baseline source:
{baseline_text}
"""


def build_repair_prompt(previous_patch: str, validation_error: str) -> str:
    return f"""The previous structured patch failed validation or compilation.
Correct it and output only code-implementation-patch-v2 JSON.
Validation error: {validation_error}
Previous patch: {previous_patch}
"""
