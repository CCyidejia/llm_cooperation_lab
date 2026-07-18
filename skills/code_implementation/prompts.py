from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a careful coding agent implementing a reviewed mechanism-transfer plan.
Return ONLY compact valid JSON. No markdown.
Generate structured Python code edits, not a whole file.
Preserve unrelated baseline behavior and do not change baseline-native game settings.
"""


def _compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(plan: dict[str, Any], review: dict[str, Any] | None, baseline_text: str, baseline_name: str, output_name: str) -> str:
    review_text = _compact_json(review) if review else "null"
    return f"""Implement the reviewed plan by generating structured edits for the baseline file.

Baseline file: {baseline_name}
Output file: {output_name}

Hard constraints:
- Do not change the original baseline file directly.
- Do not change baseline-native game settings: num_agents, initial_endowment, public_pool_multiplier, num_rounds, random_seed, result directory conventions, or the existing experiment-loop shape.
- Preserve existing imports unless a standard-library import is already available; prefer existing json/re/logging/asyncio imports.
- Preserve existing public result fields and add mechanism-specific fields required by the plan without removing old fields.
- Implement the mechanism described by the implementation plan, not a different mechanism.
- Preserve the baseline public-goods contribution stage and payoff formula first; apply any mechanism-specific payoff adjustments only after the unchanged baseline payoff is computed and clearly logged.
- Use current baseline participants and group architecture unless the plan explicitly adds an internal mechanism stage. Do not repartition agents, change paper-specific group sizes, or change the number of public-goods rounds.
- If the plan defines a second-stage mechanism decision, collect the relevant first-stage public-goods decisions first, then collect mechanism decisions, then apply mechanism payoff/state updates.
- Keep generated code syntactically valid Python and consistently indented for insertion into the existing class.
- The response must be strict JSON. In every "code" value, encode line breaks as escaped \\n characters inside one JSON string. Do not put raw multi-line text inside a JSON string.

Output exactly this JSON shape:
{{
  "schema_version": "code-implementation-patch-v1",
  "summary": "",
  "implemented_change_ids": ["C1", "C2", "C3", "C4", "C5", "C6"],
  "edits": [
    {{
      "change_id": "C1",
      "operation": "insert_method_in_class",
      "class_name": "PublicGoodsAgent",
      "method_name": "choose_mechanism_action",
      "after_method": "choose_action",
      "code": "    async def choose_mechanism_action(...):\\n        ...\\n"
    }},
    {{
      "change_id": "C3",
      "operation": "replace_method",
      "class_name": "PublicGoodsEnvironment",
      "method_name": "run_interaction",
      "code": "    async def run_interaction(...):\\n        ...\\n"
    }}
  ],
  "notes": []
}}

Recommended edits:
- Read required_code_changes, new_state_variables, new_logs, new_metrics, implementation_order, and tests_or_checks from the plan.
- Insert any new agent method required by the plan, for example a punishment, reputation, help, reward, communication, or institution-choice decision method. Use the exact method name requested by the plan when possible.
- Replace PublicGoodsAgent.update_state and PublicGoodsAgent.get_state_summary only as needed to add mechanism state while keeping existing contribution/outcome behavior compatible.
- Replace PublicGoodsAgent._build_history_string or choose_action only if the plan requires mechanism information to be visible in contribution prompts.
- Replace PublicGoodsEnvironment.run_interaction only as needed to add the mechanism stage after validated contributions and after the unchanged baseline payoff has been computed.
- Replace PublicGoodsEnvironment.get_game_summary only as needed to include mechanism summary metrics while preserving existing summary keys.
- Keep mechanism history compact in prompts but preserve full machine-readable logs in game_logs or state summaries.

Implementation plan JSON:
{_compact_json(plan)}

Consistency review JSON:
{review_text}

Baseline source code:
{baseline_text}
"""


def build_repair_prompt(previous_patch: str, validation_error: str) -> str:
    return f"""The previous patch JSON failed validation or produced invalid code.
Fix the patch and output only valid JSON with schema_version code-implementation-patch-v1.

Validation or compile error:
{validation_error}

Previous patch:
{previous_patch}
"""
