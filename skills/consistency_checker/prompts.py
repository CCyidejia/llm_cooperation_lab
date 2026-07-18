from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a rigorous research-software consistency reviewer.
The project goal is mechanism transfer, not full paper replication: preserve the provided baseline game's native parameters and architecture unless the implementation plan explicitly marks a baseline parameter as intentionally configurable.
Check whether the extracted cooperation mechanism and baseline implementation plan preserve baseline invariants while adding the paper's cooperation-promoting mechanism.
Return ONLY compact valid JSON. No markdown. Do not include chain-of-thought. Be strict: mark uncertain items as missing_or_uncertain instead of guessing.
"""


def _compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(
    paper_text: str,
    paper_name: str,
    mechanism: dict[str, Any],
    mechanism_name: str,
    plan: dict[str, Any],
    plan_name: str,
    baseline_text: str,
    baseline_name: str,
) -> str:
    mechanism_json = _compact_json(mechanism)
    plan_json = _compact_json(plan)
    return f"""Review these four artifacts for the public goods game mechanism-transfer pipeline.

Paper PDF text source: {paper_name}
Mechanism JSON source: {mechanism_name}
Implementation plan source: {plan_name}
Baseline code source: {baseline_name}

Important project rule:
- This is NOT a full replication of the paper's human-subject experiment.
- The baseline game's native settings are experimental invariants: number of LLM agents, initial endowment, public pool multiplier/MPCR, total rounds, and the baseline's group architecture should remain unchanged unless the plan explicitly marks a parameter as a separate optional experiment variable.
- Do NOT require the plan to change baseline settings to match the paper's sample size, group size, rematching protocol, laboratory session design, or treatment order.
- Judge whether the cooperation-promoting mechanism itself is transferred correctly into the existing baseline.
- If a paper detail is useful only for faithful human-experiment replication but not required for mechanism transfer, put it in recommended_fixes or quality_notes, not high-priority required_fixes.

Judge:
1. Paper -> mechanism: Does mechanism.json capture the cooperation mechanism itself: timing, information needed for the mechanism, action changes, payoff changes, evidence, and uncertainties?
2. Mechanism completeness: Is mechanism.json complete enough to implement the mechanism in the provided baseline while preserving baseline game settings?
3. Mechanism -> plan: Does implementation_plan.json cover every mechanism requirement without forcing unrelated paper settings onto the baseline?
4. Plan -> baseline: Are target locations and code surfaces plausible in the baseline code? Does the plan preserve baseline invariants?
5. Next step: Is it ready for code implementation under mechanism-transfer assumptions?

Return exactly this JSON shape:
{{
  "schema_version": "consistency-review-v1",
  "overall_status": "pass|needs_revision|fail",
  "confidence": 0.0,
  "paper_mechanism_alignment": {{"status": "pass|needs_revision|fail|uncertain", "findings": [], "missing_or_uncertain": [], "quality_notes": []}},
  "mechanism_completeness": {{"status": "pass|needs_revision|fail|uncertain", "findings": [], "missing_or_uncertain": [], "quality_notes": []}},
  "plan_alignment": {{"status": "pass|needs_revision|fail|uncertain", "covered_surfaces": [], "missing_surfaces": [], "inconsistent_items": []}},
  "baseline_feasibility": {{"status": "pass|needs_revision|fail|uncertain", "valid_targets": [], "risky_targets": [], "missing_targets": []}},
  "required_fixes": [{{"priority": "high|medium|low", "target": "paper|mechanism|plan|baseline|pipeline", "issue": "", "suggested_fix": ""}}],
  "recommended_fixes": [{{"priority": "high|medium|low", "target": "paper|mechanism|plan|baseline|pipeline", "issue": "", "suggested_fix": ""}}],
  "approval_for_next_step": {{"ready_for_code_implementation": false, "reason": ""}}
}}

Use empty arrays when there are no issues. If any required fix is high priority, overall_status must not be pass. If ready_for_code_implementation is true, required_fixes must be empty.

MECHANISM_JSON:
{mechanism_json}

IMPLEMENTATION_PLAN_JSON:
{plan_json}

BASELINE_CODE_EXCERPT:
{baseline_text}

PAPER_TEXT_EXCERPT:
{paper_text}
"""


def build_repair_prompt(invalid_json_text: str, validation_error: str) -> str:
    return f"""The previous answer was not valid consistency-review JSON.
Fix it and output only valid JSON using schema_version consistency-review-v1.

Validation error:
{validation_error}

Previous answer:
{invalid_json_text}
"""
