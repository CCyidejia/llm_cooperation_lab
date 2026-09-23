from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a rigorous research-software consistency reviewer for public-goods, prisoner's-dilemma, and trust-game mechanism transfer.
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
    return f"""Review these four artifacts for a game-adapter mechanism-transfer pipeline.

Paper PDF text source: {paper_name}
Mechanism JSON source: {mechanism_name}
Implementation plan source: {plan_name}
Baseline code source: {baseline_name}

Important project rule:
- This is NOT a full replication of the paper's human-subject experiment.
- The baseline game's native settings are experimental invariants: population, rounds, native payoff constants, repeat count, and interaction topology remain unchanged unless the plan has explicit architecture-extension authorization.
- Do NOT require the plan to change baseline settings to match the paper's sample size, group size, rematching protocol, laboratory session design, or treatment order.
- Judge whether the cooperation-promoting mechanism itself is transferred correctly into the existing baseline.
- If a paper detail is useful only for faithful human-experiment replication but not required for mechanism transfer, put it in recommended_fixes or quality_notes, not high-priority required_fixes.

Judge:
1. Paper -> mechanism: Does mechanism.json capture game family, topology, treatment matrix, timing, information, actions, payoffs, evidence, and uncertainties?
2. Mechanism completeness: Is mechanism.json complete enough to implement the mechanism in the provided baseline while preserving baseline game settings?
3. Mechanism -> plan: Does implementation_plan.json cover every mechanism requirement without forcing unrelated paper settings onto the baseline?
4. Plan -> baseline: Are target locations and code surfaces plausible in the baseline code? Does the plan preserve baseline invariants?
5. Next step: Is it ready for code implementation under mechanism-transfer assumptions?

Game-specific checks:
- For public goods, preserve the contribution/public-pool base payoff before mechanism effects.
- For prisoner's dilemma, preserve the baseline payoff matrix before mechanism effects.
- For trust games, preserve the Trustor-transfer -> Trustee-return order and both native payoff formulas. A punishment opportunity after return must be a distinct Trustor response stage.
- Require exact preservation of the trust baseline's neutral, Trustor, Trustee, and role-specific profiles and their constructor binding. Reject plans that target protected profiles; mechanism context must remain separate.
- Distinguish strategy-method elicitation and fixed laboratory roles from the mechanism itself; do not require them to replace a direct-response baseline unless the plan explicitly includes full protocol replication.
- Preserve native baseline parameters while allowing the plan to add evidence-grounded mechanism-specific cost/effect parameters.
- A simultaneous C/D/P design must not be converted into a post-decision punishment stage.
- A network-dependent treatment cannot pass against a non-network baseline unless transfer mode is extend_topology and architecture_change_authorized is true.
- Copy a deterministic blocked/requires_approval compatibility result into a non-approved review; never override that gate.

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
