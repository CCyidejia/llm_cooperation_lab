from __future__ import annotations

import json
from pathlib import Path


def load_baseline_capabilities() -> dict:
    path = Path(__file__).with_name("baseline_capabilities.json")
    return json.loads(path.read_text(encoding="utf-8-sig"))


SYSTEM_PROMPT = """Extract cooperation mechanisms from experimental-game literature.
Return ONLY compact valid JSON. No markdown and no hidden or long reasoning.
Separate the paper's native game design from the requirements for transferring the mechanism.
Do not assume that every punishment mechanism is a second-stage public-goods punishment stage:
punishment may instead be a same-stage action in a prisoner's dilemma.
Treat a trust game as a role-specific sequential game: Trustor transfer, Trustee return, and any
later Trustor sanction are distinct stages even when a strategy method elicits conditional choices ex ante.
Use null or empty lists for missing evidence and record uncertainty explicitly.
"""


def build_user_prompt(source_text: str, source_name: str, review_feedback: str = "") -> str:
    capabilities = load_baseline_capabilities()
    feedback_block = f"\nReviewer feedback to correct in this extraction:\n{review_feedback}\n" if review_feedback else ""
    return f"""Source file: {source_name}
Supported target game adapters: {capabilities['supported_game_adapters']}
Supported target topologies: {capabilities['supported_topologies']}
Allowed mechanism types: {capabilities['mechanism_types']}
Allowed implementation surfaces: {capabilities['implementation_surfaces']}
{feedback_block}
Extract the paper's game structure, treatments, and main cooperation-relevant mechanism(s).
Return exactly this JSON shape with concise, evidence-grounded values:
{{
  "schema_version": "cooperation-mechanism-v2",
  "source": {{"input_type": "pdf|text", "file_name": "{source_name}", "extraction_method": "llm_structured_extraction"}},
  "paper": {{"title": "", "authors": [], "year": null, "research_question": "", "study_type": "", "sample": "", "setting": ""}},
  "game_context": {{
    "game_family": "public_goods|prisoners_dilemma|trust_game|mixed|other",
    "topology": "group|dyadic_random_matching|fixed_network|dynamic_network|other",
    "rounds": null, "group_size": null, "endowment": null,
    "native_actions": [], "payoff_structure": "", "real_incentives": null,
    "decision_protocol": "direct_response|strategy_method|mixed|other|unknown",
    "baseline_condition": "",
    "decision_stages": [{{"stage_id": "S1", "stage_type": "primary_action|same_stage_action|post_action_response|matching_or_network|information_update", "actions": [], "observed_before_decision": []}}]
  }},
  "treatment_matrix": [{{"treatment_id": "B", "label": "", "active_mechanism_ids": [], "information_available": [], "network_rule": ""}}],
  "mechanisms": [{{
    "mechanism_id": "M1", "name": "",
    "type": "punishment|reward|reputation|social_knowledge|network_formation|partner_selection|communication|history_visibility|institution_choice|mixed|other",
    "summary": "",
    "actors": {{"decision_maker": "", "target": "", "observer": ""}},
    "timing": {{"stage": "", "stage_type": "primary_action|same_stage_action|post_action_response|matching_or_network|information_update", "frequency": "", "simultaneous_with_primary_action": false}},
    "information_structure": {{"anonymity": "", "public_information": [], "private_information": [], "history_visibility": ""}},
    "action_space_changes": [{{"action": "", "range_or_options": "", "condition": ""}}],
    "payoff_changes": [{{"who_pays": "", "who_receives_or_loses": "", "amount_or_formula": "", "condition": ""}}],
    "state_variables": [{{"name": "", "type": "agent_state|environment_state|round_log|metric|network_state", "update_rule": ""}}],
    "prompt_requirements": [],
    "metrics": [{{"name": "", "definition": ""}}],
    "effect_profile": {{"cooperation_direction": "increase|decrease|mixed|null", "conditions": [], "heterogeneity": [], "reported_evidence": ""}},
    "transfer_requirements": {{"required_game_features": [], "required_topology_features": [], "required_information_features": [], "requires_network": false, "requires_new_decision_stage": false}},
    "implementation_notes": {{"candidate_surfaces": [], "minimal_code_changes": [], "risks_or_assumptions": []}},
    "confidence": 0.0
  }}],
  "experiment_design": {{"baseline_condition": "", "treatment_conditions": [], "control_variables": [], "suggested_rounds": null, "suggested_runs": null}},
  "evidence": {{"cooperation_outcomes": [], "reported_effect_direction": "", "reported_effect_size": "", "key_passages": [{{"page": null, "paraphrase": "", "supports": ""}}]}},
  "uncertainties": []
}}

Critical distinctions:
- Record C/D/P chosen in one simultaneous decision as same_stage_action, not post_action_response.
- For trust games, record Trustor transfer as primary_action and Trustee return as post_action_response.
- Record punishment after the Trustee return as a separate post_action_response stage with the Trustor as actor; for NPT/PT designs, preserve both treatments.
- Record strategy-method elicitation in decision_protocol and conditional action descriptions; do not collapse the underlying sequential game into simultaneous play.
- Put personality-associated behavior and beliefs in effect_profile.heterogeneity; do not mislabel a measured personality trait as an experimental treatment or require rewriting the target baseline's agent profiles.
- Preserve B/R/N/RN or equivalent factorial treatments in treatment_matrix.
- Set requires_network=true for endogenous links, rewiring, neighbor-only interaction, or topology-dependent social knowledge.
- Extraction describes the paper. Baseline compatibility is decided later by the mapper.

Source text:
{source_text}
"""


def build_repair_prompt(invalid_json_text: str, validation_error: str) -> str:
    return f"""The previous answer was not valid cooperation-mechanism-v2 JSON.
Fix it and output only one valid JSON object using the same v2 schema.

Validation error:
{validation_error}

Previous answer:
{invalid_json_text}
"""
