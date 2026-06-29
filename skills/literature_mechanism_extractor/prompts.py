from __future__ import annotations

import json
from pathlib import Path


def load_baseline_capabilities() -> dict:
    path = Path(__file__).with_name("baseline_capabilities.json")
    return json.loads(path.read_text(encoding="utf-8-sig"))


SYSTEM_PROMPT = """Extract cooperation-promoting mechanisms from public goods game literature.
Return ONLY compact valid JSON. No markdown. Do not write analysis or long reasoning.
Use null/empty lists for missing evidence. Keep the whole JSON concise.
"""


def build_user_prompt(source_text: str, source_name: str) -> str:
    capabilities = load_baseline_capabilities()
    surfaces = [item["name"] for item in capabilities["implementation_surfaces"]]
    mechanism_types = capabilities["preferred_mechanism_types"] + ["mixed", "other"]
    return f"""Source file: {source_name}
Target baseline: env_main_public_goods_group.py
Allowed mechanism types: {mechanism_types}
Allowed implementation surfaces: {surfaces}

Extract the main cooperation-promoting mechanism(s) from the source.
Return exactly this JSON shape, with concise values:
{{
  "schema_version": "public-goods-mechanism-v1",
  "source": {{"input_type": "pdf", "file_name": "{source_name}", "extraction_method": "llm_structured_extraction"}},
  "paper": {{"title": "", "authors": [], "year": null, "research_question": "", "study_type": "", "sample": "", "setting": ""}},
  "game_context": {{"game_type": "public_goods_game", "rounds": null, "group_size": null, "endowment": null, "mpcr_or_multiplier": null, "real_incentives": null, "baseline_condition": ""}},
  "mechanisms": [{{
    "mechanism_id": "M1",
    "name": "",
    "type": "punishment|reward|reputation|image_scoring|indirect_reciprocity|institution_choice|partner_selection|communication|history_visibility|mixed|other",
    "summary": "",
    "actors": {{"decision_maker": "", "target": "", "observer": ""}},
    "timing": {{"stage": "", "frequency": ""}},
    "information_structure": {{"anonymity": "", "public_information": [], "private_information": [], "history_visibility": ""}},
    "action_space_changes": [{{"action": "", "range_or_options": "", "condition": ""}}],
    "payoff_changes": [{{"who_pays": "", "who_receives_or_loses": "", "amount_or_formula": "", "condition": ""}}],
    "state_variables": [{{"name": "", "type": "agent_state|environment_state|round_log|metric", "update_rule": ""}}],
    "prompt_requirements": [],
    "metrics": [{{"name": "", "definition": ""}}],
    "implementation_notes": {{"directly_implementable_in_current_baseline": false, "required_surfaces": [], "minimal_code_changes": [], "risks_or_assumptions": []}},
    "confidence": 0.0
  }}],
  "implementation_mapping": {{"target_baseline": "env_main_public_goods_group.py", "required_surfaces": [], "directly_implementable": [], "requires_baseline_extension": [], "implementation_order": []}},
  "experiment_design": {{"baseline_condition": "", "treatment_conditions": [], "control_variables": [], "suggested_rounds": null, "suggested_runs": null}},
  "evidence": {{"cooperation_outcomes": [], "reported_effect_direction": "", "reported_effect_size": "", "key_quotes_or_passages": []}},
  "uncertainties": []
}}

Source text:
{source_text}
"""

def build_repair_prompt(invalid_json_text: str, validation_error: str) -> str:
    return f"""The previous answer was not valid mechanism JSON.
Fix it. Output only valid JSON with the same schema.

Validation error:
{validation_error}

Previous answer:
{invalid_json_text}
"""
