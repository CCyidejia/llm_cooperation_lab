from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version", "source", "paper", "game_context", "mechanisms",
    "treatment_matrix", "experiment_design", "evidence", "uncertainties",
}

REQUIRED_MECHANISM_KEYS = {
    "mechanism_id", "name", "type", "summary", "actors", "timing",
    "information_structure", "action_space_changes", "payoff_changes",
    "state_variables", "prompt_requirements", "metrics", "effect_profile",
    "transfer_requirements", "implementation_notes", "confidence",
}

ALLOWED_MECHANISM_TYPES = {
    "punishment", "reward", "reputation", "image_scoring", "indirect_reciprocity",
    "institution_choice", "partner_selection", "communication", "history_visibility",
    "network_formation", "social_knowledge", "mixed", "other",
}

ALLOWED_GAME_FAMILIES = {
    "public_goods", "prisoners_dilemma", "trust_game", "mixed", "other",
}

ALLOWED_TOPOLOGIES = {"group", "dyadic_random_matching", "fixed_network", "dynamic_network", "other"}

ALLOWED_STAGE_TYPES = {
    "primary_action", "same_stage_action", "post_action_response",
    "matching_or_network", "information_update",
}

ALLOWED_DECISION_PROTOCOLS = {
    "direct_response", "strategy_method", "mixed", "other", "unknown",
}

ALLOWED_IMPLEMENTATION_SURFACES = {
    "prompt_context", "action_space", "payoff_function", "state_memory",
    "matching_or_grouping", "network_topology", "logging_metrics", "experiment_config",
}


@dataclass
class ValidationIssue:
    path: str
    message: str


class MechanismValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        details = "\n".join(f"- {i.path}: {i.message}" for i in issues)
        super().__init__(f"Invalid mechanism JSON:\n{details}")


def empty_mechanism_document() -> dict[str, Any]:
    return {
        "schema_version": "cooperation-mechanism-v2",
        "source": {"input_type": "", "file_name": "", "extraction_method": "llm_structured_extraction"},
        "paper": {"title": "", "authors": [], "year": None, "research_question": "", "study_type": "", "sample": "", "setting": ""},
        "game_context": {
            "game_family": "other", "topology": "other", "rounds": None,
            "group_size": None, "endowment": None, "native_actions": [], "payoff_structure": "",
            "decision_protocol": "unknown", "decision_stages": [], "real_incentives": None,
            "baseline_condition": "",
        },
        "mechanisms": [],
        "treatment_matrix": [],
        "experiment_design": {"baseline_condition": "", "treatment_conditions": [], "control_variables": [], "suggested_rounds": None, "suggested_runs": None},
        "evidence": {"cooperation_outcomes": [], "reported_effect_direction": "", "reported_effect_size": "", "key_passages": []},
        "uncertainties": [],
    }


def validate_mechanism_document(data: dict[str, Any]) -> None:
    issues: list[ValidationIssue] = []
    if not isinstance(data, dict):
        raise MechanismValidationError([ValidationIssue("$", "document must be a JSON object")])

    for key in sorted(REQUIRED_TOP_LEVEL_KEYS - set(data)):
        issues.append(ValidationIssue(f"$.{key}", "required top-level key is missing"))

    if data.get("schema_version") != "cooperation-mechanism-v2":
        issues.append(ValidationIssue("$.schema_version", "must be cooperation-mechanism-v2"))

    game_context = data.get("game_context", {})
    if not isinstance(game_context, dict):
        issues.append(ValidationIssue("$.game_context", "must be an object"))
    else:
        if game_context.get("game_family") not in ALLOWED_GAME_FAMILIES:
            issues.append(ValidationIssue("$.game_context.game_family", f"must be one of {sorted(ALLOWED_GAME_FAMILIES)}"))
        if game_context.get("topology") not in ALLOWED_TOPOLOGIES:
            issues.append(ValidationIssue("$.game_context.topology", f"must be one of {sorted(ALLOWED_TOPOLOGIES)}"))
        decision_protocol = game_context.get("decision_protocol")
        if decision_protocol is not None and decision_protocol not in ALLOWED_DECISION_PROTOCOLS:
            issues.append(ValidationIssue("$.game_context.decision_protocol", f"must be one of {sorted(ALLOWED_DECISION_PROTOCOLS)}"))
        stages = game_context.get("decision_stages")
        if not isinstance(stages, list):
            issues.append(ValidationIssue("$.game_context.decision_stages", "must be a list"))
        else:
            seen_stage_ids: set[Any] = set()
            for idx, stage in enumerate(stages):
                if not isinstance(stage, dict) or stage.get("stage_type") not in ALLOWED_STAGE_TYPES:
                    issues.append(ValidationIssue(f"$.game_context.decision_stages[{idx}].stage_type", f"must be one of {sorted(ALLOWED_STAGE_TYPES)}"))
                    continue
                stage_id = stage.get("stage_id")
                if not isinstance(stage_id, str) or not stage_id:
                    issues.append(ValidationIssue(f"$.game_context.decision_stages[{idx}].stage_id", "must be a non-empty string"))
                elif stage_id in seen_stage_ids:
                    issues.append(ValidationIssue(f"$.game_context.decision_stages[{idx}].stage_id", "duplicate stage_id"))
                seen_stage_ids.add(stage_id)
                for key in ("actions", "observed_before_decision"):
                    if not isinstance(stage.get(key), list):
                        issues.append(ValidationIssue(f"$.game_context.decision_stages[{idx}].{key}", "must be a list"))
            if game_context.get("game_family") == "trust_game":
                if len(stages) < 2:
                    issues.append(ValidationIssue("$.game_context.decision_stages", "trust_game must describe at least transfer and return stages"))
                stage_types = {
                    stage.get("stage_type") for stage in stages if isinstance(stage, dict)
                }
                if not {"primary_action", "post_action_response"}.issubset(stage_types):
                    issues.append(ValidationIssue("$.game_context.decision_stages", "trust_game must include primary_action and post_action_response stages"))

    mechanisms = data.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        issues.append(ValidationIssue("$.mechanisms", "must be a non-empty list"))
    else:
        seen_mechanism_ids: set[Any] = set()
        for idx, mechanism in enumerate(mechanisms):
            if not isinstance(mechanism, dict):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}]", "mechanism must be an object"))
                continue
            for key in sorted(REQUIRED_MECHANISM_KEYS - set(mechanism)):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].{key}", "required key is missing"))
            mechanism_id = mechanism.get("mechanism_id")
            if not isinstance(mechanism_id, str) or not mechanism_id:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].mechanism_id", "must be a non-empty string"))
            elif mechanism_id in seen_mechanism_ids:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].mechanism_id", "duplicate mechanism_id"))
            seen_mechanism_ids.add(mechanism_id)
            if mechanism.get("type") not in ALLOWED_MECHANISM_TYPES:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].type", f"must be one of {sorted(ALLOWED_MECHANISM_TYPES)}"))
            timing = mechanism.get("timing", {})
            if not isinstance(timing, dict) or timing.get("stage_type") not in ALLOWED_STAGE_TYPES:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].timing.stage_type", f"must be one of {sorted(ALLOWED_STAGE_TYPES)}"))
            effect_profile = mechanism.get("effect_profile")
            if not isinstance(effect_profile, dict):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].effect_profile", "must be an object"))
            transfer_requirements = mechanism.get("transfer_requirements")
            if not isinstance(transfer_requirements, dict):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].transfer_requirements", "must be an object"))
            else:
                for key in ("required_game_features", "required_topology_features", "required_information_features"):
                    if not isinstance(transfer_requirements.get(key), list):
                        issues.append(ValidationIssue(f"$.mechanisms[{idx}].transfer_requirements.{key}", "must be a list"))
                for key in ("requires_network", "requires_new_decision_stage"):
                    if not isinstance(transfer_requirements.get(key), bool):
                        issues.append(ValidationIssue(f"$.mechanisms[{idx}].transfer_requirements.{key}", "must be boolean"))
            confidence = mechanism.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].confidence", "must be a number from 0 to 1"))

    treatments = data.get("treatment_matrix")
    if not isinstance(treatments, list):
        issues.append(ValidationIssue("$.treatment_matrix", "must be a list"))
    else:
        known_mechanisms = {
            item.get("mechanism_id") for item in mechanisms or [] if isinstance(item, dict)
        }
        seen_treatments: set[Any] = set()
        for idx, treatment in enumerate(treatments):
            if not isinstance(treatment, dict) or not isinstance(treatment.get("active_mechanism_ids"), list):
                issues.append(ValidationIssue(f"$.treatment_matrix[{idx}]", "must contain active_mechanism_ids list"))
                continue
            treatment_id = treatment.get("treatment_id")
            if not isinstance(treatment_id, str) or not treatment_id:
                issues.append(ValidationIssue(f"$.treatment_matrix[{idx}].treatment_id", "must be a non-empty string"))
            elif treatment_id in seen_treatments:
                issues.append(ValidationIssue(f"$.treatment_matrix[{idx}].treatment_id", "duplicate treatment_id"))
            seen_treatments.add(treatment_id)
            unknown = set(treatment["active_mechanism_ids"]) - known_mechanisms
            if unknown:
                issues.append(ValidationIssue(f"$.treatment_matrix[{idx}].active_mechanism_ids", f"unknown mechanism ids: {sorted(unknown)}"))

    if issues:
        raise MechanismValidationError(issues)
