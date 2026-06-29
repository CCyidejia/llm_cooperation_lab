from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version", "source", "paper", "game_context", "mechanisms",
    "implementation_mapping", "experiment_design", "evidence", "uncertainties",
}

REQUIRED_MECHANISM_KEYS = {
    "mechanism_id", "name", "type", "summary", "actors", "timing",
    "information_structure", "action_space_changes", "payoff_changes",
    "state_variables", "prompt_requirements", "metrics", "implementation_notes", "confidence",
}

ALLOWED_MECHANISM_TYPES = {
    "punishment", "reward", "reputation", "image_scoring", "indirect_reciprocity",
    "institution_choice", "partner_selection", "communication", "history_visibility",
    "mixed", "other",
}

ALLOWED_IMPLEMENTATION_SURFACES = {
    "prompt_context", "action_space", "payoff_function", "state_memory",
    "matching_or_grouping", "logging_metrics",
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
        "schema_version": "public-goods-mechanism-v1",
        "source": {"input_type": "", "file_name": "", "extraction_method": "llm_structured_extraction"},
        "paper": {"title": "", "authors": [], "year": None, "research_question": "", "study_type": "", "sample": "", "setting": ""},
        "game_context": {"game_type": "public_goods_game", "rounds": None, "group_size": None, "endowment": None, "mpcr_or_multiplier": None, "real_incentives": None, "baseline_condition": ""},
        "mechanisms": [],
        "implementation_mapping": {"target_baseline": "env_main_public_goods_group.py", "required_surfaces": [], "directly_implementable": [], "requires_baseline_extension": [], "implementation_order": []},
        "experiment_design": {"baseline_condition": "", "treatment_conditions": [], "control_variables": [], "suggested_rounds": None, "suggested_runs": None},
        "evidence": {"cooperation_outcomes": [], "reported_effect_direction": "", "reported_effect_size": "", "key_quotes_or_passages": []},
        "uncertainties": [],
    }


def validate_mechanism_document(data: dict[str, Any]) -> None:
    issues: list[ValidationIssue] = []
    if not isinstance(data, dict):
        raise MechanismValidationError([ValidationIssue("$", "document must be a JSON object")])

    for key in sorted(REQUIRED_TOP_LEVEL_KEYS - set(data)):
        issues.append(ValidationIssue(f"$.{key}", "required top-level key is missing"))

    mechanisms = data.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        issues.append(ValidationIssue("$.mechanisms", "must be a non-empty list"))
    else:
        for idx, mechanism in enumerate(mechanisms):
            if not isinstance(mechanism, dict):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}]", "mechanism must be an object"))
                continue
            for key in sorted(REQUIRED_MECHANISM_KEYS - set(mechanism)):
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].{key}", "required key is missing"))
            if mechanism.get("type") not in ALLOWED_MECHANISM_TYPES:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].type", f"must be one of {sorted(ALLOWED_MECHANISM_TYPES)}"))
            confidence = mechanism.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                issues.append(ValidationIssue(f"$.mechanisms[{idx}].confidence", "must be a number from 0 to 1"))

    mapping = data.get("implementation_mapping", {})
    if not isinstance(mapping, dict):
        issues.append(ValidationIssue("$.implementation_mapping", "must be an object"))
    else:
        surfaces = mapping.get("required_surfaces", [])
        if not isinstance(surfaces, list):
            issues.append(ValidationIssue("$.implementation_mapping.required_surfaces", "must be a list"))
        else:
            for surface in surfaces:
                if surface not in ALLOWED_IMPLEMENTATION_SURFACES:
                    issues.append(ValidationIssue("$.implementation_mapping.required_surfaces", f"unknown surface '{surface}'"))

    if issues:
        raise MechanismValidationError(issues)
