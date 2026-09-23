from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMMON_REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version", "target_baseline", "mechanism_source", "baseline_summary",
    "required_code_changes", "new_state_variables", "new_logs", "new_metrics",
    "implementation_order", "tests_or_checks", "risks", "out_of_scope",
}
V2_REQUIRED_TOP_LEVEL_KEYS = {
    "game_adapter", "transfer_policy", "compatibility", "mechanism_summaries",
    "baseline_invariants", "validation_requirements",
}
REQUIRED_CHANGE_KEYS = {
    "change_id", "surface", "target_location", "operation", "description",
    "details", "depends_on", "acceptance_criteria",
}
ALLOWED_SURFACES = {
    "prompt_context", "action_space", "payoff_function", "state_memory",
    "matching_or_grouping", "network_topology", "logging_metrics", "experiment_config",
}
ALLOWED_COMPATIBILITY = {
    "compatible", "compatible_with_extension", "requires_approval", "blocked",
}


@dataclass
class PlanValidationIssue:
    path: str
    message: str


class ImplementationPlanValidationError(ValueError):
    def __init__(self, issues: list[PlanValidationIssue]):
        self.issues = issues
        details = "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)
        super().__init__(f"Invalid implementation plan:\n{details}")


def validate_implementation_plan(data: dict[str, Any]) -> None:
    issues: list[PlanValidationIssue] = []
    if not isinstance(data, dict):
        raise ImplementationPlanValidationError([PlanValidationIssue("$", "plan must be a JSON object")])

    version = data.get("schema_version")
    if version not in {"implementation-plan-v1", "implementation-plan-v2"}:
        issues.append(PlanValidationIssue("$.schema_version", "must be implementation-plan-v1 or implementation-plan-v2"))

    required = set(COMMON_REQUIRED_TOP_LEVEL_KEYS)
    if version == "implementation-plan-v2":
        required |= V2_REQUIRED_TOP_LEVEL_KEYS
    else:
        required.add("mechanism_summary")
    for key in sorted(required - set(data)):
        issues.append(PlanValidationIssue(f"$.{key}", "required top-level key is missing"))

    if version == "implementation-plan-v2":
        adapter = data.get("game_adapter")
        if not isinstance(adapter, dict) or adapter.get("game_type") not in {"public_goods", "prisoners_dilemma", "trust_game"}:
            issues.append(PlanValidationIssue("$.game_adapter.game_type", "must be public_goods, prisoners_dilemma, or trust_game"))
        compatibility = data.get("compatibility")
        if not isinstance(compatibility, dict) or compatibility.get("status") not in ALLOWED_COMPATIBILITY:
            issues.append(PlanValidationIssue("$.compatibility.status", f"must be one of {sorted(ALLOWED_COMPATIBILITY)}"))
        if not isinstance(data.get("mechanism_summaries"), list) or not data.get("mechanism_summaries"):
            issues.append(PlanValidationIssue("$.mechanism_summaries", "must be a non-empty list"))
        if not isinstance(data.get("validation_requirements"), list):
            issues.append(PlanValidationIssue("$.validation_requirements", "must be a list"))
        if isinstance(adapter, dict) and adapter.get("game_type") == "trust_game":
            invariants = data.get("baseline_invariants")
            if not isinstance(invariants, dict):
                issues.append(PlanValidationIssue("$.baseline_invariants", "must be an object"))
            else:
                if invariants.get("preserve_agent_profiles") is not True:
                    issues.append(PlanValidationIssue("$.baseline_invariants.preserve_agent_profiles", "must be true for trust_game"))
                profile_invariants = invariants.get("agent_profile_invariants")
                if not isinstance(profile_invariants, list) or not profile_invariants:
                    issues.append(PlanValidationIssue("$.baseline_invariants.agent_profile_invariants", "must contain trust-game profile fingerprints"))

    changes = data.get("required_code_changes")
    if not isinstance(changes, list) or not changes:
        issues.append(PlanValidationIssue("$.required_code_changes", "must be a non-empty list"))
    else:
        seen_ids: set[Any] = set()
        for idx, change in enumerate(changes):
            path = f"$.required_code_changes[{idx}]"
            if not isinstance(change, dict):
                issues.append(PlanValidationIssue(path, "change must be an object"))
                continue
            for key in sorted(REQUIRED_CHANGE_KEYS - set(change)):
                issues.append(PlanValidationIssue(f"{path}.{key}", "required key is missing"))
            change_id = change.get("change_id")
            if not isinstance(change_id, str) or not change_id:
                issues.append(PlanValidationIssue(f"{path}.change_id", "must be a non-empty string"))
            elif change_id in seen_ids:
                issues.append(PlanValidationIssue(f"{path}.change_id", "duplicate change_id"))
            seen_ids.add(change_id)
            if change.get("surface") not in ALLOWED_SURFACES:
                issues.append(PlanValidationIssue(f"{path}.surface", f"must be one of {sorted(ALLOWED_SURFACES)}"))

    for list_key in (
        "new_state_variables", "new_logs", "new_metrics", "implementation_order",
        "tests_or_checks", "risks", "out_of_scope", "validation_requirements",
    ):
        if list_key in data and not isinstance(data[list_key], list):
            issues.append(PlanValidationIssue(f"$.{list_key}", "must be a list"))

    if issues:
        raise ImplementationPlanValidationError(issues)
