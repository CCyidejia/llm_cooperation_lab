from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "target_baseline",
    "mechanism_source",
    "baseline_summary",
    "mechanism_summary",
    "required_code_changes",
    "new_state_variables",
    "new_logs",
    "new_metrics",
    "implementation_order",
    "tests_or_checks",
    "risks",
    "out_of_scope",
}

REQUIRED_CHANGE_KEYS = {
    "change_id",
    "surface",
    "target_location",
    "operation",
    "description",
    "details",
    "depends_on",
    "acceptance_criteria",
}

ALLOWED_SURFACES = {
    "prompt_context",
    "action_space",
    "payoff_function",
    "state_memory",
    "matching_or_grouping",
    "logging_metrics",
    "experiment_config",
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

    for key in sorted(REQUIRED_TOP_LEVEL_KEYS - set(data)):
        issues.append(PlanValidationIssue(f"$.{key}", "required top-level key is missing"))

    changes = data.get("required_code_changes")
    if not isinstance(changes, list) or not changes:
        issues.append(PlanValidationIssue("$.required_code_changes", "must be a non-empty list"))
    elif isinstance(changes, list):
        seen_ids = set()
        for idx, change in enumerate(changes):
            if not isinstance(change, dict):
                issues.append(PlanValidationIssue(f"$.required_code_changes[{idx}]", "change must be an object"))
                continue
            for key in sorted(REQUIRED_CHANGE_KEYS - set(change)):
                issues.append(PlanValidationIssue(f"$.required_code_changes[{idx}].{key}", "required key is missing"))
            change_id = change.get("change_id")
            if change_id in seen_ids:
                issues.append(PlanValidationIssue(f"$.required_code_changes[{idx}].change_id", "duplicate change_id"))
            seen_ids.add(change_id)
            surface = change.get("surface")
            if surface not in ALLOWED_SURFACES:
                issues.append(PlanValidationIssue(f"$.required_code_changes[{idx}].surface", f"must be one of {sorted(ALLOWED_SURFACES)}"))

    for list_key in ["new_state_variables", "new_logs", "new_metrics", "implementation_order", "tests_or_checks", "risks", "out_of_scope"]:
        if list_key in data and not isinstance(data[list_key], list):
            issues.append(PlanValidationIssue(f"$.{list_key}", "must be a list"))

    if issues:
        raise ImplementationPlanValidationError(issues)