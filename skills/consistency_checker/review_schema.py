from __future__ import annotations

from typing import Any


ALLOWED_OVERALL_STATUSES = {"pass", "needs_revision", "fail"}
ALLOWED_SECTION_STATUSES = {"pass", "needs_revision", "fail", "uncertain"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
ALLOWED_TARGETS = {"paper", "mechanism", "plan", "baseline", "pipeline"}


class ReviewValidationError(ValueError):
    """Raised when a consistency review document does not match the expected schema."""


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewValidationError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReviewValidationError(f"{path} must be a list")
    return value


def _require_status(value: Any, path: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ReviewValidationError(f"{path} must be one of {sorted(allowed)}")
    return value


def _validate_findings_section(data: dict[str, Any], path: str) -> None:
    _require_status(data.get("status"), f"{path}.status", ALLOWED_SECTION_STATUSES)
    for key in ("findings", "missing_or_uncertain", "quality_notes"):
        if key in data:
            _require_list(data[key], f"{path}.{key}")


def _validate_plan_section(data: dict[str, Any], path: str) -> None:
    _require_status(data.get("status"), f"{path}.status", ALLOWED_SECTION_STATUSES)
    for key in ("covered_surfaces", "missing_surfaces", "inconsistent_items"):
        _require_list(data.get(key), f"{path}.{key}")


def _validate_baseline_section(data: dict[str, Any], path: str) -> None:
    _require_status(data.get("status"), f"{path}.status", ALLOWED_SECTION_STATUSES)
    for key in ("valid_targets", "risky_targets", "missing_targets"):
        _require_list(data.get(key), f"{path}.{key}")


def _validate_fix_items(items: Any, path: str) -> None:
    for idx, item in enumerate(_require_list(items, path)):
        item = _require_dict(item, f"{path}[{idx}]")
        _require_status(item.get("priority"), f"{path}[{idx}].priority", ALLOWED_PRIORITIES)
        _require_status(item.get("target"), f"{path}[{idx}].target", ALLOWED_TARGETS)
        for key in ("issue", "suggested_fix"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ReviewValidationError(f"{path}[{idx}].{key} must be a non-empty string")


def validate_review_document(data: dict[str, Any]) -> None:
    _require_dict(data, "review")
    if data.get("schema_version") != "consistency-review-v1":
        raise ReviewValidationError("schema_version must be consistency-review-v1")
    _require_status(data.get("overall_status"), "overall_status", ALLOWED_OVERALL_STATUSES)
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ReviewValidationError("confidence must be a number between 0 and 1")

    _validate_findings_section(_require_dict(data.get("paper_mechanism_alignment"), "paper_mechanism_alignment"), "paper_mechanism_alignment")
    _validate_findings_section(_require_dict(data.get("mechanism_completeness"), "mechanism_completeness"), "mechanism_completeness")
    _validate_plan_section(_require_dict(data.get("plan_alignment"), "plan_alignment"), "plan_alignment")
    _validate_baseline_section(_require_dict(data.get("baseline_feasibility"), "baseline_feasibility"), "baseline_feasibility")
    _validate_fix_items(data.get("required_fixes"), "required_fixes")
    _validate_fix_items(data.get("recommended_fixes"), "recommended_fixes")

    approval = _require_dict(data.get("approval_for_next_step"), "approval_for_next_step")
    if not isinstance(approval.get("ready_for_code_implementation"), bool):
        raise ReviewValidationError("approval_for_next_step.ready_for_code_implementation must be boolean")
    if not isinstance(approval.get("reason"), str):
        raise ReviewValidationError("approval_for_next_step.reason must be a string")
