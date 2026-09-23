from __future__ import annotations

from typing import Any


ALLOWED_OPERATIONS = {
    "replace_method", "insert_method_in_class", "replace_function", "insert_function",
    "replace_module_assignment",
}


class PatchValidationError(ValueError):
    """Raised when LLM-generated code edits do not match the patch schema."""


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_patch_document(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise PatchValidationError("patch document must be a JSON object")
    if data.get("schema_version") not in {"code-implementation-patch-v1", "code-implementation-patch-v2"}:
        raise PatchValidationError("schema_version must be code-implementation-patch-v1 or v2")
    edits = data.get("edits")
    if not isinstance(edits, list) or not edits:
        raise PatchValidationError("edits must be a non-empty list")

    seen_change_ids: set[str] = set()
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise PatchValidationError(f"edits[{idx}] must be an object")
        op = edit.get("operation")
        if op not in ALLOWED_OPERATIONS:
            raise PatchValidationError(f"edits[{idx}].operation must be one of {sorted(ALLOWED_OPERATIONS)}")
        change_id = edit.get("change_id")
        if not _nonempty_string(change_id):
            raise PatchValidationError(f"edits[{idx}].change_id must be a non-empty string")
        seen_change_ids.add(change_id)
        if not _nonempty_string(edit.get("code")):
            raise PatchValidationError(f"edits[{idx}].code must be a non-empty string")

        if op in {"replace_method", "insert_method_in_class"}:
            for field in ("class_name", "method_name"):
                if not _nonempty_string(edit.get(field)):
                    raise PatchValidationError(f"edits[{idx}].{field} must be a non-empty string")
            if edit.get("after_method") is not None and not isinstance(edit.get("after_method"), str):
                raise PatchValidationError(f"edits[{idx}].after_method must be a string when provided")
        elif op in {"replace_function", "insert_function"}:
            if not _nonempty_string(edit.get("function_name")):
                raise PatchValidationError(f"edits[{idx}].function_name must be a non-empty string")
            if edit.get("after_function") is not None and not isinstance(edit.get("after_function"), str):
                raise PatchValidationError(f"edits[{idx}].after_function must be a string when provided")
        elif op == "replace_module_assignment":
            if not _nonempty_string(edit.get("assignment_name")):
                raise PatchValidationError(f"edits[{idx}].assignment_name must be a non-empty string")

    declared = data.get("implemented_change_ids", [])
    if declared and (not isinstance(declared, list) or not all(_nonempty_string(item) for item in declared)):
        raise PatchValidationError("implemented_change_ids must be a list of non-empty strings")
    if declared and not set(declared).issubset(seen_change_ids):
        raise PatchValidationError("every implemented_change_id must have at least one edit")
