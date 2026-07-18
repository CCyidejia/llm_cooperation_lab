from __future__ import annotations

from typing import Any


ALLOWED_OPERATIONS = {"replace_method", "insert_method_in_class"}


class PatchValidationError(ValueError):
    """Raised when LLM-generated code edits do not match the patch schema."""


def validate_patch_document(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise PatchValidationError("patch document must be a JSON object")
    if data.get("schema_version") != "code-implementation-patch-v1":
        raise PatchValidationError("schema_version must be code-implementation-patch-v1")
    edits = data.get("edits")
    if not isinstance(edits, list) or not edits:
        raise PatchValidationError("edits must be a non-empty list")

    seen = set()
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise PatchValidationError(f"edits[{idx}] must be an object")
        op = edit.get("operation")
        if op not in ALLOWED_OPERATIONS:
            raise PatchValidationError(f"edits[{idx}].operation must be one of {sorted(ALLOWED_OPERATIONS)}")
        change_id = edit.get("change_id")
        if not isinstance(change_id, str) or not change_id.strip():
            raise PatchValidationError(f"edits[{idx}].change_id must be a non-empty string")
        seen.add(change_id)
        class_name = edit.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            raise PatchValidationError(f"edits[{idx}].class_name must be a non-empty string")
        code = edit.get("code")
        if not isinstance(code, str) or not code.strip():
            raise PatchValidationError(f"edits[{idx}].code must be a non-empty string")
        if op == "replace_method":
            method_name = edit.get("method_name")
            if not isinstance(method_name, str) or not method_name.strip():
                raise PatchValidationError(f"edits[{idx}].method_name must be a non-empty string")
        if op == "insert_method_in_class":
            method_name = edit.get("method_name")
            after_method = edit.get("after_method")
            if not isinstance(method_name, str) or not method_name.strip():
                raise PatchValidationError(f"edits[{idx}].method_name must be a non-empty string")
            if after_method is not None and not isinstance(after_method, str):
                raise PatchValidationError(f"edits[{idx}].after_method must be a string when provided")

    if "C1" not in seen:
        raise PatchValidationError("patch must include at least C1 for the punishment action-space change")
