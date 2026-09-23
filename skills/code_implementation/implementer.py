from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any

from skills.literature_mechanism_extractor.extractor import extract_json_object
from skills.baseline_mapper.plan_schema import validate_implementation_plan
from skills.consistency_checker.review_schema import validate_review_document
from skills.game_adapters import collect_agent_profile_invariants, profile_baseline

from .patch_schema import PatchValidationError, validate_patch_document
from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt


class CodeApplicationError(ValueError):
    """Raised when structured edits cannot be applied to the baseline source."""


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def summarize_plan_for_llm(plan: dict[str, Any]) -> dict[str, Any]:
    """Keep only implementation-relevant plan fields for the coding prompt."""
    return {
        "schema_version": plan.get("schema_version"),
        "target_baseline": plan.get("target_baseline"),
        "mechanism_source": plan.get("mechanism_source"),
        "baseline_invariants": plan.get("baseline_invariants"),
        "game_adapter": plan.get("game_adapter"),
        "transfer_policy": plan.get("transfer_policy"),
        "compatibility": plan.get("compatibility"),
        "mechanism_summaries": plan.get("mechanism_summaries", [plan.get("mechanism_summary")]),
        "required_code_changes": plan.get("required_code_changes"),
        "new_state_variables": plan.get("new_state_variables"),
        "new_logs": plan.get("new_logs"),
        "new_metrics": plan.get("new_metrics"),
        "implementation_order": plan.get("implementation_order"),
        "tests_or_checks": plan.get("tests_or_checks"),
        "validation_requirements": plan.get("validation_requirements"),
    }


def summarize_review_for_llm(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if review is None:
        return None
    return {
        "overall_status": review.get("overall_status"),
        "approval_for_next_step": review.get("approval_for_next_step"),
        "plan_alignment": review.get("plan_alignment"),
        "baseline_feasibility": review.get("baseline_feasibility"),
        "required_fixes": review.get("required_fixes"),
        "recommended_fixes": review.get("recommended_fixes"),
    }


def call_llm(system_prompt: str, user_prompt: str, provider: str, api_retries: int = 3, retry_delay: int = 5) -> str:
    if provider != "zgc":
        raise ValueError(f"Unsupported provider: {provider}")
    from LLMAPI.zgc import LLMAgent

    agent = LLMAgent(name="code-implementation")
    last_response = ""
    for attempt in range(api_retries + 1):
        last_response = agent.get_llm_response(system_prompt, user_prompt)
        if not last_response.startswith("API Call Failed"):
            return last_response
        retryable_markers = ["HTTP Error: 500", "get_channel_failed", "Empty content", "finish_reason=stop", "finish_reason=length", "timeout", "Connection", "temporarily"]
        if not any(marker in last_response for marker in retryable_markers):
            return last_response
        if attempt < api_retries:
            print(f"[code-implementation] LLM API failed on attempt {attempt + 1}/{api_retries + 1}; retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    return last_response


def _source_maps(source: str) -> tuple[
    dict[str, ast.ClassDef], dict[tuple[str, str], ast.AST], dict[str, ast.AST], dict[str, ast.AST]
]:
    tree = ast.parse(source)
    classes: dict[str, ast.ClassDef] = {}
    methods: dict[tuple[str, str], ast.AST] = {}
    functions: dict[str, ast.AST] = {}
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[(node.name, item.name)] = item
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node
    return classes, methods, functions, assignments

def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    decorators = getattr(node, "decorator_list", [])
    first_line = min([getattr(node, "lineno", 1)] + [item.lineno for item in decorators])
    start = first_line - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])


def build_baseline_excerpt_for_llm(source: str, plan: dict[str, Any], max_chars: int = 50000) -> str:
    """Expose complete target code without hard-coding either game's class names."""
    if len(source) <= max_chars:
        return source
    classes, methods, functions, assignments = _source_maps(source)
    sections = ["# FILE HEADER / IMPORTS\n" + "\n".join(source.splitlines()[:60])]
    wanted = {
        str(change.get("target_location", ""))
        for change in plan.get("required_code_changes", []) if isinstance(change, dict)
    }
    adapter = plan.get("game_adapter", {})
    wanted.update(adapter.get("decision_methods", []))
    wanted.update(adapter.get("round_methods", []))
    wanted.update(adapter.get("payoff_locations", []))
    for location in sorted(wanted):
        if "." in location:
            class_name, method_name = location.rsplit(".", 1)
            node = methods.get((class_name, method_name))
        else:
            node = functions.get(location) or assignments.get(location) or classes.get(location)
        if node is not None:
            sections.append(f"# TARGET {location}\n{_node_source(source, node)}")
    return "\n\n".join(sections)[:max_chars]


def _ensure_definition_code(code: str, expected_name: str, indent: bool) -> str:
    if not code.endswith("\n"):
        code += "\n"
    dedented = textwrap.dedent(code).strip("\n") + "\n"
    try:
        parsed = ast.parse(dedented)
    except SyntaxError as exc:
        raise CodeApplicationError(f"Generated code for {expected_name} is not valid as a standalone definition: {exc}") from exc
    defs = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(defs) != 1 or defs[0].name != expected_name:
        raise CodeApplicationError(f"Generated code must contain exactly one function/method named {expected_name}")
    return textwrap.indent(dedented, "    ") if indent else dedented


def _ensure_assignment_code(code: str, expected_name: str) -> str:
    normalized = textwrap.dedent(code).strip("\n") + "\n"
    try:
        tree = ast.parse(normalized)
    except SyntaxError as exc:
        raise CodeApplicationError(f"Generated assignment {expected_name} is invalid: {exc}") from exc
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    if len(tree.body) != 1 or expected_name not in names:
        raise CodeApplicationError(f"Generated code must contain exactly one assignment to {expected_name}")
    return normalized


def apply_edits(source: str, patch: dict[str, Any]) -> tuple[str, list[str]]:
    lines = source.splitlines(keepends=True)
    classes, methods, functions, assignments = _source_maps(source)
    applied: list[str] = []

    # Apply bottom-up so line numbers remain stable.
    normalized_edits = []
    for edit in patch["edits"]:
        op = edit["operation"]
        if op == "replace_method":
            class_name = edit["class_name"]
            method_name = edit["method_name"]
            code = _ensure_definition_code(edit["code"], method_name, indent=True)
            node = methods.get((class_name, method_name))
            if node is None:
                raise CodeApplicationError(f"Cannot replace missing method {class_name}.{method_name}")
            start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])]) - 1
            normalized_edits.append((start, node.end_lineno, code, edit["change_id"], f"replace {class_name}.{method_name}"))
        elif op == "insert_method_in_class":
            class_name = edit["class_name"]
            method_name = edit["method_name"]
            code = _ensure_definition_code(edit["code"], method_name, indent=True)
            if (class_name, method_name) in methods:
                raise CodeApplicationError(f"Cannot insert {class_name}.{method_name}; method already exists")
            after_method = edit.get("after_method")
            anchor = methods.get((class_name, after_method)) if after_method else None
            if anchor is None:
                cls = classes.get(class_name)
                if cls is None:
                    raise CodeApplicationError(f"Cannot insert into missing class {class_name}")
                insert_at = cls.body[-1].end_lineno
            else:
                insert_at = anchor.end_lineno
            normalized_edits.append((insert_at, insert_at, "\n" + code, edit["change_id"], f"insert {class_name}.{method_name}"))
        elif op == "replace_function":
            name = edit["function_name"]
            node = functions.get(name)
            if node is None:
                raise CodeApplicationError(f"Cannot replace missing function {name}")
            code = _ensure_definition_code(edit["code"], name, indent=False)
            start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])]) - 1
            normalized_edits.append((start, node.end_lineno, code, edit["change_id"], f"replace function {name}"))
        elif op == "insert_function":
            name = edit["function_name"]
            if name in functions:
                raise CodeApplicationError(f"Cannot insert function {name}; it already exists")
            code = _ensure_definition_code(edit["code"], name, indent=False)
            after = edit.get("after_function")
            anchor = functions.get(after) if after else None
            insert_at = anchor.end_lineno if anchor is not None else len(lines)
            normalized_edits.append((insert_at, insert_at, "\n" + code, edit["change_id"], f"insert function {name}"))
        elif op == "replace_module_assignment":
            name = edit["assignment_name"]
            node = assignments.get(name)
            if node is None:
                raise CodeApplicationError(f"Cannot replace missing module assignment {name}")
            code = _ensure_assignment_code(edit["code"], name)
            normalized_edits.append((node.lineno - 1, node.end_lineno, code, edit["change_id"], f"replace assignment {name}"))
        else:
            raise CodeApplicationError(f"Unsupported operation: {op}")

    for start, end, code, change_id, label in sorted(normalized_edits, key=lambda item: item[0], reverse=True):
        lines[start:end] = [code]
        applied.append(f"{change_id}: {label}")

    return "".join(lines), list(reversed(applied))


def validate_generated_code(
    source: str, baseline_text: str, plan: dict[str, Any], patch: dict[str, Any] | None = None,
) -> list[str]:
    issues: list[str] = []
    try:
        compile(source, "generated_env.py", "exec")
    except SyntaxError as exc:
        issues.append(f"syntax error: {exc}")
        return issues

    if not plan.get("compatibility", {}).get("code_implementation_allowed", True):
        issues.append("compatibility gate does not allow code implementation")

    invariant_snippets = plan.get("baseline_invariants", {}).get("required_source_snippets", [])
    for snippet in invariant_snippets:
        if snippet in baseline_text and snippet not in source:
            issues.append(f"baseline invariant changed or removed: {snippet}")

    baseline_invariants = plan.get("baseline_invariants", {})
    expected_profiles = baseline_invariants.get("agent_profile_invariants", [])
    if baseline_invariants.get("preserve_agent_profiles") and expected_profiles:
        specs = [item.get("spec", {}) for item in expected_profiles if isinstance(item, dict)]
        actual_profiles = collect_agent_profile_invariants(source, specs)
        for expected, actual in zip(expected_profiles, actual_profiles):
            if expected != actual:
                issues.append(f"agent profile invariant changed: {expected.get('spec', {})}")
        if len(expected_profiles) != len(actual_profiles):
            issues.append("agent profile invariant set changed")

    for requirement in plan.get("validation_requirements", []):
        if isinstance(requirement, str) and requirement.startswith("source_contains:"):
            snippet = requirement.split(":", 1)[1].strip()
            if snippet and snippet not in source:
                issues.append(f"missing plan validation snippet: {snippet}")

    if patch is not None:
        required_ids = {
            change.get("change_id") for change in plan.get("required_code_changes", [])
            if isinstance(change, dict) and change.get("change_id")
        }
        edited_ids = {
            edit.get("change_id") for edit in patch.get("edits", [])
            if isinstance(edit, dict) and edit.get("change_id")
        }
        missing_ids = sorted(required_ids - edited_ids)
        if missing_ids:
            issues.append(f"required plan changes have no structured edit: {missing_ids}")

    return issues


def build_report(
    baseline_path: Path,
    output_path: Path,
    plan_path: Path,
    review_path: Path | None,
    baseline_text: str,
    output_text: str,
    patch: dict[str, Any],
    applied_edits: list[str],
    validation_issues: list[str],
) -> dict[str, Any]:
    implemented = patch.get("implemented_change_ids") or sorted({edit.get("change_id") for edit in patch.get("edits", [])})
    return {
        "schema_version": "code-implementation-report-v1",
        "source_baseline": str(baseline_path),
        "implementation_plan": str(plan_path),
        "review_report": str(review_path) if review_path else None,
        "output_file": str(output_path),
        "output_written": not validation_issues,
        "baseline_sha256_before": sha256_text(baseline_text),
        "output_sha256": sha256_text(output_text),
        "implemented_change_ids": implemented,
        "applied_edits": applied_edits,
        "preserved_baseline_file": sha256_text(baseline_path.read_text(encoding="utf-8-sig")) == sha256_text(baseline_text),
        "syntax_check": "pass" if not any(issue.startswith("syntax error") for issue in validation_issues) else "fail",
        "validation_status": "pass" if not validation_issues else "fail",
        "validation_issues": validation_issues,
        "manual_review_needed": patch.get("notes", []),
    }


def generate_patch(
    plan: dict[str, Any],
    review: dict[str, Any] | None,
    baseline_text: str,
    baseline_name: str,
    output_name: str,
    provider: str,
    repair_attempts: int,
    api_retries: int = 3,
) -> dict[str, Any]:
    baseline_excerpt = build_baseline_excerpt_for_llm(baseline_text, plan)
    prompt_plan = summarize_plan_for_llm(plan)
    prompt_review = summarize_review_for_llm(review)
    raw = call_llm(SYSTEM_PROMPT, build_user_prompt(prompt_plan, prompt_review, baseline_excerpt, baseline_name, output_name), provider, api_retries=api_retries)
    if raw.startswith("API Call Failed"):
        hint = ""
        if "get_channel_failed" in raw or "HTTP Error: 500" in raw:
            model = os.getenv("ZGC_DEFAULT_MODEL", "")
            hint = (
                " ZGC returned a gateway/channel failure. This is usually a provider-side temporary outage "
                "or an unavailable model route. Retry later or set ZGC_DEFAULT_MODEL to an available model id "
                f"from `python -m LLMAPI.zgc`. Current ZGC_DEFAULT_MODEL={model!r}."
            )
        raise RuntimeError(f"LLM API call failed before patch extraction: {raw}.{hint}")

    last_raw = raw
    for attempt in range(repair_attempts + 1):
        try:
            patch = extract_json_object(last_raw)
            validate_patch_document(patch)
            return patch
        except (json.JSONDecodeError, PatchValidationError) as exc:
            if attempt >= repair_attempts:
                raise
            last_raw = call_llm(SYSTEM_PROMPT, build_repair_prompt(last_raw, str(exc)), provider, api_retries=api_retries)
            if last_raw.startswith("API Call Failed"):
                raise RuntimeError(f"LLM repair call failed: {last_raw}") from exc
    raise RuntimeError("unreachable")


def implement_code(
    plan_path: Path,
    baseline_path: Path,
    output_path: Path,
    report_path: Path,
    review_path: Path | None,
    provider: str,
    repair_attempts: int,
    api_retries: int,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_implementation_plan(plan)
    compatibility = plan.get("compatibility", {})
    if compatibility and not compatibility.get("code_implementation_allowed", False):
        raise ValueError(
            f"compatibility gate does not approve implementation: {compatibility.get('status')} - "
            f"{compatibility.get('reason')}"
        )

    planned_game = plan.get("game_adapter", {}).get("game_type")
    detected_profile: dict[str, Any] | None = None
    if planned_game:
        detected_profile = profile_baseline(baseline_path, "auto")
        if detected_profile.get("game_family") != planned_game:
            raise ValueError(
                f"plan targets game adapter {planned_game}, but baseline was detected as "
                f"{detected_profile.get('game_family')}"
            )
        if detected_profile.get("agent_profile_policy") == "preserve_exact":
            invariants = plan.setdefault("baseline_invariants", {})
            invariants["preserve_agent_profiles"] = True
            invariants["agent_profile_invariants"] = detected_profile.get("agent_profile_invariants", [])

    review = None
    if review_path:
        review = load_json(review_path)
        validate_review_document(review)
        approval = review.get("approval_for_next_step", {})
        if not approval.get("ready_for_code_implementation"):
            raise ValueError("review report does not approve code implementation")

    baseline_text = baseline_path.read_text(encoding="utf-8-sig")
    last_error = ""
    last_patch: dict[str, Any] | None = None

    for attempt in range(repair_attempts + 1):
        if attempt == 0:
            patch = generate_patch(plan, review, baseline_text, str(baseline_path), str(output_path), provider, repair_attempts, api_retries=api_retries)
        else:
            repair_review = dict(review or {})
            repair_review["previous_code_error"] = last_error
            patch = generate_patch(plan, repair_review, baseline_text, str(baseline_path), str(output_path), provider, repair_attempts, api_retries=api_retries)
        last_patch = patch
        try:
            output_text, applied_edits = apply_edits(baseline_text, patch)
            validation_issues = validate_generated_code(output_text, baseline_text, plan, patch)
            if validation_issues:
                last_error = "; ".join(validation_issues)
                if attempt < repair_attempts:
                    continue
            report = build_report(
                baseline_path,
                output_path,
                plan_path,
                review_path,
                baseline_text,
                output_text,
                patch,
                applied_edits,
                validation_issues,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if validation_issues:
                return report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            return report
        except CodeApplicationError as exc:
            last_error = str(exc)
            if attempt >= repair_attempts:
                raise

    raise RuntimeError(f"implementation failed after repair attempts; last_error={last_error}; last_patch={last_patch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM coding agent to implement an approved implementation_plan.json into a new env file.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--report", type=Path, default=Path("implementation_logs/code_implementation_report.json"))
    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--api-retries", type=int, default=3, help="Retries for transient provider/channel failures before giving up.")
    args = parser.parse_args()

    report = implement_code(
        plan_path=args.plan,
        baseline_path=args.baseline,
        output_path=args.output,
        report_path=args.report,
        review_path=args.review,
        provider=args.provider,
        repair_attempts=args.repair_attempts,
        api_retries=args.api_retries,
    )
    print(f"Saved implementation report to {args.report}")
    print(f"Validation status: {report['validation_status']}")
    if report["validation_status"] == "pass":
        print(f"Saved implemented env file to {args.output}")
    else:
        print("Validation failed; no generated environment file was written.")


if __name__ == "__main__":
    main()

