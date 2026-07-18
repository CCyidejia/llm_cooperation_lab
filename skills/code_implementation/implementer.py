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
        "mechanism_summary": plan.get("mechanism_summary"),
        "required_code_changes": plan.get("required_code_changes"),
        "new_state_variables": plan.get("new_state_variables"),
        "new_logs": plan.get("new_logs"),
        "new_metrics": plan.get("new_metrics"),
        "implementation_order": plan.get("implementation_order"),
        "tests_or_checks": plan.get("tests_or_checks"),
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


def _class_and_method_maps(source: str) -> tuple[dict[str, ast.ClassDef], dict[tuple[str, str], ast.AST]]:
    tree = ast.parse(source)
    classes: dict[str, ast.ClassDef] = {}
    methods: dict[tuple[str, str], ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[(node.name, item.name)] = item
    return classes, methods

def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])


def build_baseline_excerpt_for_llm(source: str) -> str:
    """Return only code surfaces needed by the coding agent to reduce empty/length responses."""
    tree = ast.parse(source)
    classes, methods = _class_and_method_maps(source)
    sections: list[str] = []

    header_lines = []
    for line in source.splitlines()[:45]:
        header_lines.append(line)
    sections.append("# FILE HEADER / IMPORTS\n" + "\n".join(header_lines))

    wanted_methods = [
        ("PublicGoodsAgent", "__init__"),
        ("PublicGoodsAgent", "update_state"),
        ("PublicGoodsAgent", "update_history"),
        ("PublicGoodsAgent", "get_state_summary"),
        ("PublicGoodsAgent", "_build_history_string"),
        ("PublicGoodsAgent", "choose_action"),
        ("PublicGoodsAgent", "_call_llm_with_retry"),
        ("PublicGoodsEnvironment", "__init__"),
        ("PublicGoodsEnvironment", "run_interaction"),
        ("PublicGoodsEnvironment", "get_game_summary"),
    ]
    for class_name, method_name in wanted_methods:
        node = methods.get((class_name, method_name))
        if node is not None:
            method_source = _node_source(source, node)
            if method_name == "choose_action":
                method_lines = method_source.splitlines()
                method_source = "\n".join(method_lines[:75]) + "\n        # ... contribution parsing and fallback logic unchanged ...\n        return contribution, explanation"
            sections.append(f"# {class_name}.{method_name}\n" + method_source)

    for node in tree.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in {"main", "run_experiment"}:
            method_source = _node_source(source, node)
            if node.name == "main":
                method_source = "\n".join(method_source.splitlines()[:90]) + "\n# ... main continues with unchanged save logic ..."
            sections.append(f"# module function {node.name}\n" + method_source)

    return "\n\n".join(sections)


def _ensure_method_code(code: str, expected_method_name: str) -> str:
    if not code.endswith("\n"):
        code += "\n"
    dedented = textwrap.dedent(code).strip("\n") + "\n"
    try:
        parsed = ast.parse(dedented)
    except SyntaxError as exc:
        raise CodeApplicationError(f"Generated code for {expected_method_name} is not valid as a standalone method block: {exc}") from exc
    defs = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(defs) != 1 or defs[0].name != expected_method_name:
        raise CodeApplicationError(f"Generated code must contain exactly one method named {expected_method_name}")
    return textwrap.indent(dedented, "    ")


def apply_edits(source: str, patch: dict[str, Any]) -> tuple[str, list[str]]:
    lines = source.splitlines(keepends=True)
    _, methods = _class_and_method_maps(source)
    applied: list[str] = []

    # Apply bottom-up so line numbers remain stable.
    normalized_edits = []
    for edit in patch["edits"]:
        op = edit["operation"]
        class_name = edit["class_name"]
        method_name = edit["method_name"]
        code = _ensure_method_code(edit["code"], method_name)
        if op == "replace_method":
            node = methods.get((class_name, method_name))
            if node is None:
                raise CodeApplicationError(f"Cannot replace missing method {class_name}.{method_name}")
            normalized_edits.append((node.lineno - 1, node.end_lineno, code, edit["change_id"], f"replace {class_name}.{method_name}"))
        elif op == "insert_method_in_class":
            if (class_name, method_name) in methods:
                raise CodeApplicationError(f"Cannot insert {class_name}.{method_name}; method already exists")
            after_method = edit.get("after_method")
            anchor = methods.get((class_name, after_method)) if after_method else None
            if anchor is None:
                classes, _ = _class_and_method_maps(source)
                cls = classes.get(class_name)
                if cls is None:
                    raise CodeApplicationError(f"Cannot insert into missing class {class_name}")
                insert_at = cls.body[-1].end_lineno
            else:
                insert_at = anchor.end_lineno
            normalized_edits.append((insert_at, insert_at, "\n" + code, edit["change_id"], f"insert {class_name}.{method_name}"))
        else:
            raise CodeApplicationError(f"Unsupported operation: {op}")

    for start, end, code, change_id, label in sorted(normalized_edits, key=lambda item: item[0], reverse=True):
        lines[start:end] = [code]
        applied.append(f"{change_id}: {label}")

    return "".join(lines), list(reversed(applied))


def validate_generated_code(source: str, baseline_text: str, plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    try:
        compile(source, "generated_env.py", "exec")
    except SyntaxError as exc:
        issues.append(f"syntax error: {exc}")
        return issues

    mechanism = plan.get("mechanism_summary", {})
    mechanism_type = str(mechanism.get("type", "")).lower()
    mechanism_name = str(mechanism.get("name", "")).lower()
    target_locations = {
        str(change.get("target_location", "")).lower()
        for change in plan.get("required_code_changes", [])
        if isinstance(change, dict)
    }
    required_snippets = ["final_payoffs"]

    requires_punishment = (
        "punishment" in mechanism_type
        or "punishment" in mechanism_name
        or any("choose_punishment_actions" in item for item in target_locations)
    )
    requires_ir_help = (
        any(token in mechanism_type for token in ("reputation", "indirect", "reciprocity"))
        or any(token in mechanism_name for token in ("reputation", "indirect", "reciprocity"))
        or any("choose_ir_help_action" in item for item in target_locations)
    )
    requires_reward = (
        "reward" in mechanism_type
        or "reward" in mechanism_name
        or any("choose_reward_actions" in item for item in target_locations)
    )

    if requires_punishment:
        required_snippets.extend(
            [
                "choose_punishment_actions",
                "punishment_matrix",
                "punishment_costs",
                "punishment_penalties",
                "base_payoffs",
                "punishment_frequency",
                "average_punishment_sent",
                "average_punishment_received",
            ]
        )

    if requires_ir_help:
        required_snippets.extend(
            [
                "choose_ir_help_action",
                "public_reputation_ledger",
                "ir_logs",
                "reputation",
                "help",
                "ir_payoff_adjustments",
            ]
        )

    if requires_reward:
        required_snippets.extend(
            [
                "choose_reward_actions",
                "reward_actions",
                "reward_cost",
                "reward_benefit",
                "baseline_payoffs",
                "final_payoffs",
            ]
        )

    for snippet in required_snippets:
        if snippet not in source:
            issues.append(f"missing required snippet: {snippet}")

    invariant_snippets = [
        '"num_agents": 24',
        '"initial_endowment": 20',
        '"public_pool_multiplier": 9.6',
        '"num_rounds": 30',
    ]
    for snippet in invariant_snippets:
        if snippet in baseline_text and snippet not in source:
            issues.append(f"baseline invariant changed or removed: {snippet}")

    if plan.get("baseline_invariants", {}).get("preserve_existing_group_architecture") and "num_agents=EXPERIMENT_SETTINGS[\"num_agents\"]" not in source:
        issues.append("environment construction no longer uses EXPERIMENT_SETTINGS['num_agents']")

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
    baseline_excerpt = build_baseline_excerpt_for_llm(baseline_text)
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
            validation_issues = validate_generated_code(output_text, baseline_text, plan)
            if validation_issues:
                last_error = "; ".join(validation_issues)
                if attempt < repair_attempts:
                    continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
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
    print(f"Saved implemented env file to {args.output}")
    print(f"Saved implementation report to {args.report}")
    print(f"Validation status: {report['validation_status']}")


if __name__ == "__main__":
    main()

