from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any

from skills.game_adapters import evaluate_compatibility, profile_baseline
from skills.literature_mechanism_extractor.extractor import extract_json_object
from skills.literature_mechanism_extractor.schema import validate_mechanism_document

from .plan_schema import ImplementationPlanValidationError, validate_implementation_plan


SYSTEM_PROMPT = """You are a baseline mapper for an automated experimental-game research pipeline.
Map a structured cooperation mechanism onto the detected target game adapter. Return one JSON object only.
Transfer the mechanism, not the paper's population size, rounds, payoff parameters, or matching regime.
Do not silently add network topology. Architecture changes are controlled by the supplied compatibility gate.
Use actual target classes, methods, functions, and module assignments from the baseline profile.
The plan must distinguish same-stage actions from post-action punishment stages.
For trust games, preserve role order and distinguish Trustor transfer, Trustee return, and later sanctions.
Treat the baseline agent profile as immutable when the adapter policy is preserve_exact.
"""


PLAN_SCHEMA_INSTRUCTIONS = """Return an implementation-plan-v2 object:
{
  "schema_version": "implementation-plan-v2",
  "target_baseline": "",
  "mechanism_source": "",
  "game_adapter": {"game_type": "public_goods|prisoners_dilemma|trust_game", "interaction_topology": "", "native_actions": [], "decision_methods": [], "round_methods": [], "payoff_locations": [], "decision_stage_model": "", "agent_profile_policy": "preserve_exact|unspecified"},
  "transfer_policy": {"mode": "preserve_baseline|extend_topology", "architecture_change_authorized": false, "paper_parameters_replace_baseline": false},
  "compatibility": {"status": "compatible|compatible_with_extension|requires_approval|blocked", "code_implementation_allowed": true, "reason": ""},
  "baseline_summary": {},
  "baseline_invariants": {"preserve_game_parameters": true, "preserve_experiment_parameters": true, "preserve_existing_topology": true, "preserve_agent_profiles": true, "required_source_snippets": [], "agent_profile_invariants": [], "rule": ""},
  "mechanism_summaries": [{"mechanism_id": "M1", "name": "", "type": "", "timing_stage_type": "", "summary": ""}],
  "required_code_changes": [{
    "change_id": "C1",
    "surface": "prompt_context|action_space|payoff_function|state_memory|matching_or_grouping|network_topology|logging_metrics|experiment_config",
    "target_location": "ExistingClass.method, module_function, or MODULE_ASSIGNMENT",
    "operation": "replace_method|insert_method_in_class|replace_function|insert_function|replace_module_assignment",
    "description": "",
    "details": {},
    "depends_on": [],
    "acceptance_criteria": []
  }],
  "new_state_variables": [], "new_logs": [], "new_metrics": [],
  "implementation_order": [], "validation_requirements": [], "tests_or_checks": [],
  "risks": [], "out_of_scope": []
}

Rules:
- Cover every active mechanism and every treatment interaction requested by treatment_matrix.
- A C/D/P prisoner's-dilemma design is one action-space/payoff change in the primary decision stage; do not invent a second stage.
- In a trust game, preserve Trustor transfer before Trustee return. Add punishment observed after the return as a separate Trustor response stage.
- Never rewrite protected neutral, Trustor, Trustee, or role-specific agent profiles. Put mechanism information in separate decision-context or treatment fields without changing profile text, order, or constructor binding.
- Strategy-method schedules and fixed laboratory roles are protocol details, not automatic replacements for a direct-response baseline or its role assignment.
- Network formation, rewiring, and neighbor-specific knowledge require network_topology changes and explicit extension approval.
- Keep the baseline's original payoff formula as the base before adding mechanism effects. Transfer mechanism-specific cost/effect parameters without replacing native payoff constants.
- Add logs and metrics sufficient to separate actions, treatments, mechanism use, and cooperation.
- Each required change must point to a location in the supplied profile whenever one exists.
"""


def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = getattr(node, "end_lineno", getattr(node, "lineno", 1))
    return "\n".join(lines[start:end])


def build_profiled_baseline_excerpt(
    source: str,
    baseline_profile: dict[str, Any],
    max_chars: int,
) -> str:
    """Keep complete adapter-relevant nodes when a baseline exceeds the prompt budget."""
    if max_chars <= 0 or len(source) <= max_chars:
        return source

    tree = ast.parse(source)
    methods: dict[str, ast.AST] = {}
    functions: dict[str, ast.AST] = {}
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[f"{node.name}.{item.name}"] = item
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node

    wanted_locations: list[str] = []
    for key in (
        "decision_method_candidates",
        "environment_round_method_candidates",
        "payoff_location_candidates",
        "state_update_candidates",
        "summary_or_logging_candidates",
    ):
        for location in baseline_profile.get(key, []):
            if location not in wanted_locations:
                wanted_locations.append(location)

    helper_terms = ("agent", "config", "result", "statistic", "summary", "setup")
    helper_functions = [
        name for name in functions if any(term in name.lower() for term in helper_terms)
    ]

    sections = ["# FILE HEADER / IMPORTS\n" + "\n".join(source.splitlines()[:60])]
    for location in wanted_locations:
        node = methods.get(location) or functions.get(location)
        if node is not None:
            sections.append(f"# ADAPTER TARGET: {location}\n{_node_source(source, node)}")
    for name in helper_functions:
        if name not in wanted_locations:
            sections.append(f"# SUPPORTING FUNCTION: {name}\n{_node_source(source, functions[name])}")
    for name, node in assignments.items():
        sections.append(f"# MODULE ASSIGNMENT: {name}\n{_node_source(source, node)}")

    selected: list[str] = []
    used = 0
    for section in sections:
        separator = 2 if selected else 0
        if used + separator + len(section) > max_chars:
            continue
        selected.append(section)
        used += separator + len(section)
    return "\n\n".join(selected) + f"\n\n[PROFILED EXCERPT: selected {used} of {len(source)} characters.]"


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    max_output_tokens: int | None = None,
    request_timeout: float | None = None,
) -> str:
    if provider != "zgc":
        raise ValueError(f"Unsupported provider: {provider}")
    from LLMAPI.zgc import LLMAgent

    return LLMAgent(
        name="baseline-mapper",
        max_tokens=max_output_tokens,
        request_timeout=request_timeout,
    ).get_llm_response(system_prompt, user_prompt)


def _is_transient_api_failure(raw: str) -> bool:
    if not raw.startswith("API Call Failed"):
        return False
    lower = raw.lower()
    return any(
        marker in lower
        for marker in (
            "timed out", "timeout", "connection", "temporarily", "rate limit",
            "too many requests", "http error: 429", "http error: 500",
            "http error: 502", "http error: 503", "http error: 504",
            "empty response", "no choices",
        )
    )


def call_llm_with_api_retry(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    max_output_tokens: int | None,
    request_timeout: float | None,
    api_retries: int,
) -> str:
    attempts = max(1, api_retries)
    last_raw = ""
    for attempt in range(1, attempts + 1):
        last_raw = call_llm(
            system_prompt,
            user_prompt,
            provider,
            max_output_tokens,
            request_timeout,
        )
        if not _is_transient_api_failure(last_raw) or attempt >= attempts:
            return last_raw
        delay = min(2 ** (attempt - 1), 8)
        print(
            "Mapper API request failed transiently; retrying the same request "
            f"in {delay}s ({attempt}/{attempts})..."
        )
        time.sleep(delay)
    return last_raw


def summarize_baseline_profile_for_llm(baseline_profile: dict[str, Any]) -> dict[str, Any]:
    """Hide large deterministic fingerprints that the mapper must not reproduce."""
    summary = dict(baseline_profile)
    invariants = baseline_profile.get("agent_profile_invariants", [])
    summary["agent_profile_invariants"] = [
        {
            "spec": item.get("spec", {}),
            "match_count": len(item.get("matches", [])),
        }
        for item in invariants
        if isinstance(item, dict)
    ]
    return summary


def build_user_prompt(
    mechanism_doc: dict[str, Any], mechanism_path: Path, baseline_path: Path,
    baseline_text: str, baseline_profile: dict[str, Any], compatibility: dict[str, Any],
    transfer_mode: str, allow_architecture_change: bool, review_feedback: str = "",
) -> str:
    feedback = f"\nCHECKER FEEDBACK TO ADDRESS:\n{review_feedback}\n" if review_feedback else ""
    prompt_profile = summarize_baseline_profile_for_llm(baseline_profile)
    return f"""{PLAN_SCHEMA_INSTRUCTIONS}

OUTPUT BUDGET RULES:
- Return compact JSON. Do not include prose, source code, duplicated findings, or long explanations.
- Use empty placeholders for `game_adapter`, `baseline_summary`, and `baseline_invariants`; the host overwrites them from the baseline. Keep `compatibility` to its supplied status, permission, and a short reason.
- Keep each description, criterion, risk, and note to one short sentence.
- Include only changes required to cover the mechanism and treatments.

MECHANISM DOCUMENT ({mechanism_path}):
{json.dumps(mechanism_doc, ensure_ascii=False, indent=2)}

TARGET BASELINE ({baseline_path}) PROFILE:
{json.dumps(prompt_profile, ensure_ascii=False, separators=(",", ":"))}

DETERMINISTIC COMPATIBILITY GATE:
{json.dumps(compatibility, ensure_ascii=False, indent=2)}
Transfer mode: {transfer_mode}; architecture change authorized: {allow_architecture_change}
{feedback}
TARGET SOURCE:
```python
{baseline_text}
```
Return the v2 plan only. Copy the compatibility gate's status and permission exactly.
"""


def _mapper_retry_limits(max_baseline_chars: int) -> list[int]:
    if max_baseline_chars <= 0:
        candidates = [0, 24000, 16000, 10000, 6000]
    else:
        candidates = [max_baseline_chars, 24000, 16000, 10000, 6000]
    limits: list[int] = []
    for candidate in candidates:
        if candidate > 0 and max_baseline_chars > 0 and candidate >= max_baseline_chars and candidate != max_baseline_chars:
            continue
        if candidate not in limits:
            limits.append(candidate)
    return limits


def call_llm_with_length_retry(
    mechanism_doc: dict[str, Any],
    mechanism_path: Path,
    baseline_path: Path,
    baseline_source: str,
    baseline_profile: dict[str, Any],
    compatibility: dict[str, Any],
    transfer_mode: str,
    allow_architecture_change: bool,
    review_feedback: str,
    provider: str,
    max_baseline_chars: int,
    max_output_tokens: int | None = None,
    request_timeout: float | None = None,
    api_retries: int = 3,
) -> str:
    last_raw = ""
    retry_limits = _mapper_retry_limits(max_baseline_chars)
    for index, limit in enumerate(retry_limits):
        baseline_text = build_profiled_baseline_excerpt(
            baseline_source, baseline_profile, limit,
        )
        raw = call_llm_with_api_retry(
            SYSTEM_PROMPT,
            build_user_prompt(
                mechanism_doc, mechanism_path, baseline_path, baseline_text, baseline_profile,
                compatibility, transfer_mode, allow_architecture_change, review_feedback,
            ),
            provider,
            max_output_tokens,
            request_timeout,
            api_retries,
        )
        last_raw = raw
        if not raw.startswith("API Call Failed"):
            return raw
        if "finish_reason=length" not in raw and "output was truncated" not in raw.lower():
            return raw
        if index < len(retry_limits) - 1:
            output_budget = max_output_tokens if max_output_tokens is not None else "provider default"
            print(
                "Mapper response exhausted the model output budget "
                f"(max_output_tokens={output_budget}, baseline_excerpt_chars={limit}); "
                "retrying with a shorter profiled baseline to free context capacity..."
            )
    return last_raw


def build_repair_prompt(previous_output: str, validation_error: str) -> str:
    return f"""{PLAN_SCHEMA_INSTRUCTIONS}
The previous mapper output was invalid. Correct it and return JSON only.
Validation error: {validation_error}
Previous output: {previous_output}
"""


def ensure_required_context(
    plan: dict[str, Any], mechanism_doc: dict[str, Any], mechanism_path: Path,
    baseline_path: Path, baseline_profile: dict[str, Any], compatibility: dict[str, Any],
    transfer_mode: str, allow_architecture_change: bool,
) -> dict[str, Any]:
    plan["schema_version"] = "implementation-plan-v2"
    plan["target_baseline"] = str(baseline_path)
    plan["mechanism_source"] = str(mechanism_path)
    plan["game_adapter"] = {
        "game_type": baseline_profile["game_family"],
        "interaction_topology": baseline_profile["interaction_topology"],
        "native_actions": baseline_profile["native_actions"],
        "decision_methods": baseline_profile["decision_method_candidates"],
        "round_methods": baseline_profile["environment_round_method_candidates"],
        "payoff_locations": baseline_profile.get("payoff_location_candidates", []),
        "decision_stage_model": baseline_profile.get("decision_stage_model", ""),
        "agent_profile_policy": baseline_profile.get("agent_profile_policy", "unspecified"),
    }
    plan["transfer_policy"] = {
        "mode": transfer_mode,
        "architecture_change_authorized": allow_architecture_change,
        "paper_parameters_replace_baseline": False,
    }
    plan["compatibility"] = compatibility
    plan["baseline_summary"] = baseline_profile
    plan["baseline_invariants"] = {
        "preserve_game_parameters": True,
        "preserve_experiment_parameters": True,
        "preserve_existing_topology": not compatibility.get("requires_architecture_change", False),
        "preserve_agent_profiles": baseline_profile.get("agent_profile_policy") == "preserve_exact",
        "required_source_snippets": baseline_profile.get("available_core_invariants", []),
        "agent_profile_invariants": baseline_profile.get("agent_profile_invariants", []),
        "rule": "Keep baseline parameters and protected agent profiles unchanged; implement only the mechanism and explicitly authorized topology extensions.",
    }
    if not plan.get("mechanism_summaries"):
        plan["mechanism_summaries"] = [
            {
                "mechanism_id": item.get("mechanism_id"), "name": item.get("name"),
                "type": item.get("type"), "timing_stage_type": item.get("timing", {}).get("stage_type"),
                "summary": item.get("summary"),
            }
            for item in mechanism_doc.get("mechanisms", [])
        ]
    plan.setdefault("validation_requirements", [])
    plan.setdefault("out_of_scope", [])
    note = "Replacing baseline population, rounds, payoff constants, repeat count, or unapproved topology with paper settings."
    if note not in plan["out_of_scope"]:
        plan["out_of_scope"].append(note)
    return plan


def build_plan(
    mechanism_path: Path, baseline_path: Path, provider: str = "zgc", repair_attempts: int = 1,
    max_baseline_chars: int = 24000, game_type: str = "auto", transfer_mode: str = "preserve_baseline",
    allow_architecture_change: bool = False, review_feedback: str = "",
    max_output_tokens: int | None = None, request_timeout: float | None = None,
    api_retries: int = 3,
) -> dict[str, Any]:
    mechanism_doc = json.loads(mechanism_path.read_text(encoding="utf-8-sig"))
    validate_mechanism_document(mechanism_doc)
    baseline_profile = profile_baseline(baseline_path, game_type)
    compatibility = evaluate_compatibility(mechanism_doc, baseline_profile, transfer_mode, allow_architecture_change)
    baseline_source = baseline_path.read_text(encoding="utf-8-sig")
    raw = call_llm_with_length_retry(
        mechanism_doc, mechanism_path, baseline_path, baseline_source, baseline_profile,
        compatibility, transfer_mode, allow_architecture_change, review_feedback,
        provider, max_baseline_chars, max_output_tokens, request_timeout, api_retries,
    )
    if raw.startswith("API Call Failed"):
        raise RuntimeError(f"LLM API call failed before plan extraction: {raw}")

    last_raw = raw
    for attempt in range(repair_attempts + 1):
        try:
            plan = extract_json_object(last_raw)
            plan = ensure_required_context(
                plan, mechanism_doc, mechanism_path, baseline_path, baseline_profile,
                compatibility, transfer_mode, allow_architecture_change,
            )
            validate_implementation_plan(plan)
            return plan
        except (json.JSONDecodeError, ImplementationPlanValidationError) as exc:
            if attempt >= repair_attempts:
                raise
            last_raw = call_llm_with_api_retry(
                SYSTEM_PROMPT,
                build_repair_prompt(last_raw, str(exc)),
                provider,
                max_output_tokens,
                request_timeout,
                api_retries,
            )
            if last_raw.startswith("API Call Failed"):
                raise RuntimeError(f"LLM repair call failed: {last_raw}") from exc
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map a cooperation mechanism onto a public-goods, prisoner's-dilemma, or trust-game baseline.")
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--max-baseline-chars", type=int, default=24000)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Mapper response budget. Defaults to ZGC_MAX_TOKENS or 16384; use 0 to omit the client cap.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Seconds per ZGC request. Defaults to ZGC_REQUEST_TIMEOUT or 600; use 0 for no client timeout.",
    )
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--game-type", default="auto", choices=["auto", "public_goods", "prisoners_dilemma", "trust_game"])
    parser.add_argument("--transfer-mode", default="preserve_baseline", choices=["preserve_baseline", "extend_topology"])
    parser.add_argument("--allow-architecture-change", action="store_true")
    parser.add_argument("--feedback-file", type=Path)
    args = parser.parse_args()

    feedback = args.feedback_file.read_text(encoding="utf-8-sig") if args.feedback_file else ""
    plan = build_plan(
        args.mechanism, args.baseline, args.provider, args.repair_attempts,
        args.max_baseline_chars, args.game_type, args.transfer_mode,
        args.allow_architecture_change, feedback, args.max_output_tokens,
        args.request_timeout, args.api_retries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved implementation plan to {args.output}")
    if not plan["compatibility"]["code_implementation_allowed"]:
        print(f"Compatibility gate: {plan['compatibility']['status']} - {plan['compatibility']['reason']}")


if __name__ == "__main__":
    main()
