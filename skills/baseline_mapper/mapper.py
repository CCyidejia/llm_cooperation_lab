from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from skills.literature_mechanism_extractor.extractor import extract_json_object

from .plan_schema import ImplementationPlanValidationError, validate_implementation_plan


SYSTEM_PROMPT = """You are the baseline-mapper skill for an automated research-agent pipeline.

Your task is to map a structured cooperation mechanism JSON onto an existing game baseline source file.
You must output one valid JSON object only. Do not output markdown, comments, or prose outside JSON.

Important research constraint:
- Transfer only the cooperation-promoting mechanism.
- The baseline game's own settings must remain unchanged.
- Do NOT change the baseline number of agents, initial endowment, multiplier/MPCR/payoff constants, number of rounds, experiment repeat count, matching/group structure, or other baseline game parameters unless the baseline already exposes them as treatment toggles.
- Paper experiment settings are evidence/context, not replacement values for the baseline.

The output is an implementation plan for a later coding-agent skill. It must not contain code patches.
"""


PLAN_SCHEMA_INSTRUCTIONS = """Return a JSON object with this shape:
{
  "schema_version": "implementation-plan-v1",
  "target_baseline": "<baseline file path>",
  "mechanism_source": "<mechanism file path>",
  "baseline_summary": {
    "file": "<baseline file path>",
    "classes": {},
    "functions": {},
    "detected_capabilities": {}
  },
  "baseline_invariants": {
    "preserve_num_agents": true,
    "preserve_initial_endowment": true,
    "preserve_public_pool_multiplier": true,
    "preserve_total_rounds": true,
    "preserve_experiment_repeat_count": true,
    "preserve_existing_group_or_matching_architecture": true,
    "rule": "Keep all baseline game and experiment settings unchanged; add only the mechanism."
  },
  "mechanism_summary": {
    "mechanism_id": "",
    "name": "",
    "type": "",
    "summary": ""
  },
  "required_code_changes": [
    {
      "change_id": "C1",
      "surface": "action_space | prompt_context | payoff_function | state_memory | matching_or_grouping | logging_metrics | experiment_config",
      "target_location": "ClassName.method_name or module-level location",
      "operation": "short operation name",
      "description": "what to change",
      "details": {
        "specific implementation guidance": "include enough detail for the coding skill"
      },
      "depends_on": [],
      "acceptance_criteria": []
    }
  ],
  "new_state_variables": [],
  "new_logs": [],
  "new_metrics": [],
  "implementation_order": [],
  "tests_or_checks": [],
  "risks": [],
  "out_of_scope": []
}

Rules:
- required_code_changes must be concrete and point to actual baseline classes/methods when possible.
- Include required new methods if the mechanism needs a new decision stage.
- Describe how the payoff function changes while preserving the baseline payoff formula before mechanism effects.
- Include new logs and metrics needed to analyze cooperation and the mechanism.
- Include implementation order.
- Include tests/checks.
- Include risks and assumptions.
- Include out_of_scope items, especially unchanged baseline game parameters.
"""


class BaselineVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: dict[str, dict[str, Any]] = {}
        self.functions: dict[str, int] = {}
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        previous = self.current_class
        self.current_class = node.name
        self.classes[node.name] = {"line": node.lineno, "methods": {}}
        self.generic_visit(node)
        self.current_class = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._record_function(node)
        self.generic_visit(node)

    def _record_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "")
        lineno = getattr(node, "lineno", None)
        if self.current_class:
            self.classes[self.current_class]["methods"][name] = lineno
        else:
            self.functions[name] = lineno


def _locations(visitor: BaselineVisitor) -> set[str]:
    locations = set(visitor.functions)
    for class_name, info in visitor.classes.items():
        for method in info.get("methods", {}):
            locations.add(f"{class_name}.{method}")
    return locations


def analyze_baseline(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    visitor = BaselineVisitor()
    visitor.visit(tree)
    locations = _locations(visitor)
    return {
        "file": str(path),
        "classes": visitor.classes,
        "functions": visitor.functions,
        "detected_capabilities": {
            "agent_decision_method_candidates": sorted(
                loc for loc in locations if "Agent." in loc and ("choose" in loc or "action" in loc)
            ),
            "environment_round_method_candidates": sorted(
                loc for loc in locations if "Environment." in loc and ("run" in loc or "interaction" in loc)
            ),
            "state_update_candidates": sorted(loc for loc in locations if "update" in loc or "history" in loc),
            "summary_or_logging_candidates": sorted(loc for loc in locations if "summary" in loc or "log" in loc),
        },
    }


def trim_text(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[TRUNCATED: {label} had {len(text)} characters; first {max_chars} characters were sent to the LLM.]"


def call_llm(system_prompt: str, user_prompt: str, provider: str) -> str:
    if provider != "zgc":
        raise ValueError(f"Unsupported provider: {provider}")
    from LLMAPI.zgc import LLMAgent

    agent = LLMAgent(name="baseline-mapper")
    return agent.get_llm_response(system_prompt, user_prompt)


def build_user_prompt(
    mechanism_doc: dict[str, Any],
    mechanism_path: Path,
    baseline_path: Path,
    baseline_text: str,
    baseline_summary: dict[str, Any],
) -> str:
    return f"""{PLAN_SCHEMA_INSTRUCTIONS}

INPUT 1: mechanism.json path
{mechanism_path}

INPUT 2: mechanism.json
{json.dumps(mechanism_doc, ensure_ascii=False, indent=2)}

INPUT 3: baseline code path
{baseline_path}

INPUT 4: baseline AST/code structure summary
{json.dumps(baseline_summary, ensure_ascii=False, indent=2)}

INPUT 5: baseline source code
```python
{baseline_text}
```

Produce implementation_plans/altruistic_punishment_public_goods_plan.json style output.
Again: keep the baseline game's own settings and experiment rounds unchanged.
"""


def build_repair_prompt(previous_output: str, validation_error: str) -> str:
    return f"""{PLAN_SCHEMA_INSTRUCTIONS}

The previous baseline-mapper output was invalid.

Validation error:
{validation_error}

Previous output:
{previous_output}

Return a corrected JSON object only.
"""


def ensure_required_context(
    plan: dict[str, Any],
    mechanism_path: Path,
    baseline_path: Path,
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    plan.setdefault("schema_version", "implementation-plan-v1")
    plan["target_baseline"] = str(baseline_path)
    plan["mechanism_source"] = str(mechanism_path)
    plan.setdefault("baseline_summary", baseline_summary)
    plan.setdefault(
        "baseline_invariants",
        {
            "preserve_num_agents": True,
            "preserve_initial_endowment": True,
            "preserve_public_pool_multiplier": True,
            "preserve_total_rounds": True,
            "preserve_experiment_repeat_count": True,
            "preserve_existing_group_or_matching_architecture": True,
            "rule": "Keep all baseline game and experiment settings unchanged; add only the mechanism.",
        },
    )
    plan.setdefault("out_of_scope", [])
    invariant_note = (
        "Changing baseline number of agents, initial endowment, multiplier/MPCR/payoff constants, "
        "number of rounds, experiment repeat count, or group/matching architecture."
    )
    if invariant_note not in plan["out_of_scope"]:
        plan["out_of_scope"].append(invariant_note)
    return plan


def build_plan(
    mechanism_path: Path,
    baseline_path: Path,
    provider: str = "zgc",
    repair_attempts: int = 1,
    max_baseline_chars: int = 24000,
) -> dict[str, Any]:
    mechanism_doc = json.loads(mechanism_path.read_text(encoding="utf-8-sig"))
    baseline_text = trim_text(baseline_path.read_text(encoding="utf-8-sig"), max_baseline_chars, "baseline code")
    baseline_summary = analyze_baseline(baseline_path)

    raw = call_llm(
        SYSTEM_PROMPT,
        build_user_prompt(mechanism_doc, mechanism_path, baseline_path, baseline_text, baseline_summary),
        provider,
    )
    if raw.startswith("API Call Failed"):
        raise RuntimeError(
            "LLM API call failed before implementation plan JSON extraction. "
            "Check provider API key, model availability, and output token limits. "
            f"Provider response: {raw}"
        )

    last_raw = raw
    for attempt in range(repair_attempts + 1):
        try:
            plan = extract_json_object(last_raw)
            plan = ensure_required_context(plan, mechanism_path, baseline_path, baseline_summary)
            validate_implementation_plan(plan)
            return plan
        except (json.JSONDecodeError, ImplementationPlanValidationError) as exc:
            if attempt >= repair_attempts:
                raise
            last_raw = call_llm(SYSTEM_PROMPT, build_repair_prompt(last_raw, str(exc)), provider)
            if last_raw.startswith("API Call Failed"):
                raise RuntimeError(f"LLM repair call failed: {last_raw}") from exc

    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM to map mechanism.json plus a baseline env file into implementation_plan.json.")
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--max-baseline-chars", type=int, default=24000, help="Maximum baseline source characters sent to the LLM; use 0 for no limit.")
    args = parser.parse_args()

    plan = build_plan(
        mechanism_path=args.mechanism,
        baseline_path=args.baseline,
        provider=args.provider,
        repair_attempts=args.repair_attempts,
        max_baseline_chars=args.max_baseline_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved implementation plan to {args.output}")


if __name__ == "__main__":
    main()
