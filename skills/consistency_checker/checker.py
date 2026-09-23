from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skills.literature_mechanism_extractor.extractor import extract_json_object, read_pdf_input, read_text_input

from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .review_schema import ReviewValidationError, validate_review_document


def enforce_compatibility_gate(review: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Prevent an LLM review from approving a plan blocked by deterministic topology checks."""
    compatibility = plan.get("compatibility", {})
    if compatibility.get("code_implementation_allowed", True):
        return review
    reason = str(compatibility.get("reason", "The mapper compatibility gate did not allow implementation."))
    review["overall_status"] = "fail" if compatibility.get("status") == "blocked" else "needs_revision"
    review["approval_for_next_step"] = {
        "ready_for_code_implementation": False,
        "reason": reason,
    }
    fixes = review.setdefault("required_fixes", [])
    if not any(isinstance(item, dict) and item.get("issue") == "Compatibility gate blocks implementation." for item in fixes):
        fixes.append({
            "priority": "high",
            "target": "pipeline",
            "issue": "Compatibility gate blocks implementation.",
            "suggested_fix": reason,
        })
    return review


def call_llm(system_prompt: str, user_prompt: str, provider: str) -> str:
    if provider != "zgc":
        raise ValueError(f"Unsupported provider: {provider}")
    from LLMAPI.zgc import LLMAgent

    agent = LLMAgent(name="consistency-checker")
    return agent.get_llm_response(system_prompt, user_prompt)


def trim_text(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[TRUNCATED: {label} had {len(text)} characters; first {max_chars} characters were sent to the LLM.]"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def call_llm_with_length_retry(
    paper_text: str,
    paper_name: str,
    mechanism: dict[str, Any],
    mechanism_name: str,
    plan: dict[str, Any],
    plan_name: str,
    baseline_text: str,
    baseline_name: str,
    provider: str,
    max_paper_chars: int,
    max_baseline_chars: int,
) -> str:
    paper_sizes = [max_paper_chars, max(2500, max_paper_chars // 2), 1800] if max_paper_chars > 0 else [0]
    baseline_sizes = [max_baseline_chars, max(5000, max_baseline_chars // 2), 3000] if max_baseline_chars > 0 else [0]
    attempts = list(dict.fromkeys(zip(paper_sizes, baseline_sizes)))
    last_raw = ""

    for paper_limit, baseline_limit in attempts:
        paper_excerpt = trim_text(paper_text, paper_limit, "paper text")
        baseline_excerpt = trim_text(baseline_text, baseline_limit, "baseline code")
        raw = call_llm(
            SYSTEM_PROMPT,
            build_user_prompt(
                paper_excerpt,
                paper_name,
                mechanism,
                mechanism_name,
                plan,
                plan_name,
                baseline_excerpt,
                baseline_name,
            ),
            provider,
        )
        last_raw = raw
        if not raw.startswith("API Call Failed"):
            return raw
        if "finish_reason=length" not in raw:
            return raw
        print(
            "LLM response hit length limit; retrying with shorter excerpts "
            f"(paper={paper_limit}, baseline={baseline_limit})..."
        )

    return last_raw


def check_consistency(
    paper_text: str,
    paper_name: str,
    mechanism: dict[str, Any],
    mechanism_name: str,
    plan: dict[str, Any],
    plan_name: str,
    baseline_text: str,
    baseline_name: str,
    provider: str = "zgc",
    repair_attempts: int = 1,
    max_paper_chars: int = 7000,
    max_baseline_chars: int = 12000,
) -> dict[str, Any]:
    raw = call_llm_with_length_retry(
        paper_text,
        paper_name,
        mechanism,
        mechanism_name,
        plan,
        plan_name,
        baseline_text,
        baseline_name,
        provider,
        max_paper_chars,
        max_baseline_chars,
    )
    if raw.startswith("API Call Failed"):
        raise RuntimeError(
            "LLM API call failed before consistency JSON extraction. "
            "Check provider key, model, and output token limits. "
            f"Provider response: {raw}"
        )

    last_raw = raw
    for attempt in range(repair_attempts + 1):
        try:
            data = extract_json_object(last_raw)
            data = enforce_compatibility_gate(data, plan)
            validate_review_document(data)
            return data
        except (json.JSONDecodeError, ReviewValidationError) as exc:
            if attempt >= repair_attempts:
                raise
            last_raw = call_llm(SYSTEM_PROMPT, build_repair_prompt(last_raw, str(exc)), provider)
            if last_raw.startswith("API Call Failed"):
                raise RuntimeError(f"LLM repair call failed: {last_raw}") from exc
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency across PDF, mechanism.json, implementation_plan.json, and baseline code using an LLM.")
    paper_group = parser.add_mutually_exclusive_group(required=True)
    paper_group.add_argument("--pdf", type=Path, help="Path to the source PDF paper.")
    paper_group.add_argument("--paper-text", type=Path, help="Path to pre-extracted UTF-8 paper text.")
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--max-paper-chars", type=int, default=7000, help="Maximum paper text characters sent to the LLM; use 0 for no limit.")
    parser.add_argument("--max-baseline-chars", type=int, default=12000, help="Maximum baseline code characters sent to the LLM; use 0 for no limit.")
    args = parser.parse_args()

    if args.pdf:
        paper_path = args.pdf
        paper_text = read_pdf_input(paper_path)
    else:
        paper_path = args.paper_text
        paper_text = read_text_input(paper_path)
    if not paper_text.strip():
        raise ValueError(f"No paper text extracted from {paper_path}")

    mechanism = load_json(args.mechanism)
    plan = load_json(args.plan)
    baseline_text = args.baseline.read_text(encoding="utf-8-sig")

    review = check_consistency(
        paper_text=paper_text,
        paper_name=paper_path.name,
        mechanism=mechanism,
        mechanism_name=str(args.mechanism),
        plan=plan,
        plan_name=str(args.plan),
        baseline_text=baseline_text,
        baseline_name=str(args.baseline),
        provider=args.provider,
        repair_attempts=args.repair_attempts,
        max_paper_chars=args.max_paper_chars,
        max_baseline_chars=args.max_baseline_chars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved consistency review to {args.output}")


if __name__ == "__main__":
    main()
