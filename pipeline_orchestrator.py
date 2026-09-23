from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SKILL1_MODULE = "skills.literature_mechanism_extractor.extractor"
SKILL2_MODULE = "skills.baseline_mapper.mapper"
SKILL3_MODULE = "skills.consistency_checker.checker"
SKILL4_MODULE = "skills.code_implementation.implementer"


def slugify(value: str) -> str:
    value = Path(value).stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "paper"


def run_command(command: list[str], step_name: str, model: str | None = None) -> None:
    print(f"\n[pipeline] Running {step_name}")
    print("[pipeline] " + " ".join(command))
    env = os.environ.copy()
    if model:
        env["ZGC_DEFAULT_MODEL"] = model
        print(f"[pipeline] ZGC_DEFAULT_MODEL={model}")
    completed = subprocess.run(command, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {completed.returncode}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build_default_paths(args: argparse.Namespace) -> None:
    source_path = args.pdf or args.text
    source_slug = slugify(source_path)
    baseline_slug = slugify(args.baseline)

    if args.mechanism is None:
        args.mechanism = Path("mechanisms") / f"{source_slug}_mechanism.json"
    if args.plan is None:
        args.plan = Path("implementation_plans") / f"{source_slug}_{baseline_slug}_plan.json"
    if args.review is None:
        args.review = Path("review_reports") / f"{source_slug}_{baseline_slug}_review.json"
    if args.output is None:
        args.output = Path(f"{baseline_slug}_{source_slug}_implemented.py")
    if args.report is None:
        args.report = Path("implementation_logs") / f"{source_slug}_{baseline_slug}_code_report.json"
    if args.pipeline_report is None:
        args.pipeline_report = Path("pipeline_logs") / f"{source_slug}_{baseline_slug}_pipeline_report.json"


def build_skill1_command(args: argparse.Namespace, with_feedback: bool = False) -> list[str]:
    command = [sys.executable, "-m", SKILL1_MODULE]
    if args.pdf:
        command.extend(["--pdf", str(args.pdf)])
    else:
        command.extend(["--text", str(args.text)])
    command.extend(
        [
            "--output",
            str(args.mechanism),
            "--provider",
            args.provider,
            "--repair-attempts",
            str(args.skill1_repair_attempts),
            "--max-source-chars",
            str(args.max_source_chars),
        ]
    )
    if with_feedback and args.review.exists():
        command.extend(["--feedback-file", str(args.review)])
    return command


def build_skill2_command(args: argparse.Namespace, with_feedback: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        SKILL2_MODULE,
        "--mechanism",
        str(args.mechanism),
        "--baseline",
        str(args.baseline),
        "--output",
        str(args.plan),
        "--provider",
        args.provider,
        "--repair-attempts",
        str(args.skill2_repair_attempts),
        "--api-retries",
        str(args.api_retries),
        "--max-baseline-chars",
        str(args.max_baseline_chars),
        "--game-type",
        args.game_type,
        "--transfer-mode",
        args.transfer_mode,
    ]
    if args.allow_architecture_change:
        command.append("--allow-architecture-change")
    if args.max_output_tokens is not None:
        command.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if args.request_timeout is not None:
        command.extend(["--request-timeout", str(args.request_timeout)])
    if with_feedback and args.review.exists():
        command.extend(["--feedback-file", str(args.review)])
    return command


def build_skill3_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", SKILL3_MODULE]
    if args.pdf:
        command.extend(["--pdf", str(args.pdf)])
    else:
        command.extend(["--paper-text", str(args.text)])
    command.extend(
        [
            "--mechanism",
            str(args.mechanism),
            "--plan",
            str(args.plan),
            "--baseline",
            str(args.baseline),
            "--output",
            str(args.review),
            "--provider",
            args.provider,
            "--repair-attempts",
            str(args.skill3_repair_attempts),
            "--max-paper-chars",
            str(args.max_paper_chars),
            "--max-baseline-chars",
            str(args.max_baseline_chars),
        ]
    )
    return command


def build_skill4_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        SKILL4_MODULE,
        "--plan",
        str(args.plan),
        "--review",
        str(args.review),
        "--baseline",
        str(args.baseline),
        "--output",
        str(args.output),
        "--report",
        str(args.report),
        "--provider",
        args.provider,
        "--repair-attempts",
        str(args.skill4_repair_attempts),
        "--api-retries",
        str(args.api_retries),
    ]


def review_allows_code_implementation(review: dict[str, Any]) -> bool:
    approval = review.get("approval_for_next_step", {})
    return bool(approval.get("ready_for_code_implementation"))


def choose_rerun_target(review: dict[str, Any]) -> str:
    required_fixes = review.get("required_fixes", [])
    if not required_fixes:
        return "skill2"

    targets = {
        str(item.get("target", "")).strip().lower()
        for item in required_fixes
        if isinstance(item, dict)
    }
    if targets & {"paper", "mechanism"}:
        return "skill1"
    if targets & {"plan", "baseline", "pipeline"}:
        return "skill2"
    return "skill2"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    build_default_paths(args)
    ensure_parent_dirs(args.mechanism, args.plan, args.review, args.output, args.report, args.pipeline_report)

    events: list[dict[str, Any]] = []

    def record(step: str, status: str, detail: str = "") -> None:
        events.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )

    run_command(build_skill1_command(args), "Skill 1 literature-mechanism-extractor", args.reasoning_model)
    record("skill1", "completed", str(args.mechanism))

    run_command(build_skill2_command(args), "Skill 2 baseline-mapper", args.reasoning_model)
    record("skill2", "completed", str(args.plan))

    initial_plan = read_json(args.plan)
    compatibility = initial_plan.get("compatibility", {})
    if compatibility and not compatibility.get("code_implementation_allowed", False):
        reason = compatibility.get("reason", "Compatibility gate blocked this transfer.")
        record("compatibility_gate", compatibility.get("status", "blocked"), reason)
        pipeline_report = {
            "status": "blocked_by_compatibility_gate",
            "reason": reason,
            "paths": collect_paths(args),
            "events": events,
        }
        write_json(args.pipeline_report, pipeline_report)
        return pipeline_report

    approved = False
    review: dict[str, Any] = {}
    for attempt in range(1, args.max_review_cycles + 1):
        run_command(build_skill3_command(args), f"Skill 3 consistency-checker cycle {attempt}", args.reasoning_model)
        review = read_json(args.review)
        approved = review_allows_code_implementation(review)
        record("skill3", "approved" if approved else "needs_revision", f"cycle={attempt}")

        if approved:
            break

        if attempt >= args.max_review_cycles:
            break

        rerun_target = choose_rerun_target(review)
        record("gate", "rerun", rerun_target)
        if rerun_target == "skill1":
            run_command(build_skill1_command(args, with_feedback=True), f"Skill 1 rerun after review cycle {attempt}", args.reasoning_model)
            record("skill1", "rerun_completed", str(args.mechanism))
            run_command(build_skill2_command(args, with_feedback=True), f"Skill 2 rerun after Skill 1 cycle {attempt}", args.reasoning_model)
            record("skill2", "rerun_completed", str(args.plan))
        else:
            run_command(build_skill2_command(args, with_feedback=True), f"Skill 2 rerun after review cycle {attempt}", args.reasoning_model)
            record("skill2", "rerun_completed", str(args.plan))

    if not approved:
        reason = review.get("approval_for_next_step", {}).get("reason", "Review did not approve code implementation.")
        record("skill4", "skipped", reason)
        pipeline_report = {
            "status": "blocked_before_code_implementation",
            "reason": reason,
            "paths": collect_paths(args),
            "events": events,
        }
        write_json(args.pipeline_report, pipeline_report)
        return pipeline_report

    code_passed = False
    code_report: dict[str, Any] = {}
    for code_attempt in range(1, args.max_code_cycles + 1):
        run_command(build_skill4_command(args), f"Skill 4 code-implementation cycle {code_attempt}", args.coding_model)
        code_report = read_json(args.report)
        validation_status = str(code_report.get("validation_status", "")).lower()
        validation_issues = code_report.get("validation_issues", [])
        if validation_status == "pass":
            code_passed = True
            record("skill4", "completed", f"cycle={code_attempt}; {args.output}")
            break

        record("skill4", "validation_failed", f"cycle={code_attempt}; issues={validation_issues}")
        if code_attempt < args.max_code_cycles:
            print(f"[pipeline] Code validation failed; retrying Skill 4 ({code_attempt}/{args.max_code_cycles})")

    if not code_passed:
        reason = "Code implementation validation did not pass."
        pipeline_report = {
            "status": "blocked_after_code_implementation",
            "reason": reason,
            "code_validation_issues": code_report.get("validation_issues", []),
            "paths": collect_paths(args),
            "events": events,
        }
        write_json(args.pipeline_report, pipeline_report)
        return pipeline_report

    pipeline_report = {
        "status": "completed",
        "paths": collect_paths(args),
        "events": events,
    }
    write_json(args.pipeline_report, pipeline_report)
    return pipeline_report


def collect_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "source": str(args.pdf or args.text),
        "baseline": str(args.baseline),
        "mechanism": str(args.mechanism),
        "plan": str(args.plan),
        "review": str(args.review),
        "implemented_env": str(args.output),
        "implementation_report": str(args.report),
        "pipeline_report": str(args.pipeline_report),
        "reasoning_model": str(args.reasoning_model or ""),
        "coding_model": str(args.coding_model or ""),
        "game_type": args.game_type,
        "transfer_mode": args.transfer_mode,
        "allow_architecture_change": str(args.allow_architecture_change),
        "max_output_tokens": str(args.max_output_tokens if args.max_output_tokens is not None else ""),
        "request_timeout": str(args.request_timeout if args.request_timeout is not None else ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the literature-to-code skill pipeline with review gating."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="Source PDF paper.")
    source.add_argument("--text", type=Path, help="Source text or table-summary file.")

    parser.add_argument("--baseline", type=Path, required=True, help="Baseline env file.")
    parser.add_argument("--mechanism", type=Path, help="Output mechanism.json path.")
    parser.add_argument("--plan", type=Path, help="Output implementation_plan.json path.")
    parser.add_argument("--review", type=Path, help="Output consistency review path.")
    parser.add_argument("--output", type=Path, help="Output implemented env file path.")
    parser.add_argument("--report", type=Path, help="Output code implementation report path.")
    parser.add_argument("--pipeline-report", type=Path, help="Output pipeline report path.")

    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--game-type", default="auto", choices=["auto", "public_goods", "prisoners_dilemma", "trust_game"])
    parser.add_argument("--transfer-mode", default="preserve_baseline", choices=["preserve_baseline", "extend_topology"])
    parser.add_argument("--allow-architecture-change", action="store_true")
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Model used for Skill 1-3. Example: gpt-5.5.",
    )
    parser.add_argument(
        "--coding-model",
        default=None,
        help="Model used for Skill 4. Example: kimi-k2.7-code.",
    )
    parser.add_argument("--max-review-cycles", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--max-paper-chars", type=int, default=12000)
    parser.add_argument("--max-baseline-chars", type=int, default=0)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="ZGC response budget for Skill 2. Defaults to ZGC_MAX_TOKENS or 16384; 0 omits the client cap.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Seconds per Skill 2 ZGC request. Defaults to ZGC_REQUEST_TIMEOUT or 600; 0 disables the client timeout.",
    )
    parser.add_argument("--skill1-repair-attempts", type=int, default=1)
    parser.add_argument("--skill2-repair-attempts", type=int, default=1)
    parser.add_argument("--skill3-repair-attempts", type=int, default=1)
    parser.add_argument("--skill4-repair-attempts", type=int, default=2)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--max-code-cycles",
        type=int,
        default=2,
        help="Maximum Skill 4 runs if generated code fails validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(args)
    print(f"\n[pipeline] Status: {report['status']}")
    print(f"[pipeline] Report: {report['paths']['pipeline_report']}")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
