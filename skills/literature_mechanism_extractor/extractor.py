from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .schema import MechanismValidationError, validate_mechanism_document


def read_text_input(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def read_pdf_input(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf. Install it or pass a text/table summary with --text.") from exc

    reader = pypdf.PdfReader(str(path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[Page {idx}]\n{text}")
    return "\n".join(pages).strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def call_llm(system_prompt: str, user_prompt: str, provider: str) -> str:
    if provider != "zgc":
        raise ValueError(f"Unsupported provider: {provider}")
    from LLMAPI.zgc import LLMAgent

    agent = LLMAgent(name="literature-mechanism-extractor")
    return agent.get_llm_response(system_prompt, user_prompt)


def prepare_source_text(source_text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(source_text) <= max_chars:
        return source_text
    return (
        source_text[:max_chars]
        + f"\n\n[TRUNCATED: source text had {len(source_text)} characters; first {max_chars} characters were used for this extraction pass.]"
    )


def call_llm_with_length_retry(
    source_text: str,
    source_name: str,
    provider: str,
    max_source_chars: int,
) -> str:
    if max_source_chars <= 0:
        retry_sizes = [0]
    else:
        retry_sizes = []
        size = max_source_chars
        while size >= 2500:
            retry_sizes.append(size)
            size = size // 2
        if 1800 not in retry_sizes:
            retry_sizes.append(1800)

    last_raw = ""
    for size in retry_sizes:
        prepared_text = prepare_source_text(source_text, size)
        raw = call_llm(SYSTEM_PROMPT, build_user_prompt(prepared_text, source_name), provider)
        last_raw = raw
        if not raw.startswith("API Call Failed"):
            return raw
        if "finish_reason=length" not in raw:
            return raw
        print(f"LLM response hit length limit with max_source_chars={size}; retrying with shorter source text...")

    return last_raw


def extract_mechanism(
    source_text: str,
    source_name: str,
    provider: str = "zgc",
    repair_attempts: int = 1,
    max_source_chars: int = 8000,
) -> dict[str, Any]:
    raw = call_llm_with_length_retry(source_text, source_name, provider, max_source_chars)
    if raw.startswith("API Call Failed"):
        raise RuntimeError(
            "LLM API call failed before JSON extraction. "
            "Check model output limits, provider settings, and API key/environment variables. "
            f"Provider response: {raw}"
        )

    last_raw = raw
    for attempt in range(repair_attempts + 1):
        try:
            data = extract_json_object(last_raw)
            validate_mechanism_document(data)
            return data
        except (json.JSONDecodeError, MechanismValidationError) as exc:
            if attempt >= repair_attempts:
                raise
            last_raw = call_llm(SYSTEM_PROMPT, build_repair_prompt(last_raw, str(exc)), provider)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract public-goods cooperation mechanisms into mechanism.json.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=Path, help="Path to a UTF-8 text or table-summary file.")
    group.add_argument("--pdf", type=Path, help="Path to a PDF file.")
    parser.add_argument("--output", type=Path, default=Path("mechanism.json"))
    parser.add_argument("--provider", default="zgc", choices=["zgc"])
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=8000,
        help="Maximum extracted source characters sent to the LLM; use 0 for no limit.",
    )
    args = parser.parse_args()

    if args.text:
        source_path = args.text
        source_text = read_text_input(source_path)
    else:
        source_path = args.pdf
        source_text = read_pdf_input(source_path)

    if not source_text.strip():
        raise ValueError(f"No text extracted from {source_path}")

    data = extract_mechanism(source_text, source_path.name, args.provider, args.repair_attempts, args.max_source_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved mechanism JSON to {args.output}")


if __name__ == "__main__":
    main()
