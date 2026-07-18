# code_implementation

Use this skill to turn an approved `implementation_plan.json` into a new mechanism-specific env file with an LLM coding agent plus deterministic program checks.

Inputs:
- `implementation_plan.json` from `skills/baseline_mapper`.
- Source baseline env file.
- Optional `review_report.json` from `skills/consistency_checker`; when provided, it must approve code implementation.

Output:
- A new env file. The source baseline is never overwritten.
- A JSON implementation report recording applied edits, preserved baseline hash, syntax status, validation issues, and manual-review notes.

Run example:

```powershell
python -m skills.code_implementation.implementer --plan "implementation_plans\altruistic_punishment_public_goods_plan.json" --review "review_reports\altruistic_punishment_public_goods_review.json" --baseline "env_main_public_goods_group.py" --output "env_main_public_goods_group_altruistic_punishment.py" --report "implementation_logs\altruistic_punishment_public_goods_code_report.json"
```

Provider:
- Uses `LLMAPI.zgc` by default.
- Set `ZGC_LLM_API_KEY` in the same terminal before running.
- Optional model override: `ZGC_DEFAULT_MODEL`.
- Optional output budget override: `ZGC_MAX_TOKENS`.

Contract:
- Use the LLM only to generate structured local edits.
- Apply edits by program against the baseline copy.
- Preserve baseline-native settings: agent count, initial endowment, public pool multiplier/MPCR, total rounds, seed, and existing experiment loop.
- Preserve original result fields and add mechanism fields instead of deleting old fields.
- Fail closed when syntax or required mechanism checks fail.
