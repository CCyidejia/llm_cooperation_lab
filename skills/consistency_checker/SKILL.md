# consistency_checker

Purpose: use an LLM to review whether the extracted cooperation mechanism, baseline mapping plan, source paper, and baseline code are mutually consistent under a mechanism-transfer goal, not full paper replication.

Inputs:
- Source paper PDF or pre-extracted paper text.
- `mechanism.json` from `skills/literature_mechanism_extractor`.
- `implementation_plan.json` from `skills/baseline_mapper`.
- Target baseline env file, usually `env_main_public_goods_group.py`.

Output:
- A structured `review_report.json` with pass/needs_revision/fail status, missing items, inconsistent items, baseline feasibility risks, and approval for the next code-implementation step.

Run example:

```powershell
python -m skills.consistency_checker.checker --pdf "papers\Altruistic punishment in humans.pdf" --mechanism "mechanisms\altruistic_punishment_in_humans_mechanism.json" --plan "implementation_plans\altruistic_punishment_public_goods_plan.json" --baseline "env_main_public_goods_group.py" --output "review_reports\altruistic_punishment_public_goods_review.json"
```

Provider:
- Uses `LLMAPI.zgc` by default.
- Set `ZGC_LLM_API_KEY` in the same terminal before running.
- Optional model override: `ZGC_DEFAULT_MODEL`.
- Optional output budget override: `ZGC_MAX_TOKENS`.

Contract:
- This skill does not modify baseline code.
- It is a gate before automatic code implementation. It should preserve baseline-native game settings such as agent count, endowment, multiplier/MPCR, rounds, and architecture unless the user explicitly asks to change them.
- If `approval_for_next_step.ready_for_code_implementation` is false, fix the reported mechanism or plan issues before generating code.
