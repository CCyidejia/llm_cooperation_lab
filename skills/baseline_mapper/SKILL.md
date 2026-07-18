---
name: baseline-mapper
description: Use an LLM to map a validated mechanism.json plus an env baseline file into implementation_plan.json. Use before code generation to identify target locations, required code changes, new state variables, logs, metrics, tests, and risks without modifying baseline code.
---

# Baseline Mapper

Use this skill after `literature_mechanism_extractor` has produced a valid `mechanism.json`.

This skill does not edit environment code. It creates an implementation plan for the next code-implementation skill.

The mapping is performed by an LLM. The program reads inputs, summarizes the baseline structure, calls the LLM, validates the returned JSON, and saves the plan.

## Inputs

- `mechanism.json`: structured mechanism extraction output.
- `env_main_public_goods_group.py`: public goods game baseline.

## Output

- `implementation_plan.json`: LLM-generated plan with target locations, required code changes, state variables, logs, metrics, implementation order, tests, and risks.

The plan must explicitly preserve baseline game settings and experiment settings:

- number of agents
- initial endowment
- multiplier / MPCR / payoff constants
- number of rounds
- experiment repeat count
- existing group or matching architecture

## CLI

```powershell
python -m skills.baseline_mapper.mapper --mechanism mechanisms\altruistic_punishment_in_humans_mechanism.json --baseline env_main_public_goods_group.py --output implementation_plans\altruistic_punishment_public_goods_plan.json --provider zgc --repair-attempts 1
```

## Current Scope

The mapper is mechanism-general because the mapping is delegated to an LLM, but the output is still validated against the implementation plan schema.
