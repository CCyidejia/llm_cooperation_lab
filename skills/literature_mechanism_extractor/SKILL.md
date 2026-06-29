---
name: literature-mechanism-extractor
description: Extract cooperation-promoting mechanisms from public goods game literature PDFs, text notes, or literature tables into validated mechanism.json for the llm_cooperation_lab public goods baseline.
---

# Literature Mechanism Extractor

Use this skill to convert a public-goods-game paper, PDF text, or literature summary table into a structured `mechanism.json`.

The skill is scoped to the current baseline `env_main_public_goods_group.py`. It does not modify baseline code. It only extracts mechanisms and maps them to implementation surfaces.

## Workflow

1. Read the source paper or summary.
2. Extract only mechanisms that can plausibly promote cooperation.
3. Output valid JSON following `schema.py`.
4. Validate the JSON with `validate_mechanism_document`.
5. Save the result as `mechanism.json`.

## Mechanism Requirements

Each mechanism must specify mechanism type, timing, actors and targets, information structure, action-space changes, payoff changes, state variables, prompt requirements, metrics, implementation surfaces, and uncertainties.

## Baseline Mapping

Use `baseline_capabilities.json` when deciding whether a mechanism is directly implementable or requires extension.

Current baseline supports contribution-only decisions. Punishment, reward, reputation, image scoring, and institution choice usually require adding state variables, prompt context, payoff rules, and logging metrics.

## CLI

Text or table-summary input:

```powershell
python -m skills.literature_mechanism_extractor.extractor --text path\to\paper_summary.txt --output mechanisms\paper_mechanism.json
```

PDF input requires `pypdf`:

```powershell
python -m skills.literature_mechanism_extractor.extractor --pdf path\to\paper.pdf --output mechanisms\paper_mechanism.json
```
