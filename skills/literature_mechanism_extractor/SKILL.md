---
name: literature-mechanism-extractor
description: Extract cooperation mechanisms from public-goods, prisoner's-dilemma, or trust-game papers, PDFs, text notes, and treatment tables into validated cooperation-mechanism-v2 JSON. Use when identifying roles, sequential or simultaneous actions, timing, information, payoffs, treatments, network requirements, or transfer requirements before mapping code.
---

# Literature Mechanism Extractor

Extract the paper's native game and mechanism design without assuming a target baseline.

## Workflow

1. Read the source and preserve page labels where available.
2. Identify game family, topology, roles, decision stages, native actions, decision protocol, and treatments.
3. Extract every mechanism into `cooperation-mechanism-v2` format.
4. Distinguish simultaneous actions from later response stages.
5. Mark network, matching, history, and information requirements explicitly.
6. Validate with `schema.validate_mechanism_document` and save JSON.

Read [game-adapters.md](references/game-adapters.md) when the source uses a trust game, punishment, reputation, social knowledge, partner choice, or networks.

## CLI

```powershell
python -m skills.literature_mechanism_extractor.extractor --pdf "papers\paper.pdf" --output "mechanisms\paper.json"
```

For trust games, keep Trustor transfer, Trustee return, and any later Trustor sanction as separate stages. Record strategy-method elicitation separately from the game-tree sequence and record measured personality traits as heterogeneity rather than randomized treatments.

On a checker-directed rerun, pass `--feedback-file review.json` so required fixes affect the new extraction.
