---
name: consistency-checker
description: Review a paper, cooperation-mechanism-v2 extraction, implementation-plan-v2, and public-goods, prisoner's-dilemma, or trust-game baseline for mechanism-transfer consistency. Use as a fail-closed gate before code implementation.
---

# Consistency Checker

Review paper-to-mechanism fidelity, treatment completeness, mechanism-to-plan coverage, actual code targets, baseline invariants, stage timing, and topology compatibility.

Never approve a plan whose deterministic compatibility result is `blocked` or `requires_approval`. Treat simultaneous C/D/P as one decision stage. Require explicit network extension for neighbor, link, or rewiring mechanisms.

For trust games, require transfer-before-return timing, preserved native payoff formulas, and exact preservation of all baseline agent profiles. Treat punishment after return as a separate Trustor response, and keep personality heterogeneity or mechanism context out of protected profiles.

```powershell
python -m skills.consistency_checker.checker --pdf papers\paper.pdf --mechanism mechanisms\paper.json --plan implementation_plans\paper.json --baseline env_main_trust_game_group.py --output review_reports\paper.json
```
