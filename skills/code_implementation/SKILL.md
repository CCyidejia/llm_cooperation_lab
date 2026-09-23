---
name: code-implementation
description: Implement an approved implementation-plan-v2 into a new public-goods, prisoner's-dilemma, or trust-game environment file using validated structured Python edits. Use only after compatibility and consistency approval.
---

# Code Implementation

Generate and apply structured edits to a copy of the baseline. Supported targets include class methods, module functions, and module assignments. Class names and decision methods come from the selected game adapter rather than fixed public-goods names.

For trust games, preserve the Trustor-transfer -> Trustee-return sequence, role-specific bounds and histories, transfer multiplier, both base payoff formulas, and all baseline agent profiles exactly. Add any approved later sanction as a separate Trustor response and keep mechanism context separate from protected profiles.

Fail closed when the review is not approved, compatibility blocks implementation, the detected baseline game differs from the plan, a required change lacks an edit, syntax fails, a baseline invariant disappears, or an agent-profile AST fingerprint changes.

```powershell
python -m skills.code_implementation.implementer --plan implementation_plans\paper.json --review review_reports\paper.json --baseline env_main_trust_game_group.py --output generated\paper_trust.py --report implementation_logs\paper.json
```
