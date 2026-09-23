---
name: baseline-mapper
description: Map validated cooperation-mechanism-v2 JSON onto a public-goods, prisoner's-dilemma, or trust-game Python baseline. Use to detect the game adapter, preserve baseline invariants and sequential role order, gate network/topology extensions, and produce a concrete implementation-plan-v2 without editing code.
---

# Baseline Mapper

Use after mechanism extraction and before consistency checking or code generation.

## Workflow

1. Validate the mechanism document.
2. Detect `public_goods`, `prisoners_dilemma`, or `trust_game` from the baseline AST.
3. Profile actual agent, environment, decision, round, function, and assignment locations.
4. Evaluate topology compatibility deterministically.
5. Ask the mapper model for concrete edits while preserving baseline game parameters.
6. Overwrite model-supplied adapter and compatibility fields with deterministic results.
7. Validate and save `implementation-plan-v2` JSON.

For trust games, preserve transfer-before-return timing, the native Trustor/Trustee payoff formulas, and every baseline agent profile exactly. Map a punishment opportunity after return to a distinct Trustor response stage; do not replace baseline-native parameters or profiles with paper settings.

Use a 16,384-token ZGC response budget by default. If the provider truncates mapper output, retry automatically with progressively shorter AST-profiled baseline excerpts. Retry transient timeouts, connection failures, rate limits, and 5xx responses without shrinking the excerpt. Keep exact agent-profile fingerprints host-side and restore them deterministically after mapping.

Set `--max-output-tokens` to override the response budget. A value of `0` omits the client-side `max_tokens` field, but it cannot remove the provider's model/context limits. Set `--request-timeout` to override the default 600-second request timeout; use `0` only when an indefinitely waiting request is explicitly acceptable.

The default `preserve_baseline` mode blocks network-dependent mechanisms on a non-network baseline. Use `--transfer-mode extend_topology --allow-architecture-change` only when topology expansion is explicitly intended.

```powershell
python -m skills.baseline_mapper.mapper --mechanism mechanisms\paper.json --baseline env_main_trust_game_group.py --output implementation_plans\paper_trust.json --game-type auto --max-output-tokens 16384 --api-retries 3
```
