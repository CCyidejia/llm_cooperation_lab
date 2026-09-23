# Game adapter distinctions

## Public goods

- Native action: contribution to a group account.
- Native payoff: retained endowment plus public-pool return.
- Punishment or reward commonly occurs after contributions, but timing must come from the paper.

## Prisoner's dilemma

- Native action: cooperate or defect in a dyadic interaction.
- Native payoff: payoff-matrix lookup.
- A paper can add punishment as a third simultaneous action (C/D/P); do not convert it into a public-goods-style second stage.

## Trust game

- Native roles: Trustor (investor) and Trustee (recipient/agent).
- Native sequence: the Trustor transfers first; the transfer is multiplied; the Trustee then chooses a return. Preserve these as distinct `primary_action` and `post_action_response` stages.
- Preserve role-specific actions and payoffs. The supplied baseline uses `TrustGamePopulationAgent.act` for both roles and calculates the multiplied transfer and both payoffs in `TrustGamePopulationEnvironment.run_interaction`.
- A punishment opportunity after the Trustee's return is a third stage controlled by the Trustor. It is not simultaneous with either native action.
- Preserve NPT/PT or equivalent no-punishment/punishment treatments. Transfer evidence-grounded punishment cost and effect parameters, while retaining the target baseline's native endowment, transfer multiplier, action ranges, rounds, population, matching, and repeat count.
- Record strategy-method elicitation separately from game-tree timing: conditional choices made ex ante do not turn the sequential trust game into simultaneous play. Treat exact strategy-method replication as an experimental-protocol choice unless the mechanism explicitly requires it.
- Record measured personality traits as heterogeneity, not as randomized treatments or target-profile rewrites. In the antisocial-personality paper, punishment is the situational mechanism; antisociality moderates beliefs, trust, returns, and sanctions. Preserve the supplied trust baseline's neutral, Trustor, Trustee, and role-specific profiles exactly.

## Network compatibility

Public reputation alone may transfer without changing topology if it is globally visible. Endogenous links, rewiring, neighbor-only information, or network-conditioned social knowledge require persistent network state. A mapper must block those in `preserve_baseline` mode and allow them only under explicit topology extension authorization.

Factorial designs such as B/R/N/RN must remain separable treatments rather than being collapsed into one mechanism.
