from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


GAME_ADAPTERS: dict[str, dict[str, Any]] = {
    "public_goods": {
        "game_family": "public_goods",
        "interaction_topology": "group",
        "decision_stage_model": "simultaneous_group",
        "role_hints": ["group member"],
        "agent_class_hints": ["PublicGoodsAgent"],
        "environment_class_hints": ["PublicGoodsEnvironment"],
        "decision_method_hints": ["choose_action"],
        "round_method_hints": ["run_interaction"],
        "payoff_function_hints": [],
        "payoff_method_hints": ["run_interaction"],
        "native_actions": ["integer contribution"],
        "native_decision_stages": [
            {
                "stage_id": "contribution",
                "role": "group member",
                "stage_type": "primary_action",
                "observes": [],
            },
        ],
        "core_invariant_snippets": [
            '"num_agents": 24',
            '"initial_endowment": 20',
            '"public_pool_multiplier": 9.6',
            '"num_rounds": 30',
        ],
        "allowed_mechanism_surfaces": [
            "prompt_context",
            "action_space",
            "payoff_function",
            "state_memory",
            "matching_or_grouping",
            "network_topology",
            "logging_metrics",
            "experiment_config",
        ],
    },
    "prisoners_dilemma": {
        "game_family": "prisoners_dilemma",
        "interaction_topology": "dyadic_random_matching",
        "decision_stage_model": "simultaneous_dyadic",
        "role_hints": ["player"],
        "agent_class_hints": ["PrisonersDilemmaAgent"],
        "environment_class_hints": ["PrisonersDilemmaEnvironment"],
        "decision_method_hints": ["make_decision"],
        "round_method_hints": ["run_interaction"],
        "payoff_function_hints": ["get_payoff"],
        "payoff_method_hints": [],
        "native_actions": ["Yes", "No"],
        "native_decision_stages": [
            {
                "stage_id": "cooperate_or_defect",
                "role": "player",
                "stage_type": "primary_action",
                "observes": [],
            },
        ],
        "core_invariant_snippets": [
            '("Yes", "Yes"): (3, 3)',
            '("Yes", "No"): (0, 5)',
            '("No", "Yes"): (5, 0)',
            '("No", "No"): (1, 1)',
            "NUM_AGENTS = 24",
            "TOTAL_POPULATION_ROUNDS = 10",
            "REPEAT_TIMES = 3",
        ],
        "allowed_mechanism_surfaces": [
            "prompt_context",
            "action_space",
            "payoff_function",
            "state_memory",
            "matching_or_grouping",
            "network_topology",
            "logging_metrics",
            "experiment_config",
        ],
    },
    "trust_game": {
        "game_family": "trust_game",
        "interaction_topology": "dyadic_random_matching",
        "decision_stage_model": "sequential_dyadic",
        "agent_profile_policy": "preserve_exact",
        "agent_profile_invariant_specs": [
            {"kind": "assignment", "scope": "TrustGamePopulationAgent.act", "name": "role_profile"},
            {"kind": "assignment", "scope": "_create_agents", "name": "trustor_profile"},
            {"kind": "assignment", "scope": "_create_agents", "name": "trustee_profile"},
            {"kind": "assignment", "scope": "_create_agents", "name": "neutral_profile"},
            {
                "kind": "call_keyword",
                "scope": "_create_agents",
                "call": "TrustGamePopulationAgent",
                "keyword": "profile",
            },
            {"kind": "attribute_assignment", "scope": "<module>", "attribute": "_profile"},
        ],
        "role_hints": ["Trustor", "Trustee"],
        "agent_class_hints": ["TrustGamePopulationAgent"],
        "environment_class_hints": ["TrustGamePopulationEnvironment"],
        "decision_method_hints": ["act"],
        "round_method_hints": ["run_interaction"],
        "payoff_function_hints": [],
        "payoff_method_hints": ["run_interaction"],
        "native_actions": ["Trustor integer transfer", "Trustee integer return"],
        "native_decision_stages": [
            {
                "stage_id": "trustor_transfer",
                "role": "Trustor",
                "stage_type": "primary_action",
                "observes": [],
            },
            {
                "stage_id": "trustee_return",
                "role": "Trustee",
                "stage_type": "post_action_response",
                "observes": ["Trustor transfer", "multiplied transfer"],
            },
        ],
        "core_invariant_snippets": [
            '"num_agents": 24',
            '"initial_funds": 10',
            '"multiplication_factor": 3',
            '"num_rounds": 10',
            "NUM_GAMES = 3",
            "trustee_received = trustor_amount * self.multiplication_factor",
            "trustor_payoff = self.initial_funds - trustor_amount + trustee_amount",
            "trustee_payoff = trustee_received - trustee_amount",
        ],
        "allowed_mechanism_surfaces": [
            "prompt_context",
            "action_space",
            "payoff_function",
            "state_memory",
            "matching_or_grouping",
            "network_topology",
            "logging_metrics",
            "experiment_config",
        ],
    },
}


GAME_TYPE_ALIASES = {
    "public_goods": "public_goods",
    "public_goods_game": "public_goods",
    "repeated_public_goods_game": "public_goods",
    "prisoners_dilemma": "prisoners_dilemma",
    "prisoner's_dilemma": "prisoners_dilemma",
    "prisoner_dilemma": "prisoners_dilemma",
    "networked_prisoners_dilemma": "prisoners_dilemma",
    "trust_game": "trust_game",
    "trust_game_population": "trust_game",
    "sequential_trust_game": "trust_game",
    "investment_game": "trust_game",
}


def normalize_game_type(value: str | None) -> str:
    normalized = re.sub(r"[^a-z_']+", "_", (value or "").strip().lower()).strip("_")
    return GAME_TYPE_ALIASES.get(normalized, normalized)


def _class_roles(tree: ast.Module) -> tuple[list[str], list[str]]:
    agents: list[str] = []
    environments: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        lower = node.name.lower()
        if lower.endswith("agent") or "agent" in lower:
            agents.append(node.name)
        if lower.endswith("environment") or "environment" in lower:
            environments.append(node.name)
    return agents, environments


def _profile_scopes(tree: ast.Module) -> dict[str, ast.AST]:
    scopes: dict[str, ast.AST] = {"<module>": tree}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scopes[f"{node.name}.{item.name}"] = item
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes[node.name] = node
    return scopes


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def collect_agent_profile_invariants(
    source: str,
    specs: list[dict[str, str]],
    *,
    require_matches: bool = False,
) -> list[dict[str, Any]]:
    """Fingerprint profile definitions and bindings without depending on formatting."""
    tree = ast.parse(source)
    scopes = _profile_scopes(tree)
    invariants: list[dict[str, Any]] = []

    for spec in specs:
        scope_name = spec.get("scope", "")
        scope = scopes.get(scope_name)
        if scope is None:
            if require_matches:
                raise ValueError(f"Missing agent-profile scope: {scope_name}")
            matches: list[dict[str, str]] = []
        else:
            matches = []
            kind = spec.get("kind")
            for node in ast.walk(scope):
                if kind == "assignment" and isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(target, ast.Name) and target.id == spec.get("name") for target in targets):
                        matches.append({"value_ast": ast.dump(node.value, include_attributes=False)})
                elif kind == "attribute_assignment" and isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute) and target.attr == spec.get("attribute"):
                            matches.append({
                                "target_ast": ast.dump(target, include_attributes=False),
                                "value_ast": ast.dump(node.value, include_attributes=False),
                            })
                elif kind == "call_keyword" and isinstance(node, ast.Call):
                    if _call_name(node.func) != spec.get("call"):
                        continue
                    for keyword in node.keywords:
                        if keyword.arg == spec.get("keyword"):
                            matches.append({"value_ast": ast.dump(keyword.value, include_attributes=False)})

        if require_matches and not matches:
            raise ValueError(f"Agent-profile invariant did not match baseline source: {spec}")
        invariants.append({"spec": dict(spec), "matches": matches})

    return invariants


def detect_game_type(source: str, path: str | Path = "") -> str:
    haystack = f"{path}\n{source[:12000]}".lower()
    if (
        "trustgame" in haystack
        or "trust_game" in haystack
        or ("trust game" in haystack and "trustor" in haystack and "trustee" in haystack)
    ):
        return "trust_game"
    if "prisonersdilemma" in haystack or "prisoner's dilemma" in haystack or "payoff_matrix" in haystack:
        return "prisoners_dilemma"
    if "publicgoods" in haystack or "public goods" in haystack or "public_pool_multiplier" in haystack:
        return "public_goods"
    raise ValueError(f"Unable to detect supported game type for {path or '<source>'}")


def profile_baseline(path: Path, requested_game_type: str = "auto") -> dict[str, Any]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    game_type = (
        detect_game_type(source, path)
        if requested_game_type == "auto"
        else normalize_game_type(requested_game_type)
    )
    if game_type not in GAME_ADAPTERS:
        raise ValueError(f"Unsupported game type: {game_type}")

    adapter = dict(GAME_ADAPTERS[game_type])
    agent_classes, environment_classes = _class_roles(tree)
    classes: dict[str, dict[str, Any]] = {}
    functions: dict[str, int] = {}
    assignments: dict[str, int] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {
                "line": node.lineno,
                "methods": {
                    item.name: item.lineno
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                },
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node.lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.lineno

    decision_terms = ("choose", "action", "decision", "decide", "make_decision")
    decision_candidates = []
    state_candidates = []
    logging_candidates = []
    round_candidates = []
    payoff_candidates = []
    decision_hints = set(adapter.get("decision_method_hints", []))
    round_hints = set(adapter.get("round_method_hints", []))
    payoff_method_hints = set(adapter.get("payoff_method_hints", []))
    for class_name, info in classes.items():
        for method_name in info["methods"]:
            location = f"{class_name}.{method_name}"
            lower = method_name.lower()
            if class_name in agent_classes and (
                method_name in decision_hints or any(term in lower for term in decision_terms)
            ):
                decision_candidates.append(location)
            if "update" in lower or "history" in lower or "memory" in lower or "state" in lower:
                state_candidates.append(location)
            if "summary" in lower or "log" in lower or "save" in lower:
                logging_candidates.append(location)
            if class_name in environment_classes and (
                method_name in round_hints or "run" in lower or "interaction" in lower or "round" in lower
            ):
                round_candidates.append(location)
            if method_name in payoff_method_hints:
                payoff_candidates.append(location)

    payoff_function_hints = set(adapter.get("payoff_function_hints", []))
    payoff_candidates.extend(name for name in functions if name in payoff_function_hints)

    profile_specs = adapter.get("agent_profile_invariant_specs", [])
    profile_invariants = collect_agent_profile_invariants(
        source, profile_specs, require_matches=True,
    ) if profile_specs else []

    adapter.update(
        {
            "baseline_file": str(path),
            "classes": classes,
            "functions": functions,
            "module_assignments": assignments,
            "detected_agent_classes": agent_classes,
            "detected_environment_classes": environment_classes,
            "decision_method_candidates": sorted(decision_candidates),
            "environment_round_method_candidates": sorted(round_candidates),
            "payoff_location_candidates": sorted(payoff_candidates),
            "state_update_candidates": sorted(state_candidates),
            "summary_or_logging_candidates": sorted(logging_candidates),
            "available_core_invariants": [
                snippet for snippet in adapter["core_invariant_snippets"] if snippet in source
            ],
            "agent_profile_invariants": profile_invariants,
        }
    )
    return adapter


def mechanism_requires_network(mechanism_doc: dict[str, Any]) -> bool:
    game_context = mechanism_doc.get("game_context", {})
    topology = str(game_context.get("topology", game_context.get("interaction_topology", ""))).lower()
    if topology in {"network", "fixed_network", "dynamic_network"}:
        return True
    for mechanism in mechanism_doc.get("mechanisms", []):
        requirements = mechanism.get("transfer_requirements", {})
        if requirements.get("requires_network") or requirements.get("requires_matching_change"):
            return True
        if mechanism.get("type") in {"partner_selection", "network_formation"}:
            return True
    return False


def evaluate_compatibility(
    mechanism_doc: dict[str, Any],
    baseline_profile: dict[str, Any],
    transfer_mode: str,
    allow_architecture_change: bool,
) -> dict[str, Any]:
    requires_network = mechanism_requires_network(mechanism_doc)
    target_has_network = baseline_profile.get("interaction_topology") in {"network", "fixed_network", "dynamic_network"}
    if not requires_network or target_has_network:
        return {
            "status": "compatible",
            "code_implementation_allowed": True,
            "requires_network": requires_network,
            "requires_architecture_change": False,
            "reason": "The target baseline exposes all essential mechanism surfaces.",
        }
    if transfer_mode != "extend_topology":
        return {
            "status": "blocked",
            "code_implementation_allowed": False,
            "requires_network": True,
            "requires_architecture_change": True,
            "reason": (
                "The paper mechanism requires a persistent interaction network, but the target baseline "
                "has no network. Use transfer_mode=extend_topology and explicitly allow architecture changes, "
                "or transfer only a non-network reputation treatment."
            ),
        }
    if not allow_architecture_change:
        return {
            "status": "requires_approval",
            "code_implementation_allowed": False,
            "requires_network": True,
            "requires_architecture_change": True,
            "reason": "Topology extension was requested but matching/group architecture changes were not approved.",
        }
    return {
        "status": "compatible_with_extension",
        "code_implementation_allowed": True,
        "requires_network": True,
        "requires_architecture_change": True,
        "reason": "The required network topology extension was explicitly authorized.",
    }
