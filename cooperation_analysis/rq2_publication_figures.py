#!/usr/bin/env python3
"""Build publication figures for RQ2 without modifying source experiment data.

Inputs
------
``RQ2_result`` contains the real-mechanism treatment runs.  The no-mechanism
cooperation baselines are the retained RQ1 summary values reported in the
manuscript table; matched baseline payoff logs are not available for every
model.  Consequently:

* Figure 3A--E is estimable.  Forest-plot intervals use independent run means
  and treat the retained RQ1 baseline point estimate as fixed.
* Figure 4A uses game-accounting identities to express net welfare relative to
  the retained RQ1 baseline on a common, normalized social-surplus scale.
* Figure 4B (change in payoff equality) is not identified without matched RQ1
  agent-level payoff logs and is explicitly marked unavailable.
* Figure 4C is a two-stage (model/run) bootstrap of round-specific deviations
  from each model's retained overall RQ1 baseline.
* Figure 4D is not identified because no Sham/Corrupted conditions are present.

All outputs are written outside ``RQ2_result``.  Source files are opened only
for reading, and a SHA-256 manifest is emitted for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t


MODEL_ORDER = [
    "GLM-5.1",
    "GPT-5.4",
    "Gemini-3.1-Pro-Preview",
    "Claude-Opus-4-6",
    "Kimi-K2.6",
    "Grok-4.3",
]
MODEL_LABELS = {
    "GLM-5.1": "GLM-5.1",
    "GPT-5.4": "GPT-5.4",
    "Gemini-3.1-Pro-Preview": "Gemini 3.1 Pro",
    "Claude-Opus-4-6": "Claude Opus 4.6",
    "Kimi-K2.6": "Kimi K2.6",
    "Grok-4.3": "Grok 4.3",
}
MODEL_TOKENS = {
    "glm": "GLM-5.1",
    "gpt": "GPT-5.4",
    "gemini": "Gemini-3.1-Pro-Preview",
    "claude": "Claude-Opus-4-6",
    "kimi": "Kimi-K2.6",
    "grok": "Grok-4.3",
}

MECHANISM_ORDER = ["reputation", "reward", "punishment"]
MECHANISM_LABELS = {
    "reputation": "Reputation",
    "reward": "Reward",
    "punishment": "Punishment",
}
MECHANISM_COLORS = {
    "reputation": "#377EB8",
    "reward": "#2CA25F",
    "punishment": "#E6550D",
}

PANEL_ORDER = [
    ("PD", "cooperation", "A  Prisoner's dilemma: cooperation"),
    ("PGG", "contribution", "B  Public-goods game: contribution"),
    ("TG", "sender", "C  Trust game: Sender transfer"),
    ("TG", "receiver", "D  Trust game: Receiver conditional return"),
]

# Retained RQ1 no-institution means (percentage points), transcribed from the
# RQ1 summary table. They are intentionally centralized and written to CSV.
BASELINES = {
    ("PD", "cooperation"): {
        "GLM-5.1": 0.00,
        "GPT-5.4": 0.00,
        "Gemini-3.1-Pro-Preview": 4.86,
        "Claude-Opus-4-6": 100.00,
        "Kimi-K2.6": 0.00,
        "Grok-4.3": 0.00,
    },
    ("PGG", "contribution"): {
        "GLM-5.1": 0.00,
        "GPT-5.4": 0.00,
        "Gemini-3.1-Pro-Preview": 0.00,
        "Claude-Opus-4-6": 2.25,
        "Kimi-K2.6": 0.00,
        "Grok-4.3": 0.00,
    },
    ("TG", "sender"): {
        "GLM-5.1": 30.21,
        "GPT-5.4": 11.12,
        "Gemini-3.1-Pro-Preview": 27.68,
        "Claude-Opus-4-6": 27.53,
        "Kimi-K2.6": 11.73,
        "Grok-4.3": 3.70,
    },
    ("TG", "receiver"): {
        "GLM-5.1": 9.85,
        "GPT-5.4": 10.96,
        "Gemini-3.1-Pro-Preview": 26.16,
        "Claude-Opus-4-6": 37.57,
        "Kimi-K2.6": 20.69,
        "Grok-4.3": 0.74,
    },
}

# Values displayed in the supplied RQ2 table.  They are used only for an audit
# of the raw-log calculation; they are never substituted for calculated data.
EXPECTED_TREATMENT = {
    ("PD", "punishment", "cooperation"): {
        "GLM-5.1": (1.25, 0.72), "GPT-5.4": (0.00, 0.00),
        "Gemini-3.1-Pro-Preview": (9.17, 4.51), "Claude-Opus-4-6": (100.00, 0.00),
        "Kimi-K2.6": (1.39, 0.64), "Grok-4.3": (36.11, 3.15),
    },
    ("PD", "reputation", "cooperation"): {
        "GLM-5.1": (6.25, 1.91), "GPT-5.4": (0.14, 0.24),
        "Gemini-3.1-Pro-Preview": (100.00, 0.00), "Claude-Opus-4-6": (100.00, 0.00),
        "Kimi-K2.6": (1.39, 0.24), "Grok-4.3": (0.00, 0.00),
    },
    ("PD", "reward", "cooperation"): {
        "GLM-5.1": (8.33, 1.82), "GPT-5.4": (0.00, 0.00),
        "Gemini-3.1-Pro-Preview": (87.78, 1.05), "Claude-Opus-4-6": (100.00, 0.00),
        "Kimi-K2.6": (70.14, 2.68), "Grok-4.3": (3.75, 0.00),
    },
    ("PGG", "punishment", "contribution"): {
        "GLM-5.1": (98.20, 1.16), "GPT-5.4": (0.00, 0.00),
        "Gemini-3.1-Pro-Preview": (99.95, 0.08), "Claude-Opus-4-6": (75.00, 0.00),
        "Kimi-K2.6": (0.14, 0.24), "Grok-4.3": (21.73, 3.96),
    },
    ("PGG", "reputation", "contribution"): {
        "GLM-5.1": (2.07, 3.35), "GPT-5.4": (97.08, 0.69),
        "Gemini-3.1-Pro-Preview": (99.86, 0.24), "Claude-Opus-4-6": (50.00, 0.00),
        "Kimi-K2.6": (0.78, 0.41), "Grok-4.3": (0.00, 0.00),
    },
    ("PGG", "reward", "contribution"): {
        "GLM-5.1": (10.03, 2.11), "GPT-5.4": (2.31, 1.43),
        "Gemini-3.1-Pro-Preview": (3.56, 0.76), "Claude-Opus-4-6": (13.75, 3.76),
        "Kimi-K2.6": (0.41, 0.36), "Grok-4.3": (0.00, 0.00),
    },
    ("TG", "punishment", "sender"): {
        "GLM-5.1": (64.19, 0.66), "GPT-5.4": (78.60, 3.46),
        "Gemini-3.1-Pro-Preview": (100.00, 0.00), "Claude-Opus-4-6": (99.58, 0.05),
        "Kimi-K2.6": (80.35, 1.08), "Grok-4.3": (91.57, 5.13),
    },
    ("TG", "punishment", "receiver"): {
        "GLM-5.1": (56.83, 1.31), "GPT-5.4": (43.84, 1.03),
        "Gemini-3.1-Pro-Preview": (63.47, 0.16), "Claude-Opus-4-6": (45.32, 0.80),
        "Kimi-K2.6": (55.42, 1.14), "Grok-4.3": (66.82, 3.54),
    },
    ("TG", "reputation", "sender"): {
        "GLM-5.1": (44.21, 3.80), "GPT-5.4": (3.51, 0.53),
        "Gemini-3.1-Pro-Preview": (56.81, 5.25), "Claude-Opus-4-6": (56.48, 1.56),
        "Kimi-K2.6": (28.12, 5.09), "Grok-4.3": (2.56, 0.29),
    },
    ("TG", "reputation", "receiver"): {
        "GLM-5.1": (25.23, 2.52), "GPT-5.4": (0.00, 0.00),
        "Gemini-3.1-Pro-Preview": (42.56, 1.49), "Claude-Opus-4-6": (25.39, 0.40),
        "Kimi-K2.6": (26.57, 1.00), "Grok-4.3": (0.00, 0.00),
    },
    ("TG", "reward", "sender"): {
        "GLM-5.1": (53.15, 1.19), "GPT-5.4": (19.08, 1.52),
        "Gemini-3.1-Pro-Preview": (99.03, 0.48), "Claude-Opus-4-6": (59.54, 0.34),
        "Kimi-K2.6": (48.42, 1.37), "Grok-4.3": (34.54, 0.75),
    },
    ("TG", "reward", "receiver"): {
        "GLM-5.1": (43.05, 0.37), "GPT-5.4": (36.65, 0.67),
        "Gemini-3.1-Pro-Preview": (49.93, 0.12), "Claude-Opus-4-6": (51.86, 0.29),
        "Kimi-K2.6": (49.73, 2.85), "Grok-4.3": (46.98, 0.55),
    },
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "RQ2_result")
    parser.add_argument("--output", type=Path, default=project_root / "RQ2_figure_output")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def model_from_path(path: Path) -> str:
    lower = str(path).lower()
    for token, model in MODEL_TOKENS.items():
        if token in lower:
            return model
    raise ValueError(f"Cannot infer model from {path}")


def game_mechanism(condition_name: str) -> tuple[str, str]:
    if condition_name.startswith("prisoners_dilemma_"):
        return "PD", condition_name.removeprefix("prisoners_dilemma_")
    if condition_name.startswith("public_goods_"):
        return "PGG", condition_name.removeprefix("public_goods_")
    if condition_name.startswith("trust_game_"):
        return "TG", condition_name.removeprefix("trust_game_")
    raise ValueError(f"Unknown condition directory: {condition_name}")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def log_files(experiment: Path) -> list[Path]:
    pattern = re.compile(r"^game_logs?(\d+)\.json$", re.IGNORECASE)
    files = [p for p in experiment.rglob("*.json") if pattern.match(p.name)]
    return sorted(files, key=lambda p: int(pattern.match(p.name).group(1)))


def config_for(experiment: Path) -> dict:
    configs = list(experiment.rglob("experiment_config.json"))
    return read_json(configs[0]) if configs else {}


def gini_nonnegative(values: list[float]) -> float:
    """Return the Gini coefficient for nonnegative cumulative payoffs."""
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.any(x < 0) or np.allclose(x.sum(), 0):
        return math.nan
    x = np.sort(x)
    n = x.size
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def extract_pd(log: list[dict]) -> dict:
    decisions: list[float] = []
    welfare: list[float] = []
    cumulative = defaultdict(float)
    round_values: list[list[float]] = []
    interactions_per_round = 12
    for start in range(0, len(log), interactions_per_round):
        block = log[start : start + interactions_per_round]
        current: list[float] = []
        for item in block:
            detail = item.get("detail", item)
            for suffix in ("1", "2"):
                choice = detail.get(f"choice{suffix}")
                if choice is not None:
                    value = float(str(choice).strip().lower() in {"yes", "cooperate", "cooperation", "c"})
                    current.append(value)
                    decisions.append(value)
                payoff = detail.get(f"payoff{suffix}")
                if payoff is not None:
                    name = detail.get(f"agent{suffix}_name", f"agent_{suffix}")
                    cumulative[str(name)] += float(payoff)
            if detail.get("payoff1") is not None and detail.get("payoff2") is not None:
                # Net welfare is based on final payoffs.  DD total payoff is 2;
                # CC total payoff is 6, so 4 is the full-cooperation surplus.
                pair_total = float(detail["payoff1"]) + float(detail["payoff2"])
                welfare.append(100.0 * (pair_total - 2.0) / 4.0)
        round_values.append(current)
    return {
        "outcomes": {"cooperation": 100.0 * float(np.mean(decisions))},
        "rounds": {"cooperation": np.array([100.0 * np.mean(x) for x in round_values])},
        "welfare": float(np.mean(welfare)),
        "equality_level": 1.0 - gini_nonnegative(list(cumulative.values())),
    }


def extract_pgg(log: list[dict], config: dict) -> dict:
    game_cfg = config.get("game_settings", config)
    endowment = float(game_cfg.get("initial_endowment", 20))
    multiplier = float(game_cfg.get("public_pool_multiplier", 9.6))
    all_scores: list[float] = []
    round_rates: list[float] = []
    welfare: list[float] = []
    cumulative = defaultdict(float)
    for item in log:
        contributions = item.get("agent_contributions", {})
        scores = [float(v) / endowment for v in contributions.values()]
        all_scores.extend(scores)
        round_rates.append(100.0 * float(np.mean(scores)))
        final_payoffs = item.get("final_payoffs") or item.get("payoffs")
        if final_payoffs:
            for name, payoff in final_payoffs.items():
                cumulative[str(name)] += float(payoff)
            n_agents = len(final_payoffs)
            zero_contribution_total = n_agents * endowment
            full_cooperation_surplus = (multiplier - 1.0) * n_agents * endowment
            welfare.append(
                100.0 * (sum(map(float, final_payoffs.values())) - zero_contribution_total)
                / full_cooperation_surplus
            )
    return {
        "outcomes": {"contribution": 100.0 * float(np.mean(all_scores))},
        "rounds": {"contribution": np.asarray(round_rates)},
        "welfare": float(np.mean(welfare)),
        "equality_level": 1.0 - gini_nonnegative(list(cumulative.values())),
    }


def extract_tg(log: list[dict], config: dict) -> dict:
    game_cfg = config.get("game_settings", config)
    initial_funds = float(game_cfg.get("initial_funds", 10))
    multiplier = float(game_cfg.get("multiplication_factor", 3))
    total_rounds = int(game_cfg.get("total_population_rounds", 30))
    per_round = len(log) // total_rounds
    if per_round <= 0 or per_round * total_rounds != len(log):
        raise ValueError(f"Cannot divide {len(log)} TG interactions into {total_rounds} rounds")
    sender_all: list[float] = []
    receiver_all: list[float] = []
    welfare: list[float] = []
    cumulative = defaultdict(float)
    sender_rounds: list[float] = []
    receiver_rounds: list[float] = []
    for start in range(0, len(log), per_round):
        sender_current: list[float] = []
        receiver_current: list[float] = []
        for item in log[start : start + per_round]:
            detail = item.get("detail", item)
            sent = float(detail.get("sent_amount", 0))
            returned = float(detail.get("returned_amount", 0))
            received = float(detail.get("trustee_received", sent * multiplier))
            sender_value = sent / initial_funds
            sender_current.append(sender_value)
            sender_all.append(sender_value)
            if received > 0:
                receiver_value = returned / received
                receiver_current.append(receiver_value)
                receiver_all.append(receiver_value)

            payoff_keys = [("trustor", "trustor_payoff"), ("trustee", "trustee_payoff")]
            if "rewarder_payoff" in detail:
                payoff_keys.append(("rewarder", "rewarder_payoff"))
            total_payoff = 0.0
            initial_total = initial_funds
            for role, payoff_key in payoff_keys:
                payoff = float(detail[payoff_key])
                total_payoff += payoff
                name = detail.get(f"{role}_name", f"{role}_{detail.get(f'{role}_id', '')}")
                cumulative[str(name)] += payoff
                if role == "rewarder":
                    initial_total += initial_funds
            # Full trust creates (multiplier - 1) * initial_funds = 20
            # units of surplus. Final payoffs already contain mechanism costs.
            full_trust_surplus = (multiplier - 1.0) * initial_funds
            welfare.append(100.0 * (total_payoff - initial_total) / full_trust_surplus)
        sender_rounds.append(100.0 * float(np.mean(sender_current)))
        receiver_rounds.append(
            100.0 * float(np.mean(receiver_current)) if receiver_current else math.nan
        )
    return {
        "outcomes": {
            "sender": 100.0 * float(np.mean(sender_all)),
            "receiver": 100.0 * float(np.mean(receiver_all)) if receiver_all else math.nan,
        },
        "rounds": {
            "sender": np.asarray(sender_rounds),
            "receiver": np.asarray(receiver_rounds),
        },
        "welfare": float(np.mean(welfare)),
        "equality_level": 1.0 - gini_nonnegative(list(cumulative.values())),
    }


def source_manifest(input_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(p for p in input_root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({
            "relative_path": path.relative_to(input_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return pd.DataFrame(records)


def collect_runs(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_records = []
    welfare_records = []
    round_store: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    conditions = sorted(p for p in input_root.iterdir() if p.is_dir())
    for condition in conditions:
        game, mechanism = game_mechanism(condition.name)
        if mechanism not in MECHANISM_ORDER:
            continue
        for experiment in sorted(p for p in condition.iterdir() if p.is_dir()):
            model = model_from_path(experiment)
            config = config_for(experiment)
            files = log_files(experiment)
            if not files:
                raise FileNotFoundError(f"No game logs under {experiment}")
            for run_index, path in enumerate(files, start=1):
                log = read_json(path)
                if not isinstance(log, list):
                    raise TypeError(f"Expected a JSON list in {path}")
                if game == "PD":
                    result = extract_pd(log)
                elif game == "PGG":
                    result = extract_pgg(log, config)
                else:
                    result = extract_tg(log, config)
                common = {
                    "game": game,
                    "mechanism": mechanism,
                    "model": model,
                    "experiment_dir": experiment.relative_to(input_root).as_posix(),
                    "run": run_index,
                    "log_file": path.relative_to(input_root).as_posix(),
                    "records": len(log),
                }
                for outcome, value in result["outcomes"].items():
                    run_records.append({**common, "outcome": outcome, "rate_pp": value})
                    round_store[(game, outcome, mechanism, model)].append(result["rounds"][outcome])
                welfare_records.append({
                    **common,
                    "net_welfare_normalized_pp": result["welfare"],
                    "payoff_equality_level": result["equality_level"],
                })
    runs = pd.DataFrame(run_records)
    welfare = pd.DataFrame(welfare_records)
    return runs, welfare, round_store


def summarize_effects(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["game", "outcome", "mechanism", "model"]
    for key, group in runs.groupby(keys, sort=False):
        game, outcome, mechanism, model = key
        values = group["rate_pp"].to_numpy(float)
        n = len(values)
        treatment_mean = float(values.mean())
        treatment_sd = float(values.std(ddof=1)) if n >= 2 else math.nan
        if n >= 2:
            half_width = float(t.ppf(0.975, n - 1) * treatment_sd / math.sqrt(n))
            treatment_low_raw = treatment_mean - half_width
            treatment_high_raw = treatment_mean + half_width
            treatment_low = max(0.0, treatment_low_raw)
            treatment_high = min(100.0, treatment_high_raw)
        else:
            treatment_low_raw = treatment_high_raw = math.nan
            treatment_low = treatment_high = math.nan
        baseline = BASELINES[(game, outcome)][model]
        records.append({
            "game": game,
            "outcome": outcome,
            "mechanism": mechanism,
            "model": model,
            "baseline_pp": baseline,
            "treatment_mean_pp": treatment_mean,
            "treatment_sd_pp": treatment_sd,
            "n_runs": n,
            "delta_pp": treatment_mean - baseline,
            "ci95_low_pp": treatment_low - baseline,
            "ci95_high_pp": treatment_high - baseline,
            "ci95_low_unbounded_pp": treatment_low_raw - baseline,
            "ci95_high_unbounded_pp": treatment_high_raw - baseline,
            "ci_note": "t CI across independent treatment runs; fixed RQ1 baseline; bounded to [0,100] before differencing",
        })
    result = pd.DataFrame(records)
    result["model"] = pd.Categorical(result["model"], MODEL_ORDER, ordered=True)
    result["mechanism"] = pd.Categorical(result["mechanism"], MECHANISM_ORDER, ordered=True)
    return result.sort_values(["game", "outcome", "model", "mechanism"]).reset_index(drop=True)


def audit_expected(effects: pd.DataFrame) -> pd.DataFrame:
    records = []
    lookup = effects.set_index(["game", "mechanism", "outcome", "model"])
    for (game, mechanism, outcome), expected_by_model in EXPECTED_TREATMENT.items():
        for model, (expected_mean, expected_sd) in expected_by_model.items():
            row = lookup.loc[(game, mechanism, outcome, model)]
            calc_mean = float(row["treatment_mean_pp"])
            calc_sd = float(row["treatment_sd_pp"])
            mean_match = round(calc_mean + 1e-12, 2) == round(expected_mean, 2)
            sd_match = round(calc_sd + 1e-12, 2) == round(expected_sd, 2)
            records.append({
                "game": game,
                "mechanism": mechanism,
                "outcome": outcome,
                "model": model,
                "calculated_mean_pp": calc_mean,
                "table_mean_pp": expected_mean,
                "calculated_sd_pp": calc_sd,
                "table_sd_pp": expected_sd,
                "mean_matches_2dp": mean_match,
                "sd_matches_2dp": sd_match,
                "fully_matches_2dp": mean_match and sd_match,
            })
    return pd.DataFrame(records)


def effect_type(baseline: float, treatment: float, delta: float) -> str:
    """Mutually exclusive descriptive classification for Figure 3E."""
    if delta <= -5.0:
        return "Mechanism harm"
    if baseline >= 95.0 and treatment >= 95.0:
        return "Ceiling"
    if baseline <= 5.0 and treatment <= 5.0:
        return "Floor"
    if baseline < 20.0 and treatment >= 50.0 and delta >= 5.0:
        return "Institutional rescue"
    if delta >= 5.0:
        return "Cooperation amplification"
    return "No material effect"


def classify_effects(effects: pd.DataFrame) -> pd.DataFrame:
    result = effects.copy()
    result["effect_type"] = [
        effect_type(float(b), float(y), float(d))
        for b, y, d in zip(result["baseline_pp"], result["treatment_mean_pp"], result["delta_pp"])
    ]
    return result


def baseline_welfare(game: str, model: str) -> tuple[float, str]:
    if game == "PD":
        p = BASELINES[("PD", "cooperation")][model] / 100.0
        # The marginal cooperation rate does not reveal CC frequency.  The
        # independence estimate is exact for p=0 and p=1 and only affects the
        # Gemini baseline among these six models.
        return 100.0 * (1.5 * p - 0.5 * p * p), "PD independence reconstruction from marginal cooperation"
    if game == "PGG":
        return BASELINES[("PGG", "contribution")][model], "exact PGG accounting identity"
    return BASELINES[("TG", "sender")][model], "exact TG accounting identity from Sender transfer"


def welfare_effects(welfare_runs: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    welfare_summary = (
        welfare_runs.groupby(["game", "mechanism", "model"], observed=True)
        .agg(
            treatment_welfare_pp=("net_welfare_normalized_pp", "mean"),
            treatment_welfare_sd_pp=("net_welfare_normalized_pp", lambda x: x.std(ddof=1)),
            n_runs=("net_welfare_normalized_pp", "size"),
            treatment_equality_level=("payoff_equality_level", "mean"),
        )
        .reset_index()
    )
    effect_lookup = effects.set_index(["game", "outcome", "mechanism", "model"])
    records = []
    for row in welfare_summary.itertuples(index=False):
        game, mechanism, model = row.game, row.mechanism, row.model
        if game == "TG":
            sender = effect_lookup.loc[(game, "sender", mechanism, model)]
            receiver = effect_lookup.loc[(game, "receiver", mechanism, model)]
            cooperation_delta = 0.5 * (float(sender.delta_pp) + float(receiver.delta_pp))
            cooperation_definition = "mean of Sender and Receiver changes"
        else:
            outcome = "cooperation" if game == "PD" else "contribution"
            erow = effect_lookup.loc[(game, outcome, mechanism, model)]
            cooperation_delta = float(erow.delta_pp)
            cooperation_definition = outcome
        base_welfare, method = baseline_welfare(game, str(model))
        records.append({
            "game": game,
            "mechanism": mechanism,
            "model": str(model),
            "delta_cooperation_pp": cooperation_delta,
            "cooperation_definition": cooperation_definition,
            "baseline_welfare_normalized_pp": base_welfare,
            "treatment_welfare_normalized_pp": float(row.treatment_welfare_pp),
            "delta_net_welfare_normalized_pp": float(row.treatment_welfare_pp) - base_welfare,
            "treatment_welfare_sd_pp": float(row.treatment_welfare_sd_pp),
            "n_runs": int(row.n_runs),
            "baseline_welfare_method": method,
            "treatment_payoff_equality_level": float(row.treatment_equality_level),
            "delta_equality": math.nan,
            "delta_equality_note": "not identified: matched no-institution agent-level payoff baseline unavailable",
        })
    return pd.DataFrame(records)


def dynamic_effects(
    round_store: dict[tuple[str, str, str, str], list[np.ndarray]],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for game, outcome, _ in PANEL_ORDER:
        for mechanism in MECHANISM_ORDER:
            arrays_by_model = []
            models = []
            for model in MODEL_ORDER:
                arrays = round_store[(game, outcome, mechanism, model)]
                if not arrays:
                    continue
                matrix = np.vstack(arrays)
                arrays_by_model.append(matrix)
                models.append(model)
            min_rounds = min(x.shape[1] for x in arrays_by_model)
            arrays_by_model = [x[:, :min_rounds] for x in arrays_by_model]
            model_curves = np.vstack([
                x.mean(axis=0) - BASELINES[(game, outcome)][model]
                for x, model in zip(arrays_by_model, models)
            ])
            estimate = model_curves.mean(axis=0)
            draws = np.empty((bootstrap, min_rounds), dtype=float)
            for b in range(bootstrap):
                sampled_models = rng.integers(0, len(models), size=len(models))
                sampled_curves = []
                for model_index in sampled_models:
                    matrix = arrays_by_model[model_index]
                    sampled_runs = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
                    sampled_curves.append(
                        matrix[sampled_runs].mean(axis=0)
                        - BASELINES[(game, outcome)][models[model_index]]
                    )
                draws[b] = np.vstack(sampled_curves).mean(axis=0)
            low, high = np.percentile(draws, [2.5, 97.5], axis=0)
            for round_index in range(min_rounds):
                records.append({
                    "game": game,
                    "outcome": outcome,
                    "mechanism": mechanism,
                    "round": round_index + 1,
                    "delta_pp": estimate[round_index],
                    "ci95_low_pp": low[round_index],
                    "ci95_high_pp": high[round_index],
                    "n_models": len(models),
                    "bootstrap_draws": bootstrap,
                    "method": "two-stage model/run percentile bootstrap; fixed overall RQ1 baseline",
                })
    return pd.DataFrame(records)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def plot_forest(ax: plt.Axes, effects: pd.DataFrame, game: str, outcome: str, title: str) -> None:
    subset = effects[(effects["game"] == game) & (effects["outcome"] == outcome)]
    y = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = {"reputation": -0.22, "reward": 0.0, "punishment": 0.22}
    ax.axvspan(-5, 5, color="#D9D9D9", alpha=0.55, zorder=0)
    ax.axvline(0, color="#4D4D4D", lw=1.0, zorder=1)
    for mechanism in MECHANISM_ORDER:
        part = subset[subset["mechanism"] == mechanism].set_index("model")
        xs = np.array([float(part.loc[m, "delta_pp"]) for m in MODEL_ORDER])
        lows = np.array([float(part.loc[m, "ci95_low_pp"]) for m in MODEL_ORDER])
        highs = np.array([float(part.loc[m, "ci95_high_pp"]) for m in MODEL_ORDER])
        ax.errorbar(
            xs, y + offsets[mechanism],
            xerr=np.vstack([xs - lows, highs - xs]),
            fmt="o", ms=5.3, capsize=2.2, lw=1.15,
            color=MECHANISM_COLORS[mechanism],
            label=MECHANISM_LABELS[mechanism], zorder=3,
        )
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-105, 105)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("Absolute change from no-institution baseline (percentage points)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)


TYPE_ORDER = [
    "Institutional rescue",
    "Cooperation amplification",
    "No material effect",
    "Mechanism harm",
    "Ceiling",
    "Floor",
]
TYPE_COLORS = ["#1B9E77", "#66C2A5", "#D9D9D9", "#D73027", "#756BB1", "#636363"]
TYPE_CODES = {
    "Institutional rescue": "R",
    "Cooperation amplification": "A",
    "No material effect": "N",
    "Mechanism harm": "H",
    "Ceiling": "C",
    "Floor": "F",
}


def plot_effect_matrix(ax: plt.Axes, classified: pd.DataFrame) -> None:
    rows = []
    for game, outcome, short in [
        ("PD", "cooperation", "PD"),
        ("PGG", "contribution", "PGG"),
        ("TG", "sender", "TG-S"),
        ("TG", "receiver", "TG-R"),
    ]:
        for model in MODEL_ORDER:
            rows.append((game, outcome, model, f"{short}  |  {MODEL_LABELS[model]}"))
    code = {name: i for i, name in enumerate(TYPE_ORDER)}
    matrix = np.empty((len(rows), len(MECHANISM_ORDER)), dtype=int)
    cell_types = []
    for i, (game, outcome, model, _) in enumerate(rows):
        cell_row = []
        for j, mechanism in enumerate(MECHANISM_ORDER):
            item = classified[
                (classified["game"] == game)
                & (classified["outcome"] == outcome)
                & (classified["model"].astype(str) == model)
                & (classified["mechanism"].astype(str) == mechanism)
            ].iloc[0]
            kind = item["effect_type"]
            matrix[i, j] = code[kind]
            cell_row.append(kind)
        cell_types.append(cell_row)
    cmap = ListedColormap(TYPE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(TYPE_ORDER) + 0.5), cmap.N)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(3), [MECHANISM_LABELS[x] for x in MECHANISM_ORDER])
    ax.set_yticks(range(len(rows)), [x[3] for x in rows])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0)
    for i in range(len(rows)):
        for j in range(3):
            kind = cell_types[i][j]
            color = "white" if kind in {"Mechanism harm", "Ceiling", "Floor"} else "#222222"
            ax.text(j, i, TYPE_CODES[kind], ha="center", va="center", weight="bold", color=color)
    for split in [5.5, 11.5, 17.5]:
        ax.axhline(split, color="white", lw=2.2)
    ax.set_title("E  Descriptive institution-effect types (|Δ| < 5 pp = no material change)", loc="left", pad=30)
    handles = [Patch(facecolor=c, label=n) for n, c in zip(TYPE_ORDER, TYPE_COLORS)]
    ax.legend(handles=handles, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.055))
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figure3(effects: pd.DataFrame, classified: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(15.5, 21.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 2.15])
    forest_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                   fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, (game, outcome, title) in zip(forest_axes, PANEL_ORDER):
        plot_forest(ax, effects, game, outcome, title)
    forest_axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(1.05, 1.11))
    matrix_ax = fig.add_subplot(grid[2, :])
    plot_effect_matrix(matrix_ax, classified)
    fig.suptitle("Figure 3. RQ2 institution effects on cooperation", fontsize=16, weight="bold")
    fig.text(
        0.5, -0.012,
        "Points are absolute changes from retained RQ1 no-institution means. Error bars are bounded 95% t intervals "
        "across treatment runs (baseline treated as fixed); gray band denotes ±5 percentage points.",
        ha="center", va="bottom", fontsize=8.5,
    )
    save_figure(fig, output, "Figure3_RQ2_core_institution_effects")

    forest_fig, forest_axes_grid = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    forest_axes = list(forest_axes_grid.flat)
    for ax, (game, outcome, title) in zip(forest_axes, PANEL_ORDER):
        plot_forest(ax, effects, game, outcome, title)
    forest_axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(1.05, 1.10))
    forest_fig.suptitle("Figure 3A-D. Treatment effects relative to no-institution baselines",
                        fontsize=15, weight="bold")
    save_figure(forest_fig, output, "Figure3A-D_RQ2_forest_plots")

    matrix_fig, matrix_ax = plt.subplots(figsize=(10.5, 12.5), constrained_layout=True)
    plot_effect_matrix(matrix_ax, classified)
    matrix_fig.suptitle("Figure 3E. Institution-effect types", fontsize=15, weight="bold")
    save_figure(matrix_fig, output, "Figure3E_RQ2_effect_type_matrix")


def unavailable_panel(ax: plt.Axes, title: str, heading: str, body: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#BDBDBD")
    ax.set_facecolor("#F7F7F7")
    ax.text(0.5, 0.60, heading, ha="center", va="center", fontsize=13, weight="bold", color="#555555")
    ax.text(0.5, 0.40, body, ha="center", va="center", fontsize=9.5, color="#555555", wrap=True,
            transform=ax.transAxes)


def plot_welfare_scatter(ax: plt.Axes, welfare: pd.DataFrame, include_legend: bool = True) -> None:
    shapes = {"PD": "o", "PGG": "s", "TG": "^"}
    for mechanism in MECHANISM_ORDER:
        for game in ["PD", "PGG", "TG"]:
            part = welfare[(welfare["mechanism"] == mechanism) & (welfare["game"] == game)]
            ax.scatter(
                part["delta_cooperation_pp"], part["delta_net_welfare_normalized_pp"],
                s=48, marker=shapes[game], c=MECHANISM_COLORS[mechanism],
                edgecolors="white", linewidths=0.7, alpha=0.88,
            )
    ax.axhline(0, color="#555555", lw=0.9)
    ax.axvline(0, color="#555555", lw=0.9)
    ax.grid(color="#ECECEC", lw=0.7)
    ax.set_xlabel("Δ cooperation (percentage points)")
    ax.set_ylabel("Δ normalized net welfare\n(pp of full-cooperation social surplus)")
    ax.set_title("A  Cooperation change and cost-adjusted welfare change", loc="left")
    if include_legend:
        mech_handles = [Line2D([], [], marker="o", linestyle="", color=MECHANISM_COLORS[m],
                               label=MECHANISM_LABELS[m], markersize=7) for m in MECHANISM_ORDER]
        game_handles = [Line2D([], [], marker=shapes[g], linestyle="", color="#555555",
                               label=g, markersize=7) for g in ["PD", "PGG", "TG"]]
        ax.legend(handles=mech_handles + game_handles, ncol=2, frameon=False, loc="best")


def plot_dynamic_axis(ax: plt.Axes, dynamic: pd.DataFrame, game: str, outcome: str, title: str) -> None:
    part = dynamic[(dynamic["game"] == game) & (dynamic["outcome"] == outcome)]
    ax.axhspan(-5, 5, color="#D9D9D9", alpha=0.45)
    ax.axhline(0, color="#555555", lw=0.8)
    for mechanism in MECHANISM_ORDER:
        line = part[part["mechanism"] == mechanism].sort_values("round")
        x = line["round"].to_numpy(float)
        y = line["delta_pp"].to_numpy(float)
        low = line["ci95_low_pp"].to_numpy(float)
        high = line["ci95_high_pp"].to_numpy(float)
        ax.plot(x, y, color=MECHANISM_COLORS[mechanism], lw=1.4, label=MECHANISM_LABELS[mechanism])
        ax.fill_between(x, low, high, color=MECHANISM_COLORS[mechanism], alpha=0.12, linewidth=0)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("Round", fontsize=8.5)
    ax.set_ylabel("Δ pp", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)


def make_figure4(
    welfare: pd.DataFrame,
    dynamic: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.72, 0.48])
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    cgrid = outer[1, :].subgridspec(1, 4, wspace=0.30)
    caxes = [fig.add_subplot(cgrid[0, j]) for j in range(4)]
    ax_d = fig.add_subplot(outer[2, :])

    plot_welfare_scatter(ax_a, welfare)

    unavailable_panel(
        ax_b,
        "B  Efficiency–equality trade-off",
        "Δ Equality is not estimable",
        "RQ2_result contains treatment payoff histories, but matched no-institution agent-level payoff histories "
        "are unavailable for all 18 model–game baselines. Plotting Δ Equality would require inventing a baseline.",
    )

    for index, (ax, (game, outcome, title)) in enumerate(zip(caxes, PANEL_ORDER)):
        prefix = "C  " if index == 0 else ""
        plot_dynamic_axis(ax, dynamic, game, outcome, prefix + title.split("  ", 1)[1])
    caxes[0].legend(frameon=False, fontsize=7, ncol=1, loc="best")

    unavailable_panel(
        ax_d,
        "D  Real versus sham/corrupted mechanisms",
        "No Sham or Corrupted conditions found",
        "All archived RQ2 cells are real-mechanism treatments. Real–Sham and Real–Corrupted paired contrasts "
        "require new matched runs using the same models, games, seeds, and repetition structure.",
    )

    fig.suptitle("Figure 4. RQ2 cooperation quality and evidential coverage", fontsize=16, weight="bold")
    fig.text(
        0.5, -0.012,
        "A: final payoffs are used, so recorded reward/punishment costs are not deducted twice. "
        "C: 95% percentile intervals from a two-stage model/run bootstrap; each model's retained overall RQ1 "
        "baseline is fixed across rounds. B and D are deliberately not imputed.",
        ha="center", va="bottom", fontsize=8.5,
    )
    save_figure(fig, output, "Figure4_RQ2_cooperation_quality")

    welfare_fig, welfare_ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    plot_welfare_scatter(welfare_ax, welfare)
    welfare_fig.suptitle("Figure 4A. Cooperation and net-welfare change", fontsize=14, weight="bold")
    save_figure(welfare_fig, output, "Figure4A_RQ2_cooperation_welfare")

    dynamic_fig, dynamic_axes_grid = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    dynamic_axes = list(dynamic_axes_grid.flat)
    for ax, (game, outcome, title) in zip(dynamic_axes, PANEL_ORDER):
        plot_dynamic_axis(ax, dynamic, game, outcome, title.split("  ", 1)[1])
    dynamic_axes[0].legend(frameon=False, ncol=3, loc="best")
    dynamic_fig.suptitle("Figure 4C. Model-averaged dynamic institution effects",
                         fontsize=14, weight="bold")
    save_figure(dynamic_fig, output, "Figure4C_RQ2_dynamic_effects")


def write_baselines(output: Path) -> None:
    records = []
    for (game, outcome), values in BASELINES.items():
        for model, value in values.items():
            records.append({
                "game": game,
                "outcome": outcome,
                "model": model,
                "baseline_pp": value,
                "source": "retained RQ1 summary table supplied by researcher",
            })
    pd.DataFrame(records).to_csv(output / "rq1_fixed_cooperation_baselines.csv", index=False, encoding="utf-8-sig")


def write_readme(output: Path, audit: pd.DataFrame, manifest_rows: int) -> None:
    matched = int(audit["fully_matches_2dp"].sum())
    total = len(audit)
    text = f"""# RQ2 publication-figure output

This directory was generated by `cooperation_analysis/rq2_publication_figures.py`.
The script reads `RQ2_result` without writing to it. The input manifest contains
SHA-256 hashes for {manifest_rows} source files.

## Main outputs

- `Figure3_RQ2_core_institution_effects.png/.pdf`: forest plots A-D and the
  descriptive effect-type matrix E.
- `Figure4_RQ2_cooperation_quality.png/.pdf`: welfare-change panel A, dynamic
  panel C, and explicit data-availability panels B/D.
- CSV files contain run-level values, effect estimates, classifications,
  welfare calculations, dynamic estimates, baseline values, and the raw-log
  versus supplied-table audit.

## Statistical definitions

- Cooperation effects are absolute percentage-point differences from the
  retained RQ1 no-institution means.
- Figure 3 intervals are 95% t intervals across independent treatment runs,
  bounded to the outcome support before differencing. Because the retained RQ1
  table has no replicate uncertainty, its baseline is treated as fixed.
- The practical-equivalence band is -5 to +5 percentage points.
- Net welfare uses final recorded payoffs, so mechanism costs already included
  in those payoffs are not subtracted again. Welfare is normalized by each
  game's full-cooperation social surplus for cross-game comparability.
- For PD, marginal baseline cooperation does not identify mutual-cooperation
  frequency. The welfare baseline uses independent action reconstruction; this
  matters only for Gemini because all other PD baselines are 0% or 100%.
- Figure 4C is a two-stage percentile bootstrap: resample models, then treatment
  runs within model. It measures round-specific treatment outcomes relative to
  each model's retained *overall* RQ1 mean, not a matched round-specific
  baseline trajectory.

## Non-identifiable requested panels

- Figure 4B: treatment equality levels can be calculated, but **change in
  equality** cannot be calculated without matched no-institution agent-level
  payoff histories for all model-game baselines.
- Figure 4D: no Sham or Corrupted experimental conditions occur in the archive.
  No contrast was fabricated.

## Audit result

Raw-log means and sample SDs match the supplied RQ2 table to two decimals for
**{matched}/{total}** model-game-role-mechanism cells. See
`rq2_table_value_audit.csv` for any mismatches and their exact magnitudes.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output = args.output.resolve()
    if output == input_root or input_root in output.parents:
        raise ValueError("Output must be outside RQ2_result to preserve source data")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    output.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    manifest = source_manifest(input_root)
    manifest.to_csv(output / "input_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    runs, welfare_runs, round_store = collect_runs(input_root)
    effects = summarize_effects(runs)
    audit = audit_expected(effects)
    classified = classify_effects(effects)
    welfare = welfare_effects(welfare_runs, effects)
    dynamic = dynamic_effects(round_store, args.bootstrap, args.seed)

    runs.to_csv(output / "rq2_run_level_cooperation.csv", index=False, encoding="utf-8-sig")
    welfare_runs.to_csv(output / "rq2_run_level_welfare.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output / "figure3_forest_effects.csv", index=False, encoding="utf-8-sig")
    classified.to_csv(output / "figure3_effect_types.csv", index=False, encoding="utf-8-sig")
    welfare.to_csv(output / "figure4_welfare_effects.csv", index=False, encoding="utf-8-sig")
    dynamic.to_csv(output / "figure4_dynamic_effects.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "rq2_table_value_audit.csv", index=False, encoding="utf-8-sig")
    write_baselines(output)

    make_figure3(effects, classified, output)
    make_figure4(welfare, dynamic, output)
    write_readme(output, audit, len(manifest))

    matched = int(audit["fully_matches_2dp"].sum())
    print(f"Wrote outputs to: {output}")
    print(f"RQ2 table audit: {matched}/{len(audit)} cells match mean and SD to 2 decimals")
    if matched != len(audit):
        print(audit.loc[~audit["fully_matches_2dp"], [
            "game", "mechanism", "outcome", "model",
            "calculated_mean_pp", "table_mean_pp", "calculated_sd_pp", "table_sd_pp",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
