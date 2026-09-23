#!/usr/bin/env python3
"""Audit the RQ1/RQ2 table cells against raw logs and compare group welfare.

The experiment directories below are selected to reproduce the values in the
four supplied RQ1/RQ2 result-table images.  Each game-log file is one
independent run.  The primary raw outcome is final payoff per participant per
population round.  A normalized, resource-adjusted welfare measure is also
reported because the trust-game reward condition introduces a third role with
an additional endowment.

Outputs are written to ``RQ1_RQ2_welfare_output`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ORDER = [
    "Llama-3-8B",
    "Llama-3-70B",
    "Qwen2.5-7B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct",
    "GLM-5.1",
    "GPT-5.4",
    "Gemini-3.1-Pro-Preview",
    "Claude-Opus-4-6",
    "Kimi-K2.6",
    "Grok-4.3",
    "DeepSeek-V4-Pro",
]


@dataclass(frozen=True)
class Cell:
    rq: str
    game: str
    mechanism: str
    model: str
    relative_dir: str | None
    target_primary_mean_pct: float
    target_primary_sd_pct: float | None
    target_receiver_mean_pct: float | None = None
    target_receiver_sd_pct: float | None = None
    note: str = ""


def rq1_cells() -> list[Cell]:
    rows: list[Cell] = []

    pd = {
        "Llama-3-8B": ("RQ1_old_result/result_prisoners_dilemma_group/10轮_3次_012404_llama3-8b", 2.22, 1.20),
        "Llama-3-70B": ("RQ1_old_result/result_prisoners_dilemma_group/10轮_3次_012403_llama3-70b", 0.14, 0.24),
        "Qwen2.5-7B-Instruct": ("RQ1_old_result/result_prisoners_dilemma_group/10lun_3ci_020501_Qwen2.5-7B-Instruct", 0.00, 0.00),
        "GLM-5.1": ("RQ1_result/result/prisoners_dilemma/090201_glm-5.1", 0.00, 0.00),
        "GPT-5.4": ("RQ1_result/result/prisoners_dilemma/082004_gpt-5.4", 0.00, 0.00),
        "Gemini-3.1-Pro-Preview": ("RQ1_result/result/prisoners_dilemma/082003_gemini-3.1-pro-preview", 3.19, 0.87),
        "Claude-Opus-4-6": ("RQ1_result/result/prisoners_dilemma/070101_claude-opus-4-6", 100.00, 0.00),
        "Kimi-K2.6": ("RQ1_result/result/prisoners_dilemma/072201_kimi-k2.6", 0.00, 0.00),
        # The current attachment reports exact zero cooperation, but the only
        # available Grok raw directory gives 0.69% +/- 0.24%.
        "Grok-4.3": (None, 0.00, 0.00),
        "DeepSeek-V4-Pro": ("RQ1_result/result/prisoners_dilemma/091501_deepseek-v4-pro", 0.14, 0.24),
    }
    for model, (path, mean, sd) in pd.items():
        note = ""
        if model == "Grok-4.3":
            note = "Current attachment reports 0.00% +/- 0.00%; available RQ1_result/result/prisoners_dilemma/072701_grok-4.3 logs give 0.69% +/- 0.24%. Exact DD payoff reconstructed from the current attachment value."
        rows.append(Cell("RQ1", "PD", "none", model, path, mean, sd, note=note))

    pgg = {
        "Llama-3-8B": ("RQ1_old_result/result_public_goods_group/llama3-8b-three", 60.18, 3.66),
        "Llama-3-70B": ("RQ1_old_result/result_public_goods_group/llama3-70b-three", 47.39, 0.54),
        "Qwen2.5-7B-Instruct": ("RQ1_old_result/result_public_goods_group/qwen2.5-7b-instruct-three", 3.75, 2.97),
        "GLM-5.1": ("RQ1_result/result/public_goods/result_090801_glm-5.1", 0.00, 0.00),
        "GPT-5.4": ("RQ1_result/result/public_goods/result_070108_gpt-5.4", 0.00, 0.00),
        "Gemini-3.1-Pro-Preview": ("RQ1_result/result/public_goods/result_082001_gemini-3.1-pro-preview", 0.00, 0.00),
        "Claude-Opus-4-6": ("RQ1_result/result/public_goods/result_082001_claude-opus-4-6", 0.00, 0.00),
        "Kimi-K2.6": ("RQ1_result/result/public_goods/result_072301_kimi-k2.6", 0.00, 0.00),
        "Grok-4.3": ("RQ1_result/result/public_goods/result_072202_grok-4.3", 0.00, 0.00),
        "DeepSeek-V4-Pro": ("RQ1_result/result/public_goods/result_063004_deepseek-v4-pro", 0.00, 0.00),
    }
    for model, (path, mean, sd) in pgg.items():
        note = ""
        rows.append(Cell("RQ1", "PGG", "none", model, path, mean, sd, note=note))

    tg = {
        "Llama-3-8B": ("RQ1_old_result/result_trust_game_population/30lun_3ci_012401_llama3-8b", 55.65, 0.34, 57.99, 2.32),
        "Llama-3-70B": ("RQ1_old_result/result_trust_game_population/30lun_3ci_012402_llama3-70b", 54.32, 0.62, 36.51, 0.47),
        "Qwen2.5-7B-Instruct": ("RQ1_old_result/result_trust_game_population/020601_30lun_3ci_Qwen2.5-7B-Instruct", 46.19, 1.69, 37.50, 2.97),
        "GLM-5.1": ("RQ1_result/result/trust_game/082904_glm-5.1_final", 30.21, 0.97, 9.85, 0.57),
        "GPT-5.4": ("RQ1_result/result/trust_game/083001_gpt-5.4", 10.60, 0.25, 0.00, 0.00),
        "Gemini-3.1-Pro-Preview": ("RQ1_result/result/trust_game/091601_gemini-3.1-pro-preview", 61.06, 2.09, 40.82, 1.78),
        "Claude-Opus-4-6": ("RQ1_result/result/trust_game/083102_claude-opus-4-6", 32.29, 2.29, 31.92, 0.49),
        "Kimi-K2.6": ("RQ1_result/result/trust_game/072201_kimi-k2.6", 23.45, 0.91, 20.69, 1.47),
        "Grok-4.3": ("RQ1_result/result/trust_game/072601_grok-4.3", 7.41, 0.88, 0.74, 0.83),
        "DeepSeek-V4-Pro": ("RQ1_result/result/trust_game/091601_deepseek-v4-pro", 40.52, 2.05, 1.74, 1.19),
    }
    for model, values in tg.items():
        path, sender, sender_sd, receiver, receiver_sd = values
        note = ""
        rows.append(Cell("RQ1", "TG", "none", model, path, sender, sender_sd, receiver, receiver_sd, note))
    return rows


def rq2_cells() -> list[Cell]:
    rows: list[Cell] = []

    mappings = {
        ("PD", "punishment"): {
            "GLM-5.1": ("RQ2_result/prisoners_dilemma_punishment/080408_glm-5.1-three", 1.25, 0.72),
            "GPT-5.4": ("RQ2_result/prisoners_dilemma_punishment/080403_gpt-5.4-three", 0.00, 0.00),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/prisoners_dilemma_punishment/080402_gemini-3.1-pro-preview-three", 9.17, 4.51),
            "Claude-Opus-4-6": ("RQ2_result/prisoners_dilemma_punishment/080401_claude-opus-4-6-three", 100.00, 0.00),
            "Kimi-K2.6": ("RQ2_result/prisoners_dilemma_punishment/080404_kimi-k2.6-three", 1.39, 0.64),
            "Grok-4.3": ("RQ2_result/prisoners_dilemma_punishment/080405_grok-4.3-three", 36.11, 3.15),
            "DeepSeek-V4-Pro": ("RQ2_result/prisoners_dilemma_punishment/091601_deepseek-v4-pro", 31.25, 1.91),
        },
        ("PD", "reputation"): {
            "GLM-5.1": ("RQ2_result/prisoners_dilemma_reputation/072806_glm-5.1-three", 6.25, 1.91),
            "GPT-5.4": ("RQ2_result/prisoners_dilemma_reputation/072801_gpt-5.4_three", 0.14, 0.24),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/prisoners_dilemma_reputation/072802_gemini-3.1-pro-preview-three", 100.00, 0.00),
            "Claude-Opus-4-6": ("RQ2_result/prisoners_dilemma_reputation/072803_claude-opus-4-6-three", 100.00, 0.00),
            "Kimi-K2.6": ("RQ2_result/prisoners_dilemma_reputation/072804_kimi-k2.6-three", 1.39, 0.24),
            "Grok-4.3": ("RQ2_result/prisoners_dilemma_reputation/072805_grok-4.3-three", 0.00, 0.00),
            "DeepSeek-V4-Pro": ("RQ2_result/prisoners_dilemma_reputation/091601_deepseek-v4-pro", 68.33, 28.32),
        },
        ("PD", "reward"): {
            "GLM-5.1": ("RQ2_result/prisoners_dilemma_reward/082606_glm-5.1-three", 8.33, 1.82),
            "GPT-5.4": ("RQ2_result/prisoners_dilemma_reward/082603_gpt-5.4-three", 0.00, 0.00),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/prisoners_dilemma_reward/082602_gemini-3.1-pro-preview-three", 87.78, 1.05),
            "Claude-Opus-4-6": ("RQ2_result/prisoners_dilemma_reward/082604_claude-opus-4-6-three", 100.00, 0.00),
            "Kimi-K2.6": ("RQ2_result/prisoners_dilemma_reward/082605_kimi-k2.6-three", 70.14, 2.68),
            "Grok-4.3": ("RQ2_result/prisoners_dilemma_reward/082801_grok-4.3-three", 3.75, 0.00),
            "DeepSeek-V4-Pro": ("RQ2_result/prisoners_dilemma_reward/091501_deepseek-v4-pro", 67.08, 6.71),
        },
        ("PGG", "punishment"): {
            "GLM-5.1": ("RQ2_result/public_goods_punishment/result_072003_glm-5.1-three", 98.20, 1.16),
            "GPT-5.4": ("RQ2_result/public_goods_punishment/result_070301_gpt-5.4-three", 0.00, 0.00),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/public_goods_punishment/result_gemini-3.1-pro-preview-three", 99.95, 0.08),
            "Claude-Opus-4-6": ("RQ2_result/public_goods_punishment/result_071905_claude-opus-4-6-three", 75.00, 0.00),
            "Kimi-K2.6": ("RQ2_result/public_goods_punishment/result_072001_kimi-k2.6-three", 0.14, 0.24),
            "Grok-4.3": ("RQ2_result/public_goods_punishment/result_081601_grok-4.3-three", 24.65, 2.36),
            "DeepSeek-V4-Pro": ("RQ2_result/public_goods_punishment/result_091601_deepseek-v4-pro", 1.16, 0.40),
        },
        ("PGG", "reputation"): {
            "GLM-5.1": ("RQ2_result/public_goods_reputation/result_080601_glm-5.1-three-new", 2.07, 3.35),
            "GPT-5.4": ("RQ2_result/public_goods_reputation/result_080602_gpt-5.4-three-new", 97.08, 0.69),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/public_goods_reputation/result_080603_gemini-3.1-pro-preview-three-new", 99.86, 0.24),
            "Claude-Opus-4-6": ("RQ2_result/public_goods_reputation/result_claude-opus-4-6-three", 50.00, 0.00),
            "Kimi-K2.6": ("RQ2_result/public_goods_reputation/result_080604_kimi-k2.6-three-new", 0.78, 0.41),
            "Grok-4.3": ("RQ2_result/public_goods_reputation/result_072102_grok-4.3-three", 0.00, 0.00),
            "DeepSeek-V4-Pro": ("RQ2_result/public_goods_reputation/result_091601_deepseek-v4-pro", 62.72, 0.99),
        },
        ("PGG", "reward"): {
            "GLM-5.1": ("RQ2_result/public_goods_reward/result_070504_glm-5.1-three", 10.03, 2.11),
            "GPT-5.4": ("RQ2_result/public_goods_reward/result_081601_gpt-5.4-three", 2.31, 1.43),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/public_goods_reward/result_072302_gemini-3.1-pro-preview-three", 3.56, 0.76),
            "Claude-Opus-4-6": ("RQ2_result/public_goods_reward/result_070703_claude-opus-4-6-three", 13.75, 3.76),
            "Kimi-K2.6": ("RQ2_result/public_goods_reward/result_072201_kimi-k2.6-three", 0.41, 0.36),
            "Grok-4.3": ("RQ2_result/public_goods_reward/result_072204_grok-4.3-three", 0.00, 0.00),
            "DeepSeek-V4-Pro": ("RQ2_result/public_goods_reward/result_091601_deepseek-v4-pro", 61.84, 0.40),
        },
    }
    for (game, mechanism), model_rows in mappings.items():
        for model, (path, mean, sd) in model_rows.items():
            rows.append(Cell("RQ2", game, mechanism, model, path, mean, sd))

    trust = {
        "punishment": {
            "GLM-5.1": ("RQ2_result/trust_game_punishment/080501-glm-5.1-three", 64.19, 0.66, 56.83, 1.31),
            "GPT-5.4": ("RQ2_result/trust_game_punishment/072903_gpt-5.4-three", 78.60, 3.46, 43.84, 1.03),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/trust_game_punishment/072905_gemini-3.1-pro-preview-three", 100.00, 0.00, 63.47, 0.16),
            "Claude-Opus-4-6": ("RQ2_result/trust_game_punishment/072906_claude-opus-4-6-three", 99.58, 0.05, 45.32, 0.80),
            "Kimi-K2.6": ("RQ2_result/trust_game_punishment/072902_kimi-k2.6-three", 80.35, 1.08, 55.42, 1.14),
            "Grok-4.3": ("RQ2_result/trust_game_punishment/072904_grok-4.3-three", 91.57, 5.13, 66.82, 3.54),
            "DeepSeek-V4-Pro": ("RQ2_result/trust_game_punishment/091601_deespeek-v4-pro", 86.44, 2.81, 66.31, 0.37),
        },
        "reputation": {
            "GLM-5.1": ("RQ2_result/trust_game_reputation/090101_glm-5.1", 94.06, 0.62, 56.18, 0.77),
            "GPT-5.4": ("RQ2_result/trust_game_reputation/090102_gpt-5.4", 86.95, 1.18, 40.19, 0.03),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/trust_game_reputation/090204_gemini-3.1-pro-preview", 94.91, 1.26, 49.85, 0.12),
            "Claude-Opus-4-6": ("RQ2_result/trust_game_reputation/090201_claude-opus-4-6", 92.15, 1.04, 50.13, 0.16),
            "Kimi-K2.6": ("RQ2_result/trust_game_reputation/090203_kimi-k2.6", 91.46, 0.99, 49.34, 0.25),
            "Grok-4.3": ("RQ2_result/trust_game_reputation/090202_grok-4.3", 68.70, 3.65, 42.90, 0.74),
            "DeepSeek-V4-Pro": ("RQ2_result/trust_game_reputation/091701_deepseek-v4-pro", 91.41, 1.01, 61.31, 0.81),
        },
        "reward": {
            "GLM-5.1": ("RQ2_result/trust_game_reward/082803_glm-5.1-three", 53.15, 1.19, 43.05, 0.37),
            "GPT-5.4": ("RQ2_result/trust_game_reward/082806_gpt-5.4-three", 19.08, 1.52, 36.65, 0.67),
            "Gemini-3.1-Pro-Preview": ("RQ2_result/trust_game_reward/082805_gemini-3.1-pro-preview-three", 99.03, 0.48, 49.93, 0.12),
            "Claude-Opus-4-6": ("RQ2_result/trust_game_reward/082804_claude-opus-4-6-three", 59.54, 0.34, 51.86, 0.29),
            "Kimi-K2.6": ("RQ2_result/trust_game_reward/082802_kimi-k2.6-three", 48.42, 1.37, 49.73, 2.85),
            "Grok-4.3": ("RQ2_result/trust_game_reward/082801_grok-4.3-three", 34.54, 0.75, 46.98, 0.55),
            "DeepSeek-V4-Pro": ("RQ2_result/trust_game_reward/091701_deepseek-v4-pro", 75.17, 1.30, 44.64, 0.45),
        },
    }
    for mechanism, model_rows in trust.items():
        for model, values in model_rows.items():
            path, sender, sender_sd, receiver, receiver_sd = values
            rows.append(Cell("RQ2", "TG", mechanism, model, path, sender, sender_sd, receiver, receiver_sd))
    return rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "RQ1_RQ2_welfare_output")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def log_files(directory: Path) -> list[Path]:
    pattern = re.compile(r"^game_logs?(\d+)\.json$", re.IGNORECASE)
    files = [p for p in directory.rglob("*.json") if pattern.match(p.name)]
    return sorted(files, key=lambda p: int(pattern.match(p.name).group(1)))


def fsum_dict(data: Any) -> float:
    if not isinstance(data, dict):
        return math.nan
    return float(sum(float(value) for value in data.values()))


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def infer_pgg_multiplier(log: list[dict[str, Any]]) -> float:
    ratios = []
    for item in log:
        contribution = float(item.get("total_contribution", 0))
        gain = float(item.get("public_pool_gain", 0))
        if contribution > 0:
            ratios.append(gain / contribution)
    return mean(ratios) if ratios else 9.6


def run_pd(log: list[dict[str, Any]]) -> dict[str, float]:
    decisions: list[float] = []
    total_payoff = 0.0
    names: set[str] = set()
    for item in log:
        detail = item.get("detail", item)
        for suffix in ("1", "2"):
            choice = str(detail.get(f"choice{suffix}", "")).strip().lower()
            decisions.append(float(choice in {"yes", "cooperate", "cooperation", "c"}))
            total_payoff += float(detail.get(f"payoff{suffix}", 0))
            names.add(str(detail.get(f"agent{suffix}_name", detail.get(f"agent{suffix}_id", suffix))))
    n_agents = len(names) or 24
    pairs_per_round = n_agents / 2
    rounds = len(log) / pairs_per_round
    group_round = total_payoff / rounds
    per_agent_round = group_round / n_agents
    normalized = 100.0 * (per_agent_round - 1.0) / 2.0
    return {
        "primary_cooperation_pct": 100.0 * mean(decisions),
        "receiver_cooperation_pct": math.nan,
        "n_agents": float(n_agents),
        "n_rounds": rounds,
        "multiplier": math.nan,
        "final_group_payoff_per_round": group_round,
        "final_payoff_per_agent_round": per_agent_round,
        "base_payoff_per_agent_round": per_agent_round,
        "mechanism_adjustment_per_agent_round": 0.0,
        "resource_adjusted_surplus_per_agent_round": per_agent_round - 1.0,
        "normalized_net_welfare_pct": normalized,
        "normalized_base_welfare_pct": normalized,
    }


def run_pgg(log: list[dict[str, Any]]) -> dict[str, float]:
    contribution_scores: list[float] = []
    final_sums: list[float] = []
    base_sums: list[float] = []
    normalized_final: list[float] = []
    normalized_base: list[float] = []
    multiplier = infer_pgg_multiplier(log)
    n_agents_values: list[int] = []
    endowment = 20.0
    for item in log:
        contributions = item.get("agent_contributions", item.get("contributions", {}))
        n_agents = int(item.get("num_agents", len(contributions) or 24))
        n_agents_values.append(n_agents)
        contribution_scores.extend(float(v) / endowment for v in contributions.values())
        total_contribution = float(item.get("total_contribution", sum(map(float, contributions.values()))))
        public_gain = float(item.get("public_pool_gain", multiplier * total_contribution))
        formula_base = n_agents * endowment - total_contribution + public_gain
        explicit_base = (
            item.get("base_payoffs")
            or item.get("baseline_pg_payoffs")
            or item.get("baseline_payoffs")
        )
        base_total = fsum_dict(explicit_base) if explicit_base else formula_base
        final_map = item.get("final_payoffs") or item.get("payoffs")
        final_total = fsum_dict(final_map) if final_map else base_total
        full_surplus = (multiplier - 1.0) * n_agents * endowment
        final_sums.append(final_total)
        base_sums.append(base_total)
        normalized_final.append(100.0 * (final_total - n_agents * endowment) / full_surplus)
        normalized_base.append(100.0 * (base_total - n_agents * endowment) / full_surplus)
    n_agents = int(round(mean([float(x) for x in n_agents_values]))) if n_agents_values else 24
    group_round = mean(final_sums)
    base_group_round = mean(base_sums)
    return {
        "primary_cooperation_pct": 100.0 * mean(contribution_scores),
        "receiver_cooperation_pct": math.nan,
        "n_agents": float(n_agents),
        "n_rounds": float(len(log)),
        "multiplier": multiplier,
        "final_group_payoff_per_round": group_round,
        "final_payoff_per_agent_round": group_round / n_agents,
        "base_payoff_per_agent_round": base_group_round / n_agents,
        "mechanism_adjustment_per_agent_round": (group_round - base_group_round) / n_agents,
        "resource_adjusted_surplus_per_agent_round": group_round / n_agents - endowment,
        "normalized_net_welfare_pct": mean(normalized_final),
        "normalized_base_welfare_pct": mean(normalized_base),
    }


def run_tg(log: list[dict[str, Any]]) -> dict[str, float]:
    sender: list[float] = []
    receiver: list[float] = []
    final_totals: list[float] = []
    base_totals: list[float] = []
    adjusted_surpluses: list[float] = []
    normalized_final: list[float] = []
    normalized_base: list[float] = []
    roles_per_interaction: list[int] = []
    names: set[str] = set()
    initial_funds = 10.0
    multiplier = 3.0
    full_trust_surplus = (multiplier - 1.0) * initial_funds
    for item in log:
        detail = item.get("detail", item)
        sent = float(detail.get("sent_amount", 0))
        returned = float(detail.get("returned_amount", 0))
        received = float(detail.get("trustee_received", multiplier * sent))
        sender.append(sent / initial_funds)
        if received > 0:
            receiver.append(returned / received)
        has_rewarder = "rewarder_payoff" in detail
        roles = 3 if has_rewarder else 2
        roles_per_interaction.append(roles)
        final_total = float(detail.get("trustor_payoff", 0)) + float(detail.get("trustee_payoff", 0))
        if has_rewarder:
            final_total += float(detail.get("rewarder_payoff", 0))
        if "trustor_base_payoff" in detail and "trustee_base_payoff" in detail:
            base_total = float(detail["trustor_base_payoff"]) + float(detail["trustee_base_payoff"])
        elif "trustee_payoff_before_reward" in detail:
            base_total = (
                float(detail.get("trustor_payoff", 0))
                + float(detail["trustee_payoff_before_reward"])
                + initial_funds
            )
        else:
            base_total = float(detail.get("trustor_payoff", 0)) + float(detail.get("trustee_payoff", 0))
        initial_total = initial_funds + (initial_funds if has_rewarder else 0.0)
        final_totals.append(final_total)
        base_totals.append(base_total)
        adjusted_surpluses.append(final_total - initial_total)
        normalized_final.append(100.0 * (final_total - initial_total) / full_trust_surplus)
        normalized_base.append(100.0 * (base_total - initial_total) / full_trust_surplus)
        for role in ("trustor", "trustee", "rewarder"):
            if f"{role}_name" in detail:
                names.add(str(detail[f"{role}_name"]))
    roles = int(round(mean([float(x) for x in roles_per_interaction])))
    n_agents = len(names) or 24
    rounds = len(log) * roles / n_agents
    group_round = sum(final_totals) / rounds
    base_group_round = sum(base_totals) / rounds
    return {
        "primary_cooperation_pct": 100.0 * mean(sender),
        "receiver_cooperation_pct": 100.0 * mean(receiver),
        "n_agents": float(n_agents),
        "n_rounds": rounds,
        "multiplier": multiplier,
        "final_group_payoff_per_round": group_round,
        "final_payoff_per_agent_round": group_round / n_agents,
        "base_payoff_per_agent_round": base_group_round / n_agents,
        "mechanism_adjustment_per_agent_round": (group_round - base_group_round) / n_agents,
        "resource_adjusted_surplus_per_agent_round": sum(adjusted_surpluses) / (rounds * n_agents),
        "normalized_net_welfare_pct": mean(normalized_final),
        "normalized_base_welfare_pct": mean(normalized_base),
    }


def reconstruct_from_table(cell: Cell) -> list[dict[str, float]]:
    """Use an exact game-accounting identity when the matching raw log is absent."""
    n_agents = 24.0
    if cell.game == "PD":
        if not math.isclose(cell.target_primary_mean_pct, 0.0):
            raise ValueError("PD welfare cannot be reconstructed from a nonzero marginal cooperation rate")
        row = {
            "primary_cooperation_pct": 0.0,
            "receiver_cooperation_pct": math.nan,
            "n_agents": n_agents,
            "n_rounds": 10.0,
            "multiplier": math.nan,
            "final_group_payoff_per_round": 24.0,
            "final_payoff_per_agent_round": 1.0,
            "base_payoff_per_agent_round": 1.0,
            "mechanism_adjustment_per_agent_round": 0.0,
            "resource_adjusted_surplus_per_agent_round": 0.0,
            "normalized_net_welfare_pct": 0.0,
            "normalized_base_welfare_pct": 0.0,
        }
        return [dict(row) for _ in range(3)]
    if cell.game == "PGG":
        if not math.isclose(cell.target_primary_mean_pct, 0.0):
            raise ValueError("PGG welfare reconstruction is only exact for zero contribution")
        row = {
            "primary_cooperation_pct": 0.0,
            "receiver_cooperation_pct": math.nan,
            "n_agents": n_agents,
            "n_rounds": 30.0,
            "multiplier": 9.6,
            "final_group_payoff_per_round": 480.0,
            "final_payoff_per_agent_round": 20.0,
            "base_payoff_per_agent_round": 20.0,
            "mechanism_adjustment_per_agent_round": 0.0,
            "resource_adjusted_surplus_per_agent_round": 0.0,
            "normalized_net_welfare_pct": 0.0,
            "normalized_base_welfare_pct": 0.0,
        }
        return [dict(row) for _ in range(3)]
    # In the base trust game, pair payoff is 10 + 2*sent.  Therefore group
    # welfare depends only on Sender transfer and not on Receiver return.
    sender_fraction = cell.target_primary_mean_pct / 100.0
    surplus_per_agent = 10.0 * sender_fraction
    payoff_per_agent = 5.0 + surplus_per_agent
    row = {
        "primary_cooperation_pct": cell.target_primary_mean_pct,
        "receiver_cooperation_pct": float(cell.target_receiver_mean_pct),
        "n_agents": n_agents,
        "n_rounds": 30.0,
        "multiplier": 3.0,
        "final_group_payoff_per_round": payoff_per_agent * n_agents,
        "final_payoff_per_agent_round": payoff_per_agent,
        "base_payoff_per_agent_round": payoff_per_agent,
        "mechanism_adjustment_per_agent_round": 0.0,
        "resource_adjusted_surplus_per_agent_round": surplus_per_agent,
        "normalized_net_welfare_pct": cell.target_primary_mean_pct,
        "normalized_base_welfare_pct": cell.target_primary_mean_pct,
    }
    # The table omits Gemini's run-level SD, so this is one aggregate record,
    # not an invented set of replicate observations.
    return [row]


METRIC_KEYS = [
    "primary_cooperation_pct",
    "receiver_cooperation_pct",
    "n_agents",
    "n_rounds",
    "multiplier",
    "final_group_payoff_per_round",
    "final_payoff_per_agent_round",
    "base_payoff_per_agent_round",
    "mechanism_adjustment_per_agent_round",
    "resource_adjusted_surplus_per_agent_round",
    "normalized_net_welfare_pct",
    "normalized_base_welfare_pct",
]


def summarize_runs(run_rows: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {"n_runs": float(len(run_rows))}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in run_rows if math.isfinite(float(row[key]))]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_sd"] = sample_sd(values)
    return result


def matches(target: float | None, actual: float) -> bool | None:
    if target is None:
        return None
    return round(target + 1e-12, 2) == round(actual + 1e-12, 2)


def analyze_cell(root: Path, cell: Cell) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cell.relative_dir is None:
        run_metrics = reconstruct_from_table(cell)
        source_status = "accounting_reconstruction_from_attachment; no exactly matching raw directory"
        files: list[Path] = []
    else:
        directory = root / cell.relative_dir
        files = log_files(directory) if directory.is_dir() else []
        if not files:
            raise FileNotFoundError(f"No game logs found for {cell.model} {cell.rq} {cell.game}: {directory}")
        extractor = {"PD": run_pd, "PGG": run_pgg, "TG": run_tg}[cell.game]
        run_metrics = [extractor(read_json(path)) for path in files]
        source_status = "raw_logs"

    summary = summarize_runs(run_metrics)
    primary_mean_match = matches(cell.target_primary_mean_pct, summary["primary_cooperation_pct_mean"])
    primary_sd_match = matches(cell.target_primary_sd_pct, summary["primary_cooperation_pct_sd"])
    receiver_mean_match = matches(cell.target_receiver_mean_pct, summary["receiver_cooperation_pct_mean"])
    receiver_sd_match = matches(cell.target_receiver_sd_pct, summary["receiver_cooperation_pct_sd"])
    checks = [x for x in (primary_mean_match, primary_sd_match, receiver_mean_match, receiver_sd_match) if x is not None]
    if cell.relative_dir is None:
        audit_status = "reconstructed_no_matching_raw"
    else:
        audit_status = "match" if checks and all(checks) else "mismatch"
    result: dict[str, Any] = {
        "rq": cell.rq,
        "game": cell.game,
        "mechanism": cell.mechanism,
        "model": cell.model,
        "source_dir": cell.relative_dir or "",
        "source_status": source_status,
        "source_note": cell.note,
        "log_files": ";".join(str(path.relative_to(root)).replace("\\", "/") for path in files),
        "target_primary_mean_pct": cell.target_primary_mean_pct,
        "target_primary_sd_pct": cell.target_primary_sd_pct,
        "calculated_primary_mean_pct": summary["primary_cooperation_pct_mean"],
        "calculated_primary_sd_pct": summary["primary_cooperation_pct_sd"],
        "target_receiver_mean_pct": cell.target_receiver_mean_pct,
        "target_receiver_sd_pct": cell.target_receiver_sd_pct,
        "calculated_receiver_mean_pct": summary["receiver_cooperation_pct_mean"],
        "calculated_receiver_sd_pct": summary["receiver_cooperation_pct_sd"],
        "audit_status": audit_status,
        **summary,
    }
    detailed = []
    for index, metrics in enumerate(run_metrics, start=1):
        detailed.append({
            "rq": cell.rq,
            "game": cell.game,
            "mechanism": cell.mechanism,
            "model": cell.model,
            "run": index,
            "log_file": str(files[index - 1].relative_to(root)).replace("\\", "/") if files else "",
            "source_status": source_status,
            **metrics,
        })
    return result, detailed


def difference_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rq1 = {
        (row["game"], row["model"]): row
        for row in summaries
        if row["rq"] == "RQ1" and row["mechanism"] == "none"
    }
    metrics = [
        "final_group_payoff_per_round_mean",
        "final_payoff_per_agent_round_mean",
        "base_payoff_per_agent_round_mean",
        "mechanism_adjustment_per_agent_round_mean",
        "resource_adjusted_surplus_per_agent_round_mean",
        "normalized_net_welfare_pct_mean",
        "normalized_base_welfare_pct_mean",
    ]
    rows = []
    for treatment in summaries:
        if treatment["rq"] != "RQ2":
            continue
        baseline = rq1.get((treatment["game"], treatment["model"]))
        if baseline is None:
            continue
        row: dict[str, Any] = {
            "game": treatment["game"],
            "mechanism": treatment["mechanism"],
            "model": treatment["model"],
            "rq1_source_dir": baseline["source_dir"],
            "rq2_source_dir": treatment["source_dir"],
            "rq1_multiplier": baseline["multiplier_mean"],
            "rq2_multiplier": treatment["multiplier_mean"],
        }
        for metric in metrics:
            short = metric.removesuffix("_mean")
            before = float(baseline[metric])
            after = float(treatment[metric])
            row[f"rq1_{short}"] = before
            row[f"rq2_{short}"] = after
            row[f"delta_{short}"] = after - before
        rows.append(row)
    game_order = {"PD": 0, "PGG": 1, "TG": 2}
    mechanism_order = {"punishment": 0, "reputation": 1, "reward": 2}
    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda row: (game_order[row["game"]], mechanism_order[row["mechanism"]], model_order[row["model"]]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def aggregate_differences(differences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for game in ("PD", "PGG", "TG"):
        for mechanism in ("punishment", "reputation", "reward"):
            group = [row for row in differences if row["game"] == game and row["mechanism"] == mechanism]
            if not group:
                continue
            deltas = [float(row["delta_normalized_net_welfare_pct"]) for row in group]
            raw = [float(row["delta_final_payoff_per_agent_round"]) for row in group]
            adjusted = [float(row["delta_resource_adjusted_surplus_per_agent_round"]) for row in group]
            rows.append({
                "game": game,
                "mechanism": mechanism,
                "n_models": len(group),
                "mean_delta_final_payoff_per_agent_round": mean(raw),
                "mean_delta_resource_adjusted_surplus_per_agent_round": mean(adjusted),
                "mean_delta_normalized_net_welfare_pct": mean(deltas),
                "median_delta_normalized_net_welfare_pct": statistics.median(deltas),
                "models_positive_normalized_welfare": sum(value > 0 for value in deltas),
                "models_zero_normalized_welfare": sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in deltas),
                "models_negative_normalized_welfare": sum(value < 0 for value in deltas),
            })
    return rows


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    mismatches = [row for row in summaries if row["audit_status"] == "mismatch"]
    reconstructed = [row for row in summaries if row["audit_status"] == "reconstructed_no_matching_raw"]
    raw_matches = [row for row in summaries if row["audit_status"] == "match"]
    aggregate_lookup = {(row["game"], row["mechanism"]): row for row in aggregates}
    grok_pgg_punishment = next(
        row for row in differences
        if row["game"] == "PGG" and row["mechanism"] == "punishment" and row["model"] == "Grok-4.3"
    )
    lines = [
        "# RQ1–RQ2 群体收益分析",
        "",
        "## 口径",
        "",
        "- 原始群体收益：每次独立运行中，所有参与者最终 payoff 的总和；主表换算为“人均每个 population round 的最终收益”。",
        "- 机制成本只扣一次：优先读取 `final_payoffs` 或日志中已经调整后的角色 payoff。",
        "- 标准化净福利：相对于各博弈无合作起点、除以完全合作可创造的基础社会剩余。该指标允许惩罚导致负值，也允许奖励创造额外收益而超过 100%。",
        "- 信任博弈奖励条件包含第三方奖励者及额外 10 单位禀赋；因此跨条件判断优先看资源调整后的标准化净福利，而不是只看原始最终 payoff。",
        "- 差值统一定义为 `RQ2 − RQ1`；正值表示机制组更高。每个模型等权。",
        "",
        "## 附件数值与原始文件核验",
        "",
        f"共分析 {len(summaries)} 个 RQ1/RQ2 单元格；按附件均值与 SD 四舍五入到两位核验后，{len(raw_matches)} 个由原始日志精确匹配，{len(reconstructed)} 个没有精确匹配日志而采用博弈恒等式重建，{len(mismatches)} 个仍不匹配。",
        "",
    ]
    if reconstructed:
        lines.extend([
            "没有精确匹配原始日志的项目：",
            "",
            "| RQ | 博弈 | 模型 | 处理方式 | 搜索结果 |",
            "|---|---|---|---|---|",
        ])
        for row in reconstructed:
            lines.append(
                f"| {row['rq']} | {row['game']} | {row['model']} | 博弈收益恒等式重建 | {row['source_note']} |"
            )
        lines.append("")
    if mismatches:
        lines.extend([
            "不匹配项目：",
            "",
            "| RQ | 博弈 | 机制 | 模型 | 目标主指标 | 计算主指标 | 来源 |",
            "|---|---|---|---|---:|---:|---|",
        ])
        for row in mismatches:
            lines.append(
                f"| {row['rq']} | {row['game']} | {row['mechanism']} | {row['model']} | "
                f"{fmt(row['target_primary_mean_pct'], 2)} | {fmt(row['calculated_primary_mean_pct'], 2)} | {row['source_dir'] or row['source_status']} |"
            )
        lines.append("")

    pd_pun = aggregate_lookup[("PD", "punishment")]
    pd_rep = aggregate_lookup[("PD", "reputation")]
    pd_rew = aggregate_lookup[("PD", "reward")]
    pgg_pun = aggregate_lookup[("PGG", "punishment")]
    pgg_rep = aggregate_lookup[("PGG", "reputation")]
    pgg_rew = aggregate_lookup[("PGG", "reward")]
    tg_pun = aggregate_lookup[("TG", "punishment")]
    tg_rep = aggregate_lookup[("TG", "reputation")]
    tg_rew = aggregate_lookup[("TG", "reward")]
    lines.extend([
        "## 主要发现",
        "",
        f"- 囚徒困境：三种机制的跨模型平均标准化净福利差均非负；惩罚、声誉、奖励依次为 +{fmt(pd_pun['mean_delta_normalized_net_welfare_pct'])}、+{fmt(pd_rep['mean_delta_normalized_net_welfare_pct'])}、+{fmt(pd_rew['mean_delta_normalized_net_welfare_pct'])} 个百分点，奖励的平均改善最大。",
        f"- 公共物品：声誉的平均改善最大（+{fmt(pgg_rep['mean_delta_normalized_net_welfare_pct'])} 个百分点），其次为惩罚（+{fmt(pgg_pun['mean_delta_normalized_net_welfare_pct'])}）和奖励（+{fmt(pgg_rew['mean_delta_normalized_net_welfare_pct'])}）。但惩罚不是稳健改善：Grok-4.3 虽达到 24.65% 贡献率，标准化净福利却下降 {fmt(abs(float(grok_pgg_punishment['delta_normalized_net_welfare_pct'])))} 个百分点，人均每轮最终收益下降 {fmt(abs(float(grok_pgg_punishment['delta_final_payoff_per_agent_round'])))}。",
        f"- 信任博弈：三个机制均纳入 {tg_rep['n_models']} 个模型；惩罚、声誉、奖励的平均标准化净福利差分别为 {fmt(tg_pun['mean_delta_normalized_net_welfare_pct'])}、{fmt(tg_rep['mean_delta_normalized_net_welfare_pct'])}、{fmt(tg_rew['mean_delta_normalized_net_welfare_pct'])} 个百分点。",
        f"- 信任博弈奖励增加了第三方奖励者和额外预算。扣除这部分初始资源后，奖励的人均社会剩余平均增量为 +{fmt(tg_rew['mean_delta_resource_adjusted_surplus_per_agent_round'])}，低于惩罚的 +{fmt(tg_pun['mean_delta_resource_adjusted_surplus_per_agent_round'])} 和声誉的 +{fmt(tg_rep['mean_delta_resource_adjusted_surplus_per_agent_round'])}。",
        "",
    ])

    lines.extend([
        "## RQ2 − RQ1：逐模型标准化净福利差值",
        "",
        "单位为完全合作基础社会剩余的百分点。这个指标是跨机制比较的主要结论口径。",
        "",
        "| 博弈 | 机制 | 模型 | RQ1 | RQ2 | 差值 | 方向 |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in differences:
        delta = float(row["delta_normalized_net_welfare_pct"])
        direction = "提高" if delta > 1e-12 else "降低" if delta < -1e-12 else "不变"
        lines.append(
            f"| {row['game']} | {row['mechanism']} | {row['model']} | "
            f"{fmt(row['rq1_normalized_net_welfare_pct'])} | {fmt(row['rq2_normalized_net_welfare_pct'])} | "
            f"{fmt(delta)} | {direction} |"
        )

    lines.extend([
        "",
        "## 跨模型描述性汇总",
        "",
        "| 博弈 | 机制 | 模型数 | 平均人均最终收益差 | 平均资源调整后人均剩余差 | 平均标准化净福利差 | 中位数差 | 正/零/负 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in aggregates:
        lines.append(
            f"| {row['game']} | {row['mechanism']} | {row['n_models']} | "
            f"{fmt(row['mean_delta_final_payoff_per_agent_round'])} | "
            f"{fmt(row['mean_delta_resource_adjusted_surplus_per_agent_round'])} | "
            f"{fmt(row['mean_delta_normalized_net_welfare_pct'])} | "
            f"{fmt(row['median_delta_normalized_net_welfare_pct'])} | "
            f"{row['models_positive_normalized_welfare']}/{row['models_zero_normalized_welfare']}/{row['models_negative_normalized_welfare']} |"
        )

    lines.extend([
        "",
        "## 解释限制",
        "",
        "- 最新 RQ1 附件不再包含 Qwen3。Grok-4.3 囚徒困境在附件中为 0.00% ± 0.00%，但现有原始日志为 0.69% ± 0.24%；本次按用户指定的最新附件零合作值，用标准 DD 收益恒等式重建，并在 CSV 中明确标记。",
        "- 老模型的公共物品实验使用的乘数可能与新模型不同；逐模型 RQ2−RQ1 比较使用各自对应模型的日志，跨模型原始 payoff 不应直接横比。",
        "- 本报告核验保存结果与附件数值的一致性，但不自动剔除包含 API 调用失败或默认决策的日志；这类运行在用于正式推断前仍需单独做数据质量审查。",
        "- 当前差值是描述性均值差。正式显著性检验应以独立 game-log 为样本，并根据是否共享随机种子选择配对检验或 Welch/Bootstrap 区间。",
        "",
        "## 机器可读文件",
        "",
        "- `source_match_and_welfare_summary.csv`：文件匹配、附件数值核验和每个单元格的收益汇总。",
        "- `run_level_welfare.csv`：每个独立 game-log 的收益。",
        "- `rq2_minus_rq1_welfare.csv`：逐模型、逐博弈、逐机制差值。",
        "- `mechanism_aggregate_welfare.csv`：跨模型描述性汇总。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for cell in rq1_cells() + rq2_cells():
        summary, details = analyze_cell(root, cell)
        summaries.append(summary)
        run_rows.extend(details)
        print(
            f"{cell.rq} {cell.game:3s} {cell.mechanism:10s} {cell.model:28s} "
            f"target={cell.target_primary_mean_pct:6.2f} calculated={summary['calculated_primary_mean_pct']:6.2f} "
            f"{summary['audit_status']}"
        )

    differences = difference_rows(summaries)
    aggregates = aggregate_differences(differences)
    write_csv(output / "source_match_and_welfare_summary.csv", summaries)
    write_csv(output / "run_level_welfare.csv", run_rows)
    write_csv(output / "rq2_minus_rq1_welfare.csv", differences)
    write_csv(output / "mechanism_aggregate_welfare.csv", aggregates)
    write_report(output / "RQ1_RQ2_group_welfare_report.md", summaries, differences, aggregates)

    mismatch_count = sum(row["audit_status"] == "mismatch" for row in summaries)
    reconstructed_count = sum(row["audit_status"] == "reconstructed_no_matching_raw" for row in summaries)
    print(f"Wrote {output}")
    print(f"Attachment audit mismatches: {mismatch_count}/{len(summaries)}")
    print(f"Accounting reconstructions without matching raw logs: {reconstructed_count}/{len(summaries)}")


if __name__ == "__main__":
    main()
