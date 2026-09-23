#!/usr/bin/env python3
"""Create an analysis-ready dataset from the first N trust-game rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def resolve_source(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    log_path = source / "data" / "game_logs.json"
    config_path = source / "experiment_config.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing source log: {log_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    return log_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--pairs-per-round", type=int, default=12)
    args = parser.parse_args()

    if args.rounds <= 0 or args.pairs_per_round <= 0:
        raise ValueError("rounds and pairs-per-round must be positive")

    keep_count = args.rounds * args.pairs_per_round
    prepared = []
    for source in args.source:
        log_path, config_path = resolve_source(source)
        logs = load_json(log_path)
        if not isinstance(logs, list):
            raise TypeError(f"Expected a JSON array in {log_path}")
        if len(logs) < keep_count:
            raise ValueError(
                f"{log_path} has {len(logs)} interactions; {keep_count} required"
            )
        kept_logs = logs[:keep_count]
        expected_interactions = list(range(1, keep_count + 1))
        actual_interactions = [item.get("interaction") for item in kept_logs]
        if actual_interactions != expected_interactions:
            raise ValueError(f"Unexpected interaction sequence in {log_path}")
        prepared.append((source.resolve(), log_path, config_path, logs, kept_logs))

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    data_dir = output / "data"
    data_dir.mkdir(parents=True)

    manifest_sources = []
    for index, (source, log_path, _, logs, kept_logs) in enumerate(prepared, 1):
        output_name = f"game_log{index}.json"
        save_json(data_dir / output_name, kept_logs)
        manifest_sources.append(
            {
                "output_file": f"data/{output_name}",
                "source_directory": str(source),
                "source_file": str(log_path),
                "source_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                "original_interactions": len(logs),
                "kept_interactions": len(kept_logs),
                "kept_population_rounds": args.rounds,
                "first_timestamp": kept_logs[0].get("timestamp"),
                "last_timestamp": kept_logs[-1].get("timestamp"),
            }
        )

    config = load_json(prepared[0][2])
    config["experiment_name"] = "TrustGame_Population_First_30_Rounds"
    config["description"] = (
        "Derived dataset containing the first 30 population rounds from longer "
        "50- and 100-round experiments. These are truncated observations, not "
        "new independent 30-round experiments."
    )
    game_settings = config.setdefault("game_settings", {})
    game_settings["num_rounds"] = args.rounds
    game_settings["total_population_rounds"] = args.rounds
    config["derived_data"] = {
        "operation": "prefix truncation",
        "rounds_kept": args.rounds,
        "pairs_per_round": args.pairs_per_round,
        "interactions_kept_per_source": keep_count,
        "horizon_caveat": (
            "Agents originally made these decisions while assigned to longer "
            "50- or 100-round experiments."
        ),
    }
    save_json(output / "experiment_config.json", config)

    manifest = {
        "dataset_type": "derived_prefix_dataset",
        "rounds_kept": args.rounds,
        "pairs_per_round": args.pairs_per_round,
        "sources": manifest_sources,
    }
    save_json(output / "source_manifest.json", manifest)

    print(f"Created: {output}")
    for source in manifest_sources:
        print(
            f"  {source['output_file']}: "
            f"{source['kept_interactions']}/{source['original_interactions']} interactions"
        )


if __name__ == "__main__":
    main()
