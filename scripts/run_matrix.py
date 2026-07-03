#!/usr/bin/env python3
"""Run the configured StableSelect pilot matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=ROOT / "configs" / "experiment_matrix.yaml", type=Path)
    parser.add_argument("--models", default=ROOT / "configs" / "models.yaml", type=Path)
    parser.add_argument("--tasks", default=ROOT / "configs" / "tasks.yaml", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def command_for_run(
    model: dict[str, Any],
    precision: str,
    task_group: str,
    dataset: str,
    matrix: dict[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_custom_qa.py"),
        "--model-name",
        model["name"],
        "--model-id",
        model["model_id"],
        "--precision",
        precision,
        "--task-group",
        task_group,
        "--dataset",
        str(ROOT / dataset),
        "--max-examples",
        str(matrix["max_examples_per_task"]),
        "--max-new-tokens",
        str(matrix["max_new_tokens"]),
    ]


def main() -> None:
    args = parse_args()
    matrix = read_yaml(args.matrix)
    model_config = read_yaml(args.models)
    task_config = read_yaml(args.tasks)

    models_by_name = {model["name"]: model for model in model_config["models"]}
    datasets = task_config["local_qa_datasets"]
    continue_on_error = bool(matrix.get("continue_on_error", False))

    commands: list[list[str]] = []
    for model_name in matrix["models"]:
        if model_name not in models_by_name:
            raise KeyError(f"Unknown model in matrix: {model_name}")
        for precision in matrix["precisions"]:
            for task_group in matrix["task_groups"]:
                if task_group not in datasets:
                    raise KeyError(f"Unknown local dataset task group: {task_group}")
                commands.append(
                    command_for_run(
                        model=models_by_name[model_name],
                        precision=precision,
                        task_group=task_group,
                        dataset=datasets[task_group],
                        matrix=matrix,
                    )
                )

    for command in commands:
        printable = " ".join(command)
        if args.dry_run:
            print(printable)
            continue
        print(f"running: {printable}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0 and not continue_on_error:
            raise SystemExit(completed.returncode)
        if completed.returncode != 0:
            print(f"run failed with exit code {completed.returncode}: {printable}")


if __name__ == "__main__":
    main()

