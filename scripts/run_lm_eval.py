#!/usr/bin/env python3
"""Build or run lm-evaluation-harness commands from StableSelect configs."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name from configs/models.yaml")
    parser.add_argument("--precision", required=True, help="Precision from configs/precisions.yaml")
    parser.add_argument("--task-group", required=True, help="Task group from configs/tasks.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task for pilots.")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def main() -> None:
    args = parse_args()
    models = read_yaml(ROOT / "configs" / "models.yaml")["models"]
    precisions = read_yaml(ROOT / "configs" / "precisions.yaml")["precisions"]
    task_groups = read_yaml(ROOT / "configs" / "tasks.yaml")["lm_eval_task_groups"]

    model_by_name = {model["name"]: model for model in models}
    if args.model not in model_by_name:
        raise KeyError(f"Unknown model: {args.model}")
    if args.precision not in precisions:
        raise KeyError(f"Unknown precision: {args.precision}")
    if args.task_group not in task_groups:
        raise KeyError(f"Unknown task group: {args.task_group}")

    tasks = task_groups[args.task_group]
    if not tasks:
        raise ValueError(
            f"No lm-eval tasks configured for task group {args.task_group}. "
            "Use the custom QA runner or update configs/tasks.yaml."
        )

    model = model_by_name[args.model]
    model_args = {"pretrained": model["model_id"]}
    model_args.update(precisions[args.precision]["lm_eval_model_args"])
    model_args_text = ",".join(
        f"{key}={format_value(value)}" for key, value in model_args.items()
    )

    run_id = (
        f"{args.model}__{args.precision}__{args.task_group}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_path = ROOT / "results" / "lm_eval" / run_id

    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args_text,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(args.num_fewshot),
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(output_path),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    printable = " ".join(command)
    if args.dry_run:
        print(printable)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"running: {printable}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

