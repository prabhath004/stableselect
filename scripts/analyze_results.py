#!/usr/bin/env python3
"""Aggregate StableSelect pilot metrics into ranking-stability tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=ROOT / "results" / "metrics", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results" / "tables", type=Path)
    parser.add_argument("--baseline-task", default="english")
    parser.add_argument("--baseline-precision", default="bf16")
    return parser.parse_args()


def read_metric_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["language"] == "all":
                    row["score"] = float(row["score"])
                    row["num_examples"] = int(row["num_examples"])
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ranks_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: row["score"], reverse=True)
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(sorted_rows, start=1):
        ranked.append(
            {
                "task_group": row["task_group"],
                "precision": row["precision"],
                "model_name": row["model_name"],
                "score": f"{row['score']:.6f}",
                "rank": index,
            }
        )
    return ranked


def spearman_against_baseline(
    baseline_rank: dict[str, int],
    setting_rank: dict[str, int],
) -> float | None:
    common_models = sorted(set(baseline_rank) & set(setting_rank))
    n = len(common_models)
    if n < 2:
        return None
    squared_diffs = [
        (baseline_rank[model] - setting_rank[model]) ** 2 for model in common_models
    ]
    return 1 - (6 * sum(squared_diffs)) / (n * (n * n - 1))


def main() -> None:
    args = parse_args()
    rows = read_metric_rows(args.input_dir)
    if not rows:
        raise SystemExit(
            f"No metric CSVs found in {args.input_dir}. "
            "Run scripts/run_custom_qa.py or scripts/run_matrix.py first."
        )

    by_setting: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[(row["task_group"], row["precision"])].append(row)
        by_model[row["model_name"]].append(row)

    baseline_key = (args.baseline_task, args.baseline_precision)
    if baseline_key not in by_setting:
        fallback_key = None
        for candidate_precision in ("fp16", "bf16", "4bit"):
            candidate_key = (args.baseline_task, candidate_precision)
            if candidate_key in by_setting:
                fallback_key = candidate_key
                break
        if fallback_key is None and len(by_setting) == 1:
            fallback_key = next(iter(by_setting))
        if fallback_key is None:
            available = ", ".join(
                f"{task}/{precision}" for task, precision in sorted(by_setting)
            )
            raise SystemExit(
                f"Missing baseline setting: {baseline_key}. "
                f"Available settings: {available}"
            )
        print(
            f"requested baseline {baseline_key} not found; "
            f"using {fallback_key} instead"
        )
        baseline_key = fallback_key
    baseline_task, baseline_precision = baseline_key

    setting_rank_rows: list[dict[str, Any]] = []
    setting_rank_maps: dict[tuple[str, str], dict[str, int]] = {}
    for setting, setting_rows in sorted(by_setting.items()):
        ranked = ranks_for(setting_rows)
        setting_rank_rows.extend(ranked)
        setting_rank_maps[setting] = {
            row["model_name"]: int(row["rank"]) for row in ranked
        }

    baseline_ranked = ranks_for(by_setting[baseline_key])
    baseline_top = baseline_ranked[0]["model_name"]
    baseline_rank_map = {
        row["model_name"]: int(row["rank"]) for row in baseline_ranked
    }

    stability_rows: list[dict[str, Any]] = []
    for setting, rank_map in sorted(setting_rank_maps.items()):
        ranked = ranks_for(by_setting[setting])
        top_model = ranked[0]["model_name"]
        rho = spearman_against_baseline(baseline_rank_map, rank_map)
        stability_rows.append(
            {
                "task_group": setting[0],
                "precision": setting[1],
                "top_model": top_model,
                "baseline_top_model": baseline_top,
                "top1_flip": str(top_model != baseline_top),
                "spearman_vs_baseline": "" if rho is None else f"{rho:.6f}",
            }
        )

    model_stability_rows: list[dict[str, Any]] = []
    for model_name, model_rows in sorted(by_model.items()):
        clean_rows = [
            row
            for row in model_rows
            if row["task_group"] == baseline_task
            and row["precision"] == baseline_precision
        ]
        if not clean_rows:
            continue
        clean_score = clean_rows[0]["score"]
        scores = [row["score"] for row in model_rows]
        worst_group_drop = max(clean_score - score for score in scores)
        average_score = mean(scores)
        deployment_stability_score = average_score - worst_group_drop
        model_stability_rows.append(
            {
                "model_name": model_name,
                "clean_baseline_score": f"{clean_score:.6f}",
                "average_score": f"{average_score:.6f}",
                "worst_group_drop": f"{worst_group_drop:.6f}",
                "deployment_stability_score": f"{deployment_stability_score:.6f}",
            }
        )

    quant_rows: list[dict[str, Any]] = []
    for model_name, model_rows in sorted(by_model.items()):
        by_task_precision = {
            (row["task_group"], row["precision"]): row["score"] for row in model_rows
        }
        task_groups = sorted({row["task_group"] for row in model_rows})
        for task_group in task_groups:
            fp_score = by_task_precision.get((task_group, baseline_precision))
            q4_score = by_task_precision.get((task_group, "4bit"))
            if fp_score is None or q4_score is None:
                continue
            quant_rows.append(
                {
                    "model_name": model_name,
                    "task_group": task_group,
                    "baseline_precision": baseline_precision,
                    "baseline_score": f"{fp_score:.6f}",
                    "quantized_precision": "4bit",
                    "quantized_score": f"{q4_score:.6f}",
                    "quantization_drop": f"{fp_score - q4_score:.6f}",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "setting_ranks.csv", setting_rank_rows)
    write_csv(args.output_dir / "rank_stability.csv", stability_rows)
    write_csv(args.output_dir / "model_stability.csv", model_stability_rows)
    write_csv(args.output_dir / "quantization_drop.csv", quant_rows)

    top1_flips = sum(row["top1_flip"] == "True" for row in stability_rows)
    print(f"wrote analysis tables to {args.output_dir}")
    print(f"baseline top model: {baseline_top}")
    print(f"top-1 flips: {top1_flips}/{len(stability_rows)} settings")


if __name__ == "__main__":
    main()
