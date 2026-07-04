#!/usr/bin/env python3
"""Extract compact CSV tables from lm-eval result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_LABELS = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "aya_expanse_8b": "Aya-Expanse-8B",
}

TASK_LABELS = {
    "arc_easy": "ARC-Easy English",
    "belebele_hin_Deva": "Belebele Hindi",
    "belebele_arb_Arab": "Belebele Arabic",
    "belebele_spa_Latn": "Belebele Spanish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=Path("results/manual/stableselect_final_pilot_from_base64.zip"),
        type=Path,
        help="Zip archive or directory containing lm-eval result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/manual/tables"),
        type=Path,
        help="Directory for extracted CSV tables.",
    )
    parser.add_argument(
        "--metric",
        default="acc_norm",
        choices=["acc", "acc_norm"],
        help="Primary metric for ranking and stability tables.",
    )
    return parser.parse_args()


def result_jsons(input_path: Path) -> list[tuple[str, dict[str, Any]]]:
    if input_path.is_dir():
        items = []
        for path in sorted(input_path.glob("**/results_*.json")):
            try:
                source_name = str(path.relative_to(input_path))
            except ValueError:
                source_name = str(path)
            items.append((source_name, json.loads(path.read_text(encoding="utf-8"))))
        return items
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    items = []
    with zipfile.ZipFile(input_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.startswith("results_") and name.endswith(".json")
        )
        for name in names:
            items.append((name, json.loads(archive.read(name).decode("utf-8"))))
    return items


def infer_model(source_name: str) -> str:
    for part in source_name.replace("\\", "/").split("/"):
        if part.startswith("qwen2_5_7b_instruct"):
            return "qwen2_5_7b_instruct"
        if part.startswith("aya_expanse_8b"):
            return "aya_expanse_8b"
    raise ValueError(f"Could not infer model from path: {source_name}")


def infer_precision(source_name: str) -> str:
    for part in source_name.replace("\\", "/").split("/"):
        pieces = part.split("__")
        if len(pieces) >= 2 and pieces[1] in {"4bit", "bf16", "fp16"}:
            return pieces[1]
    return "unknown"


def read_rows(results: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, data in results:
        model_name = infer_model(source_name)
        precision = infer_precision(source_name)
        sample_counts = data.get("n-samples", {})
        for task_name, metrics in data.get("results", {}).items():
            task_samples = sample_counts.get(task_name, {})
            for metric_name in ("acc", "acc_norm"):
                score_key = f"{metric_name},none"
                stderr_key = f"{metric_name}_stderr,none"
                if score_key not in metrics:
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "model_label": MODEL_LABELS.get(model_name, model_name),
                        "precision": precision,
                        "task_name": task_name,
                        "task_label": TASK_LABELS.get(task_name, task_name),
                        "examples": task_samples.get("effective", ""),
                        "metric": metric_name,
                        "score": float(metrics[score_key]),
                        "stderr": metrics.get(stderr_key, ""),
                        "source_json": source_name,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: (item["task_label"], item["metric"], item["model_label"])):
        copied = dict(row)
        copied["score"] = f"{row['score']:.6f}"
        if copied["stderr"] != "":
            copied["stderr"] = f"{float(copied['stderr']):.6f}"
        out.append(copied)
    return out


def rank_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == metric:
            groups[(row["task_name"], row["task_label"], row["precision"])].append(row)

    out: list[dict[str, Any]] = []
    for (_task_name, task_label, precision), items in sorted(groups.items()):
        ranked = sorted(items, key=lambda item: (-item["score"], item["model_label"]))
        current_rank = 0
        previous_score: float | None = None
        for index, item in enumerate(ranked, start=1):
            if previous_score is None or abs(item["score"] - previous_score) > 1e-12:
                current_rank = index
                previous_score = item["score"]
            out.append(
                {
                    "task_label": task_label,
                    "precision": precision,
                    "metric": metric,
                    "rank": current_rank,
                    "model_name": item["model_name"],
                    "model_label": item["model_label"],
                    "score": f"{item['score']:.6f}",
                }
            )
    return out


def wide_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["metric"] == metric:
            groups[(row["task_name"], row["task_label"], row["precision"])][row["model_name"]] = row

    out: list[dict[str, Any]] = []
    for (_task_name, task_label, precision), by_model in sorted(groups.items()):
        qwen = by_model.get("qwen2_5_7b_instruct")
        aya = by_model.get("aya_expanse_8b")
        if qwen is None or aya is None:
            continue
        qwen_score = qwen["score"]
        aya_score = aya["score"]
        if qwen_score > aya_score:
            winner = "Qwen"
        elif aya_score > qwen_score:
            winner = "Aya"
        else:
            winner = "Tie"
        out.append(
            {
                "task_label": task_label,
                "precision": precision,
                "metric": metric,
                "examples": qwen["examples"] or aya["examples"],
                "qwen_score": f"{qwen_score:.6f}",
                "aya_score": f"{aya_score:.6f}",
                "winner": winner,
                "qwen_minus_aya": f"{qwen_score - aya_score:.6f}",
            }
        )
    return out


def stability_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    model_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == metric:
            model_scores[row["model_name"]].append(row)

    out: list[dict[str, Any]] = []
    for model_name, items in sorted(model_scores.items()):
        english = [item for item in items if item["task_name"] == "arc_easy"]
        baseline = english[0]["score"] if english else max(item["score"] for item in items)
        scores = [item["score"] for item in items]
        worst_drop = max(baseline - score for score in scores)
        avg_score = mean(scores)
        out.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "metric": metric,
                "baseline_task": "ARC-Easy English",
                "baseline_score": f"{baseline:.6f}",
                "average_score": f"{avg_score:.6f}",
                "worst_group_drop": f"{worst_drop:.6f}",
                "deployment_stability_score": f"{avg_score - worst_drop:.6f}",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    results = result_jsons(args.input)
    if not results:
        raise SystemExit(f"No lm-eval result JSON files found in {args.input}")
    rows = read_rows(results)
    write_csv(args.output_dir / "pilot_lm_eval_scores.csv", score_rows(rows))
    write_csv(args.output_dir / "pilot_lm_eval_ranks.csv", rank_rows(rows, args.metric))
    write_csv(args.output_dir / "pilot_lm_eval_wide.csv", wide_rows(rows, args.metric))
    write_csv(
        args.output_dir / "pilot_lm_eval_stability.csv",
        stability_rows(rows, args.metric),
    )
    print(f"read {len(results)} result files")
    print(f"wrote tables to {args.output_dir}")


if __name__ == "__main__":
    main()
