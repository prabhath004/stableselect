#!/usr/bin/env python3
"""Run a small exact-answer QA evaluation with a Hugging Face causal LM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Short model name for result files.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "4bit"],
        required=True,
        help="Inference precision to evaluate.",
    )
    parser.add_argument("--task-group", required=True, help="Logical task group name.")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL QA dataset.")
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def load_jsonl(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise ValueError(f"No examples found in {path}")
    return records


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z\u0600-\u06ff\u0900-\u097f\u0c00-\u0c7f]+", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, answers: list[str]) -> bool:
    normalized_prediction = normalize(prediction)
    for answer in answers:
        normalized_answer = normalize(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return True
    return False


def build_prompt(tokenizer: Any, question: str) -> str:
    instruction = (
        f"{question}\n\n"
        "Give the shortest exact answer. Do not explain your reasoning."
    )
    messages = [{"role": "user", "content": instruction}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {instruction}\nAnswer:"


def dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if precision in {"fp16", "4bit"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8 and torch.cuda.is_bf16_supported()


def quantization_compute_dtype() -> torch.dtype:
    return torch.bfloat16 if supports_bf16() else torch.float16


def load_model(model_id: str, precision: str, device_map: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = dtype_for_precision(precision)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
    }

    if precision == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit bitsandbytes inference needs a CUDA GPU. "
                "Run bf16/fp16 locally or move this run to a CUDA machine."
            )
        compute_dtype = quantization_compute_dtype()
        print(f"using bitsandbytes 4-bit compute dtype: {compute_dtype}")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def model_input_device(model: Any) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_answer(
    tokenizer: Any,
    model: Any,
    question: str,
    max_new_tokens: int,
) -> tuple[str, float, float | None]:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_input_device(model))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_s = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return prediction, latency_s, peak_memory_mb


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(
    predictions: list[dict[str, Any]],
    run_id: str,
    args: argparse.Namespace,
    predictions_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {"all": predictions}
    for pred in predictions:
        groups.setdefault(pred["language"], []).append(pred)

    timestamp = datetime.now(timezone.utc).isoformat()
    for language, items in sorted(groups.items()):
        correct = sum(1 for item in items if item["correct"])
        latencies = [float(item["latency_s"]) for item in items]
        memory_values = [
            float(item["peak_memory_mb"])
            for item in items
            if item.get("peak_memory_mb") is not None
        ]
        accuracy = correct / len(items)
        rows.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "model_name": args.model_name,
                "model_id": args.model_id,
                "precision": args.precision,
                "task_group": args.task_group,
                "language": language,
                "num_examples": len(items),
                "correct": correct,
                "accuracy": f"{accuracy:.6f}",
                "score": f"{accuracy:.6f}",
                "avg_latency_s": f"{sum(latencies) / len(latencies):.6f}",
                "peak_memory_mb": (
                    f"{max(memory_values):.2f}" if memory_values else ""
                ),
                "predictions_path": str(predictions_path),
            }
        )
    return rows


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.dataset, args.max_examples)
    tokenizer, model = load_model(args.model_id, args.precision, args.device_map)

    run_id = (
        f"{args.model_name}__{args.precision}__{args.task_group}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    predictions_path = args.output_dir / "raw" / f"{run_id}.jsonl"
    metrics_path = args.output_dir / "metrics" / f"{run_id}.csv"

    predictions: list[dict[str, Any]] = []
    for example in examples:
        prediction, latency_s, peak_memory_mb = generate_answer(
            tokenizer=tokenizer,
            model=model,
            question=example["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        answers = [str(answer) for answer in example["answers"]]
        predictions.append(
            {
                "id": example["id"],
                "language": example.get("language", "unknown"),
                "task_group": args.task_group,
                "prompt": example["prompt"],
                "answers": answers,
                "prediction": prediction,
                "correct": is_correct(prediction, answers),
                "latency_s": latency_s,
                "peak_memory_mb": peak_memory_mb,
            }
        )

    write_predictions(predictions_path, predictions)
    metric_rows = summarize(predictions, run_id, args, predictions_path)
    write_metrics(metrics_path, metric_rows)

    all_row = next(row for row in metric_rows if row["language"] == "all")
    print(
        f"wrote {metrics_path} | "
        f"score={all_row['score']} | "
        f"predictions={predictions_path}"
    )


if __name__ == "__main__":
    main()
