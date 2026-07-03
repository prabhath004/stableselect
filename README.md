# StableSelect

Deployment-aware model selection for open LLMs.

StableSelect asks whether clean leaderboard-style rankings remain stable under
real deployment constraints such as quantized inference, multilingual prompts,
and code-switched user inputs.

## Current Goal

Build a NeurIPS workshop-sized empirical paper around this claim:

> The model that wins under clean benchmark settings is not always the most
> stable deployment choice.

The first milestone is a small pilot:

```text
3 models x 2 precision settings x 3 task groups = 18 settings
```

## Repo Layout

```text
configs/                 Experiment configuration
data/pilot/              Tiny sanity-check QA sets
paper/                   Submission outline and draft skeleton
scripts/                 Evaluation and analysis scripts
results/raw/             Model predictions
results/metrics/         Per-run metrics CSVs
results/tables/          Aggregated analysis tables
```

## Setup

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Kaggle P100, use the P100-specific stack instead:

```bash
pip install -r requirements-kaggle-p100.txt
```

For gated models such as Llama, log in to Hugging Face:

```bash
.venv/bin/hf auth login
```

For 4-bit inference with `bitsandbytes`, use a CUDA GPU environment. If you are
on a Mac laptop, start with BF16/FP16 sanity runs or move the 4-bit runs to a
cloud GPU.

## First Smoke Test

Before running 7B-8B research models, validate the pipeline with a small model:

```bash
python scripts/run_custom_qa.py \
  --model-name qwen2_5_0_5b_instruct_smoke \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --precision fp16 \
  --task-group english \
  --dataset data/pilot/english_qa.jsonl \
  --max-examples 3
```

The smoke-test score is not research evidence. It only checks that downloads,
prompting, prediction writing, and metrics writing work.

## First Research Sanity Run

Run one model on the tiny English pilot set:

```bash
python scripts/run_custom_qa.py \
  --model-name qwen2_5_7b_instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --precision bf16 \
  --task-group english \
  --dataset data/pilot/english_qa.jsonl \
  --max-examples 5
```

This writes:

```text
results/raw/*.jsonl
results/metrics/*.csv
```

## Run the Pilot Matrix

Preview commands first:

```bash
python scripts/run_matrix.py --dry-run
```

Run the configured matrix:

```bash
python scripts/run_matrix.py
```

Analyze results:

```bash
python scripts/analyze_results.py
```

## Standard Benchmark Path

The tiny local datasets are only for pipeline validation. For paper evidence,
use standard tasks through `lm-evaluation-harness` where possible:

```bash
python scripts/run_lm_eval.py \
  --model qwen2_5_7b_instruct \
  --precision bf16 \
  --task-group english \
  --limit 20 \
  --dry-run
```

Check task names available in your installed harness:

```bash
lm_eval --tasks list
```

Then update `configs/tasks.yaml`.

## What Counts as a Useful Pilot Result

The pilot becomes paper-useful if it shows at least one of these:

- the top model changes under 4-bit quantization,
- the top model changes under multilingual or code-switched prompts,
- the best average-scoring model is not the most stable model,
- average scores hide a large worst-group drop.

## Submission Target

Primary target: NeurIPS workshop on LLM evaluation, foundation model evaluation,
reliable ML, efficient ML, or multilingual LLMs.

Keep the first paper tight: four pages, one clear protocol, one clear result
table, and one ranking-stability figure.
