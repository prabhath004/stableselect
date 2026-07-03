# StableSelect Pilot Protocol

## Research Question

Do clean leaderboard-style model rankings remain stable under deployment
constraints?

## Pilot Goal

Check whether model rankings change between clean BF16 English evaluation and
4-bit multilingual or code-switched evaluation.

## Models

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `CohereLabs/aya-expanse-8b`

## Precision Settings

- BF16 or FP16 baseline
- 4-bit quantized inference

## Task Groups

- English QA
- Multilingual QA
- Code-switched QA

## Main Metrics

- task score / accuracy
- quantization drop
- worst-group drop
- top-1 ranking flip
- rank correlation against clean baseline
- Deployment Stability Score

## Baseline Setting

The clean leaderboard-style baseline is:

```text
task_group = english
precision = bf16
```

## Deployment Settings

Deployment settings are every non-baseline condition:

```text
english + 4bit
multilingual + bf16
multilingual + 4bit
code_switched + bf16
code_switched + 4bit
```

## First Success Condition

The pilot is successful if it produces one table with scores for each model,
precision, and task group.

## Strong Paper Signal

The strongest result is:

> The clean baseline winner is not the top-ranked model in at least one
> deployment setting, or it has a worse Deployment Stability Score than another
> model.

