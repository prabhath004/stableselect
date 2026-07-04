# Kaggle Qwen/Aya 4-bit Pilot Summary

Date: 2026-07-04

GPU: Kaggle Tesla P100 16GB

Precision: 4-bit bitsandbytes

## Standard Benchmark Results

These values are from the recovered `lm-eval` JSON files in
`stableselect_final_pilot_from_base64.zip`.

| Task | Examples | Metric | Qwen2.5-7B-Instruct | Aya-Expanse-8B | Winner |
|---|---:|---|---:|---:|---|
| ARC-Easy English | 100 | acc | 0.74 | 0.75 | Aya |
| ARC-Easy English | 100 | acc_norm | 0.80 | 0.77 | Qwen |
| Belebele Hindi | 50 | acc_norm | 0.58 | 0.62 | Aya |
| Belebele Arabic | 50 | acc_norm | 0.76 | 0.76 | Tie |
| Belebele Spanish | 50 | acc_norm | 0.90 | 0.72 | Qwen |

## Pilot Finding

The top model changed across task, metric, and language settings. Qwen was
ahead on ARC-Easy normalized accuracy and Spanish Belebele, Aya was ahead on
ARC-Easy raw accuracy and Hindi Belebele, and the models tied on Arabic
Belebele.

Using normalized accuracy as the primary metric, Qwen has the higher average
score across the four pilot tasks (0.7600 vs. 0.7175), but Aya has the higher
deployment stability score (0.5675 vs. 0.5400) because Qwen drops more sharply
from the English baseline to Hindi.

This is pilot evidence only. It should be rerun with saved result files, larger
subsets, and a third model before being used as paper evidence.

## Extracted Tables

The recovered JSON files were converted into CSV tables:

- `results/manual/tables/pilot_lm_eval_scores.csv`
- `results/manual/tables/pilot_lm_eval_wide.csv`
- `results/manual/tables/pilot_lm_eval_ranks.csv`
- `results/manual/tables/pilot_lm_eval_stability.csv`

## Files Note

The first Kaggle result files were lost after a notebook refresh. The rerun was
downloaded through a base64 transfer and reconstructed locally as
`stableselect_final_pilot_from_base64.zip`. The archive passed `unzip -t`.
