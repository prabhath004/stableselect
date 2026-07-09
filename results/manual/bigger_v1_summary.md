# Bigger V1 3-Model 4-Bit Summary

Date: 2026-07-09

GPU: Kaggle Tesla P100 16GB

Precision: 4-bit bitsandbytes with float16 compute

Models:

- Qwen2.5-7B-Instruct
- Aya-Expanse-8B
- Llama-3.1-8B-Instruct

Tasks:

- ARC-Easy English, 500 examples
- HellaSwag English, 500 examples
- Belebele Hindi, 200 examples
- Belebele Arabic, 200 examples
- Belebele Spanish, 200 examples

Primary metric: normalized accuracy (`acc_norm`)

## Main Results

| Task | Examples | Aya | Llama | Qwen | Winner |
|---|---:|---:|---:|---:|---|
| ARC-Easy English | 500 | 0.768 | 0.752 | 0.774 | Qwen |
| HellaSwag English | 500 | 0.678 | 0.678 | 0.672 | Aya/Llama tie |
| Belebele Hindi | 200 | 0.600 | 0.585 | 0.630 | Qwen |
| Belebele Arabic | 200 | 0.700 | 0.705 | 0.810 | Qwen |
| Belebele Spanish | 200 | 0.730 | 0.780 | 0.865 | Qwen |

## Stability Table

| Model | ARC-Easy baseline | Average score | Worst drop | Deployment stability score |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.774 | 0.7502 | 0.144 | 0.6062 |
| Llama-3.1-8B-Instruct | 0.752 | 0.7000 | 0.167 | 0.5330 |
| Aya-Expanse-8B | 0.768 | 0.6952 | 0.168 | 0.5272 |

## Interpretation

The bigger run confirms that task choice can change model rankings: Qwen is top
on ARC-Easy and all three Belebele tasks, but HellaSwag ranks Aya and Llama
above Qwen. In this larger 3-model run, however, the average-score and
deployment-stability criteria both select Qwen.

This is stronger evidence than the tiny pilot because it includes a third model
and larger subsets, but it is still not final paper evidence. The next
experiment should add another deployment axis, preferably precision or
quantization comparison on stronger hardware, or add more models/tasks if only
P100 is available.

## Files

- Raw JSON archive: `results/manual/stableselect_bigger_v1_final_from_folder.zip`
- Score table: `results/manual/tables_bigger_v1/bigger_v1_lm_eval_scores.csv`
- Wide table: `results/manual/tables_bigger_v1/bigger_v1_lm_eval_wide.csv`
- Rank table: `results/manual/tables_bigger_v1/bigger_v1_lm_eval_ranks.csv`
- Stability table: `results/manual/tables_bigger_v1/bigger_v1_lm_eval_stability.csv`
