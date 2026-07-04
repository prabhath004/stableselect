# Kaggle Qwen/Aya 4-bit Pilot Summary

Date: 2026-07-04

GPU: Kaggle Tesla P100 16GB

Precision: 4-bit bitsandbytes

## Standard Benchmark Results

| Task | Examples | Metric | Qwen2.5-7B-Instruct | Aya-Expanse-8B | Winner |
|---|---:|---|---:|---:|---|
| ARC-Easy English | 100 | acc_norm | 0.79 | 0.78 | Qwen |
| Belebele Hindi | 50 | acc_norm | 0.48 | 0.62 | Aya |
| Belebele Arabic | 50 | acc_norm | 0.76 | 0.74 | Qwen |
| Belebele Spanish | 50 | acc_norm | 0.90 | 0.72 | Qwen |

## Pilot Finding

The top model changed across task and language settings. Qwen was slightly ahead
on English ARC-Easy and stronger on Arabic and Spanish Belebele, while Aya was
stronger on Hindi Belebele.

This is pilot evidence only. It should be rerun with saved result files, larger
subsets, and a third model before being used as paper evidence.

## Files Note

The Kaggle result files were lost after the notebook session refreshed. The
numeric values above were copied from the notebook output shown during the run.

