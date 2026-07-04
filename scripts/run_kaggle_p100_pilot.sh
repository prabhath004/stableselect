#!/usr/bin/env bash
set -euo pipefail

# Run this from the repo root inside a Kaggle P100 notebook.
# It saves a zip checkpoint after each benchmark pair to /kaggle/working.

MODEL_ARGS_QWEN="pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=float16,trust_remote_code=True,load_in_4bit=True,bnb_4bit_compute_dtype=float16"
MODEL_ARGS_AYA="pretrained=CohereLabs/aya-expanse-8b,dtype=float16,trust_remote_code=True,load_in_4bit=True,bnb_4bit_compute_dtype=float16"

checkpoint() {
  local name="$1"
  mkdir -p /kaggle/working
  zip -r "/kaggle/working/${name}.zip" results
  echo "wrote /kaggle/working/${name}.zip"
}

run_belebele_pair() {
  local task="$1"
  local label="$2"

  lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS_QWEN}" \
    --tasks "${task}" \
    --num_fewshot 0 \
    --batch_size auto \
    --limit 50 \
    --output_path "results/lm_eval/qwen2_5_7b_instruct__4bit__${task}"

  lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS_AYA}" \
    --tasks "${task}" \
    --num_fewshot 0 \
    --batch_size auto \
    --limit 50 \
    --output_path "results/lm_eval/aya_expanse_8b__4bit__${task}"

  checkpoint "stableselect_after_${label}"
}

python scripts/check_gpu_env.py

python scripts/run_lm_eval.py \
  --model qwen2_5_7b_instruct \
  --precision 4bit \
  --task-group english \
  --limit 100

python scripts/run_lm_eval.py \
  --model aya_expanse_8b \
  --precision 4bit \
  --task-group english \
  --limit 100

checkpoint "stableselect_after_english"

run_belebele_pair "belebele_hin_Deva" "hindi"
run_belebele_pair "belebele_arb_Arab" "arabic"
run_belebele_pair "belebele_spa_Latn" "spanish"

checkpoint "stableselect_final_pilot"

