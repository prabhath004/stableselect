#!/usr/bin/env bash
set -euo pipefail

# Bigger StableSelect run for Kaggle P100.
# Run from repo root inside a Kaggle GPU notebook.
#
# Defaults are intentionally moderate for P100:
#   ENGLISH_LIMIT=500
#   BELEBELE_LIMIT=200
#
# To run full tasks, override with:
#   ENGLISH_LIMIT=0 BELEBELE_LIMIT=0 bash scripts/run_kaggle_p100_bigger.sh
#
# To add Llama after HF access is approved:
#   RUN_LLAMA=1 bash scripts/run_kaggle_p100_bigger.sh

ENGLISH_LIMIT="${ENGLISH_LIMIT:-500}"
BELEBELE_LIMIT="${BELEBELE_LIMIT:-200}"
RUN_LLAMA="${RUN_LLAMA:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

MODEL_ARGS_QWEN="pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=float16,trust_remote_code=True,load_in_4bit=True,bnb_4bit_compute_dtype=float16"
MODEL_ARGS_AYA="pretrained=CohereLabs/aya-expanse-8b,dtype=float16,trust_remote_code=True,load_in_4bit=True,bnb_4bit_compute_dtype=float16"
MODEL_ARGS_LLAMA="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=float16,trust_remote_code=True,load_in_4bit=True,bnb_4bit_compute_dtype=float16"

ENGLISH_TASKS=(
  "arc_easy"
  # PIQA is intentionally disabled in the Kaggle P100 default run because the
  # pinned lm-eval/datasets stack can fail while loading it with a UTF-8 decode
  # error. Re-enable manually only after confirming the dataset loads.
  "hellaswag"
)

BELEBELE_TASKS=(
  "belebele_hin_Deva"
  "belebele_arb_Arab"
  "belebele_spa_Latn"
)

mkdir -p results/lm_eval results/manual /kaggle/working

checkpoint() {
  local name="$1"
  zip -qr "/kaggle/working/${name}.zip" results
  echo "wrote /kaggle/working/${name}.zip"
}

limit_args() {
  local limit="$1"
  if [[ "${limit}" != "0" ]]; then
    printf -- "--limit %s" "${limit}"
  fi
}

run_one() {
  local model_name="$1"
  local model_args="$2"
  local task="$3"
  local limit="$4"
  local task_safe="${task//[^a-zA-Z0-9_]/_}"
  local output_path="results/lm_eval/${model_name}__4bit__${task_safe}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && find "${output_path}" -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo
    echo "=== ${model_name} | ${task} | existing result found, skipping ==="
    return
  fi

  echo
  echo "=== ${model_name} | ${task} | limit=${limit} ==="
  # shellcheck disable=SC2046
  lm_eval \
    --model hf \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${output_path}" \
    $(limit_args "${limit}")
}

run_task_for_all_models() {
  local task="$1"
  local limit="$2"
  local label="$3"

  run_one "qwen2_5_7b_instruct" "${MODEL_ARGS_QWEN}" "${task}" "${limit}"
  run_one "aya_expanse_8b" "${MODEL_ARGS_AYA}" "${task}" "${limit}"

  if [[ "${RUN_LLAMA}" == "1" ]]; then
    run_one "llama3_1_8b_instruct" "${MODEL_ARGS_LLAMA}" "${task}" "${limit}"
  fi

  checkpoint "stableselect_bigger_after_${label}"
}

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu=Kaggle P100 expected"
  echo "precision=4bit bitsandbytes float16 compute"
  echo "english_limit=${ENGLISH_LIMIT}"
  echo "belebele_limit=${BELEBELE_LIMIT}"
  echo "run_llama=${RUN_LLAMA}"
  echo "english_tasks=${ENGLISH_TASKS[*]}"
  echo "belebele_tasks=${BELEBELE_TASKS[*]}"
} > results/manual/bigger_v1_run_manifest.txt

python scripts/check_gpu_env.py

for task in "${ENGLISH_TASKS[@]}"; do
  run_task_for_all_models "${task}" "${ENGLISH_LIMIT}" "english_${task}"
done

for task in "${BELEBELE_TASKS[@]}"; do
  run_task_for_all_models "${task}" "${BELEBELE_LIMIT}" "${task}"
done

checkpoint "stableselect_bigger_v1_final"

echo
echo "Done. Final zip: /kaggle/working/stableselect_bigger_v1_final.zip"
