.PHONY: setup smoke sanity matrix-dry matrix analyze lm-eval-dry

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

smoke:
	python scripts/run_custom_qa.py \
		--model-name qwen2_5_0_5b_instruct_smoke \
		--model-id Qwen/Qwen2.5-0.5B-Instruct \
		--precision fp16 \
		--task-group english \
		--dataset data/pilot/english_qa.jsonl \
		--max-examples 3

sanity:
	python scripts/run_custom_qa.py \
		--model-name qwen2_5_7b_instruct \
		--model-id Qwen/Qwen2.5-7B-Instruct \
		--precision bf16 \
		--task-group english \
		--dataset data/pilot/english_qa.jsonl \
		--max-examples 5

matrix-dry:
	python scripts/run_matrix.py --dry-run

matrix:
	python scripts/run_matrix.py

analyze:
	python scripts/analyze_results.py

lm-eval-dry:
	python scripts/run_lm_eval.py \
		--model qwen2_5_7b_instruct \
		--precision bf16 \
		--task-group english \
		--limit 20 \
		--dry-run
