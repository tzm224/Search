#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base_url=127.0.0.1:8000
model_name=search_agent-RL-step20-ckpt
dataset=test_data.jsonl


python3 "${SCRIPT_DIR}/get_response.py" \
  --dataset $dataset \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 128

python3 "${SCRIPT_DIR}/get_eval.py" \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 512

python3 "${SCRIPT_DIR}/print_acc.py" \
  --dataset $dataset \
  --model_name $model_name
