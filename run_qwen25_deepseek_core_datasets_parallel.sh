#!/usr/bin/env bash
# Run Qwen2.5-7B-Instruct as the LLM and DeepSeek-R1-Distill-Qwen-1.5B
# as the SLM, reusing the existing core-dataset parallel runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLM_PATH="${LLM_PATH:-/private/zhenningshi/model_weights/Qwen2.5-7B-Instruct}"
export SLM_PATH="${SLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B}"
export RUN_LABEL="${RUN_LABEL:-Qwen2.5-7B-Instruct + DeepSeek-R1-Distill-Qwen-1.5B}"
export RUN_DIR_PREFIX="${RUN_DIR_PREFIX:-parallel_qwen25_deepseek_core}"
export RUN_USAGE_NAME="${RUN_USAGE_NAME:-run_qwen25_deepseek_core_datasets_parallel.sh}"


exec "${SCRIPT_DIR}/run_deepseek_core_datasets_parallel.sh" "$@"
