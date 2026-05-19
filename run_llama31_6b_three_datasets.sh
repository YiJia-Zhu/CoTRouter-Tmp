#!/usr/bin/env bash
# Run Llama-3.1-6B-Instruct experiments on GSM8K, MATH, and AIME24.

set -euo pipefail

LLM_PATH="${LLM_PATH:-/mnt/8T/xgr/zhuyijia/huggingface_models/llama3.1_6B_Instruct}"
SLM_PATH="${SLM_PATH:-/mnt/8T/xgr/shizhenning/model_weights/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

LLM_GPUS="${LLM_GPUS:-0}"
SLM_GPU="${SLM_GPU:-0}"
EXPERIMENT="${EXPERIMENT:-baselines}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

DATASETS=(GSM8K MATH AIME24)

run_dataset() {
    local dataset="$1"
    local sample_arg=()

    if [[ -n "${NUM_SAMPLES}" ]]; then
        sample_arg=(--num_samples "${NUM_SAMPLES}")
    fi

    echo "================================================================"
    echo "Running ${EXPERIMENT} on ${dataset}"
    echo "LLM: ${LLM_PATH}"
    echo "SLM: ${SLM_PATH}"
    echo "LLM_GPUS: ${LLM_GPUS}; SLM_GPU: ${SLM_GPU}"
    echo "================================================================"

    python cotrouter_main.py \
        --dataset "${dataset}" \
        --experiment "${EXPERIMENT}" \
        --llm_path "${LLM_PATH}" \
        --slm_path "${SLM_PATH}" \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu "${SLM_GPU}" \
        "${sample_arg[@]}"
}

for dataset in "${DATASETS[@]}"; do
    run_dataset "${dataset}"
done

echo "================================================================"
echo "Finished Llama-3.1-6B-Instruct runs for: ${DATASETS[*]}"
echo "Results are saved under results/cotrouter_YYYYMMDD_HHMMSS/"
echo "================================================================"
