#!/usr/bin/env bash
# Run DeepSeek-R1-Distill-Qwen 7B + 1.5B on the newly added local datasets.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_deepseek_new_datasets.sh

Environment overrides:
  DATASETS="new"
  EXPERIMENT="baselines"
  NUM_SAMPLES="20"
  LLM_GPUS="1"
  SLM_GPU="2"
  PYTHON_BIN="/opt/conda/envs/thinkleap/bin/python"
  COTROUTER_DATASET_DIR="/private/zhenningshi/CoTRouter-Tmp/huggingface_datasets"
EOF
    exit 0
fi

LLM_PATH="${LLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B}"
SLM_PATH="${SLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COTROUTER_DATASET_DIR="${COTROUTER_DATASET_DIR:-${SCRIPT_DIR}/huggingface_datasets}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "/opt/conda/envs/thinkleap/bin/python" ]] && \
       /opt/conda/envs/thinkleap/bin/python -c "import vllm" >/dev/null 2>&1; then
        PYTHON_BIN="/opt/conda/envs/thinkleap/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

EXPERIMENT="${EXPERIMENT:-baselines}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DATASETS="${DATASETS:-new}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-1000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"

export COTROUTER_DATASET_DIR

if [[ ! -d "${COTROUTER_DATASET_DIR}" ]]; then
    echo "Dataset directory not found: ${COTROUTER_DATASET_DIR}" >&2
    exit 1
fi

pick_gpus() {
    if [[ -n "${LLM_GPUS:-}" && -n "${SLM_GPU:-}" ]]; then
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found. Set LLM_GPUS and SLM_GPU manually." >&2
        exit 1
    fi

    mapfile -t free_gpus < <(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F',' -v max_mem="${MAX_USED_MEM_MB}" -v max_util="${MAX_GPU_UTIL}" '
            {
                gsub(/ /, "", $1);
                gsub(/ /, "", $2);
                gsub(/ /, "", $3);
                if ($2 <= max_mem && $3 <= max_util) print $1;
            }
        ' |
        head -n 2
    )

    if [[ "${#free_gpus[@]}" -eq 0 ]]; then
        mapfile -t free_gpus < <(
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
            sort -t',' -k2,2n -k3,3n |
            awk -F',' '{ gsub(/ /, "", $1); print $1; }' |
            head -n 2
        )
    fi

    if [[ "${#free_gpus[@]}" -eq 0 ]]; then
        echo "No GPUs found by nvidia-smi." >&2
        exit 1
    fi

    LLM_GPUS="${LLM_GPUS:-${free_gpus[0]}}"
    if [[ "${#free_gpus[@]}" -ge 2 ]]; then
        SLM_GPU="${SLM_GPU:-${free_gpus[1]}}"
    else
        SLM_GPU="${SLM_GPU:-${free_gpus[0]}}"
    fi

    export LLM_GPUS SLM_GPU
}

pick_gpus

echo "================================================================"
echo "Running DeepSeek 7B + 1.5B on local new datasets"
echo "LLM: ${LLM_PATH}"
echo "SLM: ${SLM_PATH}"
echo "Dataset dir: ${COTROUTER_DATASET_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Experiment: ${EXPERIMENT}"
echo "LLM_GPUS: ${LLM_GPUS}; SLM_GPU: ${SLM_GPU}"
echo "Datasets: ${DATASETS}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

run_dataset() {
    local dataset="$1"
    local sample_arg=()

    if [[ -n "${NUM_SAMPLES}" ]]; then
        sample_arg=(--num_samples "${NUM_SAMPLES}")
    fi

    echo
    echo "================================================================"
    echo "Running ${EXPERIMENT} on ${dataset}"
    echo "================================================================"

    "${PYTHON_BIN}" cotrouter_main.py \
        --dataset "${dataset}" \
        --experiment "${EXPERIMENT}" \
        --llm_path "${LLM_PATH}" \
        --slm_path "${SLM_PATH}" \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu "${SLM_GPU}" \
        "${sample_arg[@]}"
}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
for dataset in "${DATASET_ARRAY[@]}"; do
    run_dataset "${dataset}"
done

echo
echo "================================================================"
echo "Finished datasets: ${DATASETS}"
echo "Results are saved under results/cotrouter_YYYYMMDD_HHMMSS/"
echo "================================================================"
