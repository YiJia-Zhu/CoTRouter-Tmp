#!/usr/bin/env bash
# Run DeepSeek-R1-Distill-Qwen 7B + 1.5B experiments in parallel.
# Each dataset job uses one GPU for the 7B model and one GPU for the 1.5B model.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_deepseek_new_datasets_parallel.sh

Environment overrides:
  DATASETS="ARC OpenBookQA CommonsenseQA HumanEval"
  EXPERIMENT="baselines"
  NUM_SAMPLES="20"
  GPU_PAIRS="1:2 3:4 5:6"
  PYTHON_BIN="/opt/conda/envs/thinkleap/bin/python"
  LLM_PATH="/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B"
  SLM_PATH="/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B"
  COTROUTER_DATASET_DIR="/private/zhenningshi/CoTRouter-Tmp/huggingface_datasets"

If GPU_PAIRS is unset, idle GPUs are discovered with nvidia-smi and paired in order.
EOF
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LLM_PATH="${LLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B}"
SLM_PATH="${SLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B}"
COTROUTER_DATASET_DIR="${COTROUTER_DATASET_DIR:-${SCRIPT_DIR}/huggingface_datasets}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "/opt/conda/envs/thinkleap/bin/python" ]] && \
       /opt/conda/envs/thinkleap/bin/python -c "import vllm" >/dev/null 2>&1; then
        PYTHON_BIN="/opt/conda/envs/thinkleap/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

EXPERIMENT="${EXPERIMENT:-main}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DATASETS="${DATASETS:-ARC OpenBookQA CommonsenseQA HumanEval}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-1000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"

export COTROUTER_DATASET_DIR

if [[ ! -d "${COTROUTER_DATASET_DIR}" ]]; then
    echo "Dataset directory not found: ${COTROUTER_DATASET_DIR}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Set GPU_PAIRS manually." >&2
    exit 1
fi

expand_datasets() {
    if [[ "${DATASETS}" == "new" || "${DATASETS}" == "all_new" ]]; then
        DATASETS="ARC OpenBookQA CommonsenseQA HumanEval"
    fi
}

build_gpu_pairs() {
    if [[ -n "${GPU_PAIRS:-}" ]]; then
        return
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
        '
    )

    if [[ "${#free_gpus[@]}" -lt 2 ]]; then
        echo "Need at least two idle GPUs for parallel paired jobs." >&2
        echo "Set GPU_PAIRS manually if you want to override." >&2
        exit 1
    fi

    local pairs=()
    local i=0
    while (( i + 1 < ${#free_gpus[@]} )); do
        pairs+=("${free_gpus[i]}:${free_gpus[i+1]}")
        i=$((i + 2))
    done

    GPU_PAIRS="${pairs[*]}"
}

expand_datasets
build_gpu_pairs

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a PAIR_ARRAY <<< "${GPU_PAIRS}"

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
PARALLEL_DIR="results/parallel_${RUN_ID}"
LOG_DIR="${PARALLEL_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "Running DeepSeek 7B + 1.5B parallel experiments"
echo "LLM: ${LLM_PATH}"
echo "SLM: ${SLM_PATH}"
echo "Dataset dir: ${COTROUTER_DATASET_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Experiment: ${EXPERIMENT}"
echo "Datasets: ${DATASETS}"
echo "GPU pairs: ${GPU_PAIRS}"
echo "Logs: ${LOG_DIR}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

run_dataset() {
    local dataset="$1"
    local pair="$2"
    local llm_gpu="${pair%%:*}"
    local slm_gpu="${pair##*:}"
    local sample_arg=()

    if [[ -n "${NUM_SAMPLES}" ]]; then
        sample_arg=(--num_samples "${NUM_SAMPLES}")
    fi

    echo "[START] ${dataset} on LLM GPU ${llm_gpu}, SLM GPU ${slm_gpu}"
    mkdir -p "${PARALLEL_DIR}/${dataset}"
    {
        echo "Dataset: ${dataset}"
        echo "LLM GPU: ${llm_gpu}"
        echo "SLM GPU: ${slm_gpu}"
        echo "Started: $(date)"
        "${PYTHON_BIN}" cotrouter_main.py \
            --dataset "${dataset}" \
            --experiment "${EXPERIMENT}" \
            --llm_path "${LLM_PATH}" \
            --slm_path "${SLM_PATH}" \
            --llm_gpus "${llm_gpu}" \
            --slm_gpu "${slm_gpu}" \
            --output_dir "${PARALLEL_DIR}/${dataset}" \
            "${sample_arg[@]}"
        echo "Finished: $(date)"
    } > "${LOG_DIR}/${dataset}.log" 2>&1
}

declare -a running_pids=()
declare -a available_pairs=("${PAIR_ARRAY[@]}")
declare -A pid_to_dataset=()
declare -A pid_to_pair=()

next_dataset=0
failed=0

start_available_jobs() {
    while (( next_dataset < ${#DATASET_ARRAY[@]} && ${#available_pairs[@]} > 0 )); do
        local pair="${available_pairs[0]}"
        available_pairs=("${available_pairs[@]:1}")
        local dataset="${DATASET_ARRAY[next_dataset]}"
        run_dataset "${dataset}" "${pair}" &
        local pid=$!
        running_pids+=("${pid}")
        pid_to_dataset["${pid}"]="${dataset}"
        pid_to_pair["${pid}"]="${pair}"
        next_dataset=$((next_dataset + 1))
    done
}

remove_pid() {
    local remove="$1"
    local remaining=()
    local pid
    for pid in "${running_pids[@]}"; do
        if [[ "${pid}" != "${remove}" ]]; then
            remaining+=("${pid}")
        fi
    done
    running_pids=("${remaining[@]}")
}

start_available_jobs

while (( ${#running_pids[@]} > 0 )); do
    completed_pid=""
    for pid in "${running_pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            completed_pid="${pid}"
            break
        fi
    done

    if [[ -z "${completed_pid}" ]]; then
        sleep 10
        continue
    fi

    if wait "${completed_pid}"; then
        echo "[DONE] ${pid_to_dataset[${completed_pid}]} on pair ${pid_to_pair[${completed_pid}]}"
    else
        echo "[FAIL] ${pid_to_dataset[${completed_pid}]} on pair ${pid_to_pair[${completed_pid}]}" >&2
        echo "       See ${LOG_DIR}/${pid_to_dataset[${completed_pid}]}.log" >&2
        failed=1
    fi

    available_pairs+=("${pid_to_pair[${completed_pid}]}")
    remove_pid "${completed_pid}"
    start_available_jobs
done

cat > "${PARALLEL_DIR}/summary.txt" <<EOF
Parallel run: ${RUN_ID}
Experiment: ${EXPERIMENT}
Datasets: ${DATASETS}
GPU pairs: ${GPU_PAIRS}
Python: ${PYTHON_BIN}
LLM: ${LLM_PATH}
SLM: ${SLM_PATH}
Logs: ${LOG_DIR}
EOF

echo "================================================================"
echo "Parallel run complete. Logs: ${LOG_DIR}"
echo "Summary: ${PARALLEL_DIR}/summary.txt"
echo "================================================================"

exit "${failed}"
