#!/usr/bin/env bash
# Run DeepSeek-R1-Distill-Qwen 7B + 1.5B on GSM8K, MATH, and AIME in parallel.
# Each dataset job uses one GPU for the 7B model and one GPU for the 1.5B model.

set -euo pipefail

# =========================
# Editable experiment list
# =========================

# Select datasets here. Valid values: GSM8K MATH AIME AIME24
DATASETS=(${DATASETS:-AIME})

# Select which experiment families to run.
# true + true  -> CoTRouter + LLM-Only + SLM-Only
# true + false -> CoTRouter only
# false + true -> LLM-Only + SLM-Only only
RUN_COTROUTER="${RUN_COTROUTER:-false}"
RUN_BASELINES="${RUN_BASELINES:-true}"

# Fixed main-experiment target ratio. Override per run if needed by editing here.
TARGET_RATIOS_GSM8K=(${TARGET_RATIOS_GSM8K:-0.7})
TARGET_RATIOS_MATH=(${TARGET_RATIOS_MATH:-0.8})
TARGET_RATIOS_AIME=(${TARGET_RATIOS_AIME:-0.8})
TARGET_RATIOS_AIME24=(${TARGET_RATIOS_AIME24:-0.8})

# Optional sample cap. Leave empty for full datasets.
# This controls maximum samples per dataset.
# NUM_SAMPLES="${NUM_SAMPLES:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"


# Batch sizes for CoTRouter and LLM-Only/SLM-Only baselines.
COTROUTER_BATCH_SIZE="${COTROUTER_BATCH_SIZE:-1}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-${COTROUTER_BATCH_SIZE}}"

# Per-dataset max generated/context tokens. AIME needs room for the long traces
# reported in the paper; these values are passed to Python as GLOBAL_MAX_TOKENS.
MAX_TOKENS_GSM8K="${MAX_TOKENS_GSM8K:-8000}"
MAX_TOKENS_MATH="${MAX_TOKENS_MATH:-8000}"
MAX_TOKENS_AIME="${MAX_TOKENS_AIME:-16000}"
MAX_TOKENS_AIME24="${MAX_TOKENS_AIME24:-16000}"

# Optional manual GPU pairs. Leave empty to auto-discover idle GPUs with nvidia-smi.
# Example: GPU_PAIRS="1:2 3:4 5:6"
GPU_PAIRS="${GPU_PAIRS:-6:1}"

# vLLM memory utilization is fixed here as requested.
LLM_GPU_MEMORY_UTILIZATION="0.71"
SLM_GPU_MEMORY_UTILIZATION="0.20"

# =========================
# Paths and environment
# =========================

LLM_PATH="${LLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B}"
SLM_PATH="${SLM_PATH:-/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B}"
RUN_LABEL="${RUN_LABEL:-DeepSeek 7B + 1.5B}"
RUN_DIR_PREFIX="${RUN_DIR_PREFIX:-parallel_core}"
RUN_USAGE_NAME="${RUN_USAGE_NAME:-run_deepseek_core_datasets_parallel.sh}"
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

MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-1000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"

export COTROUTER_DATASET_DIR

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: bash ${RUN_USAGE_NAME}

Top-of-file knobs:
  DATASETS=(${DATASETS[*]})
  RUN_COTROUTER=${RUN_COTROUTER}
  RUN_BASELINES=${RUN_BASELINES}
  TARGET_RATIOS_GSM8K=(${TARGET_RATIOS_GSM8K[*]})
  TARGET_RATIOS_MATH=(${TARGET_RATIOS_MATH[*]})
  TARGET_RATIOS_AIME=(${TARGET_RATIOS_AIME[*]})

Environment overrides:
  NUM_SAMPLES=20
  COTROUTER_BATCH_SIZE=150
  BASELINE_BATCH_SIZE=150
  MAX_TOKENS_AIME=16000
  GPU_PAIRS="1:2 3:4 5:6"
  PYTHON_BIN="/opt/conda/envs/thinkleap/bin/python"
  LLM_PATH="${LLM_PATH}"
  SLM_PATH="${SLM_PATH}"
  COTROUTER_DATASET_DIR="${COTROUTER_DATASET_DIR}"

vLLM memory utilization is fixed to:
  LLM: ${LLM_GPU_MEMORY_UTILIZATION}
  SLM: ${SLM_GPU_MEMORY_UTILIZATION}
EOF
    exit 0
fi

if [[ ! -d "${COTROUTER_DATASET_DIR}" ]]; then
    echo "Dataset directory not found: ${COTROUTER_DATASET_DIR}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Set GPU_PAIRS manually." >&2
    exit 1
fi

resolve_experiment() {
    local run_cotrouter
    local run_baselines
    run_cotrouter="$(echo "${RUN_COTROUTER}" | tr '[:upper:]' '[:lower:]')"
    run_baselines="$(echo "${RUN_BASELINES}" | tr '[:upper:]' '[:lower:]')"

    case "${run_cotrouter}" in
        true|1|yes|y|on) run_cotrouter="true" ;;
        false|0|no|n|off) run_cotrouter="false" ;;
        *) echo "Invalid RUN_COTROUTER=${RUN_COTROUTER}" >&2; exit 1 ;;
    esac

    case "${run_baselines}" in
        true|1|yes|y|on) run_baselines="true" ;;
        false|0|no|n|off) run_baselines="false" ;;
        *) echo "Invalid RUN_BASELINES=${RUN_BASELINES}" >&2; exit 1 ;;
    esac

    if [[ "${run_cotrouter}" == "true" && "${run_baselines}" == "true" ]]; then
        echo "all"
    elif [[ "${run_cotrouter}" == "true" && "${run_baselines}" == "false" ]]; then
        echo "main"
    elif [[ "${run_cotrouter}" == "false" && "${run_baselines}" == "true" ]]; then
        echo "baselines"
    else
        echo "RUN_COTROUTER and RUN_BASELINES cannot both be false." >&2
        exit 1
    fi
}

target_ratios_for_dataset() {
    case "$1" in
        GSM8K)
            echo "${TARGET_RATIOS_GSM8K[*]}"
            ;;
        MATH)
            echo "${TARGET_RATIOS_MATH[*]}"
            ;;
        AIME)
            echo "${TARGET_RATIOS_AIME[*]}"
            ;;
        AIME24)
            echo "${TARGET_RATIOS_AIME24[*]}"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

max_tokens_for_dataset() {
    case "$1" in
        GSM8K)
            echo "${MAX_TOKENS_GSM8K}"
            ;;
        MATH)
            echo "${MAX_TOKENS_MATH}"
            ;;
        AIME)
            echo "${MAX_TOKENS_AIME}"
            ;;
        AIME24)
            echo "${MAX_TOKENS_AIME24}"
            ;;
        *)
            echo "${GLOBAL_MAX_TOKENS:-8000}"
            ;;
    esac
}

build_gpu_pairs() {
    if [[ -n "${GPU_PAIRS}" ]]; then
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
        echo "Need at least two idle GPUs for paired jobs." >&2
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

EXPERIMENT="$(resolve_experiment)"
build_gpu_pairs
read -r -a PAIR_ARRAY <<< "${GPU_PAIRS}"

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
PARALLEL_DIR="results/${RUN_DIR_PREFIX}_${RUN_ID}"
LOG_DIR="${PARALLEL_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "Running ${RUN_LABEL} core datasets in parallel"
echo "LLM: ${LLM_PATH}"
echo "SLM: ${SLM_PATH}"
echo "Dataset dir: ${COTROUTER_DATASET_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Experiment: ${EXPERIMENT} (RUN_COTROUTER=${RUN_COTROUTER}, RUN_BASELINES=${RUN_BASELINES})"
echo "Datasets: ${DATASETS[*]}"
echo "GPU pairs: ${GPU_PAIRS}"
echo "Num samples: ${NUM_SAMPLES:-full}"
echo "CoTRouter batch size: ${COTROUTER_BATCH_SIZE}"
echo "Baseline batch size: ${BASELINE_BATCH_SIZE}"
echo "Max tokens: GSM8K=${MAX_TOKENS_GSM8K}, MATH=${MAX_TOKENS_MATH}, AIME=${MAX_TOKENS_AIME}, AIME24=${MAX_TOKENS_AIME24}"
echo "vLLM utilization: LLM=${LLM_GPU_MEMORY_UTILIZATION}, SLM=${SLM_GPU_MEMORY_UTILIZATION}"
echo "Logs: ${LOG_DIR}"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

run_dataset() {
    local dataset="$1"
    local pair="$2"
    local llm_gpu="${pair%%:*}"
    local slm_gpu="${pair##*:}"
    local sample_arg=()
    local target_ratios
    local max_tokens

    target_ratios="$(target_ratios_for_dataset "${dataset}")"
    max_tokens="$(max_tokens_for_dataset "${dataset}")"

    if [[ -n "${NUM_SAMPLES}" ]]; then
        sample_arg=(--num_samples "${NUM_SAMPLES}")
    fi

    echo "[START] ${dataset} on LLM GPU ${llm_gpu}, SLM GPU ${slm_gpu}; ratios: ${target_ratios}"
    mkdir -p "${PARALLEL_DIR}/${dataset}"
    {
        echo "Dataset: ${dataset}"
        echo "LLM GPU: ${llm_gpu}"
        echo "SLM GPU: ${slm_gpu}"
        echo "Experiment: ${EXPERIMENT}"
        echo "Target ratios: ${target_ratios}"
        echo "Num samples: ${NUM_SAMPLES:-full}"
        echo "CoTRouter batch size: ${COTROUTER_BATCH_SIZE}"
        echo "Baseline batch size: ${BASELINE_BATCH_SIZE}"
        echo "GLOBAL_MAX_TOKENS: ${max_tokens}"
        echo "vLLM utilization: LLM=${LLM_GPU_MEMORY_UTILIZATION}, SLM=${SLM_GPU_MEMORY_UTILIZATION}"
        echo "Started: $(date)"
        GLOBAL_MAX_TOKENS="${max_tokens}" "${PYTHON_BIN}" cotrouter_main.py \
            --dataset "${dataset}" \
            --experiment "${EXPERIMENT}" \
            --llm_path "${LLM_PATH}" \
            --slm_path "${SLM_PATH}" \
            --llm_gpus "${llm_gpu}" \
            --slm_gpu "${slm_gpu}" \
            --llm_gpu_memory_utilization "${LLM_GPU_MEMORY_UTILIZATION}" \
            --slm_gpu_memory_utilization "${SLM_GPU_MEMORY_UTILIZATION}" \
            --target_ratios ${target_ratios} \
            --cotrouter_batch_size "${COTROUTER_BATCH_SIZE}" \
            --baseline_batch_size "${BASELINE_BATCH_SIZE}" \
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
    while (( next_dataset < ${#DATASETS[@]} && ${#available_pairs[@]} > 0 )); do
        local pair="${available_pairs[0]}"
        available_pairs=("${available_pairs[@]:1}")
        local dataset="${DATASETS[next_dataset]}"
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
Parallel core run: ${RUN_ID}
Run label: ${RUN_LABEL}
Experiment: ${EXPERIMENT}
RUN_COTROUTER: ${RUN_COTROUTER}
RUN_BASELINES: ${RUN_BASELINES}
Datasets: ${DATASETS[*]}
GPU pairs: ${GPU_PAIRS}
Num samples: ${NUM_SAMPLES:-full}
CoTRouter batch size: ${COTROUTER_BATCH_SIZE}
Baseline batch size: ${BASELINE_BATCH_SIZE}
Max tokens: GSM8K=${MAX_TOKENS_GSM8K}, MATH=${MAX_TOKENS_MATH}, AIME=${MAX_TOKENS_AIME}, AIME24=${MAX_TOKENS_AIME24}
Python: ${PYTHON_BIN}
LLM: ${LLM_PATH}
SLM: ${SLM_PATH}
vLLM utilization: LLM=${LLM_GPU_MEMORY_UTILIZATION}, SLM=${SLM_GPU_MEMORY_UTILIZATION}
Logs: ${LOG_DIR}
EOF

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import csv

base = Path("${PARALLEL_DIR}")
files = sorted(base.glob("*/results_summary.csv"))
out = base / "merged_results_summary.csv"
if files:
    fieldnames = None
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise SystemExit(f"Header mismatch in {path}")
            rows.extend(reader)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(files)} summary files into {out}")
else:
    print(f"No results_summary.csv files found under {base}")
PY

echo "================================================================"
echo "Parallel ${RUN_LABEL} core run complete. Logs: ${LOG_DIR}"
echo "Summary: ${PARALLEL_DIR}/summary.txt"
echo "Merged CSV: ${PARALLEL_DIR}/merged_results_summary.csv"
echo "================================================================"

exit "${failed}"
