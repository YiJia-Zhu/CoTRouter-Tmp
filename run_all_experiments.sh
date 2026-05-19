#!/bin/bash
# run_all_experiments.sh - Complete experiment pipeline for CoTRouter

# Set default paths (modify these according to your setup)
LLM_PATH="/mnt/8T/xgr/zhuyijia/huggingface_models/DeepSeek-R1-Distill-Qwen-14B"
SLM_PATH="/mnt/8T/xgr/shizhenning/model_weights/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# GPU configuration
LLM_GPUS="1 2"
SLM_GPU="0"

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/cotrouter_full_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "================================================================"
echo "Running Complete CoTRouter Experiments"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "================================================================"

# Function to run experiment and check status
run_experiment() {
    echo -e "\n>>> Running: $1"
    echo "----------------------------------------------------------------"
    eval $2
    if [ $? -eq 0 ]; then
        echo "✓ Success: $1"
    else
        echo "✗ Failed: $1"
        exit 1
    fi
}

# 1. Main experiments on GSM8K
run_experiment "Main CoTRouter experiments on GSM8K" \
    "python cotrouter_main.py \
        --dataset GSM8K \
        --experiment main \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU} \
        --target_ratios 0.1 0.2 0.3 0.4 0.5 0.6"

# 2. Baseline comparisons on GSM8K
run_experiment "Baseline comparisons on GSM8K" \
    "python cotrouter_main.py \
        --dataset GSM8K \
        --experiment baselines \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU}"

# 3. Ablation studies on GSM8K
run_experiment "Ablation studies on GSM8K" \
    "python cotrouter_main.py \
        --dataset GSM8K \
        --experiment ablation \
        --num_samples 500 \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU}"

# 4. Main experiments on MATH
run_experiment "Main CoTRouter experiments on MATH" \
    "python cotrouter_main.py \
        --dataset MATH \
        --experiment main \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU} \
        --target_ratios 0.2 0.3 0.4 0.5 0.6"

# 5. Baseline comparisons on MATH
run_experiment "Baseline comparisons on MATH" \
    "python cotrouter_main.py \
        --dataset MATH \
        --experiment baselines \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU}"

# 6. Parameter sensitivity (small subset)
run_experiment "Parameter sensitivity analysis" \
    "python cotrouter_main.py \
        --dataset GSM8K \
        --experiment sensitivity \
        --num_samples 100 \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU}"

# 7. Hyperparameter sweep
run_experiment "Hyperparameter sweep on GSM8K" \
    "python cotrouter_param_sweep.py \
        --dataset GSM8K \
        --num_samples 200 \
        --output_dir ${RESULTS_DIR}/param_sweep"

# 8. Generate plots
echo -e "\n>>> Generating visualization plots"
echo "----------------------------------------------------------------"

# Find the most recent results directory
LATEST_RESULTS=$(ls -td results/cotrouter_* | grep -v "full" | head -1)

if [ -d "$LATEST_RESULTS" ]; then
    python plot_results.py \
        --results_dir ${LATEST_RESULTS} \
        --output_dir ${RESULTS_DIR}/plots
    echo "✓ Plots generated successfully"
else
    echo "✗ No results directory found for plotting"
fi

# 9. Create summary report
echo -e "\n>>> Creating summary report"
echo "----------------------------------------------------------------"

cat > ${RESULTS_DIR}/experiment_summary.txt << EOF
CoTRouter Experiment Summary
Generated: $(date)
================================================================

Configuration:
- LLM Model: ${LLM_PATH}
- SLM Model: ${SLM_PATH}
- LLM GPUs: ${LLM_GPUS}
- SLM GPU: ${SLM_GPU}

Experiments Completed:
1. Main CoTRouter (GSM8K) - Target ratios: 0.1-0.6
2. Baseline comparisons (GSM8K)
3. Ablation studies (GSM8K, 500 samples)
4. Main CoTRouter (MATH) - Target ratios: 0.2-0.6
5. Baseline comparisons (MATH)
6. Parameter sensitivity (100 samples)
7. Hyperparameter sweep (200 samples)

Results Location: ${RESULTS_DIR}

Key Files:
- Raw results: ${LATEST_RESULTS}/results.json
- Analysis: ${LATEST_RESULTS}/results_analysis.json
- Summary CSV: ${LATEST_RESULTS}/results_summary.csv
- Plots: ${RESULTS_DIR}/plots/
- Parameter sweep: ${RESULTS_DIR}/param_sweep/

================================================================
EOF

echo "✓ Summary report created"

echo -e "\n================================================================"
echo "All experiments completed successfully!"
echo "Results saved to: ${RESULTS_DIR}"
echo "================================================================"

# Optional: Create LaTeX tables for paper
echo -e "\nTo generate LaTeX tables for the paper, run:"
echo "python generate_latex_tables.py --results_dir ${LATEST_RESULTS}"
