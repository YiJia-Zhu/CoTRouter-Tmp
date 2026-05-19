#!/bin/bash
# run_test_gsm8k.sh - 在GSM8K数据集的前20个样本上进行测试

# 设置模型路径 (请根据您的设置修改)
LLM_PATH="/mnt/8T/xgr/zhuyijia/huggingface_models/DeepSeek-R1-Distill-Qwen-14B"
SLM_PATH="/mnt/8T/xgr/shizhenning/model_weights/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# GPU 配置
LLM_GPUS="1"
SLM_GPU="1"

echo "================================================================"
echo "Running TEST MODE on first 20 samples of GSM8K"
echo "================================================================"

# 函数: 运行实验并检查状态
run_experiment() {
    echo -e "\n>>> Running: $1"
    eval $2
    if [ $? -eq 0 ]; then
        echo "✓ Success: $1"
    else
        echo "✗ Failed: $1"
        exit 1
    fi
}

# 1. 在GSM8K前20个样本上运行 CoTRouter 主要实验
run_experiment "Main CoTRouter experiments on GSM8K (20 samples)" \
    "python cotrouter_main.py \
        --dataset COMMON \
        --experiment main \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU} \
        --num_samples 20" # <-- 指定样本数量

# 2. 在GSM8K前20个样本上运行基线对比
run_experiment "Baseline comparisons on GSM8K (20 samples)" \
    "python cotrouter_main.py \
        --dataset COMMON \
        --experiment baselines \
        --llm_path ${LLM_PATH} \
        --slm_path ${SLM_PATH} \
        --llm_gpus ${LLM_GPUS} \
        --slm_gpu ${SLM_GPU} \
        --num_samples 20" # <-- 指定样本数量

echo -e "\n================================================================"
echo "TEST RUN COMPLETED SUCCESSFULLY!"
echo "================================================================"