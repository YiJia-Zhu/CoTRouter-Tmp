# analyze_switch_points.py
"""
一个用于分析 CoTRouter 实验结果的脚本。

该脚本读取 `results.json` 文件，模拟 CoTRouter 的路由决策过程，
识别出模型从 SLM 切换到 LLM 的确切位置，统计在切换点
出现的 token 的词频，并将包含前后完整句子、熵和Z-score的
详细上下文示例输出到文件中。

python analyze_switch_points.py --results_file ./results/param_sweep/GSM8K_20250730_205653_7/results.json --method_name Sweep_1
python analyze_switch_points.py --results_file ./results/param_sweep/MATH_20250730_220604_7/results.json --method_name Sweep_1
python analyze_switch_points.p.py --results_file ./results/param_sweep/AIME_20250730_175240_7/results.json --method_name Sweep_1

"""
import json
import numpy as np
import argparse
import os
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# 为了使脚本自包含，我们从 cotrouter.py 中复制必要的类
# ===================================================================
# Kalman Filter and EWMA Control Chart Implementations
# ===================================================================

class KalmanFilter1D:
    """一维卡尔曼滤波器，用于潜在复杂度估计"""
    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 0.5):
        self.x = 0.0
        self.P = 1.0
        self.F = 1.0
        self.H = 1.0
        self.Q = process_variance
        self.R = measurement_variance
        
    def predict(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        
    def update(self, z: float):
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        
    def filter(self, measurement: float) -> float:
        self.predict()
        self.update(measurement)
        return self.x

class EWMAControlChart:
    """用于异常检测的指数加权移动平均控制图"""
    def __init__(self, lambda_param: float = 0.2, initial_mean: float = 0.0, process_std: float = 1.0):
        self.lambda_param = lambda_param
        self.mu = initial_mean
        self.process_std = process_std
        self.t = 0
        
    def update(self, value: float) -> Tuple[float, float]:
        self.t += 1
        self.mu = self.lambda_param * value + (1 - self.lambda_param) * self.mu
        if self.t > 30:
            variance = (self.lambda_param / (2 - self.lambda_param)) * (self.process_std ** 2)
        else:
            variance = (self.lambda_param / (2 - self.lambda_param)) * \
                      (self.process_std ** 2) * (1 - (1 - self.lambda_param) ** (2 * self.t))
        std_dev = np.sqrt(variance)
        return self.mu, std_dev
    
    def get_z_score(self, value: float, prev_mu: float) -> float:
        _, std_dev = self.update(value)
        if std_dev < 1e-6:
            return 0.0
        return (value - prev_mu) / std_dev

# ===================================================================
# Simplified CoTRouter Logic for Analysis
# ===================================================================

@dataclass
class CoTRouterState:
    """用于分析的简化 CoTRouter 状态"""
    kalman_filter: KalmanFilter1D
    ewma_chart: EWMAControlChart
    L_threshold: float = 3.0
    commitment_remaining: int = 0
    current_model: str = 'slm'
    llm_tokens: int = 0
    slm_tokens: int = 0
    # 新增：用于存储触发切换的Z-score
    last_z_score: float = 0.0

class CoTRouterAnalyzer:
    """用于分析的 CoTRouter 逻辑模拟器"""
    def __init__(self, 
                 target_llm_ratio: float = 0.3,
                 beta_gain: float = 2.0,
                 k_base: int = 20,
                 alpha_severity: float = 2.0,
                 L_min: float = 1.0,
                 L_max: float = 4.0,
                 initial_L: float = 2.0,
                 kalman_process_var: float = 0.3,
                 kalman_measurement_var: float = 0.3,
                 ewma_lambda: float = 0.02):
        self.target_llm_ratio = target_llm_ratio
        self.beta_gain = beta_gain
        self.k_base = k_base
        self.alpha_severity = alpha_severity
        self.L_min = L_min
        self.L_max = L_max
        self.initial_L = initial_L
        self.kalman_process_var = kalman_process_var
        self.kalman_measurement_var = kalman_measurement_var
        self.ewma_lambda = ewma_lambda

    def create_state(self) -> CoTRouterState:
        return CoTRouterState(
            kalman_filter=KalmanFilter1D(self.kalman_process_var, self.kalman_measurement_var),
            ewma_chart=EWMAControlChart(self.ewma_lambda),
            L_threshold=self.initial_L
        )

    def update_threshold(self, state: CoTRouterState):
        total_tokens = state.llm_tokens + state.slm_tokens
        if total_tokens == 0: return
        actual_ratio = state.llm_tokens / total_tokens
        error = actual_ratio - self.target_llm_ratio
        state.L_threshold += self.beta_gain * error
        state.L_threshold = np.clip(state.L_threshold, self.L_min, self.L_max)

    def compute_commitment_duration(self, z_score: float, threshold: float) -> int:
        severity = max(0, z_score - threshold)
        return max(self.k_base, int(np.ceil(self.k_base + self.alpha_severity * severity)))

    def make_routing_decision(self, state: CoTRouterState, entropy: float) -> str:
        if state.commitment_remaining > 0:
            state.commitment_remaining -= 1
            return 'llm'
        
        prev_mu = state.ewma_chart.mu
        filtered_value = state.kalman_filter.filter(entropy)
        _, _ = state.ewma_chart.update(filtered_value)
        z_score = state.ewma_chart.get_z_score(filtered_value, prev_mu)
        state.last_z_score = z_score # 存储Z-score
        
        self.update_threshold(state)
        
        if z_score > state.L_threshold and state.current_model == 'slm':
            commitment = self.compute_commitment_duration(z_score, state.L_threshold)
            state.commitment_remaining = commitment - 1
            return 'llm'
        
        return 'slm'

# ===================================================================
# Main Analysis and File Writing Functions
# ===================================================================

def find_sentence_context_indices(tokens: List[str], switch_idx: int) -> Tuple[int, int]:
    """根据切换点索引，查找其前后句子的起始和结束索引。"""
    boundary_chars = {'.', '?', '!', '\n', 'Ċ'}

    def is_boundary(token: str) -> bool:
        return any(c in token for c in boundary_chars)

    end_current_sent = next((i for i in range(switch_idx, len(tokens)) if is_boundary(tokens[i])), len(tokens) - 1)
    start_current_sent = next((i for i in range(switch_idx - 1, -1, -1) if is_boundary(tokens[i])), -1)
    start_prev_sent = next((i for i in range(start_current_sent - 1, -1, -1) if is_boundary(tokens[i])), -1) if start_current_sent > 0 else -1
    end_next_sent = next((i for i in range(end_current_sent + 1, len(tokens)) if is_boundary(tokens[i])), len(tokens) - 1) if end_current_sent < len(tokens) - 1 else len(tokens) - 1
    
    context_start_idx = start_prev_sent + 1
    context_end_idx = end_next_sent + 1

    return context_start_idx, context_end_idx


def write_contexts_to_file(output_path: str, sorted_counts: list, switch_contexts: dict):
    """将收集到的上下文示例和指标写入文件"""
    print(f"\n正在将详细上下文写入文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Switch Point Token Context Analysis\n")
        f.write("======================================\n\n")
        
        for token, count in sorted_counts:
            f.write(f"--- Token: '{token}' (切换次数: {count}) ---\n\n")
            contexts = switch_contexts.get(token, [])
            for i, data in enumerate(contexts):
                f.write(f"  示例 #{i+1}:\n")
                f.write(f"    指标: Entropy = {data['entropy']:.4f} | Z-score = {data['z_score']:.4f}\n")
                f.write(f"    上下文: {data['context']}\n\n")
            f.write("\n" + "-"*40 + "\n\n")
    print("文件写入完成。")

def analyze_switch_points(results_file: str, method_name: str):
    """加载结果文件，分析指定方法的模型切换点。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("错误：请先安装 'transformers' 和 'torch' 库。运行: pip install transformers torch")
        return

    print("正在加载分词器 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    
    print(f"正在加载结果文件: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    try:
        target_ratio = float(method_name.split('-R')[-1]) / 100
    except (ValueError, IndexError):
        print(f"警告：无法从 '{method_name}' 解析目标比例，将使用默认值 0.3")
        target_ratio = 0.3

    analyzer = CoTRouterAnalyzer(target_llm_ratio=target_ratio)
    
    switch_token_counts = Counter()
    switch_contexts = defaultdict(list)

    for dataset, methods in results_data.items():
        if method_name not in methods:
            continue
        
        print("\n" + "="*80)
        print(f"正在分析数据集: {dataset} | 方法: {method_name}")
        print("="*80)
        
        predictions = methods[method_name]['predictions']
        metrics = methods[method_name]['metrics']
        
        for i, (pred_item, metric_item) in enumerate(zip(predictions, metrics)):
            entropy_history = metric_item.get('entropy_history')
            if not entropy_history:
                continue

            full_text = pred_item['predicted']
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(full_text, add_special_tokens=False))
            
            state = analyzer.create_state()
            
            for t_idx, entropy in enumerate(entropy_history):
                prev_model = state.current_model
                next_model = analyzer.make_routing_decision(state, entropy)
                
                if prev_model == 'slm' and next_model == 'llm':
                    if t_idx >= len(tokens): continue
                    
                    switch_token = tokens[t_idx]
                    cleaned_token = switch_token.replace('Ġ', ' ').strip()
                    
                    if cleaned_token:
                        switch_token_counts[cleaned_token] += 1
                        
                        if len(switch_contexts[cleaned_token]) < 10:
                            start_idx, end_idx = find_sentence_context_indices(tokens, t_idx)
                            context_tokens = tokens[start_idx:end_idx]
                            
                            switch_point_in_context = t_idx - start_idx

                            if switch_point_in_context < 0 or switch_point_in_context >= len(context_tokens): continue

                            before_tokens = "".join(context_tokens[:switch_point_in_context]).replace('Ġ', ' ').replace('Ċ', '\n')
                            highlighted_token = context_tokens[switch_point_in_context].replace('Ġ', ' ').replace('Ċ', '\n')
                            after_tokens = "".join(context_tokens[switch_point_in_context+1:]).replace('Ġ', ' ').replace('Ċ', '\n')
                            
                            context_str = f"...{before_tokens}  >>>{highlighted_token}<<<  {after_tokens}..."
                            
                            context_data = {
                                'context': context_str,
                                'entropy': entropy,
                                'z_score': state.last_z_score
                            }
                            switch_contexts[cleaned_token].append(context_data)
                
                # 更新状态必须在所有检查之后
                state.current_model = next_model
                if state.current_model == 'llm': state.llm_tokens += 1
                else: state.slm_tokens += 1

    # 在所有样本分析完毕后，打印词频统计结果到控制台
    print("\n" + "="*80)
    print("切换点 Token 词频统计 (Top 30)")
    print("="*80)

    if not switch_token_counts:
        print("未找到任何切换点。")
    else:
        sorted_counts = switch_token_counts.most_common()
        
        print(f"{'Token':<20} | {'出现次数':<10}")
        print("-"*33)
        for token, count in sorted_counts[:30]:
            print(f"{token:<20} | {count:<10}")
            
        # 将详细上下文写入文件
        output_dir = os.path.dirname(os.path.abspath(results_file))
        output_filename = os.path.join(output_dir, 'switch_contexts.txt')
        write_contexts_to_file(output_filename, sorted_counts, switch_contexts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析 CoTRouter 结果中的模型切换点，统计词频并输出详细上下文到文件。")
    parser.add_argument('--results_file', type=str, required=True, help='指向 results.json 文件的路径。')
    parser.add_argument('--method_name', type=str, required=True, help='要分析的实验方法名称 (例如, CoTRouter-R30)。')
    
    args = parser.parse_args()
    
    analyze_switch_points(args.results_file, args.method_name)
