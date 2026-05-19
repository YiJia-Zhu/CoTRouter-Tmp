# analyze_switch_points.py
"""
一个用于分析 CoTRouter 实验结果的脚本。

该脚本读取 `results.json` 文件，模拟 CoTRouter 的路由决策过程，
识别出模型从 SLM 切换到 LLM 的确切位置，统计在切换点
出现的 token 的词频，并将包含上下文示例输出到文件。
最重要的是，它计算并导出细粒度（每 5%）的推理链切换分布
以及总体切换频率到 CSV 文件中。

python www2.py --results_file ./results/param_sweep/GSM8K_20250730_205653_7/results.json --method_name Sweep_1
python www2.py --results_file ./results/param_sweep/MATH_20250730_220604_7/results.json --method_name Sweep_1
python www2.py --results_file ./results/param_sweep/AIME_20250730_175240_7/results.json --method_name Sweep_1


"""
import json
import numpy as np
import argparse
import os
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import csv 
import math

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
        
        # 重新计算 EWMA 标准差 (std_dev)，使用 self.t - 1 的值
        t_for_std = self.t - 1
        if t_for_std <= 0: return 0.0

        if t_for_std > 30:
            variance = (self.lambda_param / (2 - self.lambda_param)) * (self.process_std ** 2)
        else:
            variance = (self.lambda_param / (2 - self.lambda_param)) * \
                         (self.process_std ** 2) * (1 - (1 - self.lambda_param) ** (2 * t_for_std))
        std_dev = np.sqrt(variance)

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
        
        z_score = state.ewma_chart.get_z_score(filtered_value, prev_mu)
        state.ewma_chart.update(filtered_value)
        
        state.last_z_score = z_score 
        
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
                f.write(f"  示例 #{i+1}:\n")
                f.write(f"    指标: Entropy = {data['entropy']:.4f} | Z-score = {data['z_score']:.4f}\n")
                f.write(f"    上下文: {data['context']}\n\n")
            f.write("\n" + "-"*40 + "\n\n")
    print("文件写入完成。")

def export_switch_stats_to_csv(output_path: str, method_name: str, stats: Dict):
    """将切换统计和细粒度分布结果导出到 CSV 文件"""
    print(f"\n正在将切换统计和细粒度分布导出到 CSV 文件: {output_path}")
    
    # 定义 5% 间隔的字段名 (1-5%, 6-10%, ..., 96-100%)
    fine_grained_fields = []
    for i in range(0, 100, 5):
        fine_grained_fields.append(f'Freq_{i+1:02d}-{i+5:02d}%')

    fieldnames = [
        'Method', 'Dataset', 'Total Samples', 'Total Tokens', 'Total Switches', 
        'Avg Switches per Sample', 'Switches per 100 Tokens'
    ] + fine_grained_fields
    
    csv_data = []
    for dataset, data in stats.items():
        total_switches = data['total_switches']
        
        row = {
            'Method': method_name,
            'Dataset': dataset,
            'Total Samples': data['total_samples'],
            'Total Tokens': data['total_tokens'],
            'Total Switches': data['total_switches'],
            'Avg Switches per Sample': f"{data['avg_switches_per_sample']:.4f}",
            'Switches per 100 Tokens': f"{data['switches_per_100_tokens']:.4f}",
        }
        
        # 填充细粒度分布数据
        for i, field_name in enumerate(fine_grained_fields):
            count = data['fine_grained_switches'][i]
            frequency = count / total_switches * 100 if total_switches > 0 else 0
            row[field_name] = f"{frequency:.2f}%"

        csv_data.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
        
    print("CSV 文件导出完成。")

def analyze_switch_points(results_file: str, method_name: str):
    """加载结果文件，分析指定方法的模型切换点，并计算切换率和细粒度阶段分布。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("错误：请先安装 'transformers' 和 'torch' 库。运行: pip install transformers torch")
        return

    print("正在加载分词器 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    except Exception as e:
        print(f"警告：加载分词器失败: {e}。上下文分析将不准确或失败。")
        return
    
    print(f"正在加载结果文件: {results_file}")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"错误：加载结果文件失败: {e}")
        return

    try:
        target_ratio = float(method_name.split('-R')[-1]) / 100
    except (ValueError, IndexError):
        target_ratio = 0.3 

    analyzer = CoTRouterAnalyzer(target_llm_ratio=target_ratio)
    
    switch_token_counts = Counter()
    switch_contexts = defaultdict(list)
    dataset_switch_stats = {} 

    # 定义细粒度区间的数量 (20个区间，每个5%)
    NUM_INTERVALS = 20

    for dataset, methods in results_data.items():
        if method_name not in methods:
            continue
        
        print("\n" + "="*80)
        print(f"正在分析数据集: {dataset} | 方法: {method_name}")
        print("="*80)
        
        predictions = methods[method_name]['predictions']
        metrics = methods[method_name]['metrics']
        
        total_switches = 0
        total_tokens_generated = 0
        switches_per_sample = [] 
        
        # 细粒度切换计数 (20个元素，初始化为0)
        fine_grained_switches = [0] * NUM_INTERVALS

        for i, (pred_item, metric_item) in enumerate(zip(predictions, metrics)):
            entropy_history = metric_item.get('entropy_history')
            if not entropy_history:
                continue
                
            tokens_generated = len(entropy_history)
            if tokens_generated == 0: continue

            total_tokens_generated += tokens_generated

            full_text = pred_item['predicted']
            try:
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(full_text, add_special_tokens=False))
            except Exception:
                tokens = [] 

            state = analyzer.create_state()
            sample_switches = 0
            
            for t_idx, entropy in enumerate(entropy_history):
                prev_model = state.current_model
                next_model = analyzer.make_routing_decision(state, entropy)
                
                if prev_model == 'slm' and next_model == 'llm':
                    # 识别到切换点
                    sample_switches += 1
                    total_switches += 1
                    
                    # === 统计细粒度切换位置 ===
                    progress_percentage = (t_idx + 1) / tokens_generated * 100
                    interval_index = min(math.ceil(progress_percentage / 5) - 1, NUM_INTERVALS - 1)
                    fine_grained_switches[interval_index] += 1
                    # =========================

                    # 收集上下文示例 (如果分词成功)
                    if t_idx < len(tokens): 
                        switch_token = tokens[t_idx]
                        cleaned_token = switch_token.replace('Ġ', ' ').strip()
                        
                        if cleaned_token and len(switch_contexts[cleaned_token]) < 10:
                            switch_token_counts[cleaned_token] += 1
                            
                            start_idx, end_idx = find_sentence_context_indices(tokens, t_idx)
                            
                            # **修正 f-string 语法错误：将包含 \n 的复杂逻辑移出 f-string**
                            before_tokens_str = "".join(tokens[start_idx:t_idx]).replace('Ġ', ' ').replace('Ċ', '\n')
                            highlighted_token_str = tokens[t_idx].replace('Ġ', ' ').replace('Ċ', '\n')
                            after_tokens_str = "".join(tokens[t_idx+1:end_idx]).replace('Ġ', ' ').replace('Ċ', '\n')
                            
                            context_data = {
                                'context': f"...{before_tokens_str}  >>>{highlighted_token_str}<<<  {after_tokens_str}...",
                                'entropy': entropy,
                                'z_score': state.last_z_score
                            }
                            switch_contexts[cleaned_token].append(context_data)
                
                # 更新状态必须在所有检查之后
                state.current_model = next_model
                if state.current_model == 'llm': state.llm_tokens += 1
                else: state.slm_tokens += 1
            
            if tokens_generated > 0:
                switches_per_sample.append(sample_switches)

        # 数据集级别统计计算
        num_samples = len(predictions)
        avg_switches_per_sample = np.mean(switches_per_sample) if switches_per_sample else 0
        switches_per_100_tokens = (total_switches / total_tokens_generated) * 100 if total_tokens_generated > 0 else 0
        
        dataset_switch_stats[dataset] = {
            'total_samples': num_samples,
            'total_switches': total_switches,
            'total_tokens': total_tokens_generated,
            'avg_switches_per_sample': avg_switches_per_sample,
            'switches_per_100_tokens': switches_per_100_tokens,
            'switches_per_sample_distribution': switches_per_sample, 
            'fine_grained_switches': fine_grained_switches 
        }
        
    # --- 最终统计结果输出和导出 ---
    
    # 打印数据集切换统计结果
    print("\n" + "="*80)
    print(f"数据集切换统计结果 ({method_name})")
    print("="*80)
    print(f"{'Dataset':<15} | {'总切换次数':<10} | {'平均切换':<8} | {'/100 Token':<11}")
    print("-" * 47)
    for dataset, stats in dataset_switch_stats.items():
        print(f"{dataset:<15} | {stats['total_switches']:<10} | {stats['avg_switches_per_sample']:.2f}{'':<6} | {stats['switches_per_100_tokens']:.2f}{'':<8}")

    # 导出 CSV 文件
    output_dir = os.path.dirname(os.path.abspath(results_file))
    csv_filename = os.path.join(output_dir, f'switch_distribution_fine_grained_{method_name}.csv')
    export_switch_stats_to_csv(csv_filename, method_name, dataset_switch_stats)

    # 打印词频统计结果
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
        output_filename = os.path.join(output_dir, 'switch_contexts.txt')
        write_contexts_to_file(output_filename, sorted_counts, switch_contexts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="分析 CoTRouter 结果中的模型切换点，统计词频并输出详细上下文到文件。")
    parser.add_argument('--results_file', type=str, required=True, help='指向 results.json 文件的路径。')
    parser.add_argument('--method_name', type=str, required=True, help='要分析的实验方法名称 (例如, CoTRouter-R30 或 Sweep_1)。')
    
    args = parser.parse_args()
    
    analyze_switch_points(args.results_file, args.method_name)