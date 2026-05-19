import json
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import random # Added for safety, though slicing will be used
from transformers import AutoTokenizer
import seaborn as sns

"""
python www3_case.py --results_file ./results/param_sweep/GSM8K_20250730_205653_7/results.json --method_name Sweep_1
python www3_case.py --results_file ./results/param_sweep/MATH_20250730_220604_7/results.json --method_name Sweep_1
python www3_case.py --results_file ./results/param_sweep/AIME_20250730_175240_7/results.json --method_name Sweep_1


"""

# --- 1. CoTRouter Core Components (for Routing Simulation) ---
# (Classes KalmanFilter1D, EWMAControlChart, CoTRouterState, CoTRouterAnalyzer remain unchanged)

class KalmanFilter1D:
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
        return 0.0 # Placeholder for plotting

@dataclass
class CoTRouterState:
    kalman_filter: KalmanFilter1D
    ewma_chart: EWMAControlChart
    L_threshold: float = 3.0
    commitment_remaining: int = 0
    current_model: str = 'slm'
    llm_tokens: int = 0
    slm_tokens: int = 0
    last_z_score: float = 0.0

class CoTRouterAnalyzer:
    def __init__(self, target_llm_ratio: float = 0.3, beta_gain: float = 2.0, k_base: int = 20, alpha_severity: float = 2.0, L_min: float = 1.0, L_max: float = 4.0, initial_L: float = 2.0, kalman_process_var: float = 0.3, kalman_measurement_var: float = 0.3, ewma_lambda: float = 0.02):
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
        
        # Recalculate Z-score for decision making
        t_for_std = state.ewma_chart.t
        if t_for_std <= 0: z_score = 0.0
        else:
            t_temp = t_for_std - 1
            lambda_param = state.ewma_chart.lambda_param
            process_std = state.ewma_chart.process_std
            
            if t_temp > 30:
                 variance = (lambda_param / (2 - lambda_param)) * (process_std ** 2)
            else:
                 variance = (lambda_param / (2 - lambda_param)) * \
                             (process_std ** 2) * (1 - (1 - lambda_param) ** (2 * t_temp))
            std_dev = np.sqrt(variance)
            z_score = (filtered_value - prev_mu) / std_dev if std_dev > 1e-6 else 0.0

        state.ewma_chart.update(filtered_value)
        state.last_z_score = z_score 
        
        self.update_threshold(state) # Use the dynamically adjusted threshold
        L_threshold = state.L_threshold

        if z_score > L_threshold and state.current_model == 'slm':
            commitment = self.compute_commitment_duration(z_score, L_threshold)
            state.commitment_remaining = commitment - 1
            return 'llm'
        
        return 'slm'


# --- 2. Plotting and Data Processing Functions ---

def get_ema_smooth(data: np.ndarray, span: int = 5) -> np.ndarray:
    """Calculates Exponential Moving Average (EMA) to smooth data."""
    if len(data) < 2:
        return data
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def plot_single_trajectory(
    entropy_history: List[float], 
    method_name: str, 
    dataset_name: str, 
    sample_id: int, 
    smooth: bool, 
    analyzer: CoTRouterAnalyzer,
    output_dir: str,
    colors: List[str]
):
    """
    Plots the entropy trajectory for a single sample, color-coded by model assignment.
    """
    if not entropy_history:
        return

    # 1. Simulate routing decisions
    model_assignments = []
    state = analyzer.create_state()
    
    # Pre-simulate all steps to get assignments
    for i, entropy in enumerate(entropy_history):
        # The assignment for the *current* token is the state *before* the decision is made for the *next* token
        model_assignments.append(state.current_model)
        
        # The decision for the next token is made here
        next_model = analyzer.make_routing_decision(state, entropy)
        state.current_model = next_model 
        
        # The last token in entropy_history will determine the state for a hypothetical next token, 
        # but we only care about the assignment for the generated tokens up to len(entropy_history).
        # We need to ensure we have the assignment for the very last token.
        if i == len(entropy_history) - 1 and len(model_assignments) < len(entropy_history):
            model_assignments.append(state.current_model) # This should cover the last token's assignment

    # Due to the stateful nature, we must ensure assignments match entropy length
    if len(model_assignments) < len(entropy_history):
         # If the simulation missed the last token's assignment, append the final state
         model_assignments.append(state.current_model)

    model_assignments = model_assignments[:len(entropy_history)]

    # 2. Apply smoothing
    data = np.array(entropy_history)
    if smooth:
        y_data = get_ema_smooth(data, span=5)
        title_suffix = " (EMA Smoothed)"
    else:
        y_data = data
        title_suffix = " (Raw Data)"
        
    x_data = np.arange(len(y_data))

    # 3. Split data points
    slm_x, slm_y = [], []
    llm_x, llm_y = [], []

    for x, y, model in zip(x_data, y_data, model_assignments):
        if model == 'slm':
            slm_x.append(x)
            slm_y.append(y)
        else:
            llm_x.append(x)
            llm_y.append(y)

    # 4. Plotting Configuration
    FIGSIZE = (8,5)
    FONTSIZE = 18
    MARKER_SIZE = 6
    ALLWIDTH = 1.5
    
    # Color assignments based on user's palette
    # COLORS = ["#264653", "#299D92", "#8AB17C", "#E8C56B", "#E66F51"]
    SLM_COLOR = colors[3] # blue
    LLM_COLOR = colors[1] # red

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot SLM Token scatter
    ax.scatter(slm_x, slm_y, color=SLM_COLOR, marker='o', s=MARKER_SIZE**2, 
               label='SLM Token', alpha=0.7, zorder=3)
    
    # Plot LLM Token scatter
    ax.scatter(llm_x, llm_y, color=LLM_COLOR, marker='s', s=MARKER_SIZE**2, 
               label='LLM Token', alpha=0.8, zorder=4)

    # Plot connecting line
    ax.plot(x_data, y_data, color='gray', linestyle='-', linewidth=ALLWIDTH/2, alpha=0.5, zorder=0)

    # Set labels and title
    # ax.set_title(
    #     f"Entropy Trajectory for Sample {sample_id} ({dataset_name} | {method_name}){title_suffix}", 
    #     fontsize=FONTSIZE + 2
    # )
    ax.set_xlabel("Token", fontsize=FONTSIZE)
    ax.set_ylabel("Entropy" + (" (EMA)" if smooth else ""), fontsize=FONTSIZE)
    
    # Set styles
    ax.legend(fontsize=FONTSIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE - 2)
    
    # Set Y-axis limits (ylim) dynamically
    y_min = max(-0.2, y_data.min() - 0.2)
    y_max = y_data.max() + 0.2
    ax.set_ylim(y_min, y_max)
    
    # Set spine width
    for spine in ax.spines.values():
        spine.set_linewidth(ALLWIDTH)

    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    smooth_tag = 'smooth' if smooth else 'raw'
    filename = f"entropy_trace_{dataset_name}_{method_name}_sample_{sample_id}_{smooth_tag}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"Saved: {filename}")


def analyze_and_plot_entropy(
    results_file: str, 
    method_name: str, 
    num_plots: int = 10, 
    smooth: bool = False, 
    output_dir: str = './case_entropy'
):
    """Main analysis and plotting function."""
    print(f"Loading results from: {results_file}")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return

    # Custom colors (based on user's palette)
    # COLORS = ["#264653", "#299D92", "#8AB17C", "#E8C56B", "#E66F51"]
    COLORS = sns.color_palette("Paired")
    
    # Initialize analyzer
    try:
        target_ratio = float(method_name.split('-R')[-1]) / 100
    except (ValueError, IndexError):
        target_ratio = 0.3 
    analyzer = CoTRouterAnalyzer(target_llm_ratio=target_ratio)
    
    total_plotted = 0

    for dataset, methods in results_data.items():
        if method_name not in methods:
            continue
        
        metrics = methods[method_name]['metrics']
        
        print(f"\nProcessing Dataset: {dataset} | Method: {method_name}")
        
        # 1. Filter out samples with entropy history
        traces = [(i, metric_item.get('entropy_history')) 
                  for i, metric_item in enumerate(metrics) 
                  if metric_item.get('entropy_history') and len(metric_item['entropy_history']) > 0]
        
        if not traces:
            print(f"Warning: No valid entropy traces found for {dataset} | {method_name}.")
            continue

        # 2. Select the FIRST N samples (as requested)
        selected_traces = traces[:num_plots]

        for original_index, entropy_history in selected_traces:
            plot_single_trajectory(
                entropy_history=entropy_history, 
                method_name=method_name, 
                dataset_name=dataset, 
                sample_id=original_index, 
                smooth=smooth, 
                analyzer=analyzer,
                output_dir=output_dir,
                colors=COLORS
            )
            total_plotted += 1
            
        # Stop plotting after reaching the target number across all datasets
        if total_plotted >= num_plots:
            break

    if total_plotted == 0:
        print("Warning: No samples were plotted.")

def get_token_assignment_and_text(
    prediction_text: str,
    entropy_history: List[float],
    analyzer: CoTRouterAnalyzer,
    tokenizer: AutoTokenizer
) -> List[Tuple[str, str]]:
    """
    模拟路由过程，并将结果 Token 化，然后与模型分配和文本片段配对。
    
    返回: [(model_type, text_segment), ...]
    """
    if not entropy_history:
        return []

    # 1. 对预测文本进行分词
    # 注意: is_split_into_words=True 用于处理已经预分词的输入，这里我们是对完整的预测文本进行分词
    # 特别注意：许多生成模型在生成时，第一个Token通常是接在Prompt后面的，
    # 但我们这里的entropy_history是从第一个生成的Token开始的。
    token_ids = tokenizer.encode(prediction_text, add_special_tokens=False)
    
    # 2. 模拟路由决策
    model_assignments = []
    state = analyzer.create_state()
    
    # 路由决策是基于上一个Token的熵做出的，决定当前Token使用哪个模型。
    # 假设 len(entropy_history) == len(token_ids)
    for i, entropy in enumerate(entropy_history):
        # 决策是针对下一个Token的，但在记录时，我们关注的是当前这个Token是用哪个模型生成的
        # 为了与plotting函数逻辑保持一致，我们使用 state.current_model
        model_assignments.append(state.current_model)
        
        # 做出下一个决策
        next_model = analyzer.make_routing_decision(state, entropy)
        state.current_model = next_model

    # 确保长度匹配
    if len(token_ids) != len(model_assignments):
        print(f"Warning: Token count ({len(token_ids)}) does not match entropy history length ({len(entropy_history)}). Skipping.")
        return []

    # 3. 将 Token ID 解码回文本
    token_texts = [tokenizer.decode([id_], skip_special_tokens=True) for id_ in token_ids]
    
    # 4. 合并相同模型的相邻 Token
    trace = []
    current_model = None
    current_segment = ""
    
    for model, text in zip(model_assignments, token_texts):
        if model == current_model:
            current_segment += text
        else:
            if current_model is not None:
                trace.append((current_model, current_segment.replace('\n', '\\n')))
            
            # 开始新的片段
            current_model = model
            current_segment = text
            
    # 添加最后一个片段
    if current_model is not None:
        trace.append((current_model, current_segment.replace('\n', '\\n')))
        
    return trace


def save_routing_trace_to_txt(
    results_file: str,
    method_name: str,
    num_samples: int = 10,
    output_dir: str = './case_entropy_tokens'
):
    """
    主函数：加载数据，分词，模拟路由，并将Token级切换结果保存到TXT文件。
    """
    print(f"Loading results from: {results_file}")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return

    # 1. 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer. Please ensure 'transformers' is installed: {e}")
        return
        
    # 2. 初始化 Router Analyzer
    try:
        target_ratio = float(method_name.split('-R')[-1]) / 100
    except (ValueError, IndexError):
        # 使用默认值，或从结果文件中尝试读取配置（如果可用）
        target_ratio = 0.3 
    analyzer = CoTRouterAnalyzer(target_llm_ratio=target_ratio)
    
    os.makedirs(output_dir, exist_ok=True)
    total_processed = 0

    for dataset, methods in results_data.items():
        if method_name not in methods:
            continue
            
        method_data = methods[method_name]
        metrics = method_data.get('metrics', [])
        predictions = method_data.get('predictions', [])
        
        print(f"\nProcessing Dataset: {dataset} | Method: {method_name}")
        
        # 3. 筛选并处理样本
        for i in range(min(len(metrics), len(predictions))):
            if total_processed >= num_samples:
                break
                
            metric_item = metrics[i]
            prediction_item = predictions[i]
            
            entropy_history = metric_item.get('entropy_history')
            predicted_text = prediction_item.get('predicted', '')
            
            if not entropy_history or not predicted_text:
                continue
                
            # 4. 获取 Token 级路由轨迹
            routing_trace = get_token_assignment_and_text(
                prediction_text=predicted_text,
                entropy_history=entropy_history,
                analyzer=analyzer,
                tokenizer=tokenizer
            )

            if routing_trace:
                filename = f"routing_trace_{dataset}_{method_name}_sample_{i}.txt"
                filepath = os.path.join(output_dir, filename)
                
                output_lines = [f'"{model}": "{text}",' for model, text in routing_trace]
                
                # 移除最后一个逗号
                if output_lines:
                    output_lines[-1] = output_lines[-1].rstrip(',')
                
                # 【修正点】将 replace 操作的结果预先存储
                cleaned_predicted_text = predicted_text.replace('\n', ' ')
                
                # 【修正点】使用清理后的变量
                header = f"样本 {i} 推理结果：{cleaned_predicted_text}\n\n"
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.write('\n'.join(output_lines))
                    
                print(f"Saved routing trace for sample {i} to: {filename}")
                total_processed += 1
                
        if total_processed >= num_samples:
            break

    if total_processed == 0:
        print("Warning: No samples were processed or saved.")


# --- 3. 更新主执行块 (if __name__ == '__main__':) ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plots entropy trajectories for selected samples, colored by routing decisions (SLM/LLM).")
    parser.add_argument('--results_file', type=str, required=True, help='Path to the results.json file.')
    parser.add_argument('--method_name', type=str, required=True, help='Name of the experiment method (e.g., CoTRouter-R30 or Sweep_1).')
    parser.add_argument('--num_plots', type=int, default=10, help='Number of samples to plot (selects the first N available).')
    parser.add_argument('--smooth', action='store_true', help='Apply EMA smoothing to the entropy curve.')
    parser.add_argument('--output_dir', type=str, default='./case_entropy', help='Directory to save the plots.')
    parser.add_argument('--save_tokens', action='store_true', help='Save the token-level routing trace to TXT files.')
    parser.add_argument('--token_output_dir', type=str, default='./case_entropy_tokens', help='Directory to save the token trace TXT files.')
    
    args = parser.parse_args()
    
    # 1. 执行绘图逻辑
    analyze_and_plot_entropy(
        results_file=args.results_file, 
        method_name=args.method_name, 
        num_plots=args.num_plots, 
        smooth=args.smooth, 
        output_dir=args.output_dir
    )
    
    # 2. 执行保存Token切换逻辑 (新增)
    if args.save_tokens:
        save_routing_trace_to_txt(
            results_file=args.results_file,
            method_name=args.method_name,
            num_samples=args.num_plots, # 使用相同的数量
            output_dir=args.token_output_dir
        )