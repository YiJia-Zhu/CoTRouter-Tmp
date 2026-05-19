# runner.py
"""
The main benchmark execution engine.
The BenchmarkRunner class orchestrates different inference strategies,
including DE-Cascade and standard baselines.
"""
import time
import json
import os
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List

from config import (ModelConfig, InferenceState, DE_CASCADE_BATCH_SIZE, GLOBAL_INITIAL_CHUNK_SIZE,
                    GLOBAL_LLM_INTER_CHUNK_SIZE, GLOBAL_MAX_TOKENS, GLOBAL_LAG, GLOBAL_THRESHOLD, TARGET_LLM_TOKEN_RATIO)
from utils import (OnlineEntropyPeakDetector, calculate_shannon_entropy, extract_answer, is_correct_answer)
from models import VLLMModel
from data import build_math_prompt
import random
import csv

class BenchmarkRunner:
    """Runs DE-Cascade and baselines with batched inference."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.results = {}
        self.llm = None
        self.slm = None
        
    def initialize_models(self):
        """Initializes the LLM and SLM."""
        print("Initializing models with vLLM...")
        self.llm = VLLMModel(self.config.llm_path, self.config, self.config.llm_gpu_ids, 
                               self.config.llm_tensor_parallel_size, self.config.llm_gpu_memory_utilization)
        self.slm = VLLMModel(self.config.slm_path, self.config, self.config.slm_gpu_id, 
                               self.config.slm_tensor_parallel_size, self.config.slm_gpu_memory_utilization)

    def run_de_cascade_benchmark(self, dataset_name: str, problems: List[Dict]):
        method_name = 'DE-Cascade'
        print(f"\nRunning {method_name} (Batched) on {dataset_name}" + "\n" + "="*50)
        self.results[dataset_name] = self.results.get(dataset_name, {})
        current_run_results = self.results[dataset_name][method_name] = {'correct': 0, 'total': 0, 'metrics': [], 'predictions': []}
        start_time = time.time()
        
        total_batches = (len(problems) + DE_CASCADE_BATCH_SIZE - 1) // DE_CASCADE_BATCH_SIZE
        
        for batch_idx in range(total_batches):
            print(f"\n{'='*20} Processing Batch {batch_idx + 1}/{total_batches} {'='*20}")
            batch_start = batch_idx * DE_CASCADE_BATCH_SIZE
            batch_end = min((batch_idx + 1) * DE_CASCADE_BATCH_SIZE, len(problems))
            batch_problems = problems[batch_start:batch_end]

            active_states = []
            for p in batch_problems:
                prompt_text = build_math_prompt(p['question'])
                initial_token_ids = self.slm.tokenizer.encode(prompt_text)
                state = InferenceState(
                    request_id=str(uuid.uuid4()),
                    problem=p,
                    prompt=prompt_text,
                    token_ids=initial_token_ids,
                    peak_detector=OnlineEntropyPeakDetector(lag=GLOBAL_LAG, threshold=GLOBAL_THRESHOLD)
                )
                state.metrics['llm_tokens'] = 0 # Prompt tokens are not counted for LLM
                state.metrics['slm_tokens'] = len(initial_token_ids)
                if GLOBAL_INITIAL_CHUNK_SIZE == 0:
                    state.initial_chunk_generated = True
                    state.current_model = 'slm'
                active_states.append(state)
            
            iteration = 0
            # 7.9modified: Vocabulary Mismatch
            while active_states:
                iteration += 1
                if iteration > GLOBAL_MAX_TOKENS:
                    print("Reached max iterations, ending batch.")
                    break
                
                print(f"\nBatch {batch_idx + 1} - Iteration {iteration}, Active Inferences: {len(active_states)}")
                
                llm_states = [s for s in active_states if not s.initial_chunk_generated or s.current_model == 'llm']
                slm_states = [s for s in active_states if s.initial_chunk_generated and s.current_model == 'slm']
                
                # --- 修复部分 START ---
                if llm_states:
                    # **关键改动**: 传递完整文本，而不是 token_ids
                    llm_prompts = [s.prompt + s.full_generation for s in llm_states]
                    chunk_size = GLOBAL_INITIAL_CHUNK_SIZE if not llm_states[0].initial_chunk_generated else GLOBAL_LLM_INTER_CHUNK_SIZE
                    if chunk_size > 0:
                        # **关键改动**: 使用 prompts 参数而不是 prompt_token_ids
                        llm_chunks = self.llm.generate(prompts=llm_prompts, max_tokens=chunk_size)
                        for state, chunk_text in zip(llm_states, llm_chunks):
                            if not state.initial_chunk_generated: state.initial_chunk_generated = True
                            else: state.metrics['llm_interventions'] += 1
                            
                            # 使用LLM自己的分词器计算token数量，然后附加文本
                            chunk_ids = self.llm.tokenizer.encode(chunk_text) # 仅用于统计
                            state.full_generation += chunk_text
                            state.metrics['llm_tokens'] += len(chunk_ids)
                            state.current_model = 'slm'

                if slm_states:
                    # **关键改动**: 传递完整文本，而不是 token_ids
                    slm_prompts = [s.prompt + s.full_generation for s in slm_states]
                    # **关键改动**: 使用 prompts 参数而不是 prompt_token_ids
                    slm_results = self.slm.generate_one_token(prompts=slm_prompts)
                    for state, (token, token_id, logprobs) in zip(slm_states, slm_results):
                        if token_id in [self.slm.eos_token_id, self.llm.eos_token_id]: # or not token
                            state.is_finished = True
                            continue
                        entropy = calculate_shannon_entropy(logprobs, self.slm.vocab_size)
                        state.metrics['entropy_history'].append(entropy)
                        if state.peak_detector.add_datapoint(entropy) == "PEAK":
                            state.metrics['entropy_peaks'] += 1
                            state.current_model = 'llm'
                        else:
                            # 只附加文本，不再操作 token_ids 列表
                            state.full_generation += token
                            state.metrics['slm_tokens'] += 1
            # --- 修复部分 END ---
                
                remaining_states = []
                for state in active_states:
                    current_total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
                    if state.is_finished or current_total_tokens >= GLOBAL_MAX_TOKENS:
                        predicted_answer = extract_answer(state.full_generation)
                        is_correct = is_correct_answer(predicted_answer, state.problem['answer'])
                        current_run_results['total'] += 1
                        if is_correct: current_run_results['correct'] += 1
                        
                        state.metrics['total_tokens'] = current_total_tokens
                        current_run_results['metrics'].append(state.metrics)
                        current_run_results['predictions'].append({
                            'question': state.problem['question'], 'predicted': state.full_generation,
                            'predicted_answer': predicted_answer, 'ground_truth': state.problem['answer'], 'correct': is_correct
                        })
                        print(f"  Request {state.request_id[:6]} finished. Correct: {is_correct}, Total Tokens: {current_total_tokens}")
                    else:
                        remaining_states.append(state)
                active_states = remaining_states
        
        # Calculate average time and assign to all metrics records
        total_wall_time = time.time() - start_time
        avg_time = total_wall_time / len(problems) if problems else 0
        for metric_record in current_run_results['metrics']:
            metric_record['wall_time'] = avg_time
            
        print(f"\nDE-Cascade (Batched) finished in {total_wall_time:.2f}s. Average time per problem: {avg_time:.2f}s")

    def run_baseline(self, model_type: str, dataset_name: str, problems: List[Dict]):
        method_name = f'{model_type}-Only'
        print(f"\nRunning {method_name} (Batched) baseline on {dataset_name}" + "\n" + "="*50)
        
        self.results[dataset_name] = self.results.get(dataset_name, {})
        current_run_results = self.results[dataset_name][method_name] = {'correct': 0, 'total': len(problems), 'metrics': [], 'predictions': []}
        model = self.llm if model_type == 'LLM' else self.slm

        prompts = [build_math_prompt(p['question']) for p in problems]
        start_time = time.time()
        # Reserve some tokens for the prompt to avoid exceeding model max length
        max_gen_tokens = GLOBAL_MAX_TOKENS
        responses = model.generate(prompts, max_tokens=max_gen_tokens)
        total_wall_time = time.time() - start_time
        avg_time = total_wall_time / len(problems) if problems else 0

        print(f"{method_name} batch processed in {total_wall_time:.2f}s. Average time per problem: {avg_time:.2f}s")

        for i, problem in enumerate(problems):
            response = responses[i]
            predicted_answer = extract_answer(response)
            is_correct = is_correct_answer(predicted_answer, problem['answer'])
            if is_correct: current_run_results['correct'] += 1
            
            # ** RESTORED METRICS COLLECTION **
            prompt_plus_response = prompts[i] + response
            token_count = len(model.tokenizer.encode(prompt_plus_response))
            metrics = {
                'total_tokens': token_count, 
                'llm_tokens': token_count if model_type == 'LLM' else 0,
                'slm_tokens': token_count if model_type == 'SLM' else 0,
                'llm_interventions': 0, 
                'wall_time': avg_time
            }
            current_run_results['metrics'].append(metrics)
            current_run_results['predictions'].append({
                'question': problem['question'], 'predicted': response,
                'predicted_answer': predicted_answer, 'ground_truth': problem['answer'], 'correct': is_correct
            })
    
    
    # 在 runner.py 文件中，找到并替换这个函数

    def run_token_ratio_benchmark(self, dataset_name: str, problems: List[Dict]):
        """
        新增功能：按照设定的目标Token比例，随机决定由哪个模型生成下一个Token（或块）。
        （已补充完整的打印信息）
        """
        # 从config导入目标比例和块大小
        from config import TARGET_LLM_TOKEN_RATIO, GLOBAL_LLM_INTER_CHUNK_SIZE
        
        method_name = f'Token-Ratio-Target-{int(TARGET_LLM_TOKEN_RATIO * 100)}%LLM'
        print(f"\nRunning {method_name} (Batched) on {dataset_name}" + "\n" + "="*50)

        # --- 核心逻辑：计算调整后的决策概率 ---
        R = TARGET_LLM_TOKEN_RATIO
        C_llm = GLOBAL_LLM_INTER_CHUNK_SIZE
        C_slm = 1
        
        denominator = C_llm * (1 - R) + R * C_slm
        if denominator == 0:
            adjusted_prob = 1.0 if R > 0.5 else 0.0
        else:
            adjusted_prob = (R * C_slm) / denominator
        
        print(f"  Target LLM token ratio: {R:.1%}")
        print(f"  LLM chunk size: {C_llm}, SLM chunk size: {C_slm}")
        print(f"  Calculated decision probability to choose LLM: {adjusted_prob:.3%}")
        # -----------------------------------------

        self.results[dataset_name] = self.results.get(dataset_name, {})
        current_run_results = self.results[dataset_name][method_name] = {'correct': 0, 'total': 0, 'metrics': [], 'predictions': []}
        start_time = time.time()
        
        total_batches = (len(problems) + DE_CASCADE_BATCH_SIZE - 1) // DE_CASCADE_BATCH_SIZE
        
        for batch_idx in range(total_batches):
            print(f"\n{'='*20} Processing Batch {batch_idx + 1}/{total_batches} {'='*20}")
            batch_start = batch_idx * DE_CASCADE_BATCH_SIZE
            batch_end = min((batch_idx + 1) * DE_CASCADE_BATCH_SIZE, len(problems))
            batch_problems = problems[batch_start:batch_end]

            active_states = []
            for p in batch_problems:
                prompt_text = build_math_prompt(p['question'])
                initial_token_ids = self.slm.tokenizer.encode(prompt_text)
                state = InferenceState(
                    request_id=str(uuid.uuid4()), problem=p, prompt=prompt_text,
                    token_ids=initial_token_ids
                )
                state.metrics['slm_tokens'] = len(initial_token_ids)
                active_states.append(state)
            
            iteration = 0
            while active_states:
                iteration += 1
                if iteration > GLOBAL_MAX_TOKENS:
                    print("Reached max iterations, ending batch.")
                    break
                
                # --- 新增的打印信息 ---
                # print(f"\nBatch {batch_idx + 1} - Iteration {iteration}, Active Inferences: {len(active_states)}")
                # -----------------------

                llm_batch_states = []
                slm_batch_states = []

                for state in active_states:
                    if random.random() < adjusted_prob:
                        llm_batch_states.append(state)
                    else:
                        slm_batch_states.append(state)
                
                if llm_batch_states:
                    llm_ids = [s.token_ids for s in llm_batch_states]
                    llm_chunks = self.llm.generate(prompt_token_ids=llm_ids, max_tokens=C_llm)
                    for state, chunk_text in zip(llm_batch_states, llm_chunks):
                        state.metrics['llm_interventions'] += 1
                        chunk_ids = self.llm.tokenizer.encode(chunk_text)
                        state.token_ids.extend(chunk_ids)
                        state.full_generation += chunk_text
                        state.metrics['llm_tokens'] += len(chunk_ids)

                if slm_batch_states:
                    slm_ids = [s.token_ids for s in slm_batch_states]
                    slm_single_tokens = self.slm.generate(prompt_token_ids=slm_ids, max_tokens=C_slm)
                    for i, state in enumerate(slm_batch_states):
                        token_text = slm_single_tokens[i]
                        try:
                           token_id = self.slm.tokenizer.encode(token_text)[0]
                           if token_id in [self.slm.eos_token_id, self.llm.eos_token_id]:
                                state.is_finished = True
                                continue
                           state.token_ids.append(token_id)
                           state.full_generation += token_text
                           state.metrics['slm_tokens'] += 1
                        except IndexError:
                            state.is_finished = True

                remaining_states = []
                for state in active_states:
                    current_total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
                    if state.is_finished or current_total_tokens >= GLOBAL_MAX_TOKENS:
                        predicted_answer = extract_answer(state.full_generation)
                        is_correct = is_correct_answer(predicted_answer, state.problem['answer'])
                        current_run_results['total'] += 1
                        if is_correct: current_run_results['correct'] += 1
                        state.metrics['total_tokens'] = current_total_tokens
                        current_run_results['metrics'].append(state.metrics)
                        current_run_results['predictions'].append({
                            'question': state.problem['question'], 'predicted': state.full_generation,
                            'predicted_answer': predicted_answer, 'ground_truth': state.problem['answer'], 'correct': is_correct
                        })
                        # --- 新增的打印信息 ---
                        print(f"  Request {state.request_id[:6]} finished. Correct: {is_correct}, Total Tokens: {current_total_tokens}")
                        # -----------------------
                    else:
                        remaining_states.append(state)
                active_states = remaining_states
        
        total_wall_time = time.time() - start_time
        avg_time = total_wall_time / len(problems) if problems else 0
        for metric_record in current_run_results['metrics']:
            metric_record['wall_time'] = avg_time
            
        print(f"\n{method_name} (Batched) finished in {total_wall_time:.2f}s. Average time per problem: {avg_time:.2f}s")
    
    
    def save_results(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")
    

    def _save_summary_to_csv(self, dataset, method, data, metrics_avg):
        """
        (New Helper Function) Saves the summary of a single experiment to a CSV file.
        This function is called by print_summary.
        """
        csv_path = "results/experiment_param_"+str(dataset)+".csv"
        
        # 1. Gather all configuration and hyperparameter data
        config_data = {
            'llm_path': self.config.llm_path.split('/')[-1],
            'slm_path': self.config.slm_path.split('/')[-1],
            'dataset': dataset,
            'threshold': GLOBAL_THRESHOLD,
            'llm_inter_chunk_size': GLOBAL_LLM_INTER_CHUNK_SIZE,
            'initial_chunk_size': GLOBAL_INITIAL_CHUNK_SIZE,
            'lag': GLOBAL_LAG,
            'target_token_ratio': TARGET_LLM_TOKEN_RATIO,
        }

        # 2. Gather all performance metrics
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        total_tokens = metrics_avg.get('total_tokens', 1)
        if total_tokens == 0: total_tokens = 1 # Avoid division by zero
        
        llm_perc = (metrics_avg.get('llm_tokens', 0) / total_tokens) * 100
        slm_perc = (metrics_avg.get('slm_tokens', 0) / total_tokens) * 100

        metrics_data = {
            'method': method,
            'accuracy': f"{accuracy:.4f}",
            'correct': data['correct'],
            'total': data['total'],
            'avg_wall_time_s': f"{metrics_avg.get('wall_time', 0):.2f}",
            'avg_total_tokens': f"{metrics_avg.get('total_tokens', 0):.1f}",
            'avg_llm_tokens': f"{metrics_avg.get('llm_tokens', 0):.1f}",
            'llm_token_percent': f"{llm_perc:.1f}",
            'avg_slm_tokens': f"{metrics_avg.get('slm_tokens', 0):.1f}",
            'slm_token_percent': f"{slm_perc:.1f}",
            'avg_llm_interventions': f"{metrics_avg.get('llm_interventions', 0):.1f}" if 'DE-Cascade' in method else 'N/A'
        }
        
        # 3. Combine all data and define CSV headers
        row_data = {**config_data, **metrics_data}
        fieldnames = list(row_data.keys())
        
        # 4. Write to CSV, creating the file and header if it doesn't exist
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    def print_summary(self):
        """
        (Restored Function) Prints a detailed summary of all experiment results.
        """
        print("\n" + "="*80 + "\nEXPERIMENT SUMMARY\n" + "="*80)
        for dataset, methods in self.results.items():
            print(f"\n{dataset.upper()} Results:\n" + "-"*50)
            for method, data in methods.items():
                if not data['total']:
                    print(f"\n{method}: No results to display.")
                    continue
                
                accuracy = data['correct'] / data['total']
                
                # Calculate average metrics
                metrics_avg = {
                    key: np.mean([m.get(key, 0) for m in data['metrics']]) 
                    for key in ['total_tokens', 'llm_tokens', 'slm_tokens', 'llm_interventions', 'wall_time']
                }

                print(f"\n{method}:")
                print(f"  Accuracy: {accuracy:.2%} ({data['correct']}/{data['total']})")
                print(f"  Avg Wall Time (per problem): {metrics_avg['wall_time']:.2f}s")
                print(f"  Avg Total Tokens: {metrics_avg['total_tokens']:.1f}")

                if metrics_avg['total_tokens'] > 0:
                    llm_perc = (metrics_avg['llm_tokens'] / metrics_avg['total_tokens']) * 100
                    slm_perc = (metrics_avg['slm_tokens'] / metrics_avg['total_tokens']) * 100
                    print(f"    - Avg LLM Tokens: {metrics_avg['llm_tokens']:.1f} ({llm_perc:.1f}%)")
                    print(f"    - Avg SLM Tokens: {metrics_avg['slm_tokens']:.1f} ({slm_perc:.1f}%)")
                
                if 'DE-Cascade' in method:
                    print(f"  Avg LLM Interventions: {metrics_avg['llm_interventions']:.1f}")
                
                self._save_summary_to_csv(dataset, method, data, metrics_avg)
