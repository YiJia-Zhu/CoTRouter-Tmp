"""
Complete implementation of CoTRouter: Adaptive Token Routing via Entropy Feedback
for Efficient Chain-of-Thought Reasoning

Modified version with detailed timing metrics for LLM, SLM, routing decision, and total overhead.
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import uuid

from config import InferenceState, GLOBAL_MAX_TOKENS
from utils import calculate_shannon_entropy, extract_answer, is_correct_answer
from data import build_math_prompt

# ===================================================================
# Kalman Filter Implementation for Complexity Tracking
# ===================================================================

class KalmanFilter1D:
    """One-dimensional Kalman filter for latent complexity estimation"""
    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 0.5):
        # State estimate
        self.x = 0.0  # Initial state estimate
        self.P = 1.0  # Initial error covariance
        
        # Model parameters
        self.F = 1.0  # State transition (random walk)
        self.H = 1.0  # Observation model (direct observation)
        self.Q = process_variance  # Process noise variance
        self.R = measurement_variance  # Measurement noise variance
        
    def predict(self):
        """Prediction step"""
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        
    def update(self, z: float):
        """Update step with new measurement z"""
        # Kalman gain
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        
        # State update
        self.x = self.x + K * (z - self.H * self.x)
        
        # Covariance update
        self.P = (1 - K * self.H) * self.P
        
    def filter(self, measurement: float) -> float:
        """Complete filtering step: predict then update"""
        self.predict()
        self.update(measurement)
        return self.x

# ===================================================================
# EWMA Control Chart for Anomaly Detection
# ===================================================================

class EWMAControlChart:
    """Exponentially Weighted Moving Average control chart for anomaly detection"""
    def __init__(self, lambda_param: float = 0.2, initial_mean: float = 0.0, 
                 process_std: float = 1.0):
        self.lambda_param = lambda_param
        self.mu = initial_mean  # EWMA statistic
        self.process_std = process_std
        self.t = 0  # Time step counter
        
    def update(self, value: float) -> Tuple[float, float]:
        """Update EWMA statistic and return (statistic, std_dev)"""
        self.t += 1
        self.mu = self.lambda_param * value + (1 - self.lambda_param) * self.mu
        
        # EWMA variance (converges to steady state)
        if self.t > 30:  # Use steady-state formula after convergence
            variance = (self.lambda_param / (2 - self.lambda_param)) * (self.process_std ** 2)
        else:
            # Time-varying variance for initial observations
            variance = (self.lambda_param / (2 - self.lambda_param)) * \
                      (self.process_std ** 2) * (1 - (1 - self.lambda_param) ** (2 * self.t))
        
        std_dev = np.sqrt(variance)
        return self.mu, std_dev
    
    def get_z_score(self, value: float, prev_mu: float) -> float:
        """Calculate Z-score for anomaly detection"""
        _, std_dev = self.update(value)
        if std_dev < 1e-6:
            return 0.0
        return (value - prev_mu) / std_dev

# ===================================================================
# Complete CoTRouter Implementation
# ===================================================================

@dataclass
class CoTRouterState(InferenceState):
    """Extended state for CoTRouter with additional tracking"""
    # Kalman filter for complexity tracking
    kalman_filter: KalmanFilter1D = None
    # EWMA control chart
    ewma_chart: EWMAControlChart = None
    # Control parameters
    L_threshold: float = 3.0  # Adaptive threshold
    commitment_remaining: int = 0  # Remaining commitment steps
    # Budget tracking
    target_llm_ratio: float = 0.3
    # Entropy tracking
    filtered_complexity: List[float] = field(default_factory=list)
    z_scores: List[float] = field(default_factory=list)

class CoTRouter:
    """Complete implementation of the CoTRouter algorithm"""
    
    def __init__(self, 
                 # Signal processing parameters
                 kalman_process_var: float = 1e-4,
                 kalman_measurement_var: float = 0.5,
                 ewma_lambda: float = 0.2,
                 # Control policy parameters
                 target_llm_ratio: float = 0.3,
                 beta_gain: float = 0.01,
                 k_base: int = 3,
                 alpha_severity: float = 0.5,
                 L_min: float = 1.0,
                 L_max: float = 4.0,
                 # Initial values
                 initial_L: float = 3.0):
        
        # Store parameters
        self.kalman_process_var = kalman_process_var
        self.kalman_measurement_var = kalman_measurement_var
        self.ewma_lambda = ewma_lambda
        self.target_llm_ratio = target_llm_ratio
        self.beta_gain = beta_gain
        self.k_base = k_base
        self.alpha_severity = alpha_severity
        self.L_min = L_min
        self.L_max = L_max
        self.initial_L = initial_L
        
    def create_state(self, request_id: str, problem: Dict, prompt: str, 
                    initial_token_ids: List[int]) -> CoTRouterState:
        """Create a new CoTRouter state for a problem"""
        state = CoTRouterState(
            request_id=request_id,
            problem=problem,
            prompt=prompt,
            token_ids=initial_token_ids,
            kalman_filter=KalmanFilter1D(self.kalman_process_var, self.kalman_measurement_var),
            ewma_chart=EWMAControlChart(self.ewma_lambda),
            L_threshold=self.initial_L,
            target_llm_ratio=self.target_llm_ratio
        )
        state.metrics['slm_tokens'] = len(initial_token_ids)
        # Initialize timing metrics
        state.metrics['llm_time'] = 0.0
        state.metrics['slm_time'] = 0.0
        state.metrics['routing_time'] = 0.0  # NEW: routing decision time
        state.metrics['routing_calls'] = 0   # NEW: number of routing decisions
        return state
    
    def process_entropy(self, state: CoTRouterState, entropy: float) -> Tuple[float, float]:
        """Process raw entropy through Kalman filter and EWMA"""
        # Stage 1: Kalman filtering
        filtered_value = state.kalman_filter.filter(entropy)
        state.filtered_complexity.append(filtered_value)
        
        # Stage 2: EWMA anomaly detection
        prev_mu = state.ewma_chart.mu
        _, _ = state.ewma_chart.update(filtered_value)
        z_score = state.ewma_chart.get_z_score(filtered_value, prev_mu)
        state.z_scores.append(z_score)
        
        return filtered_value, z_score
    
    def update_threshold(self, state: CoTRouterState):
        """Self-calibrating threshold update based on budget"""
        total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
        if total_tokens == 0:
            return
            
        actual_ratio = state.metrics['llm_tokens'] / total_tokens
        error = actual_ratio - state.target_llm_ratio
        
        # Proportional control
        state.L_threshold += self.beta_gain * error
        state.L_threshold = np.clip(state.L_threshold, self.L_min, self.L_max)
    
    def compute_commitment_duration(self, z_score: float, threshold: float) -> int:
        """Compute intervention commitment duration based on anomaly severity"""
        severity = max(0, z_score - threshold)
        duration = max(self.k_base, int(np.ceil(self.k_base + self.alpha_severity * severity)))
        return duration
    
    def make_routing_decision(self, state: CoTRouterState, entropy: float) -> str:
        """Main routing decision logic"""
        # Check if we're in commitment mode
        if state.commitment_remaining > 0:
            state.commitment_remaining -= 1
            return 'llm'
        
        # Process entropy signal
        filtered_value, z_score = self.process_entropy(state, entropy)
        
        # Update adaptive threshold
        self.update_threshold(state)
        
        # Check for anomaly
        if z_score > state.L_threshold and state.current_model == 'slm':
            # Trigger LLM intervention
            commitment = self.compute_commitment_duration(z_score, state.L_threshold)
            state.commitment_remaining = commitment - 1  # -1 because current step uses LLM
            state.metrics['llm_interventions'] += 1
            state.metrics['entropy_peaks'] += 1
            return 'llm'
        
        return 'slm'

# ===================================================================
# Baseline Routing Strategies
# ===================================================================

class LLMOnlyRouter:
    """Pure LLM baseline - always use LLM"""
    def __init__(self):
        pass
    
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        return 'llm'

class SLMOnlyRouter:
    """Pure SLM baseline - always use SLM"""
    def __init__(self):
        pass
    
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        return 'slm'

class RandomRouter:
    """Random routing baseline with configurable probability"""
    def __init__(self, llm_probability: float = 0.5):
        self.llm_probability = llm_probability
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        return 'llm' if np.random.random() < self.llm_probability else 'slm'

class FixedThresholdRouter:
    """Fixed threshold routing without adaptation"""
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.ewma_chart = EWMAControlChart()
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        prev_mu = self.ewma_chart.mu
        z_score = self.ewma_chart.get_z_score(entropy, prev_mu)
        return 'llm' if z_score > self.threshold else 'slm'

class PeriodicRouter:
    """Periodic switching between models"""
    def __init__(self, period: int = 10):
        self.period = period
        self.counter = 0
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        self.counter += 1
        # Use LLM every 'period' tokens
        return 'llm' if (self.counter % self.period) == 0 else 'slm'

# ===================================================================
# Unified Runner for CoTRouter and Baselines (with Timing)
# ===================================================================

def run_cotrouter_experiment(runner, dataset_name: str, problems: List[Dict], 
                           router, method_name: str, batch_size: int = 1):
    """Run experiment with specified router - WITH DETAILED TIMING METRICS INCLUDING ROUTING"""
    print(f"\nRunning {method_name} on {dataset_name}" + "\n" + "="*50)
    
    runner.results[dataset_name] = runner.results.get(dataset_name, {})
    current_run_results = runner.results[dataset_name][method_name] = {
        'correct': 0, 'total': 0, 'metrics': [], 'predictions': [],
        # Aggregate timing metrics
        'total_llm_time': 0.0,
        'total_slm_time': 0.0,
        'total_routing_time': 0.0,  # NEW: total routing decision time
        'total_routing_calls': 0,   # NEW: total routing decisions made
        'total_wall_time': 0.0,
    }
    
    start_time = time.time()
    
    total_batches = (len(problems) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        print(f"\n{'='*20} Processing Batch {batch_idx + 1}/{total_batches} {'='*20}")
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        # Initialize states
        active_states = []
        for p in batch_problems:
            prompt_text = build_math_prompt(p['question'])
            initial_token_ids = runner.slm.tokenizer.encode(prompt_text)
            
            # Create appropriate state based on router type
            if isinstance(router, CoTRouter):
                state = router.create_state(
                    request_id=str(uuid.uuid4()),
                    problem=p,
                    prompt=prompt_text,
                    initial_token_ids=initial_token_ids
                )
            else:
                state = InferenceState(
                    request_id=str(uuid.uuid4()),
                    problem=p,
                    prompt=prompt_text,
                    token_ids=initial_token_ids
                )
                state.metrics['slm_tokens'] = len(initial_token_ids)
                # Initialize timing metrics for baseline routers
                state.metrics['llm_time'] = 0.0
                state.metrics['slm_time'] = 0.0
                state.metrics['routing_time'] = 0.0  # NEW
                state.metrics['routing_calls'] = 0   # NEW
            
            state.current_model = 'slm'
            active_states.append(state)
        
        # Main generation loop
        iteration = 0
        while active_states:
            iteration += 1
            if iteration > GLOBAL_MAX_TOKENS:
                print("Reached max iterations, ending batch.")
                break
            
            # Group states by current model
            llm_states = [s for s in active_states if s.current_model == 'llm']
            slm_states = [s for s in active_states if s.current_model == 'slm']
            
            # Process LLM states with timing
            if llm_states:
                llm_prompts = [s.prompt + s.full_generation for s in llm_states]
                
                # ===== LLM TIMING START =====
                llm_start = time.time()
                llm_results = runner.llm.generate_one_token(prompts=llm_prompts)
                llm_elapsed = time.time() - llm_start
                # ===== LLM TIMING END =====
                
                # Distribute time evenly among states in batch
                time_per_state = llm_elapsed / len(llm_states)
                
                for state, (token, token_id, logprobs) in zip(llm_states, llm_results):
                    # Accumulate LLM time for this state
                    state.metrics['llm_time'] += time_per_state
                    
                    if token_id == runner.llm.eos_token_id:
                        state.is_finished = True
                        continue
                    
                    state.full_generation += token
                    state.metrics['llm_tokens'] += 1
                    
                    # Calculate entropy for next decision
                    entropy = calculate_shannon_entropy(logprobs, runner.llm.vocab_size)
                    state.metrics['entropy_history'].append(entropy)
                    
                    # ===== ROUTING DECISION TIMING START =====
                    routing_start = time.time()
                    next_model = router.make_routing_decision(state, entropy)
                    routing_elapsed = time.time() - routing_start
                    state.metrics['routing_time'] += routing_elapsed
                    state.metrics['routing_calls'] += 1
                    # ===== ROUTING DECISION TIMING END =====
                    
                    state.current_model = next_model
            
            # Process SLM states with timing
            if slm_states:
                slm_prompts = [s.prompt + s.full_generation for s in slm_states]
                
                # ===== SLM TIMING START =====
                slm_start = time.time()
                slm_results = runner.slm.generate_one_token(prompts=slm_prompts)
                slm_elapsed = time.time() - slm_start
                # ===== SLM TIMING END =====
                
                # Distribute time evenly among states in batch
                time_per_state = slm_elapsed / len(slm_states)
                
                for state, (token, token_id, logprobs) in zip(slm_states, slm_results):
                    # Accumulate SLM time for this state
                    state.metrics['slm_time'] += time_per_state
                    
                    if token_id == runner.slm.eos_token_id:
                        state.is_finished = True
                        continue
                    
                    state.full_generation += token
                    state.metrics['slm_tokens'] += 1
                    
                    # Calculate entropy
                    entropy = calculate_shannon_entropy(logprobs, runner.slm.vocab_size)
                    state.metrics['entropy_history'].append(entropy)
                    
                    # ===== ROUTING DECISION TIMING START =====
                    routing_start = time.time()
                    next_model = router.make_routing_decision(state, entropy)
                    routing_elapsed = time.time() - routing_start
                    state.metrics['routing_time'] += routing_elapsed
                    state.metrics['routing_calls'] += 1
                    # ===== ROUTING DECISION TIMING END =====
                    
                    state.current_model = next_model
            
            # Check for finished states
            remaining_states = []
            for state in active_states:
                current_total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
                if state.is_finished or current_total_tokens >= GLOBAL_MAX_TOKENS:
                    # Evaluate result
                    predicted_answer = extract_answer(state.full_generation)
                    is_correct = is_correct_answer(predicted_answer, state.problem['answer'])
                    current_run_results['total'] += 1
                    if is_correct:
                        current_run_results['correct'] += 1
                    
                    state.metrics['total_tokens'] = current_total_tokens
                    state.metrics['total_time'] = state.metrics['llm_time'] + state.metrics['slm_time']
                    state.metrics['total_time_with_routing'] = state.metrics['total_time'] + state.metrics['routing_time']
                    
                    # Accumulate to run totals
                    current_run_results['total_llm_time'] += state.metrics['llm_time']
                    current_run_results['total_slm_time'] += state.metrics['slm_time']
                    current_run_results['total_routing_time'] += state.metrics['routing_time']
                    current_run_results['total_routing_calls'] += state.metrics['routing_calls']
                    
                    current_run_results['metrics'].append(state.metrics)
                    current_run_results['predictions'].append({
                        'question': state.problem['question'],
                        'predicted': state.full_generation,
                        'predicted_answer': predicted_answer,
                        'ground_truth': state.problem['answer'],
                        'correct': is_correct
                    })
                    
                    print(f"  Request {state.request_id[:6]} finished. "
                          f"Correct: {is_correct}, Total Tokens: {current_total_tokens}, "
                          f"LLM Time: {state.metrics['llm_time']:.3f}s, "
                          f"SLM Time: {state.metrics['slm_time']:.3f}s, "
                          f"Routing Time: {state.metrics['routing_time']*1000:.3f}ms")  # Show in ms
                else:
                    remaining_states.append(state)
            
            active_states = remaining_states
    
    # Calculate total wall time
    total_wall_time = time.time() - start_time
    current_run_results['total_wall_time'] = total_wall_time
    
    # Calculate per-problem averages
    num_problems = len(problems)
    avg_wall_time = total_wall_time / num_problems if num_problems else 0
    
    for metric_record in current_run_results['metrics']:
        metric_record['wall_time'] = avg_wall_time
    
    # Print detailed timing summary
    print(f"\n{'='*70}")
    print(f"{method_name} TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Wall Time:    {total_wall_time:.2f}s")
    print(f"Total LLM Time:     {current_run_results['total_llm_time']:.2f}s")
    print(f"Total SLM Time:     {current_run_results['total_slm_time']:.2f}s")
    print(f"Total Routing Time: {current_run_results['total_routing_time']*1000:.3f}ms ({current_run_results['total_routing_calls']} calls)")
    print(f"Overhead Time:      {total_wall_time - current_run_results['total_llm_time'] - current_run_results['total_slm_time'] - current_run_results['total_routing_time']:.2f}s")
    print(f"")
    print(f"Average per problem:")
    print(f"  Wall Time:    {avg_wall_time:.3f}s")
    print(f"  LLM Time:     {current_run_results['total_llm_time']/num_problems:.3f}s")
    print(f"  SLM Time:     {current_run_results['total_slm_time']/num_problems:.3f}s")
    print(f"  Routing Time: {current_run_results['total_routing_time']/num_problems*1000:.3f}ms")
    print(f"  Routing Calls: {current_run_results['total_routing_calls']/num_problems:.1f}")
    
    # Calculate average time per routing call
    if current_run_results['total_routing_calls'] > 0:
        avg_time_per_routing_call = current_run_results['total_routing_time'] / current_run_results['total_routing_calls']
        print(f"  Avg Time per Routing Call: {avg_time_per_routing_call*1e6:.2f}μs")
    
    # Print summary statistics
    if current_run_results['metrics']:
        avg_llm_tokens = np.mean([m['llm_tokens'] for m in current_run_results['metrics']])
        avg_slm_tokens = np.mean([m['slm_tokens'] for m in current_run_results['metrics']])
        avg_total_tokens = np.mean([m['total_tokens'] for m in current_run_results['metrics']])
        llm_ratio = (avg_llm_tokens / avg_total_tokens * 100) if avg_total_tokens > 0 else 0
        accuracy = current_run_results['correct'] / current_run_results['total'] * 100
        
        avg_llm_time = np.mean([m['llm_time'] for m in current_run_results['metrics']])
        avg_slm_time = np.mean([m['slm_time'] for m in current_run_results['metrics']])
        avg_routing_time = np.mean([m['routing_time'] for m in current_run_results['metrics']])
        avg_total_time = np.mean([m['total_time'] for m in current_run_results['metrics']])
        
        print(f"\n{'='*70}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"LLM Token Ratio: {llm_ratio:.1f}%")
        print(f"Avg LLM Tokens: {avg_llm_tokens:.1f}")
        print(f"Avg SLM Tokens: {avg_slm_tokens:.1f}")
        print(f"Avg Total Tokens: {avg_total_tokens:.1f}")
        print(f"")
        print(f"Avg LLM Time per problem:     {avg_llm_time:.3f}s")
        print(f"Avg SLM Time per problem:     {avg_slm_time:.3f}s")
        print(f"Avg Routing Time per problem: {avg_routing_time*1000:.3f}ms")
        print(f"Avg Total Time per problem:   {avg_total_time:.3f}s")
        
        # Time ratio
        llm_time_ratio = (current_run_results['total_llm_time'] / total_wall_time * 100) if total_wall_time > 0 else 0
        slm_time_ratio = (current_run_results['total_slm_time'] / total_wall_time * 100) if total_wall_time > 0 else 0
        routing_time_ratio = (current_run_results['total_routing_time'] / total_wall_time * 100) if total_wall_time > 0 else 0
        print(f"")
        print(f"LLM Time Ratio:     {llm_time_ratio:.1f}%")
        print(f"SLM Time Ratio:     {slm_time_ratio:.1f}%")
        print(f"Routing Time Ratio: {routing_time_ratio:.4f}%")
        
        # Routing overhead analysis
        print(f"\n{'='*70}")
        print(f"ROUTING OVERHEAD ANALYSIS")
        print(f"{'='*70}")
        total_inference_time = current_run_results['total_llm_time'] + current_run_results['total_slm_time']
        routing_overhead_pct = (current_run_results['total_routing_time'] / total_inference_time * 100) if total_inference_time > 0 else 0
        print(f"Total Inference Time (LLM+SLM): {total_inference_time:.2f}s")
        print(f"Total Routing Time:             {current_run_results['total_routing_time']*1000:.3f}ms")
        print(f"Routing Overhead vs Inference:  {routing_overhead_pct:.4f}%")
