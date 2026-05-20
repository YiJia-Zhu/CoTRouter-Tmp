# cotrouter.py
"""
Complete implementation of CoTRouter: Adaptive Token Routing via Entropy Feedback
for Efficient Chain-of-Thought Reasoning
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import os
import uuid

from config import InferenceState, GLOBAL_MAX_TOKENS
from utils import calculate_shannon_entropy, evaluate_problem_answer
from data import build_prompt

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
# Unified Runner for CoTRouter and Baselines
# ===================================================================

CONTEXT_CHECK_MARGIN = int(os.getenv('COTROUTER_CONTEXT_CHECK_MARGIN', '3000'))
CONTEXT_GENERATION_RESERVE = int(os.getenv('COTROUTER_CONTEXT_GENERATION_RESERVE', '1'))


def _filter_context_ready_states(states, model, model_name: str) -> List[InferenceState]:
    """Drop states that are too close to the selected model's vLLM context limit."""
    ready_states = []
    context_stop_at = GLOBAL_MAX_TOKENS - CONTEXT_GENERATION_RESERVE

    for state in states:
        current_total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
        if current_total_tokens < GLOBAL_MAX_TOKENS - CONTEXT_CHECK_MARGIN:
            ready_states.append(state)
            continue

        context_tokens = len(model.tokenizer.encode(state.prompt + state.full_generation))
        state.metrics[f'{model_name}_context_tokens'] = context_tokens
        if context_tokens >= context_stop_at:
            state.is_finished = True
            state.metrics['context_limit_stop_model'] = model_name
            state.metrics['context_limit_tokens'] = context_tokens
        else:
            ready_states.append(state)

    return ready_states

def run_cotrouter_experiment(runner, dataset_name: str, problems: List[Dict],
                           router, method_name: str, batch_size: int = None):
    """Run experiment with specified router"""
    print(f"\nRunning {method_name} on {dataset_name}" + "\n" + "="*50)
    if batch_size is None:
        batch_size = int(getattr(runner, 'cotrouter_batch_size', 150))
    print(f"CoTRouter batch size: {batch_size}")
    
    runner.results[dataset_name] = runner.results.get(dataset_name, {})
    current_run_results = runner.results[dataset_name][method_name] = {
        'correct': 0, 'total': 0, 'metrics': [], 'predictions': []
    }
    
    import time
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
            prompt_text = build_prompt(p)
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
            
            # Process LLM states
            if llm_states:
                llm_states = _filter_context_ready_states(llm_states, runner.llm, 'llm')
            if llm_states:
                llm_prompts = [s.prompt + s.full_generation for s in llm_states]
                llm_results = runner.llm.generate_one_token(prompts=llm_prompts)
                
                for state, (token, token_id, logprobs) in zip(llm_states, llm_results):
                    if token_id == runner.llm.eos_token_id:
                        state.is_finished = True
                        continue
                    
                    state.full_generation += token
                    state.metrics['llm_tokens'] += 1
                    
                    # Calculate entropy for next decision
                    entropy = calculate_shannon_entropy(logprobs, runner.llm.vocab_size)
                    state.metrics['entropy_history'].append(entropy)
                    
                    # Make routing decision for next token
                    next_model = router.make_routing_decision(state, entropy)
                    state.current_model = next_model
            
            # Process SLM states
            if slm_states:
                slm_states = _filter_context_ready_states(slm_states, runner.slm, 'slm')
            if slm_states:
                slm_prompts = [s.prompt + s.full_generation for s in slm_states]
                slm_results = runner.slm.generate_one_token(prompts=slm_prompts)
                
                for state, (token, token_id, logprobs) in zip(slm_states, slm_results):
                    if token_id == runner.slm.eos_token_id:
                        state.is_finished = True
                        continue
                    
                    state.full_generation += token
                    state.metrics['slm_tokens'] += 1
                    
                    # Calculate entropy
                    entropy = calculate_shannon_entropy(logprobs, runner.slm.vocab_size)
                    state.metrics['entropy_history'].append(entropy)
                    
                    # Make routing decision for next token
                    next_model = router.make_routing_decision(state, entropy)
                    state.current_model = next_model
            
            # Check for finished states
            remaining_states = []
            for state in active_states:
                current_total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
                if state.is_finished or current_total_tokens >= GLOBAL_MAX_TOKENS:
                    # Evaluate result
                    predicted_answer, is_correct = evaluate_problem_answer(
                        state.full_generation,
                        state.problem,
                    )
                    current_run_results['total'] += 1
                    if is_correct:
                        current_run_results['correct'] += 1
                    
                    state.metrics['total_tokens'] = current_total_tokens
                    current_run_results['metrics'].append(state.metrics)
                    current_run_results['predictions'].append({
                        'question': state.problem['question'],
                        'predicted': state.full_generation,
                        'predicted_answer': predicted_answer,
                        'ground_truth': state.problem['answer'],
                        'correct': is_correct
                    })
                    
                    print(f"  Request {state.request_id[:6]} finished. "
                          f"Correct: {is_correct}, Total Tokens: {current_total_tokens}")
                else:
                    remaining_states.append(state)
            
            active_states = remaining_states
    
    # Calculate timing
    total_wall_time = time.time() - start_time
    avg_time = total_wall_time / len(problems) if problems else 0
    for metric_record in current_run_results['metrics']:
        metric_record['wall_time'] = avg_time
    
    print(f"\n{method_name} finished in {total_wall_time:.2f}s. "
          f"Average time per problem: {avg_time:.2f}s")
    
    # Print summary statistics
    if current_run_results['metrics']:
        avg_llm_tokens = np.mean([m['llm_tokens'] for m in current_run_results['metrics']])
        avg_total_tokens = np.mean([m['total_tokens'] for m in current_run_results['metrics']])
        llm_ratio = (avg_llm_tokens / avg_total_tokens * 100) if avg_total_tokens > 0 else 0
        accuracy = current_run_results['correct'] / current_run_results['total'] * 100
        
        print(f"\nSummary - Accuracy: {accuracy:.1f}%, LLM Token Ratio: {llm_ratio:.1f}%")
