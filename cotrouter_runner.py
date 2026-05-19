# cotrouter_runner.py
"""
Extended benchmark runner that includes CoTRouter and all baseline methods
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import csv

from runner import BenchmarkRunner
from cotrouter import (
    CoTRouter, RandomRouter, FixedThresholdRouter, PeriodicRouter,
    run_cotrouter_experiment
)

class CoTRouterBenchmarkRunner(BenchmarkRunner):
    """Extended runner with CoTRouter implementations"""
    
    def run_cotrouter_benchmark(self, dataset_name: str, problems: List[Dict],
                               target_ratios: List[float] = [0.2, 0.3, 0.4, 0.5]):
        """Run CoTRouter with different target LLM ratios"""
        for ratio in target_ratios:
            router = CoTRouter(
                target_llm_ratio=ratio,
                # Default parameters from paper
                kalman_process_var=0.1,
                kalman_measurement_var=0.5,
                ewma_lambda=0.1,
                beta_gain=1.0,
                k_base=20,
                alpha_severity=1.0,
                L_min=1.0,
                L_max=4.0,
                initial_L=2.0
            )
            method_name = f'CoTRouter-R{int(ratio*100)}'
            run_cotrouter_experiment(self, dataset_name, problems, router, method_name)
    
    def run_random_baseline(self, dataset_name: str, problems: List[Dict],
                           llm_probabilities: List[float] = [0.3, 0.5]):
        """Run random routing baseline with different probabilities"""
        for prob in llm_probabilities:
            router = RandomRouter(llm_probability=prob)
            method_name = f'Random-P{int(prob*100)}'
            run_cotrouter_experiment(self, dataset_name, problems, router, method_name)
    
    def run_fixed_threshold_baseline(self, dataset_name: str, problems: List[Dict],
                                   thresholds: List[float] = [2.0, 3.0, 4.0]):
        """Run fixed threshold baseline with different thresholds"""
        for threshold in thresholds:
            router = FixedThresholdRouter(threshold=threshold)
            method_name = f'FixedThreshold-T{threshold}'
            run_cotrouter_experiment(self, dataset_name, problems, router, method_name)
    
    def run_periodic_baseline(self, dataset_name: str, problems: List[Dict],
                            periods: List[int] = [5, 10, 20]):
        """Run periodic switching baseline with different periods"""
        for period in periods:
            router = PeriodicRouter(period=period)
            method_name = f'Periodic-N{period}'
            run_cotrouter_experiment(self, dataset_name, problems, router, method_name)
    
    def run_cotrouter_ablation(self, dataset_name: str, problems: List[Dict]):
        """Run ablation studies for CoTRouter components"""
        base_ratio = 0.7
        
        # Full model
        router = CoTRouter(target_llm_ratio=base_ratio)
        run_cotrouter_experiment(self, dataset_name, problems, router, 'CoTRouter-Full')
        
        # Without Kalman filter (use raw entropy)
        router_no_kalman = CoTRouter(
            target_llm_ratio=base_ratio,
            kalman_process_var=1e10,  # Very high process variance = no filtering
            kalman_measurement_var=1e-10  # Very low measurement variance
        )
        run_cotrouter_experiment(self, dataset_name, problems, router_no_kalman, 'CoTRouter-NoKalman')
        
        # Without adaptive threshold (fixed beta=0)
        router_no_adaptive = CoTRouter(
            target_llm_ratio=base_ratio,
            beta_gain=0.0  # No adaptation
        )
        run_cotrouter_experiment(self, dataset_name, problems, router_no_adaptive, 'CoTRouter-NoAdaptive')
        
        # Without commitment (k_base=1, alpha=0)
        router_no_commit = CoTRouter(
            target_llm_ratio=base_ratio,
            k_base=1,
            alpha_severity=0.0
        )
        run_cotrouter_experiment(self, dataset_name, problems, router_no_commit, 'CoTRouter-NoCommitment')
    
    def run_parameter_sensitivity(self, dataset_name: str, problems: List[Dict]):
        """Run parameter sensitivity analysis"""
        base_config = {
            'target_llm_ratio': 0.7,
            'kalman_process_var': 0.1,
            'kalman_measurement_var': 0.5,
            'ewma_lambda': 0.1,
            'beta_gain': 1.0,
            'k_base': 20,
            'alpha_severity': 1.0
        }
        
        # Test different values for each parameter
        param_ranges = {
            'ewma_lambda': [0.1, 0.2, 0.3, 0.5],
            'beta_gain': [0.001, 0.005, 0.01, 0.02, 0.05],
            'k_base': [1, 3, 5, 10],
            'alpha_severity': [0.1, 0.3, 0.5, 1.0],
        }
        
        for param_name, values in param_ranges.items():
            for value in values:
                config = base_config.copy()
                config[param_name] = value
                
                router = CoTRouter(**config)
                method_name = f'Sensitivity-{param_name}-{value}'
                run_cotrouter_experiment(self, dataset_name, problems[:100], router, method_name)  # Use subset for sensitivity
    
    def save_detailed_results(self, filename: str):
        """Save results with additional analysis"""
        # First save raw results
        self.save_results(filename)
        
        # Create detailed analysis
        analysis = {}
        for dataset, methods in self.results.items():
            analysis[dataset] = {}
            for method, data in methods.items():
                if not data['total']:
                    continue
                
                metrics = data['metrics']
                analysis[dataset][method] = {
                    'accuracy': data['correct'] / data['total'],
                    'total_problems': data['total'],
                    'avg_total_tokens': np.mean([m['total_tokens'] for m in metrics]),
                    'avg_llm_tokens': np.mean([m['llm_tokens'] for m in metrics]),
                    'avg_slm_tokens': np.mean([m['slm_tokens'] for m in metrics]),
                    'llm_token_ratio': np.mean([m['llm_tokens']/m['total_tokens'] for m in metrics]),
                    'avg_llm_interventions': np.mean([m.get('llm_interventions', 0) for m in metrics]),
                    'avg_wall_time': np.mean([m['wall_time'] for m in metrics]),
                }
                
                # Add entropy statistics if available
                if metrics and 'entropy_history' in metrics[0]:
                    all_entropies = []
                    for m in metrics:
                        all_entropies.extend(m['entropy_history'])
                    
                    analysis[dataset][method]['entropy_stats'] = {
                        'mean': np.mean(all_entropies),
                        'std': np.std(all_entropies),
                        'max': np.max(all_entropies),
                        'percentile_95': np.percentile(all_entropies, 95)
                    }
        
        # Save analysis
        analysis_file = filename.replace('.json', '_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_file}")
        
        # Create CSV for easy plotting
        csv_file = filename.replace('.json', '_summary.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Method', 'Accuracy', 'LLM_Token_Ratio', 
                           'Avg_Total_Tokens', 'Avg_Wall_Time', 'Cost_Efficiency_Score'])
            
            for dataset, methods in analysis.items():
                for method, stats in methods.items():
                    efficiency_score = stats['accuracy'] / (stats['llm_token_ratio'] + 0.01)  # Avoid div by 0
                    writer.writerow([
                        dataset, method, 
                        f"{stats['accuracy']:.4f}",
                        f"{stats['llm_token_ratio']:.4f}",
                        f"{stats['avg_total_tokens']:.1f}",
                        f"{stats['avg_wall_time']:.2f}",
                        f"{efficiency_score:.2f}"
                    ])
        print(f"Summary CSV saved to {csv_file}")
