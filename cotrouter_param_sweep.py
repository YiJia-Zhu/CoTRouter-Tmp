# cotrouter_param_sweep.py
"""
Hyperparameter sweep script for CoTRouter
"""
import os
import itertools
import json
from datetime import datetime
import numpy as np

from config import ModelConfig
from data import DatasetManager
from cotrouter_runner import CoTRouterBenchmarkRunner
from cotrouter import CoTRouter, run_cotrouter_experiment

def run_hyperparameter_sweep(runner, dataset_name, problems, param_grid, num_samples=100):
    """Run grid search over hyperparameters"""
    
    # Use subset for faster experimentation
    if num_samples and len(problems) > num_samples:
        problems = problems[:num_samples]
    
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    for i, params in enumerate(itertools.product(*param_values)):
        param_dict = dict(zip(param_names, params))
        
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}: {param_dict}")
        print(f"{'='*60}")
        
        # Create router with current parameters
        router = CoTRouter(**param_dict)
        
        # Run experiment
        method_name = f"Sweep_{i+1}"
        run_cotrouter_experiment(runner, dataset_name, problems, router, method_name)
        
        # Store results
        if method_name in runner.results[dataset_name]:
            data = runner.results[dataset_name][method_name]
            if data['total'] > 0:
                metrics = data['metrics']
                result = {
                    'params': param_dict,
                    'accuracy': data['correct'] / data['total'],
                    'llm_ratio': np.mean([m['llm_tokens']/m['total_tokens'] for m in metrics]),
                    'avg_total_tokens': np.mean([m['total_tokens'] for m in metrics]),
                    'efficiency_score': None
                }
                # Calculate efficiency score
                result['efficiency_score'] = result['accuracy'] / (result['llm_ratio'] + 0.01)
                results.append(result)
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoTRouter hyperparameter sweep")
    
    parser.add_argument('--dataset', type=str, default='MATH',
                       choices=['GSM8K', 'MATH', 'AIME','ARC','CommonsenseQA'], help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for sweep')
    parser.add_argument('--output_dir', type=str, default='results/param_sweep',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize runner
    config = ModelConfig(
        llm_gpu_ids=[0],
        slm_gpu_id=0,
        llm_tensor_parallel_size=1 # 别忘了根据GPU数量调整并行度
    )
    runner = CoTRouterBenchmarkRunner(config)
    runner.initialize_models()
    
    # Load dataset
    data_manager = DatasetManager()
    if args.dataset == 'GSM8K':
        problems = data_manager.load_GSM8K(args.num_samples * 2)  # Load extra for validation
    elif args.dataset == 'MATH':
        problems = data_manager.load_MATH(args.num_samples * 2)
    elif args.dataset == 'AIME':
        problems = data_manager.load_AIME(args.num_samples * 2)
    elif args.dataset == 'ARC':
        problems = data_manager.load_ARC_Challenge(args.num_samples * 2)
    elif args.dataset == 'CommonsenseQA':
        problems = data_manager.load_CommonsenseQA(args.num_samples * 2)
    # Define parameter grid
    '''
    param_grid = {
        'target_llm_ratio': [0.7],
        'ewma_lambda': [0.5, 0.1, 0.2],
        'beta_gain': [0.5, 1.0, 1.5],
        'k_base': [20],
        'alpha_severity': [1.0],
        'kalman_process_var': [0.05, 0.1, 0.2],  # Fixed
        'kalman_measurement_var': [0.5],  # Fixed
        'L_min': [1.0],  # Fixed
        'L_max': [4.0],  # Fixed
        'initial_L': [2.0]  # Fixed
    }
    '''
    
    param_grid = {
        'target_llm_ratio': [0.8], # gsm8k is 0.7 # 0.8
        'ewma_lambda': [0.05], # gsm8k is 0.1 # 0.05
        'beta_gain': [1.0],  # Fixed
        'k_base': [50], # gsm8k is 20 # 50
        'alpha_severity': [2.0], # gsm8k is 1.0 # 2.0
        'kalman_process_var': [0.1],  # Fixed
        'kalman_measurement_var': [0.5],  # Fixed
        'L_min': [1.0],  # Fixed
        'L_max': [4.0],  # Fixed
        'initial_L': [2.0]  # Fixed
    }
    
    # Run sweep
    print(f"Running hyperparameter sweep on {args.dataset}")
    print(f"Parameter grid: {param_grid}")
    
    sweep_results = run_hyperparameter_sweep(
        runner, args.dataset, problems, param_grid, args.num_samples
    )
    
    # Sort results by efficiency score
    sweep_results.sort(key=lambda x: x['efficiency_score'], reverse=True)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(output_dir, "sweep_results.json")
    with open(results_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    results_file = os.path.join(output_dir, "results.json")
    runner.save_detailed_results(results_file)
    runner.print_summary()
    
    # Print top configurations
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY EFFICIENCY SCORE")
    print("="*80)
    
    for i, result in enumerate(sweep_results[:10]):
        print(f"\nRank {i+1}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  LLM Ratio: {result['llm_ratio']:.4f}")
        print(f"  Efficiency Score: {result['efficiency_score']:.2f}")
        print(f"  Parameters: {result['params']}")
    
    # Find best configuration for different target ratios
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS PER TARGET RATIO")
    print("="*80)
    
    for target_ratio in [0.5, 0.6, 0.7, 0.8,]:
        filtered = [r for r in sweep_results if r['params']['target_llm_ratio'] == target_ratio]
        if filtered:
            best = max(filtered, key=lambda x: x['efficiency_score'])
            print(f"\nTarget Ratio {target_ratio}:")
            print(f"  Best Accuracy: {best['accuracy']:.4f}")
            print(f"  Actual LLM Ratio: {best['llm_ratio']:.4f}")
            print(f"  Parameters: {best['params']}")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
