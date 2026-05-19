"""
Hyperparameter sweep script for CoTRouter
Modified version with detailed timing metrics including routing decision time

python www_rebuttal_overhead_test.py --dataset GSM8K --num_samples 100
python www_rebuttal_overhead_test.py --dataset MATH --num_samples 100
python www_rebuttal_overhead_test.py --dataset AIME --num_samples 100
python www_rebuttal_overhead_test.py --dataset ARC --num_samples 100
python www_rebuttal_overhead_test.py --dataset CommonsenseQA --num_samples 100
"""
import os
import itertools
import json
from datetime import datetime
import numpy as np

from config import ModelConfig
from data import DatasetManager
from cotrouter_runner import CoTRouterBenchmarkRunner
from www_rebuttal_cotrouter import CoTRouter, run_cotrouter_experiment


def run_hyperparameter_sweep(runner, dataset_name, problems, param_grid, num_samples=100):
    """Run grid search over hyperparameters with timing metrics including routing"""
    
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
        
        # Run experiment with batch_size=1 for accurate timing
        method_name = f"Sweep_{i+1}"
        run_cotrouter_experiment(runner, dataset_name, problems, router, method_name, batch_size=1)
        
        # Store results with timing
        if method_name in runner.results[dataset_name]:
            data = runner.results[dataset_name][method_name]
            if data['total'] > 0:
                metrics = data['metrics']
                result = {
                    'params': param_dict,
                    'accuracy': data['correct'] / data['total'],
                    'llm_ratio': np.mean([m['llm_tokens']/m['total_tokens'] for m in metrics]),
                    'avg_total_tokens': np.mean([m['total_tokens'] for m in metrics]),
                    'efficiency_score': None,
                    # Timing metrics
                    'total_wall_time': data['total_wall_time'],
                    'total_llm_time': data['total_llm_time'],
                    'total_slm_time': data['total_slm_time'],
                    'total_routing_time': data['total_routing_time'],  # NEW
                    'total_routing_calls': data['total_routing_calls'],  # NEW
                    'avg_llm_time': np.mean([m['llm_time'] for m in metrics]),
                    'avg_slm_time': np.mean([m['slm_time'] for m in metrics]),
                    'avg_routing_time': np.mean([m['routing_time'] for m in metrics]),  # NEW
                    'avg_routing_calls': np.mean([m['routing_calls'] for m in metrics]),  # NEW
                    'avg_total_time': np.mean([m['total_time'] for m in metrics]),
                    'llm_time_ratio': data['total_llm_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,
                    'slm_time_ratio': data['total_slm_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,
                    'routing_time_ratio': data['total_routing_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,  # NEW
                }
                # Calculate efficiency score
                result['efficiency_score'] = result['accuracy'] / (result['llm_ratio'] + 0.01)
                
                # Calculate routing overhead vs inference time
                total_inference_time = data['total_llm_time'] + data['total_slm_time']
                result['routing_overhead_pct'] = (data['total_routing_time'] / total_inference_time * 100) if total_inference_time > 0 else 0
                result['avg_time_per_routing_call_us'] = (data['total_routing_time'] / data['total_routing_calls'] * 1e6) if data['total_routing_calls'] > 0 else 0
                
                results.append(result)
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoTRouter hyperparameter sweep with timing")
    
    parser.add_argument('--dataset', type=str, default='MATH',
                       choices=['GSM8K', 'MATH', 'AIME','ARC','CommonsenseQA'], help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for sweep')
    parser.add_argument('--output_dir', type=str, default='results/param_sweep',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize runner
    config = ModelConfig(
        llm_gpu_ids=[1],
        slm_gpu_id=1,
        llm_tensor_parallel_size=1
    )
    runner = CoTRouterBenchmarkRunner(config)
    runner.initialize_models()
    
    # Load dataset
    data_manager = DatasetManager()
    if args.dataset == 'GSM8K':
        problems = data_manager.load_GSM8K(args.num_samples * 2)
    elif args.dataset == 'MATH':
        problems = data_manager.load_MATH(args.num_samples * 2)
    elif args.dataset == 'AIME':
        problems = data_manager.load_AIME(args.num_samples * 2)
    elif args.dataset == 'ARC':
        problems = data_manager.load_ARC_Challenge(args.num_samples * 2)
    elif args.dataset == 'CommonsenseQA':
        problems = data_manager.load_CommonsenseQA(args.num_samples * 2)
    
    # Define parameter grid
    param_grid = {
        'target_llm_ratio': [0.8],
        'ewma_lambda': [0.05],
        'beta_gain': [1.0],
        'k_base': [50],
        'alpha_severity': [2.0],
        'kalman_process_var': [0.1],
        'kalman_measurement_var': [0.5],
        'L_min': [1.0],
        'L_max': [4.0],
        'initial_L': [2.0]
    }
    
    # Run sweep
    print(f"Running hyperparameter sweep on {args.dataset}")
    print(f"Parameter grid: {param_grid}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: 4")
    
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
    
    detailed_results_file = os.path.join(output_dir, "results.json")
    runner.save_detailed_results(detailed_results_file)
    runner.print_summary()
    
    # Print top configurations with timing including routing
    print("\n" + "="*80)
    print("TOP CONFIGURATIONS WITH TIMING METRICS (INCLUDING ROUTING)")
    print("="*80)
    
    for i, result in enumerate(sweep_results[:10]):
        print(f"\nRank {i+1}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  LLM Ratio: {result['llm_ratio']:.4f}")
        print(f"  Efficiency Score: {result['efficiency_score']:.2f}")
        print(f"  --- TIMING ---")
        print(f"  Total Wall Time: {result['total_wall_time']:.2f}s")
        print(f"  Total LLM Time: {result['total_llm_time']:.2f}s ({result['llm_time_ratio']*100:.1f}%)")
        print(f"  Total SLM Time: {result['total_slm_time']:.2f}s ({result['slm_time_ratio']*100:.1f}%)")
        print(f"  Total Routing Time: {result['total_routing_time']*1000:.3f}ms ({result['routing_time_ratio']*100:.4f}%)")
        print(f"  --- ROUTING DETAILS ---")
        print(f"  Total Routing Calls: {result['total_routing_calls']}")
        print(f"  Avg Time per Routing Call: {result['avg_time_per_routing_call_us']:.2f}μs")
        print(f"  Routing Overhead vs Inference: {result['routing_overhead_pct']:.4f}%")
        print(f"  --- PER PROBLEM ---")
        print(f"  Avg LLM Time/problem: {result['avg_llm_time']:.3f}s")
        print(f"  Avg SLM Time/problem: {result['avg_slm_time']:.3f}s")
        print(f"  Avg Routing Time/problem: {result['avg_routing_time']*1000:.3f}ms")
        print(f"  Avg Total Time/problem: {result['avg_total_time']:.3f}s")
        print(f"  Parameters: {result['params']}")
    
    # Print timing summary table
    print("\n" + "="*100)
    print("TIMING SUMMARY TABLE")
    print("="*100)
    print(f"{'Config':<10} {'Accuracy':>10} {'LLM Ratio':>12} {'Wall Time':>12} {'LLM Time':>12} {'SLM Time':>12} {'Routing':>12} {'Overhead%':>10}")
    print("-"*100)
    
    for i, result in enumerate(sweep_results):
        print(f"{'Sweep_'+str(i+1):<10} {result['accuracy']:>10.4f} {result['llm_ratio']:>12.4f} "
              f"{result['total_wall_time']:>12.2f}s {result['total_llm_time']:>12.2f}s {result['total_slm_time']:>12.2f}s "
              f"{result['total_routing_time']*1000:>10.3f}ms {result['routing_overhead_pct']:>9.4f}%")
    
    # Print routing overhead summary
    print("\n" + "="*80)
    print("ROUTING OVERHEAD SUMMARY")
    print("="*80)
    if sweep_results:
        avg_routing_overhead = np.mean([r['routing_overhead_pct'] for r in sweep_results])
        avg_time_per_call = np.mean([r['avg_time_per_routing_call_us'] for r in sweep_results])
        total_routing_calls = sum([r['total_routing_calls'] for r in sweep_results])
        print(f"Average Routing Overhead vs Inference: {avg_routing_overhead:.4f}%")
        print(f"Average Time per Routing Call: {avg_time_per_call:.2f}μs")
        print(f"Total Routing Calls across all experiments: {total_routing_calls}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()