"""
Hyperparameter sweep script for CoTRouter
Modified version with detailed timing metrics including routing decision time
Added baseline support: LLM Only and SLM Only

# 只跑 baselines
python www_rebuttal_baseline.py --dataset GSM8K --num_samples 100 --run_baselines --skip_sweep

# 跑 baselines + sweep
python www_rebuttal_baseline.py --dataset GSM8K --num_samples 100 --run_baselines

# 只跑 sweep（原有行为）
python www_rebuttal_baseline.py --dataset GSM8K --num_samples 100

python www_rebuttal_baseline.py --dataset GSM8K --num_samples 100
python www_rebuttal_baseline.py --dataset MATH --num_samples 100
python www_rebuttal_baseline.py --dataset AIME --num_samples 100
python www_rebuttal_baseline.py --dataset ARC --num_samples 100
python www_rebuttal_baseline.py --dataset CommonsenseQA --num_samples 100

# Run with baselines
python www_rebuttal_baseline.py --dataset GSM8K --num_samples 100 --run_baselines
python www_rebuttal_baseline.py --dataset MATH --num_samples 100 --run_baselines --skip_sweep
"""
import os
import itertools
import json
from datetime import datetime
import numpy as np

from config import ModelConfig
from data import DatasetManager
from cotrouter_runner import CoTRouterBenchmarkRunner
from www_rebuttal_cotrouter import (
    CoTRouter, 
    LLMOnlyRouter, 
    SLMOnlyRouter, 
    run_cotrouter_experiment
)


def run_baseline_experiments(runner, dataset_name, problems, num_samples=100):
    """Run LLM Only and SLM Only baseline experiments with timing metrics"""
    
    # Use subset for faster experimentation
    if num_samples and len(problems) > num_samples:
        problems = problems[:num_samples]
    
    results = {}
    
    # ============================================================
    # LLM Only Baseline
    # ============================================================
    print("\n" + "="*70)
    print("RUNNING LLM ONLY BASELINE")
    print("="*70)
    
    llm_router = LLMOnlyRouter()
    run_cotrouter_experiment(
        runner, dataset_name, problems, llm_router, 
        method_name="LLM_Only", batch_size=1
    )
    
    if "LLM_Only" in runner.results[dataset_name]:
        data = runner.results[dataset_name]["LLM_Only"]
        if data['total'] > 0:
            metrics = data['metrics']
            results['LLM_Only'] = {
                'method': 'LLM_Only',
                'accuracy': data['correct'] / data['total'],
                'llm_ratio': 1.0,  # Always 100% LLM
                'avg_total_tokens': np.mean([m['total_tokens'] for m in metrics]),
                'avg_llm_tokens': np.mean([m['llm_tokens'] for m in metrics]),
                'avg_slm_tokens': np.mean([m['slm_tokens'] for m in metrics]),
                # Timing metrics
                'total_wall_time': data['total_wall_time'],
                'total_llm_time': data['total_llm_time'],
                'total_slm_time': data['total_slm_time'],
                'total_routing_time': data['total_routing_time'],
                'total_routing_calls': data['total_routing_calls'],
                'avg_llm_time': np.mean([m['llm_time'] for m in metrics]),
                'avg_slm_time': np.mean([m['slm_time'] for m in metrics]),
                'avg_routing_time': np.mean([m['routing_time'] for m in metrics]),
                'avg_total_time': np.mean([m['total_time'] for m in metrics]),
            }
    
    # ============================================================
    # SLM Only Baseline
    # ============================================================
    print("\n" + "="*70)
    print("RUNNING SLM ONLY BASELINE")
    print("="*70)
    
    slm_router = SLMOnlyRouter()
    run_cotrouter_experiment(
        runner, dataset_name, problems, slm_router, 
        method_name="SLM_Only", batch_size=1
    )
    
    if "SLM_Only" in runner.results[dataset_name]:
        data = runner.results[dataset_name]["SLM_Only"]
        if data['total'] > 0:
            metrics = data['metrics']
            results['SLM_Only'] = {
                'method': 'SLM_Only',
                'accuracy': data['correct'] / data['total'],
                'llm_ratio': 0.0,  # Always 0% LLM
                'avg_total_tokens': np.mean([m['total_tokens'] for m in metrics]),
                'avg_llm_tokens': np.mean([m['llm_tokens'] for m in metrics]),
                'avg_slm_tokens': np.mean([m['slm_tokens'] for m in metrics]),
                # Timing metrics
                'total_wall_time': data['total_wall_time'],
                'total_llm_time': data['total_llm_time'],
                'total_slm_time': data['total_slm_time'],
                'total_routing_time': data['total_routing_time'],
                'total_routing_calls': data['total_routing_calls'],
                'avg_llm_time': np.mean([m['llm_time'] for m in metrics]),
                'avg_slm_time': np.mean([m['slm_time'] for m in metrics]),
                'avg_routing_time': np.mean([m['routing_time'] for m in metrics]),
                'avg_total_time': np.mean([m['total_time'] for m in metrics]),
            }
    
    return results


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
                    'total_routing_time': data['total_routing_time'],
                    'total_routing_calls': data['total_routing_calls'],
                    'avg_llm_time': np.mean([m['llm_time'] for m in metrics]),
                    'avg_slm_time': np.mean([m['slm_time'] for m in metrics]),
                    'avg_routing_time': np.mean([m['routing_time'] for m in metrics]),
                    'avg_routing_calls': np.mean([m['routing_calls'] for m in metrics]),
                    'avg_total_time': np.mean([m['total_time'] for m in metrics]),
                    'llm_time_ratio': data['total_llm_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,
                    'slm_time_ratio': data['total_slm_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,
                    'routing_time_ratio': data['total_routing_time'] / data['total_wall_time'] if data['total_wall_time'] > 0 else 0,
                }
                # Calculate efficiency score
                result['efficiency_score'] = result['accuracy'] / (result['llm_ratio'] + 0.01)
                
                # Calculate routing overhead vs inference time
                total_inference_time = data['total_llm_time'] + data['total_slm_time']
                result['routing_overhead_pct'] = (data['total_routing_time'] / total_inference_time * 100) if total_inference_time > 0 else 0
                result['avg_time_per_routing_call_us'] = (data['total_routing_time'] / data['total_routing_calls'] * 1e6) if data['total_routing_calls'] > 0 else 0
                
                results.append(result)
    
    return results


def print_baseline_comparison(baseline_results, sweep_results=None):
    """Print comparison table between baselines and CoTRouter"""
    
    print("\n" + "="*100)
    print("BASELINE COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Method':<15} {'Accuracy':>10} {'LLM Ratio':>12} {'Avg Tokens':>12} "
          f"{'Wall Time':>12} {'LLM Time':>12} {'SLM Time':>12}")
    print("-"*100)
    
    # Print baselines
    for method_name, result in baseline_results.items():
        print(f"{method_name:<15} {result['accuracy']:>10.4f} {result['llm_ratio']:>12.2%} "
              f"{result['avg_total_tokens']:>12.1f} {result['total_wall_time']:>12.2f}s "
              f"{result['total_llm_time']:>12.2f}s {result['total_slm_time']:>12.2f}s")
    
    # Print best CoTRouter config if available
    if sweep_results and len(sweep_results) > 0:
        best = sweep_results[0]  # Already sorted by efficiency
        print(f"{'CoTRouter':.<15} {best['accuracy']:>10.4f} {best['llm_ratio']:>12.2%} "
              f"{best['avg_total_tokens']:>12.1f} {best['total_wall_time']:>12.2f}s "
              f"{best['total_llm_time']:>12.2f}s {best['total_slm_time']:>12.2f}s")
    
    # Speedup analysis
    if 'LLM_Only' in baseline_results and 'SLM_Only' in baseline_results:
        print("\n" + "="*80)
        print("SPEEDUP ANALYSIS")
        print("="*80)
        
        llm_time = baseline_results['LLM_Only']['total_wall_time']
        slm_time = baseline_results['SLM_Only']['total_wall_time']
        llm_acc = baseline_results['LLM_Only']['accuracy']
        slm_acc = baseline_results['SLM_Only']['accuracy']
        
        print(f"LLM Only Wall Time: {llm_time:.2f}s (Accuracy: {llm_acc:.4f})")
        print(f"SLM Only Wall Time: {slm_time:.2f}s (Accuracy: {slm_acc:.4f})")
        print(f"SLM Speedup vs LLM: {llm_time/slm_time:.2f}x")
        
        if sweep_results and len(sweep_results) > 0:
            best = sweep_results[0]
            cot_time = best['total_wall_time']
            cot_acc = best['accuracy']
            
            print(f"\nCoTRouter Wall Time: {cot_time:.2f}s (Accuracy: {cot_acc:.4f})")
            print(f"CoTRouter Speedup vs LLM: {llm_time/cot_time:.2f}x")
            print(f"CoTRouter Slowdown vs SLM: {cot_time/slm_time:.2f}x")
            
            # Quality-adjusted speedup
            if cot_acc > 0:
                llm_quality_adjusted = llm_time / llm_acc
                cot_quality_adjusted = cot_time / cot_acc
                print(f"\nQuality-Adjusted Time (Time/Accuracy):")
                print(f"  LLM Only: {llm_quality_adjusted:.2f}s per accuracy point")
                print(f"  CoTRouter: {cot_quality_adjusted:.2f}s per accuracy point")
                print(f"  Improvement: {llm_quality_adjusted/cot_quality_adjusted:.2f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoTRouter hyperparameter sweep with timing")
    
    parser.add_argument('--dataset', type=str, default='MATH',
                       choices=['GSM8K', 'MATH', 'AIME','ARC','CommonsenseQA'], help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for sweep')
    parser.add_argument('--output_dir', type=str, default='results/param_sweep',
                       help='Output directory for results')
    parser.add_argument('--run_baselines', action='store_true',
                       help='Run LLM Only and SLM Only baselines')
    parser.add_argument('--skip_sweep', action='store_true',
                       help='Skip hyperparameter sweep (only run baselines)')
    
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
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_results = {}
    sweep_results = []
    
    # ============================================================
    # Run Baselines if requested
    # ============================================================
    if args.run_baselines:
        print("\n" + "#"*80)
        print("# RUNNING BASELINE EXPERIMENTS")
        print("#"*80)
        
        baseline_results = run_baseline_experiments(
            runner, args.dataset, problems, args.num_samples
        )
        
        # Save baseline results
        baseline_file = os.path.join(output_dir, "baseline_results.json")
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\nBaseline results saved to: {baseline_file}")
    
    # ============================================================
    # Run Hyperparameter Sweep
    # ============================================================
    if not args.skip_sweep:
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
        
        print("\n" + "#"*80)
        print("# RUNNING HYPERPARAMETER SWEEP")
        print("#"*80)
        print(f"Running hyperparameter sweep on {args.dataset}")
        print(f"Parameter grid: {param_grid}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Batch size: 1")
        
        sweep_results = run_hyperparameter_sweep(
            runner, args.dataset, problems, param_grid, args.num_samples
        )
        
        # Sort results by efficiency score
        sweep_results.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Save sweep results
        results_file = os.path.join(output_dir, "sweep_results.json")
        with open(results_file, 'w') as f:
            json.dump(sweep_results, f, indent=2)
    
    # Save detailed results
    detailed_results_file = os.path.join(output_dir, "results.json")
    runner.save_detailed_results(detailed_results_file)
    runner.print_summary()
    
    # ============================================================
    # Print Comparison
    # ============================================================
    if args.run_baselines:
        print_baseline_comparison(baseline_results, sweep_results if not args.skip_sweep else None)
    
    # Print top configurations with timing including routing
    if sweep_results:
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