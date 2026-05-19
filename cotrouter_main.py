# cotrouter_main.py
"""
Main script for running CoTRouter experiments and reproducing paper results
"""
import os
import argparse
from datetime import datetime

from config import ModelConfig
from data import DatasetManager
from cotrouter_runner import CoTRouterBenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run CoTRouter experiments")
    
    # Model configuration
    parser.add_argument('--llm_path', type=str, 
                       default="/mnt/8T/xgr/zhuyijia/huggingface_models/DeepSeek-R1-Distill-Qwen-7B",
                       help='Path to the LLM model')
    parser.add_argument('--slm_path', type=str,
                       default="/mnt/8T/xgr/shizhenning/model_weights/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help='Path to the SLM model')
    
    # Experiment selection
    parser.add_argument('--dataset', type=str, default='AIME',
                       choices=['GSM8K', 'MATH', 'AIME', 'AIME24', 'ARC', 'CommonsenseQA', 'both'],
                       help='Dataset to use')
    parser.add_argument('--experiment', type=str, default='baselines',
                       choices=['main', 'baselines', 'ablation', 'sensitivity', 'all'],
                       help='Which experiments to run')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use (None for full dataset)')
    
    # CoTRouter specific parameters
    parser.add_argument('--target_ratios', type=float, nargs='+', 
                       default=[0.8],
                       help='Target LLM token ratios for CoTRouter')
    
    # GPU configuration
    parser.add_argument('--llm_gpus', type=int, nargs='+', default=[0],
                       help='GPU IDs for LLM')
    parser.add_argument('--slm_gpu', type=int, default=0,
                       help='GPU ID for SLM')
    
    args = parser.parse_args()
    
    # Configure models
    config = ModelConfig(
        llm_path=args.llm_path,
        slm_path=args.slm_path,
        llm_gpu_ids=args.llm_gpus,
        slm_gpu_id=args.slm_gpu,
        llm_tensor_parallel_size=len(args.llm_gpus),
        slm_tensor_parallel_size=1,
        llm_gpu_memory_utilization=0.71,
        slm_gpu_memory_utilization=0.20,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/cotrouter_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize runner
    runner = CoTRouterBenchmarkRunner(config)
    runner.initialize_models()
    
    # Load datasets
    data_manager = DatasetManager()
    datasets = {}
    
    if args.dataset in ['GSM8K', 'both']:
        gsm8k_problems = data_manager.load_GSM8K(args.num_samples)
        datasets['GSM8K'] = gsm8k_problems
        
    if args.dataset in ['MATH', 'both']:
        math_problems = data_manager.load_MATH(args.num_samples)
        datasets['MATH'] = math_problems
    
    if args.dataset in ['AIME', 'AIME24', 'both']:
        math_problems = data_manager.load_AIME(args.num_samples)
        datasets['AIME24' if args.dataset == 'AIME24' else 'AIME'] = math_problems
    
    if args.dataset in ['ARC', 'both']:
        math_problems = data_manager.load_ARC_Challenge(args.num_samples)
        datasets['ARC'] = math_problems

    if args.dataset in ['CommonsenseQA', 'both']:
        math_problems = data_manager.load_CommonsenseQA(args.num_samples)
        datasets['CommonsenseQA'] = math_problems

    # Run experiments
    for dataset_name, problems in datasets.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name}")
        print(f"{'='*60}")
        
        # Main CoTRouter experiments
        # if args.experiment in ['main', 'all']:
        #     print("\n--- CoTRouter Main Experiments ---")
        #     runner.run_cotrouter_benchmark(dataset_name, problems, args.target_ratios)
        
        # Baseline comparisons
        if args.experiment in ['baselines', 'all']:
            print("\n--- Baseline Experiments ---")
            # Pure models
            runner.run_baseline('LLM', dataset_name, problems)
            runner.run_baseline('SLM', dataset_name, problems)
            
            # Other baselines
            # runner.run_random_baseline(dataset_name, problems, [0.7]) # 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            #runner.run_fixed_threshold_baseline(dataset_name, problems, [2.0, 3.0, 4.0])
            #runner.run_periodic_baseline(dataset_name, problems, [5, 10, 20])
            
            # Existing DE-Cascade baseline
            #runner.run_de_cascade_benchmark(dataset_name, problems)
        
        # # Ablation studies
        # if args.experiment in ['ablation', 'all']:
        #     print("\n--- Ablation Studies ---")
        #     # Use subset for ablation to save time
        #     ablation_problems = problems
        #     runner.run_cotrouter_ablation(dataset_name, ablation_problems)
        
        # # Parameter sensitivity
        # if args.experiment in ['sensitivity', 'all']:
        #     print("\n--- Parameter Sensitivity Analysis ---")
        #     # Use smaller subset for sensitivity analysis
        #     sensitivity_problems = problems
        #     runner.run_parameter_sensitivity(dataset_name, sensitivity_problems)
    
    # Save results
    results_file = os.path.join(output_dir, "results.json")
    runner.save_detailed_results(results_file)
    runner.print_summary()
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nTo generate plots, run:")
    print(f"python plot_results.py --results_dir {output_dir}")

if __name__ == "__main__":
    main()
