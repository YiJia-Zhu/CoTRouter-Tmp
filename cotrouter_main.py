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

SUPPORTED_DATASETS = [
    'GSM8K', 'MATH', 'AIME', 'AIME24', 'ARC',
    'OpenBookQA', 'CommonsenseQA', 'HumanEval'
]

DATASET_ALIASES = {
    'gsm8k': 'GSM8K',
    'math': 'MATH',
    'aime': 'AIME',
    'aime24': 'AIME24',
    'aime_24': 'AIME24',
    'arc': 'ARC',
    'arc_challenge': 'ARC',
    'openbookqa': 'OpenBookQA',
    'openbook_qa': 'OpenBookQA',
    'commonsenseqa': 'CommonsenseQA',
    'commonsense_qa': 'CommonsenseQA',
    'humaneval': 'HumanEval',
    'human_eval': 'HumanEval',
    'openai_humaneval': 'HumanEval',
    'new': 'new',
    'all_new': 'new',
    'both': 'both',
    'all': 'all',
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.replace('-', '_').replace(' ', '_').lower()
    return DATASET_ALIASES.get(key, dataset_name)


def dataset_run_list(dataset_name: str):
    if dataset_name == 'new':
        return ['ARC', 'OpenBookQA', 'CommonsenseQA', 'HumanEval']
    if dataset_name == 'all':
        return SUPPORTED_DATASETS
    if dataset_name == 'both':
        return ['GSM8K', 'MATH', 'AIME', 'ARC', 'CommonsenseQA']
    if dataset_name not in SUPPORTED_DATASETS:
        valid = ', '.join(SUPPORTED_DATASETS + ['new', 'all'])
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {valid}")
    return [dataset_name]


def load_dataset_by_name(data_manager: DatasetManager, dataset_name: str, num_samples):
    if dataset_name == 'GSM8K':
        return 'GSM8K', data_manager.load_GSM8K(num_samples)
    if dataset_name == 'MATH':
        return 'MATH', data_manager.load_MATH(num_samples)
    if dataset_name == 'AIME':
        return 'AIME', data_manager.load_AIME(num_samples)
    if dataset_name == 'AIME24':
        return 'AIME24', data_manager.load_AIME(num_samples)
    if dataset_name == 'ARC':
        return 'ARC', data_manager.load_ARC_Challenge(num_samples)
    if dataset_name == 'OpenBookQA':
        return 'OpenBookQA', data_manager.load_OpenBookQA(num_samples)
    if dataset_name == 'CommonsenseQA':
        return 'CommonsenseQA', data_manager.load_CommonsenseQA(num_samples)
    if dataset_name == 'HumanEval':
        return 'HumanEval', data_manager.load_HumanEval(num_samples)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Run CoTRouter experiments")
    
    # Model configuration
    parser.add_argument('--llm_path', type=str, 
                       default="/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B",
                       help='Path to the LLM model')
    parser.add_argument('--slm_path', type=str,
                       default="/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B",
                       help='Path to the SLM model')
    
    # Experiment selection
    parser.add_argument('--dataset', type=str, default='AIME',
                       help='Dataset to use. Supports GSM8K, MATH, AIME, ARC, OpenBookQA, CommonsenseQA, HumanEval, new, all.')
    parser.add_argument('--experiment', type=str, default='baselines',
                       choices=['main', 'baselines', 'ablation', 'sensitivity', 'all'],
                       help='Which experiments to run')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use (None for full dataset)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results. Defaults to a unique timestamped directory.')
    parser.add_argument('--llm_gpu_memory_utilization', type=float, default=0.71,
                       help='vLLM GPU memory utilization for the LLM')
    parser.add_argument('--slm_gpu_memory_utilization', type=float, default=0.20,
                       help='vLLM GPU memory utilization for the SLM')
    
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
    dataset_name = normalize_dataset_name(args.dataset)
    
    # Configure models
    config = ModelConfig(
        llm_path=args.llm_path,
        slm_path=args.slm_path,
        llm_gpu_ids=args.llm_gpus,
        slm_gpu_id=args.slm_gpu,
        llm_tensor_parallel_size=len(args.llm_gpus),
        slm_tensor_parallel_size=1,
        llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
        slm_gpu_memory_utilization=args.slm_gpu_memory_utilization,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or f"results/cotrouter_{timestamp}_{os.getpid()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize runner
    runner = CoTRouterBenchmarkRunner(config)
    runner.initialize_models()
    
    # Load datasets
    data_manager = DatasetManager()
    datasets = {}

    for requested_dataset in dataset_run_list(dataset_name):
        loaded_name, problems = load_dataset_by_name(
            data_manager,
            requested_dataset,
            args.num_samples,
        )
        datasets[loaded_name] = problems

    # Run experiments
    for dataset_name, problems in datasets.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name}")
        print(f"{'='*60}")
        
        # Main CoTRouter experiments
        if args.experiment in ['main', 'all']:
            print("\n--- CoTRouter Main Experiments ---")
            runner.run_cotrouter_benchmark(dataset_name, problems, args.target_ratios)
        
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
