# generate_latex_tables.py
"""
Generate LaTeX tables from CoTRouter results for paper
"""
import json
import os
import numpy as np

def load_analysis(results_dir):
    """Load analysis results"""
    analysis_file = os.path.join(results_dir, "results_analysis.json")
    with open(analysis_file, 'r') as f:
        return json.load(f)

def generate_main_results_table(analysis, dataset):
    """Generate main comparison table"""
    methods = [
        ('Pure SLM', 'SLM-Only'),
        ('Pure LLM', 'LLM-Only'),
        ('Random (p=0.5)', 'Random-P50'),
        ('Fixed Threshold', 'FixedThreshold-T3.0'),
        ('CoTRouter (R=0.2)', 'CoTRouter-R20'),
        ('CoTRouter (R=0.3)', 'CoTRouter-R30'),
        ('CoTRouter (R=0.4)', 'CoTRouter-R40'),
    ]
    
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Performance comparison on {dataset} dataset}}
\\label{{tab:{dataset.lower()}_results}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{LLM Ratio (\\%)}} & \\textbf{{Avg Tokens}} & \\textbf{{Efficiency}} \\\\
\\midrule
"""
    
    for display_name, method_key in methods:
        if method_key in analysis[dataset]:
            stats = analysis[dataset][method_key]
            accuracy = stats['accuracy'] * 100
            llm_ratio = stats['llm_token_ratio'] * 100
            avg_tokens = stats['avg_total_tokens']
            efficiency = stats['accuracy'] / (stats['llm_token_ratio'] + 0.01)
            
            latex += f"{display_name} & {accuracy:.1f} & {llm_ratio:.1f} & {avg_tokens:.0f} & {efficiency:.2f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

def generate_ablation_table(analysis, dataset):
    """Generate ablation study table"""
    ablation_methods = [
        ('Full Model', 'CoTRouter-Full'),
        ('w/o Kalman Filter', 'CoTRouter-NoKalman'),
        ('w/o Adaptive Threshold', 'CoTRouter-NoAdaptive'),
        ('w/o Commitment', 'CoTRouter-NoCommitment'),
    ]
    
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Ablation study on {dataset} dataset}}
\\label{{tab:{dataset.lower()}_ablation}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{LLM Ratio (\\%)}} & \\textbf{{$\\Delta$ Accuracy}} \\\\
\\midrule
"""
    
    full_accuracy = None
    for display_name, method_key in ablation_methods:
        if method_key in analysis[dataset]:
            stats = analysis[dataset][method_key]
            accuracy = stats['accuracy'] * 100
            llm_ratio = stats['llm_token_ratio'] * 100
            
            if full_accuracy is None:
                full_accuracy = accuracy
                delta = "--"
            else:
                delta = f"{accuracy - full_accuracy:+.1f}"
            
            latex += f"{display_name} & {accuracy:.1f} & {llm_ratio:.1f} & {delta} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

def generate_pareto_table(analysis, dataset):
    """Generate Pareto frontier table"""
    target_ratios = [10, 20, 30, 40, 50, 60]
    
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{CoTRouter performance at different target ratios on {dataset}}}
\\label{{tab:{dataset.lower()}_pareto}}
\\begin{{tabular}}{{ccccc}}
\\toprule
\\textbf{{Target Ratio (\\%)}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{Actual Ratio (\\%)}} & \\textbf{{Deviation}} & \\textbf{{Efficiency}} \\\\
\\midrule
"""
    
    for ratio in target_ratios:
        method_key = f'CoTRouter-R{ratio}'
        if method_key in analysis[dataset]:
            stats = analysis[dataset][method_key]
            accuracy = stats['accuracy'] * 100
            actual_ratio = stats['llm_token_ratio'] * 100
            deviation = actual_ratio - ratio
            efficiency = stats['accuracy'] / (stats['llm_token_ratio'] + 0.01)
            
            latex += f"{ratio} & {accuracy:.1f} & {actual_ratio:.1f} & {deviation:+.1f} & {efficiency:.2f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--output_file', type=str, default='tables.tex',
                       help='Output LaTeX file')
    
    args = parser.parse_args()
    
    # Load analysis
    analysis = load_analysis(args.results_dir)
    
    # Generate all tables
    with open(args.output_file, 'w') as f:
        f.write("% Generated LaTeX tables for CoTRouter paper\n\n")
        
        for dataset in analysis.keys():
            f.write(f"% {dataset} Results\n")
            f.write(generate_main_results_table(analysis, dataset))
            f.write("\n\n")
            f.write(generate_ablation_table(analysis, dataset))
            f.write("\n\n")
            f.write(generate_pareto_table(analysis, dataset))
            f.write("\n\n")
    
    print(f"LaTeX tables saved to: {args.output_file}")
    
    # Also generate a summary for the abstract
    print("\n" + "="*60)
    print("Key Results for Paper Abstract:")
    print("="*60)
    
    for dataset in analysis.keys():
        if 'CoTRouter-R30' in analysis[dataset] and 'LLM-Only' in analysis[dataset]:
            cotrouter = analysis[dataset]['CoTRouter-R30']
            llm_only = analysis[dataset]['LLM-Only']
            
            print(f"\n{dataset}:")
            print(f"  - CoTRouter (R=30%) accuracy: {cotrouter['accuracy']*100:.1f}%")
            print(f"  - Pure LLM accuracy: {llm_only['accuracy']*100:.1f}%")
            print(f"  - Relative accuracy: {cotrouter['accuracy']/llm_only['accuracy']*100:.1f}%")
            print(f"  - LLM token reduction: {(1-cotrouter['llm_token_ratio'])*100:.1f}%")
            print(f"  - Efficiency gain: {cotrouter['accuracy']/(cotrouter['llm_token_ratio']+0.01):.1f}x")

if __name__ == "__main__":
    main()
