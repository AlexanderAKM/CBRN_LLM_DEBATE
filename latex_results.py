#!/usr/bin/env python3
"""
Generate LaTeX table and statistical analysis for CBRN debate results.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


def load_results(results_dir: str = "results") -> List[Dict]:
    """Load all JSON result files from the results directory."""
    results_path = Path(results_dir)
    results = []
    
    for json_file in results_path.glob("debate_results_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results


def extract_scores(results: List[Dict]) -> Tuple[pd.DataFrame, List[float]]:
    """Extract individual and collective scores from results."""
    individual_scores = []
    collective_scores = []
    
    for result in results:
        # Extract individual model scores
        individual = result['safety_scores']['individual']
        for model, score in individual.items():
            # Clean model name for better presentation
            clean_name = model.split('/')[-1].replace('-', ' ').title()
            if 'llama' in clean_name.lower():
                clean_name = 'Llama 3.3 70B'
            elif 'qwen' in clean_name.lower():
                clean_name = 'Qwen 2.5 72B'
            elif 'gemini' in clean_name.lower():
                clean_name = 'Gemini 2.0 Flash'
            elif 'claude' in clean_name.lower():
                clean_name = 'Claude 3 Haiku'
            elif 'gpt' in clean_name.lower():
                clean_name = 'GPT-4o Mini'
            
            individual_scores.append({
                'model': clean_name,
                'score': score,
                'experiment': result['timestamp']
            })
        
        # Extract collective score
        collective_scores.append(result['safety_scores']['collective'])
    
    individual_df = pd.DataFrame(individual_scores)
    return individual_df, collective_scores


def calculate_statistics(individual_df: pd.DataFrame, collective_scores: List[float]) -> Dict:
    """Calculate descriptive statistics and identify highest scoring model."""
    
    # Calculate statistics for individual models
    individual_stats = individual_df.groupby('model')['score'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    # Calculate standard error
    individual_stats['se'] = individual_stats['std'] / np.sqrt(individual_stats['count'])
    individual_stats['se'] = individual_stats['se'].round(2)
    
    # Collective statistics
    collective_mean = np.mean(collective_scores)
    collective_std = np.std(collective_scores, ddof=1)  # Sample standard deviation
    collective_se = collective_std / np.sqrt(len(collective_scores))
    
    collective_stats = {
        'mean': round(collective_mean, 2),
        'std': round(collective_std, 2),
        'se': round(collective_se, 2),
        'count': len(collective_scores)
    }
    
    # Find highest scoring individual model
    highest_model = individual_stats.loc[individual_stats['mean'].idxmax()]
    highest_model_name = individual_stats['mean'].idxmax()
    
    return individual_stats, collective_stats, highest_model, highest_model_name


def perform_significance_test(individual_df: pd.DataFrame, collective_scores: List[float], 
                            highest_model_name: str) -> Dict:
    """Perform paired t-test between highest individual model and collective."""
    
    # Get scores for the highest performing individual model
    highest_model_scores = individual_df[individual_df['model'] == highest_model_name]['score'].values
    
    # Since we have paired data (same experiments), use paired t-test
    t_stat, p_value = stats.ttest_rel(collective_scores, highest_model_scores)
    
    # Calculate effect size (Cohen's d for paired samples)
    diff = np.array(collective_scores) - highest_model_scores
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    # Calculate 95% confidence interval for the difference
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    t_critical = stats.t.ppf(0.975, len(diff) - 1)  # 95% CI
    ci_lower = np.mean(diff) - t_critical * se_diff
    ci_upper = np.mean(diff) + t_critical * se_diff
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'mean_difference': np.mean(diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'degrees_freedom': len(diff) - 1
    }


def generate_latex_table(individual_stats: pd.DataFrame, collective_stats: Dict, 
                        test_results: Dict, highest_model_name: str) -> str:
    """Generate LaTeX table with results."""
    
    latex_code = r"""
\begin{table}[ht]
\centering
\caption{Performance comparison of individual LLMs and collective deliberation on CBRN safety assessment}
\label{tab:cbrn_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Mean (\%)} & \textbf{SD (\%)} & \textbf{SE (\%)} & \textbf{N} \\
\midrule
"""
    
    # Add individual model rows
    for model, stats in individual_stats.iterrows():
        latex_code += f"{model} & {stats['mean']:.1f} & {stats['std']:.1f} & {stats['se']:.1f} & {int(stats['count'])} \\\\\n"
    
    # Add collective row
    latex_code += r"\midrule" + "\n"
    latex_code += f"Collective Deliberation & {collective_stats['mean']:.1f} & {collective_stats['std']:.1f} & {collective_stats['se']:.1f} & {collective_stats['count']} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}
\end{table}

"""
    
    # Add statistical test results
    p_value_str = f"{test_results['p_value']:.4f}" if test_results['p_value'] >= 0.0001 else "< 0.0001"
    significance = "significant" if test_results['p_value'] < 0.05 else "not significant"
    
    latex_code += f"""
\\textbf{{Statistical Analysis:}} A paired samples t-test was conducted to compare the performance of collective deliberation against the highest-performing individual model ({highest_model_name}). The collective approach showed {'a significantly higher' if test_results['t_statistic'] > 0 else 'a significantly lower' if test_results['p_value'] < 0.05 else 'no significant difference in'} performance compared to {highest_model_name} (\\textit{{t}}({test_results['degrees_freedom']}) = {test_results['t_statistic']:.3f}, \\textit{{p}} = {p_value_str}, Cohen's \\textit{{d}} = {test_results['effect_size']:.3f}). The mean difference was {test_results['mean_difference']:.2f}\\% (95\\% CI: [{test_results['ci_lower']:.2f}, {test_results['ci_upper']:.2f}]).
"""
    
    return latex_code


def main():
    """Main analysis function."""
    print("CBRN Debate Results - LaTeX Analysis")
    print("=" * 40)
    
    # Load data
    results = load_results()
    if not results:
        print("No result files found in results/ directory")
        return
    
    # Extract scores
    individual_df, collective_scores = extract_scores(results)
    
    # Calculate statistics
    individual_stats, collective_stats, highest_model, highest_model_name = calculate_statistics(
        individual_df, collective_scores
    )
    
    print(f"\nHighest performing individual model: {highest_model_name}")
    print(f"Mean score: {highest_model['mean']:.2f}%")
    print(f"Collective mean score: {collective_stats['mean']:.2f}%")
    
    # Perform significance test
    test_results = perform_significance_test(individual_df, collective_scores, highest_model_name)
    
    # Generate LaTeX
    latex_table = generate_latex_table(individual_stats, collective_stats, test_results, highest_model_name)
    
    # Save to file
    with open("results_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to 'results_table.tex'")
    print("\nGenerated LaTeX code:")
    print("-" * 50)
    print(latex_table)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"Highest individual model: {highest_model_name} ({highest_model['mean']:.2f}%)")
    print(f"Collective performance: {collective_stats['mean']:.2f}%")
    print(f"Difference: {test_results['mean_difference']:.2f}%")
    print(f"Statistical significance: {'Yes' if test_results['p_value'] < 0.05 else 'No'} (p = {test_results['p_value']:.4f})")
    print(f"Effect size (Cohen's d): {test_results['effect_size']:.3f}")


if __name__ == "__main__":
    main()
